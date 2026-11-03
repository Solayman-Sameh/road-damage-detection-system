"""Readable Kaggle evaluation cells; evaluation only, no training or downloads."""

# %% 1. Imports and common, identical measurement settings
from pathlib import Path
from itertools import cycle
import json, math, statistics, time
import yaml
import pandas as pd
import torch
from ultralytics import YOLO

IMG_SIZE, DEVICE, CUDA_DEVICE, EVAL_BATCH = 640, 0, "cuda:0", 1
MAP_CONF, FIXED_CONF, NMS_IOU, MAX_DET = 0.001, 0.25, 0.7, 300
WARMUP_CALLS, TIMED_CALLS = 20, 100


# %% 2. Resolve the held-out image list and matching YOLO label paths
def test_images(data_yaml):
    """Resolve the frozen test split from a standard Ultralytics data YAML."""
    cfg_path = Path(data_yaml).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", cfg_path.parent))
    root = root if root.is_absolute() else (cfg_path.parent / root).resolve()
    split = cfg.get("test")
    if split is None:
        raise ValueError("Dataset YAML has no held-out test split.")
    items = split if isinstance(split, list) else [split]
    images = []
    for item in items:
        p = Path(item)
        p = p if p.is_absolute() else root / p
        if p.is_dir():
            images.extend(x for x in p.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
        elif p.suffix.lower() == ".txt":
            for line in p.read_text().splitlines():
                if line.strip():
                    item = Path(line.strip())
                    images.append(item if item.is_absolute() else (p.parent / item).resolve())
        elif p.is_file():
            images.append(p)
        else:
            raise FileNotFoundError(p)
    images = sorted({p.resolve() for p in images})
    if len(images) < TIMED_CALLS:
        raise ValueError(f"Need at least {TIMED_CALLS} held-out images; found {len(images)}.")
    return cfg, images


# %% Resolve image to label file
def label_file(image_path):
    """Map the last images path component to labels and change its suffix to .txt."""
    parts = list(Path(image_path).parts)
    low = [p.lower() for p in parts]
    if "images" not in low:
        raise ValueError(f"Expected an images/ directory in {image_path}")
    parts[len(low) - 1 - low[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


# %% Read one YOLO label file
def read_yolo_labels(image_path, class_count):
    """Read normalized YOLO boxes as pixel xyxy using the original image size."""
    from PIL import Image
    path = label_file(image_path)
    if not path.exists():
        return []
    with Image.open(image_path) as image:
        width, height = image.size
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5:
            raise ValueError(f"Expected 5 YOLO fields at {path}:{line_number}")
        cls, cx, cy, w, h = values
        if int(cls) != cls or not 0 <= int(cls) < class_count or w <= 0 or h <= 0:
            raise ValueError(f"Invalid class/box at {path}:{line_number}")
        rows.append((int(cls), ((cx-w/2)*width, (cy-h/2)*height,
                                 (cx+w/2)*width, (cy+h/2)*height)))
    return rows


# %% 3. Pure-Python box matching, independently checkable without a model/GPU
def box_iou(a, b):
    """Return xyxy intersection over union, with zero for a zero-area union."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# %% Confidence-greedy, class-aware, one-to-one matching
def match_detections(ground_truth, predictions, iou_threshold=0.5):
    """Greedily match confidence-sorted detections to unused same-class truths."""
    used, matched_ious = set(), []
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c, _ in ground_truth}
    for confidence, pred_class, pred_box in sorted(predictions, key=lambda row: -row[0]):
        candidates = [(j, box_iou(pred_box, box)) for j, (cls, box) in enumerate(ground_truth)
                      if j not in used and cls == pred_class]
        best = max(candidates, key=lambda pair: pair[1], default=None)
        if best is not None and best[1] >= iou_threshold:
            used.add(best[0]); matched_ious.append(best[1]); counts[pred_class]["tp"] += 1
        else:
            counts.setdefault(pred_class, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
    for j, (cls, _) in enumerate(ground_truth):
        if j not in used:
            counts[cls]["fn"] += 1
    return counts, matched_ious


# %% Synthetic CPU-only matching checks; no model or GPU results
def synthetic_cpu_checks():
    """Exercise class-aware, one-to-one matching without importing Torch or YOLO."""
    gt = [(0, (0, 0, 10, 10)), (1, (20, 20, 30, 30))]
    pred = [(0.9, 0, (0, 0, 10, 10)), (0.8, 0, (0, 0, 10, 10)),
            (0.7, 1, (20, 20, 30, 30)), (0.6, 0, (20, 20, 30, 30))]
    counts, matched = match_detections(gt, pred)
    assert counts[0] == {"tp": 1, "fp": 2, "fn": 0} and counts[1]["tp"] == 1
    assert len(matched) == 2 and all(abs(x-1.0) < 1e-12 for x in matched)
    wrong_class, _ = match_detections([(0, (0, 0, 10, 10))], [(0.9, 1, (0, 0, 10, 10))])
    assert wrong_class[0]["fn"] == 1 and wrong_class[1]["fp"] == 1
    assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
    return "synthetic CPU checks passed; no model or GPU results were produced"


# %% 4. Common GPU prediction path and synchronized measurement for each call
def cuda_sync():
    """Synchronize device zero before and after wall-clock GPU measurements."""
    torch.cuda.synchronize(DEVICE)


# %% Prediction with CPU outputs for matching
def predict_cpu_boxes(model, image_path, confidence=FIXED_CONF):
    """Run batch-one FP32 prediction and transfer boxes to CPU for counting."""
    result = model.predict(source=str(image_path), imgsz=IMG_SIZE, device=DEVICE,
        batch=EVAL_BATCH, half=False, rect=False, conf=confidence, iou=NMS_IOU,
        max_det=MAX_DET, nms=True, verbose=False, save=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    return list(zip(boxes.conf.cpu().tolist(), boxes.cls.cpu().int().tolist(),
                    boxes.xyxy.cpu().tolist()))


# %% Timed prediction; synchronization and CPU transfer are inside the boundary
def timed_prediction(model, image_path):
    """Time full predict plus CPU box transfer; report absolute and incremental peaks."""
    cuda_sync()
    allocated_before = torch.cuda.memory_allocated(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.perf_counter()
    results = model.predict(source=str(image_path), imgsz=IMG_SIZE, device=DEVICE,
        batch=EVAL_BATCH, half=False, rect=False, conf=FIXED_CONF, iou=NMS_IOU,
        max_det=MAX_DET, nms=True, verbose=False, save=False)
    result = results[0]
    boxes = result.boxes
    if boxes is not None and len(boxes):
        _ = (boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu())
    del boxes, result, results
    cuda_sync()
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak_alloc = torch.cuda.max_memory_allocated(DEVICE)
    peak_reserved = torch.cuda.max_memory_reserved(DEVICE)
    return elapsed_ms, peak_alloc, peak_reserved, max(0, peak_alloc-allocated_before)


# %% 5. Held-out test mAP and max-F1 per-class precision/recall/F1
def validator_test_metrics(model, data_yaml):
    """Run the single held-out Ultralytics validator pass using common FP32 settings."""
    result = model.val(data=str(data_yaml), split="test", imgsz=IMG_SIZE, device=DEVICE,
        batch=EVAL_BATCH, half=False, rect=False, conf=MAP_CONF, iou=NMS_IOU,
        max_det=MAX_DET, nms=True, plots=False, verbose=False)
    box = result.box
    metrics = {"map50": float(box.map50), "map50_95": float(box.map)}
    if not all(math.isfinite(x) for x in metrics.values()):
        raise ValueError("Validator returned non-finite test mAP.")
    per_class = {}
    for j, class_id in enumerate(box.ap_class_index):
        key = int(class_id)
        per_class[result.names[key]] = {"precision": float(box.p[j]), "recall": float(box.r[j]),
                                        "f1": float(box.f1[j])}
    for row in per_class.values():
        if not all(math.isfinite(x) for x in row.values()):
            raise ValueError("Validator returned non-finite per-class test metrics.")
    return metrics, per_class


# %% 6. Fixed-confidence confusion counts and matched IoU on the exact test split
def fixed_confidence_summary(model, cfg, images):
    """Compute class-aware TP/FP/FN at confidence .25 and matched-box IoU .50."""
    names = cfg["names"]
    names = [names[i] if isinstance(names, list) else names.get(i, names.get(str(i)))
             for i in range(len(names))]
    totals = {str(name): {"tp": 0, "fp": 0, "fn": 0} for name in names}
    all_ious, gt_total, pred_total, matched_total = [], 0, 0, 0
    for image in images:
        gt = read_yolo_labels(image, len(names))
        pred = predict_cpu_boxes(model, image, FIXED_CONF)
        counts, ious = match_detections(gt, pred, 0.5)
        gt_total += len(gt); pred_total += len(pred); matched_total += len(ious)
        all_ious.extend(ious)
        for class_id, values in counts.items():
            row = totals[str(names[class_id])]
            for key in ("tp", "fp", "fn"):
                row[key] += values[key]
    for row in totals.values():
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else None
        row["recall"] = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else None
        p, r = row["precision"], row["recall"]
        row["f1"] = 2*p*r/(p+r) if p is not None and r is not None and p+r else None
    return {"confidence": FIXED_CONF, "match_iou_threshold": 0.5,
        "matching": "class-aware confidence-greedy one-to-one per image",
        "ground_truth": gt_total, "predictions": pred_total, "matched": matched_total,
        "mean_matched_iou": statistics.mean(all_ious) if all_ious else None,
        "ground_truth_coverage": matched_total/gt_total if gt_total else None,
        "prediction_match_rate": matched_total/pred_total if pred_total else None,
        "per_class": totals}


# %% 7. Synchronized 20-warmup / 100-image latency and per-call memory peaks
def benchmark_100(model, same_100_images):
    """Measure 100 batch-one calls over the same 100 test images for every model."""
    if len(same_100_images) != TIMED_CALLS:
        raise ValueError("Benchmark requires the shared, ordered 100-image sample.")
    samples = cycle(same_100_images)
    for _ in range(WARMUP_CALLS):
        predict_cpu_boxes(model, next(samples), FIXED_CONF)
    cuda_sync()
    resident_alloc = torch.cuda.memory_allocated(DEVICE)
    resident_reserved = torch.cuda.memory_reserved(DEVICE)
    durations, alloc_peaks, reserve_peaks, increments = [], [], [], []
    for _ in range(TIMED_CALLS):
        ms, alloc, reserve, increment = timed_prediction(model, next(samples))
        durations.append(ms); alloc_peaks.append(alloc); reserve_peaks.append(reserve)
        increments.append(increment)
    mean_ms = statistics.mean(durations)
    return {"device": torch.cuda.get_device_name(DEVICE), "batch": 1, "imgsz": IMG_SIZE,
        "precision": "FP32", "rect": False, "confidence": FIXED_CONF, "nms": True, "nms_iou": NMS_IOU,
        "max_det": MAX_DET, "warmup_calls": WARMUP_CALLS, "measured_calls": TIMED_CALLS,
        "timing_scope": "synchronized Ultralytics predict through CPU box transfer and result release",
        "latency_mean_ms": mean_ms, "latency_p50_ms": statistics.median(durations),
        "latency_p95_ms": sorted(durations)[math.ceil(0.95*len(durations))-1],
        "fps_from_mean_latency": 1000/mean_ms,
        "resident_gpu_allocated_bytes_after_warmup": resident_alloc,
        "resident_gpu_reserved_bytes_after_warmup": resident_reserved,
        "max_per_call_peak_allocated_bytes": max(alloc_peaks),
        "max_per_call_peak_reserved_bytes": max(reserve_peaks),
        "max_per_call_increment_above_pre_call_allocated_bytes": max(increments),
        "validator_pr_definition": "Ultralytics per-class precision/recall/F1 at its max-F1 confidence; separate from fixed .25 counts",
        "memory_note": "Increment is peak allocated minus allocated immediately before that call; not per-image persistent memory."}


# %% Format concise per-class report rows
def per_class_rows(model_name, names, validator_classes, fixed):
    rows = []
    ordered_names = [names[i] if isinstance(names, list) else names.get(i, names.get(str(i)))
                     for i in range(len(names))]
    for name in ordered_names:
        row = {"model": model_name, "class": str(name)}
        row.update({f"validator_{k}": v for k, v in validator_classes.get(name, {}).items()})
        row.update({f"fixed_{k}": v for k, v in fixed["per_class"].get(str(name), {}).items()})
        rows.append(row)
    return rows


# %% Format concise model comparison row
def comparison_row(model_name, checkpoint, cold_load, gpu_transfer, val, fixed, timing):
    return {"model": model_name, "checkpoint": str(checkpoint.resolve()), "device": timing["device"],
        "checkpoint_bytes": checkpoint.stat().st_size, "cold_checkpoint_load_seconds": cold_load,
        "model_gpu_transfer_seconds": gpu_transfer, "test_map50": val["map50"],
        "test_map50_95": val["map50_95"], "fixed_confidence": FIXED_CONF, "fixed_match_iou": 0.5,
        "matched_iou_mean": fixed["mean_matched_iou"], "ground_truth_coverage": fixed["ground_truth_coverage"],
        "prediction_match_rate": fixed["prediction_match_rate"],
        "resident_gpu_allocated_bytes": timing["resident_gpu_allocated_bytes_after_warmup"],
        "resident_gpu_reserved_bytes": timing["resident_gpu_reserved_bytes_after_warmup"],
        "latency_mean_ms": timing["latency_mean_ms"], "latency_p50_ms": timing["latency_p50_ms"],
        "latency_p95_ms": timing["latency_p95_ms"], "fps": timing["fps_from_mean_latency"],
        "max_per_call_peak_allocated_bytes": timing["max_per_call_peak_allocated_bytes"],
        "max_per_call_peak_reserved_bytes": timing["max_per_call_peak_reserved_bytes"],
        "max_per_call_incremental_allocated_bytes": timing["max_per_call_increment_above_pre_call_allocated_bytes"]}


# %% Evaluate one saved checkpoint identically
def evaluate_one(model_name, checkpoint, data_yaml, cfg, images, timing_images):
    """Evaluate one held-out checkpoint and return compact rows plus evidence."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    start = time.perf_counter(); model = YOLO(str(checkpoint)); cuda_sync()
    cold_load = time.perf_counter() - start
    start = time.perf_counter(); model.model.to(CUDA_DEVICE); cuda_sync()
    gpu_transfer = time.perf_counter() - start
    val, val_classes = validator_test_metrics(model, data_yaml)
    fixed = fixed_confidence_summary(model, cfg, images)
    torch.cuda.empty_cache(); timing = benchmark_100(model, timing_images)
    summary = comparison_row(model_name, checkpoint, cold_load, gpu_transfer, val, fixed, timing)
    rows = per_class_rows(model_name, cfg["names"], val_classes, fixed)
    settings = {"imgsz": IMG_SIZE, "device": DEVICE, "batch": EVAL_BATCH, "precision": "FP32",
        "rect": False, "map_confidence": MAP_CONF, "fixed_confidence": FIXED_CONF,
        "nms": True, "nms_iou": NMS_IOU, "max_det": MAX_DET,
        "yolo26_path": "nms=True selects the one-to-many head with NMS"}
    details = {"settings": settings, "test_metrics": val,
        "fixed_confidence_summary": fixed, "timing_and_memory": timing}
    del model; torch.cuda.empty_cache()
    return summary, rows, details


# %% Evaluate all checkpoints once and save compact comparison/per-class tables
def evaluate_checkpoints(checkpoints, data_yaml, output_dir):
    """Evaluate all three saved checkpoints once and write concise JSON/CSV reports."""
    if len(checkpoints) != 3:
        raise ValueError("Supply exactly three named checkpoints for the fixed comparison.")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU on device 0.")
    cfg, images = test_images(data_yaml)
    timing_images = images[:TIMED_CALLS]
    summaries, per_class, details = [], [], {}
    for model_name, checkpoint in checkpoints.items():
        summary, rows, detail = evaluate_one(model_name, checkpoint, data_yaml, cfg, images, timing_images)
        summaries.append(summary); per_class.extend(rows); details[model_name] = detail
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(summaries)
    classes = pd.DataFrame(per_class)
    comparison.to_csv(out / "test_comparison.csv", index=False)
    comparison.to_json(out / "test_comparison.json", orient="records", indent=2)
    classes.to_csv(out / "test_per_class.csv", index=False)
    classes.to_json(out / "test_per_class.json", orient="records", indent=2)
    (out / "test_measurement_details.json").write_text(json.dumps(details, indent=2, allow_nan=False))
    return comparison, classes
