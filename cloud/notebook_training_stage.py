"""Paste this file's full contents into one Kaggle code cell; no project imports."""
import csv, hashlib, json, math, os, shutil, time
from pathlib import Path
from itertools import cycle
from PIL import Image
import torch, ultralytics, yaml
from ultralytics import YOLO

MODELS = ("yolov8n.pt", "yolo11n.pt", "yolo26n.pt")
IMG, TRAIN_BATCH, EVAL_BATCH = 640, 8, 1
MAP_CONF, PRED_CONF, NMS_IOU, MAX_DET = .001, .25, .7, 300
BASE_EPOCHS, BASE_MIN = 20, 120
AUDIT = None


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    q.replace(p)


def runtime_audit(expected_ultralytics="8.4.191", expected_torch="2.11.0"):
    global AUDIT
    need(ultralytics.__version__ == expected_ultralytics,
         f"Ultralytics version mismatch: expected {expected_ultralytics}, got {ultralytics.__version__}")
    if expected_torch:
        need(torch.__version__.split("+")[0] == expected_torch.split("+")[0],
             f"Torch version mismatch: expected {expected_torch}, got {torch.__version__}")
    need(os.sys.version_info[:2] == (3, 13), f"Expected Python 3.13, got {os.sys.version_info[:2]}")
    need(torch.cuda.is_available() and torch.cuda.device_count() >= 1,
         "A CUDA GPU is required; CPU/TPU is not supported by this stage.")
    AUDIT = {"python": os.sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "torchvision": __import__("importlib.metadata", fromlist=["version"]).version("torchvision"),
            "cuda": torch.version.cuda, "device0": torch.cuda.get_device_name(0),
            "visible_gpus": torch.cuda.device_count(), "device_used": 0,
            "audited_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return AUDIT


def quota_gate(remaining_gpu_minutes, multiplier=1, pilot_min=10, baseline_min=120,
               eval_reserve=30, export_reserve=10):
    need(remaining_gpu_minutes > 0 and multiplier >= 1, "Enter observed GPU quota and multiplier >=1.")
    need(0 < pilot_min <= 20 and 0 < baseline_min <= BASE_MIN, "Pilot/baseline caps exceed limits.")
    wall = pilot_min + 3 * baseline_min + eval_reserve + export_reserve
    required = wall * multiplier
    need(required <= remaining_gpu_minutes,
         f"Quota gate: {wall:g} wall min x {multiplier:g} GPU multiplier = {required:g} GPU min; {remaining_gpu_minutes:g} remain.")
    return {"remaining_gpu_minutes": remaining_gpu_minutes, "multiplier": multiplier,
            "planned_wall_minutes": wall, "planned_gpu_minutes": required,
            "unallocated_gpu_minutes": remaining_gpu_minutes - required,
            "pilot_minutes": pilot_min, "baseline_minutes_per_model": baseline_min,
            "eval_reserve": eval_reserve,
            "export_reserve": export_reserve}


def ledger_read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {"schema": "p2-stage-v1", "stages": {}}


def ledger_start(path, key, allow):
    need(allow, "Pass allow_start=True after reviewing the GPU/quota audit.")
    d = ledger_read(path)
    need(d.get("schema") == "p2-stage-v1", "Unknown ledger; refusing to overwrite it.")
    if d["stages"].get(key, {}).get("status") == "completed":
        return d, False
    d["stages"][key] = {"status": "running", "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(path, d)
    return d, True


def ledger_finish(path, key, result=None, error=None):
    d = ledger_read(path)
    d["stages"].setdefault(key, {}).update({"status": "failed" if error else "completed",
        "finish_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if result is not None: d["stages"][key]["result"] = result
    if error: d["stages"][key]["error"] = error
    write_json(path, d)


def dataset_spec(data_yaml):
    p = Path(data_yaml).resolve(); d = yaml.safe_load(p.read_text())
    names = d.get("names")
    need(isinstance(names, (dict, list)), "Dataset YAML must define ordered names.")
    names = [str(names[i] if isinstance(names, list) else names.get(i, names.get(str(i)))) for i in range(len(names))]
    need(all(x and x != "None" for x in names), "Invalid dataset class order.")
    root = Path(d.get("path", p.parent)); root = root if root.is_absolute() else (p.parent / root).resolve()
    d["names"], d["_root"] = names, root
    return d


def split_images(spec, split):
    val = spec.get(split); need(val is not None, f"Dataset YAML has no {split} split.")
    imgs = []
    for x in val if isinstance(val, list) else [val]:
        p = Path(x); p = p if p.is_absolute() else spec["_root"] / p
        if p.is_dir(): imgs += [z for z in p.rglob("*") if z.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        elif p.suffix.lower() == ".txt":
            imgs += [Path(s.strip()) if Path(s.strip()).is_absolute() else p.parent / s.strip()
                     for s in p.read_text().splitlines() if s.strip()]
        elif p.is_file(): imgs.append(p)
        else: raise RuntimeError(f"Missing split path: {p}")
    imgs = sorted(set(z.resolve() for z in imgs)); need(imgs, f"No {split} images.")
    return imgs


def label_path(image):
    p = Path(image); parts = list(p.parts); low = [x.lower() for x in parts]
    if "images" in low: parts[len(low) - 1 - low[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(p):
    if not p.exists(): return []
    out = []
    for n, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip(): continue
        x = list(map(float, line.split())); need(len(x) == 5, f"Bad YOLO label {p}:{n}")
        c, cx, cy, w, h = x; need(w > 0 and h > 0, f"Invalid box {p}:{n}")
        out.append((int(c), (cx-w/2, cy-h/2, cx+w/2, cy+h/2)))
    return out


def loss_history(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f); cols = r.fieldnames or []; rows = list(r)
    need(rows, f"No completed epoch/loss rows in {path}.")
    tr = [c for c in cols if c.lower().startswith("train/") and "loss" in c.lower()]
    va = [c for c in cols if c.lower().startswith("val/") and "loss" in c.lower()]
    need(tr and va, f"Missing train/validation loss columns in {path}.")
    out = {}
    for c in tr + va:
        xs = [float(r[c]) for r in rows if (r.get(c) or "").strip()]
        need(xs and all(math.isfinite(x) for x in xs), f"Empty/nonfinite loss column {c}.")
        out[c] = {"last": xs[-1], "min": min(xs), "entries": len(xs)}
    return {"actual_epoch_rows": len(rows), "last_epoch_index": rows[-1].get("epoch"),
            "train": {c: out[c] for c in tr}, "validation": {c: out[c] for c in va},
            "all_finite": True, "csv": str(Path(path).resolve())}


def resume_evidence(path):
    # Trusted checkpoint generated by this run; never pass an external file here.
    c = torch.load(path, map_location="cpu", weights_only=False)
    o = c.get("optimizer") if isinstance(c, dict) else None
    need(isinstance(c, dict) and isinstance(c.get("epoch"), int) and c["epoch"] >= 0,
         "Latest checkpoint has no completed epoch.")
    need(isinstance(o, dict) and "state" in o and "param_groups" in o,
         "Latest checkpoint lacks optimizer state and is not resumable.")
    return {"path": str(Path(path).resolve()), "epoch": c["epoch"], "optimizer_present": True,
            "optimizer_state_entries": len(o["state"]), "bytes": Path(path).stat().st_size, "sha256": sha(path)}


def add_caps(model, epochs, minutes):
    s = {"start": None, "resume": None, "save_elapsed": None, "mid_wall_save": False}
    def start(t): s["start"] = time.monotonic()
    def save(t):
        p = Path(t.last); need(p.is_file(), f"Missing pre-strip last checkpoint {p}")
        q = Path(t.wdir) / "last-resumable.pt"; shutil.copy2(p, q); s["resume"] = str(q)
        s["save_elapsed"] = time.monotonic() - s["start"]
    def epoch_end(t):
        if int(t.epoch) + 1 >= epochs: t.stop = True
    def batch_end(t):
        if s["start"] is not None and time.monotonic() - s["start"] >= minutes * 60:
            if not s["mid_wall_save"]:
                need(bool(t.save_model()), "Could not save optimizer checkpoint at wall cap.")
                save(t); s["mid_wall_save"] = True
            t.stop = True
    model.add_callback("on_train_start", start)
    model.add_callback("on_train_epoch_end", epoch_end)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_model_save", save)
    return s


def sync(): torch.cuda.synchronize(0)


def infer(model, image, conf=PRED_CONF):
    return model.predict(source=str(image), imgsz=IMG, device=0, batch=EVAL_BATCH,
        half=False, rect=False, conf=conf, iou=NMS_IOU, max_det=MAX_DET,
        nms=True, verbose=False, save=False)[0]


def reload_probe(checkpoint, image):
    m = YOLO(str(checkpoint)); sync(); r = infer(m, image, MAP_CONF); sync()
    return {"checkpoint": str(Path(checkpoint).resolve()), "bytes": Path(checkpoint).stat().st_size,
            "sha256": sha(checkpoint), "real_validation_image": str(image),
            "reload_inference": "passed", "detections": len(r.boxes)}


def train_one(model_id, stage, data_yaml, out, ledger, epochs, minutes, allow_start,
              seed=2026, train_args=None):
    need(AUDIT is not None and AUDIT["ultralytics"] == ultralytics.__version__, "Run runtime_audit first and keep that audited runtime unchanged.")
    need(model_id in MODELS, "Unknown model."); need(0 < epochs <= BASE_EPOCHS, "Epoch cap exceeded.")
    need(0 < minutes <= BASE_MIN, "Wall cap exceeded.")
    d, run = ledger_start(ledger, stage, allow_start)
    if not run: return d["stages"][stage]["result"]
    try:
        spec = dataset_spec(data_yaml); val_image = split_images(spec, "val")[0]
        torch.cuda.reset_peak_memory_stats(0)
        model = YOLO(model_id); cap = add_caps(model, epochs, minutes)
        extra = dict(train_args or {})
        for k in ("data", "epochs", "time", "imgsz", "device", "batch", "amp", "half", "project", "name", "seed", "resume"):
            need(k not in extra, f"Tuning may not override {k}.")
        result = model.train(data=str(data_yaml), epochs=epochs, time=minutes/60,
            imgsz=IMG, device=0, batch=TRAIN_BATCH, workers=2, project=str(Path(out)/"runs"),
            name=stage.replace("/", "-"), exist_ok=False, save=True, plots=False, val=True,
            seed=seed, deterministic=True, verbose=False, **extra)
        elapsed, peak = cap["save_elapsed"], int(torch.cuda.max_memory_allocated(0))
        need(elapsed is not None, "No checkpoint timestamp for duration evidence.")
        sd = Path(getattr(result, "save_dir", model.trainer.save_dir)); best = sd/"weights"/"best.pt"
        latest = Path(cap["resume"] or ""); need(best.is_file() and latest.is_file(), "Missing best/latest-resumable checkpoint.")
        losses = loss_history(sd/"results.csv"); resume = resume_evidence(latest)
        losses["epoch_cap"] = epochs
        losses["actual_completed_epochs_lower_bound"] = int(float(losses["last_epoch_index"])) + 1
        losses["partial_epoch_possible"] = bool(cap["mid_wall_save"])
        need(losses["actual_completed_epochs_lower_bound"] >= 1, "No completed training epoch recorded.")
        probe = reload_probe(best, val_image)
        vm = YOLO(str(best)); m = vm.val(data=str(data_yaml), split="val", imgsz=IMG, device=0,
            batch=EVAL_BATCH, half=False, rect=False, conf=MAP_CONF, iou=NMS_IOU,
            max_det=MAX_DET, nms=True, plots=False, verbose=False)
        box = m.box; p = list(box.p); r = list(box.r); idx = list(box.ap_class_index)
        pr = {spec["names"][int(i)]: {"precision": float(p[j]), "recall": float(r[j])}
              for j, i in enumerate(idx)}
        val = {"map50": float(box.map50), "map50_95": float(box.map),
               "per_class_precision_recall": pr,
               "pr_confidence": "Ultralytics validator max-F1 summary; separate from fixed .25 matching"}
        need(math.isfinite(val["map50"]) and math.isfinite(val["map50_95"]), "Nonfinite validation mAP.")
        need(all(math.isfinite(v) for x in pr.values() for v in x.values()), "Nonfinite per-class validation precision/recall.")
        result = {"model": model_id, "stage": stage, "best_checkpoint": str(best.resolve()),
            "best_bytes": best.stat().st_size, "best_sha256": sha(best),
            "latest_resumable_checkpoint": str(latest.resolve()), "latest_bytes": latest.stat().st_size,
            "latest_resume_evidence": resume, "reload_probe": probe, "loss_evidence": losses,
            "actual_training_minutes_through_last_checkpoint": elapsed/60,
            "peak_training_gpu_memory_bytes": peak, "wall_cap_checkpointed_mid_epoch": cap["mid_wall_save"],
            "validation": val, "test_used_for_selection": False,
            "settings": {"imgsz": IMG, "batch": TRAIN_BATCH, "seed": seed, "device": torch.cuda.get_device_name(0),
                "epochs_cap": epochs, "wall_minutes_cap": minutes, "deterministic": True}}
        ledger_finish(ledger, stage, result); return result
    except Exception as e:
        ledger_finish(ledger, stage, error=f"{type(e).__name__}: {e}"); raise


def run_pilot(data_yaml, out, ledger, remaining_gpu_minutes, quota_multiplier=1,
              expected_ultralytics="8.4.191", expected_torch="2.11.0", allow_start=False):
    env = runtime_audit(expected_ultralytics, expected_torch)
    need(quota_multiplier >= 1, "Quota multiplier must be >=1.")
    need(remaining_gpu_minutes >= 10 * quota_multiplier, "Insufficient quota for 10-minute pilot cap.")
    write_json(Path(out)/"environment.json", env)
    return train_one(MODELS[0], "pilot", data_yaml, out, ledger, 1, 10, allow_start)


def run_baselines(data_yaml, out, ledger, budget, allow_start=False, epochs=20, seed=2026):
    need(AUDIT is not None, "Run runtime_audit and the pilot first.")
    need(budget.get("planned_gpu_minutes", float("inf")) <= budget.get("remaining_gpu_minutes", -1), "Quota plan is not valid.")
    need(ledger_read(ledger)["stages"].get("pilot",{}).get("status")=="completed", "Complete the single-GPU pilot before baselines.")
    need(0 < epochs <= BASE_EPOCHS, "Baseline cap is at most 20 epochs.")
    stages = {}
    for model_id in MODELS:
        stages[model_id] = train_one(model_id, "base_"+Path(model_id).stem, data_yaml,
            out, ledger, epochs, budget["baseline_minutes_per_model"], allow_start, seed)
    write_json(Path(out)/"baseline_results.json",stages)
    return stages


def iou(a, b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+ab-inter) if aa+ab>inter else 0


def match_summary(model, spec, images, threshold=.25, image_groups=None):
    ious=[]; gt_total=pred_total=0; matched=0
    stats={n:{"tp":0,"fp":0,"fn":0} for n in spec["names"]}; groups={}
    for image in images:
        with Image.open(image) as im: w,h=im.size
        truth=read_labels(label_path(image)); gt_total+=len(truth)
        res=infer(model,image,threshold); boxes=res.boxes; preds=[]
        if boxes is not None and len(boxes):
            preds=sorted(zip(boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.xyxy.cpu().tolist()),reverse=True)
        pred_total+=len(preds); used=set()
        g=(image_groups or {}).get(str(image),(image_groups or {}).get(image.name))
        if g is not None: groups.setdefault(str(g),{"images":0,"ground_truth":0,"predictions":0,"matched":0,"false_positives":0,"false_negatives":0})
        if g is not None: groups[str(g)]["images"]+=1; groups[str(g)]["ground_truth"]+=len(truth); groups[str(g)]["predictions"]+=len(preds)
        for conf, cls, pb in preds:
            choices=[(j,iou(pb,(b[0]*w,b[1]*h,b[2]*w,b[3]*h))) for j,(c,b) in enumerate(truth) if j not in used and int(c)==int(cls)]
            if choices:
                j,v=max(choices,key=lambda x:x[1])
                if v>=.5:
                    used.add(j); ious.append(v); matched+=1
                    stats[spec["names"][int(cls)]]['tp']+=1
                    if g is not None: groups[str(g)]["matched"]+=1
                    continue
            if 0<=int(cls)<len(spec["names"]): stats[spec["names"][int(cls)]]['fp']+=1
            if g is not None: groups[str(g)]["false_positives"]+=1
        for j,(cls,_) in enumerate(truth):
            if j not in used and 0<=cls<len(spec["names"]):
                stats[spec["names"][cls]]['fn']+=1
                if g is not None: groups[str(g)]["false_negatives"]+=1
    for s in stats.values():
        s['precision']=s['tp']/(s['tp']+s['fp']) if s['tp']+s['fp'] else None
        s['recall']=s['tp']/(s['tp']+s['fn']) if s['tp']+s['fn'] else None
    return {"confidence": threshold,"match_iou":.5,"matching":"class-aware greedy one-to-one per image, confidence order",
        "gt_count":gt_total,"prediction_count":pred_total,"matched":matched,
        "mean_matched_iou":sum(ious)/len(ious) if ious else None,
        "gt_coverage":matched/gt_total if gt_total else None,
        "prediction_match_rate":matched/pred_total if pred_total else None,
        "per_class_confusion_counts":stats,"source_group_coverage":groups or "not supplied"}


def timing(model, images, warmup=20, repeats=100):
    need(images and repeats>=5, "Timing requires images and >=5 repeats.")
    seq=cycle(images); vals=[]
    for _ in range(warmup): infer(model,next(seq),PRED_CONF)
    sync()
    torch.cuda.reset_peak_memory_stats(0)
    for _ in range(repeats):
        image=next(seq); sync(); t=time.perf_counter(); infer(model,image,PRED_CONF); sync(); vals.append((time.perf_counter()-t)*1000)
    vals.sort(); mean=sum(vals)/len(vals)
    return {"device":torch.cuda.get_device_name(0),"batch":1,"imgsz":IMG,"half":False,"rect":False,
        "confidence":PRED_CONF,"nms_iou":NMS_IOU,"max_det":MAX_DET,"warmup":warmup,"repeats":repeats,
        "timing_scope":"synchronized Ultralytics predict call incl decode/pre/postprocess",
        "latency_ms_mean":mean,"latency_ms_p50":vals[len(vals)//2],
        "latency_ms_p95":vals[min(len(vals)-1,int(.95*len(vals)))],"fps":1000/mean,
        "peak_inference_gpu_memory_bytes":int(torch.cuda.max_memory_allocated(0))}


def evaluate_test(data_yaml, checkpoints, out_json, ledger, allow_start=False,
                  confidence=.25, warmup=20, repeats=100, image_groups=None):
    need(AUDIT is not None, "Run runtime_audit first.")
    need(set(checkpoints)==set(MODELS),"Supply one validation-selected best checkpoint per model.")
    d=ledger_read(ledger); need(all(d["stages"].get("base_"+Path(m).stem,{}).get("status")=="completed" for m in MODELS),"Complete all baselines first.")
    d,go=ledger_start(ledger,"final_test",allow_start)
    if not go: raise RuntimeError("Held-out test already completed; refusing a second test look.")
    try:
        spec=dataset_spec(data_yaml); images=split_images(spec,"test"); out={"schema":"p2-test-v1","split":"test","selection_used_test":False,"models":{}}
        for model_id in MODELS:
            ck=Path(checkpoints[model_id]); need(ck.is_file(),f"Missing {ck}"); t0=time.perf_counter(); m=YOLO(str(ck)); cold_load=time.perf_counter()-t0
            torch.cuda.reset_peak_memory_stats(0)
            val=m.val(data=str(data_yaml),split="test",imgsz=IMG,device=0,batch=1,half=False,rect=False,
                conf=MAP_CONF,iou=NMS_IOU,max_det=MAX_DET,nms=True,plots=False,verbose=False)
            sync(); peak_val=int(torch.cuda.max_memory_allocated(0)); b=val.box
            p=list(b.p); r=list(b.r); idx=list(b.ap_class_index)
            pr={spec["names"][int(i)]:{"precision":float(p[j]),"recall":float(r[j]),
                "f1":(2*float(p[j])*float(r[j])/(float(p[j])+float(r[j]))) if float(p[j])+float(r[j]) else 0}
                for j,i in enumerate(idx)}
            need(math.isfinite(float(b.map50)) and math.isfinite(float(b.map)), f"Nonfinite test mAP for {model_id}.")
            need(all(math.isfinite(v) for x in pr.values() for v in x.values()), f"Nonfinite per-class test summary for {model_id}.")
            curves={}
            for j,i in enumerate(idx):
                pc=getattr(b,"prec_values",None); rx=getattr(b,"px",None)
                if pc is not None and rx is not None:
                    pv=pc[j].tolist() if hasattr(pc[j],"tolist") else list(pc[j]); rv=rx.tolist() if hasattr(rx,"tolist") else list(rx)
                    step=max(1,len(pv)//100); curves[spec["names"][int(i)]]={"recall":rv[::step],"precision":pv[::step]}
            torch.cuda.reset_peak_memory_stats(0); match=match_summary(m,spec,images,confidence,image_groups); sync()
            peak_match=int(torch.cuda.max_memory_allocated(0))
            tm=timing(m,images[:min(5,len(images))],warmup,repeats)
            out["models"][model_id]={"checkpoint":str(ck.resolve()),"bytes":ck.stat().st_size,"sha256":sha(ck),
                "cold_checkpoint_load_seconds":cold_load,"parameters":sum(x.numel() for x in m.model.parameters()),
                "test_map50":float(b.map50),"test_map50_95":float(b.map),"per_class_pr_max_f1":pr,
                "per_class_pr_curves":curves or "not exposed by this validator build",
                "pr_confidence":"Ultralytics validator max-F1 summary",
                "matched_box_summary":match,"timing":tm,"peak_test_eval_gpu_memory_bytes":peak_val,
                "peak_matched_gpu_memory_bytes":peak_match,
                "architecture_note":"YOLO26 has optional native one-to-one NMS-free inference; this comparison explicitly used nms=True one-to-many with IoU .7." if model_id=="yolo26n.pt" else "Standard NMS path."}
        out["environment"]={"torch":torch.__version__,"ultralytics":ultralytics.__version__,"gpu":torch.cuda.get_device_name(0)}
        out["settings"]={"imgsz":IMG,"eval_batch":1,"half":False,"rect":False,"map_conf":MAP_CONF,
            "nms_iou":NMS_IOU,"max_det":MAX_DET,"fixed_conf_for_match_and_timing":confidence}
        write_json(out_json,out); ledger_finish(ledger,"final_test",{"output":str(Path(out_json).resolve()),"test_used_for_selection":False})
        return out
    except Exception as e:
        ledger_finish(ledger,"final_test",error=f"{type(e).__name__}: {e}"); raise


def select_validation(candidates):
    need(set(candidates)==set(MODELS),"Missing one validation candidate.")
    winner=max(MODELS,key=lambda k:(candidates[k]["validation"]["map50_95"],candidates[k]["validation"]["map50"]))
    return {"winner":winner,"metric":"validation map50_95 then map50","test_used":False}


def package_winner(selection, baselines, test_results, destination, companion_files=()):
    """Build compact durable package; companion files must be explicitly supplied by coordinator."""
    winner=selection["winner"]; need(winner in MODELS and not selection.get("test_used"),"Invalid/test-informed selection.")
    need(set(baselines)==set(MODELS) and set(test_results.get("models",{}))==set(MODELS),"Need all three baseline and final test records.")
    root=Path(destination); model_dir=root/"models"/"fine-tuned"; model_dir.mkdir(parents=True,exist_ok=True)
    src=Path(baselines[winner]["best_checkpoint"]); need(src.is_file(),f"Missing winner checkpoint {src}")
    target=model_dir/"best.pt"; shutil.copy2(src,target)
    files=[target]
    for source in companion_files:
        source=Path(source); need(source.is_file(),f"Missing companion source {source}")
        target_file=root/source.name; shutil.copy2(source,target_file); files.append(target_file)
    records={str(p.relative_to(root)): {"bytes":p.stat().st_size,"sha256":sha(p)} for p in files}
    total=sum(x["bytes"] for x in records.values()); need(total<200*1024*1024,"Compact package exceeds 200 MiB.")
    manifest={"schema":"p2-package-v1","winner":winner,"selection":selection,
        "model_path":"models/fine-tuned/best.pt","models":baselines,"final_test":test_results,
        "environment":AUDIT,"files":records,"total_bytes":total,
        "limits":"Only winner inference weights plus explicitly supplied companion sources; no raw data or optimizer checkpoints."}
    write_json(root/"manifest.json",manifest)
    return {"directory":str(root.resolve()),"manifest":str((root/"manifest.json").resolve()),
        "winner_checkpoint":str((root/"models"/"fine-tuned"/"best.pt").resolve()),"bytes":total,
        "sha256":sha(root/"manifest.json"),"under_200_mib":True}
