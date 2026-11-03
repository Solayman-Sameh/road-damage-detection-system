# Paste this entire file into ONE new code cell AFTER the finished Optuna cell.
# Run this cell once. It evaluates and exports; it never trains.
from pathlib import Path
import json, shutil, sqlite3, zipfile


def finalize_road_tuning():
    global EVALUATIONS
    required = ["DATA_ID", "ENVIRONMENT", "NAMES", "OUTPUT", "REPORTS", "EVALUATIONS",
                "TRIALS", "SEED", "EVAL_SIZE", "FIXED_CONF", "MAP_CONF", "NMS_IOU", "MAX_DET",
                "digest", "file_sha", "atomic_json", "immutable_json", "completed_status",
                "official_metrics", "cached_predictions", "fixed_summary", "benchmark",
                "split_paths", "dataset_yaml", "YOLO", "np", "pd", "torch", "gc", "display"]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError("Run this after the original notebook and Optuna finish. Missing: " + ", ".join(missing))
    root = Path(globals().get("HPO_ROOT", "/kaggle/working/road_models_v3_tuning"))
    plan = json.loads((root / "plan.json").read_text())
    selection = json.loads((root / "validation_selected.json").read_text())
    limits = json.loads((root / "limits.json").read_text())
    scores = pd.read_csv(root / "validation_comparison.csv")
    if plan["data_id"] != DATA_ID or plan["environment"] != ENVIRONMENT:
        raise ValueError("Current data/runtime differs from the tuning experiment.")
    if selection["plan_id"] != digest(plan):
        raise ValueError("Selection belongs to a different tuning plan.")
    expected_models = {source[0] for source in plan["sources"].values()}
    if {row["model"] for row in selection["per_model"]} != expected_models or len(selection["per_model"]) != 3:
        raise ValueError("All three model studies must finish first.")
    for model in expected_models:
        finished = scores[(scores.model == model) & (scores.kind != "original_v2")]
        if len(finished) != limits["total_trials_per_model"]:
            raise ValueError(f"{model}: not all intended trials have completed successfully.")
    if selection["overall"] not in selection["per_model"]:
        raise ValueError("Overall selection is not one of the family winners.")
    if (EVAL_SIZE, FIXED_CONF, MAP_CONF, NMS_IOU, MAX_DET) != (1024, 0.25, 0.001, 0.7, 300):
        raise ValueError("Keep the original common evaluation settings.")

    records = {}
    for row in selection["per_model"]:
        model = row["model"]
        key = next(k for k, source in plan["sources"].items() if source[0] == model)
        checkpoint = Path(row["checkpoint"])
        if file_sha(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint changed: {checkpoint}")
        base_epochs = int(TRIALS[key]["epochs"])
        extra = int(row["extra_epochs"])
        if extra:
            run = checkpoint.parent.parent
            payload = json.loads((run / "training_spec.json").read_text())
            status = completed_status(run, payload["train"])
            if payload["pretrained_sha256"] != plan["sources"][key][2]:
                raise ValueError("Fine-tune parent model differs from the recorded original.")
            extra = int(status["actual_epochs"])
            candidate_id = f"{key}_tuned_{int(row['trial']):03d}"
            power = float(payload["train"]["cls_pw"])
        else:
            run = None
            candidate_id = key + "_original_v2"
            power = 0.0
        records[candidate_id] = {
            "trial": candidate_id, "model": model, "checkpoint": str(checkpoint),
            "checkpoint_sha256": row["checkpoint_sha256"], "run": str(run) if run else None,
            "origin": "round3_tuning_selection", "train_imgsz": 1024,
            "base_epochs": base_epochs, "extra_epochs": extra,
            "actual_epochs": base_epochs + extra, "planned_epochs": base_epochs + int(row["extra_epochs"]),
            "cls_pw": power, "status": "validation_selected",
            "validation_map50_95": row["map50_95"], "validation_map50": row["map50"],
        }
    winner = next(r for r in records.values()
                  if r["checkpoint_sha256"] == selection["overall"]["checkpoint_sha256"])
    final = root / "final"
    final.mkdir(exist_ok=True)
    original_evaluations = EVALUATIONS
    EVALUATIONS = final / "evaluation"
    EVALUATIONS.mkdir(exist_ok=True)
    try:
        # Choose deployment confidence exclusively on validation, before any new test call.
        val_paths, val_boxes = cached_predictions(winner, "val")
        thresholds = []
        for confidence in np.round(np.arange(0.05, 0.801, 0.05), 2):
            summary, _, _ = fixed_summary(val_paths, val_boxes, float(confidence))
            thresholds.append({"confidence": float(confidence), **summary})
        threshold_table = pd.DataFrame(thresholds).sort_values(
            ["f1", "recall", "confidence"], ascending=False)
        chosen_confidence = float(threshold_table.iloc[0]["confidence"])
        threshold_table.to_csv(final / "validation_thresholds.csv", index=False)
        frozen = {"tuning_plan_id": selection["plan_id"], "dataset_id": DATA_ID,
                  "candidates": records, "winner": winner["trial"], "confidence": chosen_confidence,
                  "eval_imgsz": 1024, "quantize": 32, "rect": False, "nms": True,
                  "map_conf": MAP_CONF, "fixed_conf": FIXED_CONF, "iou": NMS_IOU,
                  "max_det": MAX_DET, "environment": ENVIRONMENT,
                  "timing_seed": SEED, "timed_images": 100, "warmup": 20,
                  "test_scope": "previously inspected test regression benchmark"}
        immutable_json(final / "frozen_test_plan.json", frozen)
        print(f"Validation winner: {winner['model']} | {winner['trial']} | confidence {chosen_confidence:.2f}", flush=True)
        paths = split_paths("test")
        ids = np.random.default_rng(SEED).choice(len(paths), 100, replace=False)
        sample = [paths[int(i)] for i in ids]
        rows, perclass = [], []
        for key, record in records.items():
            cache = EVALUATIONS / (key + "_final.json")
            if cache.exists():
                saved = json.loads(cache.read_text())
                if saved["plan_id"] != digest(frozen):
                    raise ValueError("Cached final result belongs to another test plan.")
                print(f"REUSE final evaluation: {key}", flush=True)
            else:
                print(f"Evaluate selected checkpoint: {key}", flush=True)
                maps = official_metrics(record, "test")
                ps, boxes = cached_predictions(record, "test")
                fixed, classes, _ = fixed_summary(ps, boxes, FIXED_CONF)
                speed = benchmark(record, sample, FIXED_CONF)
                operating_conf = chosen_confidence if key == winner["trial"] else FIXED_CONF
                operating, _, _ = fixed_summary(ps, boxes, operating_conf)
                row = {**record, "split": "test_regression", "eval_imgsz": 1024,
                       "map50": maps["map50"], "map50_95": maps["map50_95"], **fixed, **speed,
                       "operating_confidence": operating_conf,
                       "operating_precision": operating["precision"],
                       "operating_recall": operating["recall"], "operating_f1": operating["f1"]}
                if key == winner["trial"] and operating_conf != FIXED_CONF:
                    deployed = benchmark(record, sample, operating_conf)
                    row.update(deployment_latency_ms=deployed["latency_mean_ms"], deployment_fps=deployed["fps"])
                aps = {r["class"]: r for r in maps["perclass"]}
                saved = {"plan_id": digest(frozen), "row": row,
                         "perclass": [{"trial": key, **r, **aps.get(r["class"], {})} for r in classes]}
                atomic_json(cache, saved)
            rows.append(saved["row"])
            perclass.extend(saved["perclass"])
        comparison = pd.DataFrame(rows)
        comparison.to_csv(final / "test_comparison.csv", index=False)
        pd.DataFrame(perclass).to_csv(final / "test_perclass.csv", index=False)
        display(comparison[["model", "trial", "actual_epochs", "map50_95", "map50",
                            "precision", "recall", "f1", "matched_iou_mean", "fps", "checkpoint_MiB"]])

        export = final / "export"
        export.mkdir(exist_ok=True)
        shutil.copy2(winner["checkpoint"], export / "best.pt")
        if file_sha(export / "best.pt") != winner["checkpoint_sha256"]:
            raise ValueError("Exported checkpoint differs from the validation-selected winner.")
        contract = {"model": winner["model"], "trial": winner["trial"], "weights": "best.pt", "names": NAMES,
                    "imgsz": 1024, "confidence": chosen_confidence, "quantize": 32,
                    "rect": False, "nms": True, "iou": NMS_IOU, "max_det": MAX_DET,
                    "checkpoint_sha256": winner["checkpoint_sha256"], "data_id": DATA_ID,
                    "ultralytics": ENVIRONMENT["packages"]["ultralytics"], "desktop_app_verified": False,
                    "selection": "validation mAP50:95, then mAP50; validation micro-F1 confidence",
                    "evidence_limit": "small single-seed fine-tuning search; small gains may be noise"}
        atomic_json(export / "inference_contract.json", contract)
        atomic_json(export / "labels.json", NAMES)
        model = YOLO(str(export / "best.pt"))
        assert list(model.names.values()) == NAMES
        probe = model.predict(str(val_paths[0]), imgsz=1024, conf=chosen_confidence,
                              device=0, quantize=32, rect=False, nms=True, iou=NMS_IOU,
                              max_det=MAX_DET, verbose=False)
        assert len(probe) == 1
        del model, probe
        gc.collect()
        torch.cuda.empty_cache()

        for filename in ["plan.json", "limits.json", "validation_comparison.csv", "validation_selected.json"]:
            shutil.copy2(root / filename, export / filename)
        for filename in ["data_identity.json", "environment.json"]:
            shutil.copy2(OUTPUT / filename, export / filename)
        for filename in ["frozen_test_plan.json", "validation_thresholds.csv", "test_comparison.csv", "test_perclass.csv"]:
            shutil.copy2(final / filename, export / filename)
        # Preserve each tried recipe, history and per-class validation evidence.
        for key in plan["sources"]:
            for source in (root / key).rglob("*"):
                if source.is_file() and source.name in {
                    "optuna_trials.csv", "training_spec.json", "class_weights.json", "results.csv",
                    "complete.json", "validation.json", "validation_input.json"}:
                    target = export / "tuning_evidence" / source.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        # SQLite backup gives a consistent database snapshot, including any WAL state.
        with sqlite3.connect(str(root / "optuna.db")) as source_db:
            with sqlite3.connect(str(export / "optuna.db")) as destination_db:
                source_db.backup(destination_db)
        baseline_report = REPORTS / "test_1024_comparison.csv"
        if baseline_report.is_file():
            shutil.copy2(baseline_report, export / "original_v2_test_comparison.csv")
        hashes = {str(p.relative_to(export)): file_sha(p) for p in export.rglob("*")
                  if p.is_file() and p.name != "checksums.json"}
        atomic_json(export / "checksums.json", hashes)
        archive = Path(shutil.make_archive(str(final / "road_damage_v3_selected"), "zip", export))
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("Export ZIP integrity check failed.")
        print(f"\nDONE. Download: {archive} ({archive.stat().st_size / 2**20:.2f} MiB)")
        print("Save Version WITH OUTPUTS too: the ZIP contains the selected inference model "
              "and reports; the complete Kaggle output retains all trial models and optimizer checkpoints.")
        return archive
    finally:
        EVALUATIONS = original_evaluations


V3_SELECTED_PACKAGE = finalize_road_tuning()
