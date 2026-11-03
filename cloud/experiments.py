"""Bounded Kaggle GPU experiments for the Project 2 road-damage benchmark.

This module deliberately does not download data, install packages, or start work
at import time. Callers must pass the observed Kaggle quota and opt into a run.
It targets the Ultralytics YOLO API and fails closed when runtime/version/model
probes do not match the audited environment.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import csv
import math
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_IDS = ("yolov8n.pt", "yolo11n.pt", "yolo26n.pt")
MAX_BASELINE_MINUTES = 120
MAX_TUNING_MINUTES = 30
MAX_TUNING_EPOCHS = 10
MAX_BASELINE_EPOCHS = 20
MAX_PACKAGE_BYTES = 200 * 1024 * 1024
PACKAGE_SCHEMA = "project02-road-damage-export-v1"
COMMON_IMAGE_SIZE = 640
COMMON_TRAIN_BATCH = 8
COMMON_EVAL_BATCH = 1
COMMON_MAP_CONFIDENCE = 0.001
COMMON_NMS_IOU = 0.7
COMMON_MAX_DETECTIONS = 300


class ExperimentGateError(RuntimeError):
    """Raised when a safety, compatibility, quota, or evidence gate fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentGateError(message)


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp_path.replace(path)


def parse_loss_history(results_csv: str | Path) -> dict[str, Any]:
    """Validate saved train/validation loss columns and summarize finite values."""
    path = Path(results_csv)
    _require(path.is_file(), f"Ultralytics loss history is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        rows = list(reader)
    _require(bool(rows), f"Ultralytics loss history has no epoch rows: {path}")
    train_cols = [name for name in headers if name.lower().startswith("train/") and "loss" in name.lower()]
    val_cols = [name for name in headers if name.lower().startswith("val/") and "loss" in name.lower()]
    _require(bool(train_cols), f"Loss CSV has no train loss columns: {path}")
    _require(bool(val_cols), f"Loss CSV has no validation loss columns: {path}")
    summaries: dict[str, dict[str, Any]] = {}
    for column in train_cols + val_cols:
        values = []
        for row_index, row in enumerate(rows, start=1):
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError as exc:
                raise ExperimentGateError(f"Non-numeric loss value in {path}, row {row_index}, column {column}: {raw!r}") from exc
            _require(math.isfinite(value), f"Non-finite loss in {path}, row {row_index}, column {column}: {raw!r}")
            values.append(value)
        _require(bool(values), f"Loss column {column!r} has no recorded values in {path}.")
        summaries[column] = {"last": values[-1], "min": min(values), "entries": len(values)}
    epoch_values = []
    if "epoch" in headers:
        for row_index, row in enumerate(rows, start=1):
            raw = (row.get("epoch") or "").strip()
            if raw:
                try:
                    epoch = float(raw)
                except ValueError as exc:
                    raise ExperimentGateError(f"Non-numeric epoch in {path}, row {row_index}: {raw!r}") from exc
                _require(math.isfinite(epoch), f"Non-finite epoch in {path}, row {row_index}.")
                epoch_values.append(epoch)
    actual_epochs = len(rows)
    _require(actual_epochs > 0, "No completed epoch rows were saved.")
    return {"results_csv": str(path.resolve()), "actual_epochs": actual_epochs,
            "last_epoch_index": epoch_values[-1] if epoch_values else actual_epochs - 1,
            "train_loss": {name: summaries[name] for name in train_cols},
            "validation_loss": {name: summaries[name] for name in val_cols},
            "all_recorded_losses_finite": True}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_environment(expected_ultralytics_version: str,
                      expected_torch_version: str | None = None,
                      model_ids: Sequence[str] = MODEL_IDS,
                      image_size: int = 640) -> dict[str, Any]:
    """Audit Python/GPU libraries and probe all requested checkpoints on CUDA.

    ``expected_ultralytics_version`` is mandatory. This is an explicit audit
    gate rather than an implicit claim that every current Ultralytics release
    supports every model family. The function only loads pretrained weights;
    it performs no training or data download.
    """
    _require(bool(expected_ultralytics_version), "Pass the exact Ultralytics version observed and approved in the environment audit.")
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - cloud dependency gate
        raise ExperimentGateError(f"Torch/Ultralytics import failed: {type(exc).__name__}: {exc}") from exc

    actual_ultralytics = str(getattr(ultralytics, "__version__", "unknown"))
    actual_torch = str(torch.__version__)
    _require(actual_ultralytics == expected_ultralytics_version,
             f"Ultralytics version mismatch: expected {expected_ultralytics_version}, found {actual_ultralytics}; audit compatibility before continuing.")
    if expected_torch_version is not None:
        _require(actual_torch == expected_torch_version,
                 f"Torch version mismatch: expected {expected_torch_version}, found {actual_torch}.")
    _require(torch.cuda.is_available(), "A CUDA GPU is required; this workflow will not train or benchmark on CPU/TPU.")

    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": props.name, "total_memory_bytes": int(props.total_memory),
                        "capability": list(torch.cuda.get_device_capability(index))})
    _require(bool(devices), "CUDA reports no visible GPU device.")

    probes = {}
    for model_id in model_ids:
        try:
            model = YOLO(model_id)
            # A small in-memory inference checks the parsed model and CUDA path
            # without relying on a dataset or returning large prediction data.
            model.predict(source=[__import__("numpy").zeros((image_size, image_size, 3), dtype="uint8")],
                          imgsz=image_size, device=0, nms=True, verbose=False, save=False)
            probes[model_id] = {"load": "passed", "cuda_predict": "passed"}
            del model
        except Exception as exc:
            probes[model_id] = {"load": "failed", "error": f"{type(exc).__name__}: {exc}"}
            raise ExperimentGateError(f"Checkpoint/API probe failed for {model_id}: {type(exc).__name__}: {exc}") from exc
    return {
        "python": os.sys.version,
        "platform": os.name,
        "torch": actual_torch,
        "torchvision": _package_version("torchvision"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics": actual_ultralytics,
        "opencv": _package_version("opencv-python") or _package_version("opencv-python-headless"),
        "gpu_devices": devices,
        "model_probes": probes,
        "probe_image_size": image_size,
        "audited_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _assert_ultralytics_version(expected_version: str) -> None:
    _require(bool(expected_version), "Pass the exact Ultralytics version recorded by the environment audit.")
    actual = _package_version("ultralytics")
    _require(actual == expected_version,
             f"Ultralytics version changed since audit: expected {expected_version}, found {actual}.")


@dataclass(frozen=True)
class BudgetPlan:
    """Explicit observed quota and reserved GPU time, all in minutes."""
    remaining_gpu_minutes: float
    quota_gpu_multiplier: float = 1.0
    pilot_minutes: float = 10
    baseline_minutes_per_model: float = 120
    include_tuning: bool = False
    tuning_minutes_per_model: float = 30
    final_eval_reserve_minutes: float = 30
    export_reserve_minutes: float = 10

    def validate(self) -> dict[str, float]:
        _require(self.remaining_gpu_minutes > 0, "Enter the currently observed remaining GPU quota in minutes.")
        _require(self.quota_gpu_multiplier >= 1, "Quota multiplier must be at least 1. Use 2 conservatively for a T4x2 session if quota is charged per assigned GPU.")
        _require(0 < self.pilot_minutes <= 20, "Pilot cap must be greater than 0 and at most 20 minutes.")
        _require(0 < self.baseline_minutes_per_model <= MAX_BASELINE_MINUTES,
                 f"Each baseline must be capped at {MAX_BASELINE_MINUTES} minutes or less.")
        _require(self.final_eval_reserve_minutes >= 0 and self.export_reserve_minutes >= 0,
                 "Evaluation and export reserves cannot be negative.")
        tuning = self.tuning_minutes_per_model if self.include_tuning else 0
        if self.include_tuning:
            _require(0 < tuning <= MAX_TUNING_MINUTES,
                     f"Each optional tuning trial must be capped at {MAX_TUNING_MINUTES} minutes or less.")
        wall_minutes = self.pilot_minutes + 3 * self.baseline_minutes_per_model + 3 * tuning + self.final_eval_reserve_minutes + self.export_reserve_minutes
        quota_minutes = wall_minutes * self.quota_gpu_multiplier
        _require(quota_minutes <= self.remaining_gpu_minutes,
                 f"Quota gate: planned work reserves {wall_minutes:.1f} wall minutes x {self.quota_gpu_multiplier:g} GPU(s) = {quota_minutes:.1f} GPU minutes but only {self.remaining_gpu_minutes:.1f} remain. Reduce run caps or omit tuning before starting.")
        return {"remaining_gpu_minutes": float(self.remaining_gpu_minutes), "planned_minutes": float(quota_minutes),
                "planned_wall_minutes": float(wall_minutes), "quota_gpu_multiplier": float(self.quota_gpu_multiplier),
                "planned_quota_gpu_minutes": float(quota_minutes),
                "pilot_minutes": float(self.pilot_minutes), "baseline_minutes_per_model": float(self.baseline_minutes_per_model),
                "tuning_minutes_per_model": float(tuning), "final_eval_reserve_minutes": float(self.final_eval_reserve_minutes),
                "export_reserve_minutes": float(self.export_reserve_minutes),
                "unallocated_gpu_minutes": float(self.remaining_gpu_minutes - quota_minutes)}


def _cuda():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ExperimentGateError("Torch import failed; environment audit is required first.") from exc
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "CUDA GPU 0 is required.")
    return torch


def _ledger_load(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ExperimentGateError(f"Run ledger is unreadable at {path}: {exc}") from exc
        _require(value.get("schema") == "project02-run-ledger-v1", "Run ledger schema is not recognized; do not overwrite it.")
        return value
    return {"schema": "project02-run-ledger-v1", "stages": {}}


def _ledger_begin(path: Path, stage: str, allow_start: bool) -> dict[str, Any]:
    _require(allow_start, "Explicit opt-in required: pass allow_start=True after reviewing the Kaggle GPU/quota audit.")
    ledger = _ledger_load(path)
    existing = ledger["stages"].get(stage)
    _require(not existing or existing.get("status") != "completed",
             f"Stage {stage!r} is already completed in the ledger; refusing to duplicate interactive training/evaluation.")
    ledger["stages"][stage] = {"status": "running", "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _json_write(path, ledger)
    return ledger


def _ledger_finish(path: Path, stage: str, result: Mapping[str, Any] | None = None,
                   error: str | None = None) -> None:
    ledger = _ledger_load(path)
    ledger["stages"].setdefault(stage, {})
    ledger["stages"][stage].update({"status": "failed" if error else "completed",
                                     "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if error:
        ledger["stages"][stage]["error"] = error
    if result is not None:
        ledger["stages"][stage]["result"] = result
    _json_write(path, ledger)


def _assert_saved_version_if_required(require_saved_version: bool) -> None:
    if not require_saved_version:
        return
    run_type = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip().lower()
    _require(run_type == "batch", "This phase requires a clean Kaggle Save Version / Run All batch execution (KAGGLE_KERNEL_RUN_TYPE=Batch).")


def _install_training_caps(model: Any, epochs: int, minutes: float,
                           save_resumable: bool = True) -> dict[str, Any]:
    """Enforce independent epoch/wall caps and snapshot resumable weights.

    Ultralytics `time` mode recalculates its epoch count, so epochs must be
    enforced by callback as well. `on_model_save` fires after last.pt is written
    and before final trainer cleanup strips optimizer state.
    """
    state: dict[str, Any] = {"started_at": None, "resumable_copy": None,
                             "wall_cap_checkpointed_mid_epoch": False,
                             "last_checkpoint_elapsed_seconds": None}

    def on_train_start(trainer: Any) -> None:
        state["started_at"] = time.monotonic()

    def on_train_epoch_end(trainer: Any) -> None:
        if int(getattr(trainer, "epoch", -1)) + 1 >= epochs:
            trainer.stop = True

    def on_train_batch_end(trainer: Any) -> None:
        started = state["started_at"]
        if started is not None and time.monotonic() - started >= minutes * 60.0:
            # A batch-end stop can bypass the trainer's for/else epoch save.
            # Persist the optimizer state before asking it to break the loop.
            if save_resumable and not state["wall_cap_checkpointed_mid_epoch"]:
                saved = trainer.save_model()
                _require(bool(saved), "Wall-cap checkpoint could not be saved while stopping mid-epoch.")
                on_model_save(trainer)
                state["wall_cap_checkpointed_mid_epoch"] = True
            trainer.stop = True

    def on_model_save(trainer: Any) -> None:
        if not save_resumable:
            return
        last = Path(trainer.last)
        _require(last.is_file(), f"on_model_save fired without a last checkpoint at {last}.")
        destination = Path(trainer.wdir) / "last-resumable.pt"
        shutil.copy2(last, destination)
        state["resumable_copy"] = str(destination)
        if state["started_at"] is not None:
            state["last_checkpoint_elapsed_seconds"] = time.monotonic() - state["started_at"]

    _require(hasattr(model, "add_callback"), "Installed Ultralytics model lacks the callback API needed for epoch/wall caps and checkpoint persistence.")
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    if save_resumable:
        model.add_callback("on_model_save", on_model_save)
    return state


def _inspect_resumable_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Verify a trusted, locally generated checkpoint has optimizer resume state."""
    path = Path(checkpoint)
    _require(path.is_file(), f"Resumable checkpoint is missing: {path}")
    try:
        import torch
        # This checkpoint is produced by this run's on_model_save callback.
        # Full pickle loading is required to inspect optimizer state; never use
        # this helper on user-supplied or otherwise untrusted checkpoint files.
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # compatibility with audited older Torch versions
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ExperimentGateError(f"Could not inspect generated resume checkpoint: {type(exc).__name__}: {exc}") from exc
    _require(isinstance(payload, dict), "Generated checkpoint is not a trainer checkpoint mapping.")
    epoch = payload.get("epoch")
    optimizer = payload.get("optimizer")
    _require(isinstance(epoch, int) and epoch >= 0, "Generated checkpoint has no completed training epoch.")
    _require(isinstance(optimizer, dict) and "param_groups" in optimizer and "state" in optimizer,
             "Generated checkpoint has no optimizer state and cannot be treated as resumable.")
    return {"path": str(path.resolve()), "epoch": epoch, "optimizer_present": True,
            "optimizer_state_entries": len(optimizer.get("state", {})),
            "optimizer_parameter_groups": len(optimizer.get("param_groups", [])),
            "bytes": path.stat().st_size, "sha256": _sha256(path),
            "resume_evidence": "trusted generated checkpoint loaded and epoch/optimizer fields verified"}


def _require_common_image_size(image_size: int) -> None:
    _require(image_size == COMMON_IMAGE_SIZE,
             f"Project 2 comparison uses the fixed common imgsz={COMMON_IMAGE_SIZE}; got {image_size}.")


def run_pilot(data_yaml: str | Path, output_dir: str | Path, ledger_path: str | Path,
              remaining_gpu_minutes: float, expected_ultralytics_version: str,
              pilot_minutes: float = 10,
              quota_gpu_multiplier: float = 1.0,
              image_size: int = COMMON_IMAGE_SIZE, batch: int = COMMON_TRAIN_BATCH,
              allow_start: bool = False) -> dict[str, Any]:
    """Run a single-GPU, one-epoch-or-time-capped pilot on YOLOv8n only."""
    _require_common_image_size(image_size)
    _require(batch == COMMON_TRAIN_BATCH, f"Use the shared explicit training batch={COMMON_TRAIN_BATCH} for the pilot.")
    _require(quota_gpu_multiplier >= 1, "Quota multiplier must be at least 1; use 2 conservatively for T4x2 if quota is charged per assigned GPU.")
    _require(remaining_gpu_minutes >= pilot_minutes * quota_gpu_multiplier,
             "Observed remaining GPU quota is below the pilot's conservative device-minute reserve.")
    _require(0 < pilot_minutes <= 20, "Pilot cap must be 20 minutes or less.")
    _require(Path(data_yaml).is_file(), f"Dataset YAML not found: {data_yaml}")
    environment = audit_environment(expected_ultralytics_version, model_ids=MODEL_IDS, image_size=image_size)
    torch = _cuda()
    out, ledger = Path(output_dir), Path(ledger_path)
    out.mkdir(parents=True, exist_ok=True)
    _json_write(out / "environment_audit.json", environment)
    stage = "pilot_yolov8n"
    _ledger_begin(ledger, stage, allow_start)
    torch.cuda.reset_peak_memory_stats(0)
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_IDS[0])
        cap_state = _install_training_caps(model, epochs=1, minutes=pilot_minutes)
        result = model.train(data=str(data_yaml), epochs=1, time=pilot_minutes / 60.0,
                             imgsz=image_size, device=0, batch=batch, workers=2, project=str(out / "runs"),
                             name="pilot-yolov8n", exist_ok=False, save=True, plots=False,
                             val=True, seed=2026, deterministic=True, verbose=False)
        training_elapsed = cap_state["last_checkpoint_elapsed_seconds"]
        _require(training_elapsed is not None, "Pilot ended without a checkpoint timestamp for actual run duration.")
        peak_training_memory = int(torch.cuda.max_memory_allocated(0))
        trainer = getattr(model, "trainer", None)
        save_dir_value = getattr(result, "save_dir", getattr(trainer, "save_dir", None))
        _require(save_dir_value is not None, "Could not determine Ultralytics pilot output directory from the audited trainer API.")
        save_dir = Path(save_dir_value)
        best = save_dir / "weights" / "best.pt"
        latest = Path(cap_state["resumable_copy"]) if cap_state["resumable_copy"] else None
        _require(best.is_file(), f"Pilot did not persist best.pt at {best}.")
        loss_evidence = parse_loss_history(save_dir / "results.csv")
        _require(latest is not None and latest.is_file(), "Pilot did not preserve a pre-strip resumable checkpoint.")
        resume_evidence = _inspect_resumable_checkpoint(latest)
        val_image = _resolve_split_images(load_data_spec(data_yaml), "val")[0]
        reload_info = reload_and_probe(best, image_size=image_size, image_path=val_image)
        summary = {"model": MODEL_IDS[0], "save_dir": str(save_dir), "best_checkpoint": str(best),
                   "latest_checkpoint": str(latest) if latest.exists() else None,
                   "training_elapsed_minutes": training_elapsed / 60.0,
                   "peak_training_gpu_memory_bytes": peak_training_memory,
                   "best_checkpoint_bytes": best.stat().st_size,
                   "training_batch": batch, "resume_evidence": resume_evidence,
                   "loss_evidence": loss_evidence,
                   "wall_cap_checkpointed_mid_epoch": cap_state["wall_cap_checkpointed_mid_epoch"],
                   "reload_probe": reload_info}
        _ledger_finish(ledger, stage, summary)
        return summary
    except Exception as exc:
        _ledger_finish(ledger, stage, error=f"{type(exc).__name__}: {exc}")
        raise


def _extract_metric_summary(metrics: Any, class_names: Sequence[str]) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    _require(box is not None, "Ultralytics validation did not return box metrics.")
    map50 = float(getattr(box, "map50"))
    map5095 = float(getattr(box, "map"))
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        return value.tolist() if hasattr(value, "tolist") else list(value)
    p_values = _as_list(getattr(box, "p", None))
    r_values = _as_list(getattr(box, "r", None))
    class_indices = _as_list(getattr(box, "ap_class_index", None))
    per_class = {}
    for i, class_index in enumerate(class_indices):
        idx = int(class_index)
        name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
        per_class[name] = {"precision": float(p_values[i]) if i < len(p_values) else None,
                           "recall": float(r_values[i]) if i < len(r_values) else None}
    return {"map50": map50, "map50_95": map5095, "per_class_precision_recall": per_class,
            "precision_recall_confidence_definition": "Ultralytics validator summary at its max-F1 operating confidence; distinct from fixed-confidence matched-box summary",
            "fitness": float(getattr(metrics, "fitness", 0.0))}


def load_data_spec(data_yaml: str | Path) -> dict[str, Any]:
    """Read the dataset YAML and return validated ordered class names."""
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise ExperimentGateError("PyYAML is required to read the dataset YAML; inspect the cloud environment before installing anything.") from exc
    yaml_path = Path(data_yaml).resolve()
    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    _require(isinstance(spec, dict) and isinstance(spec.get("names"), (dict, list)), "Dataset YAML must define names as an ordered list or integer-key mapping.")
    names_raw = spec["names"]
    names = [str(names_raw[i] if isinstance(names_raw, list) else names_raw.get(i, names_raw.get(str(i)))) for i in range(len(names_raw))]
    _require(all(name and name != "None" for name in names), "Dataset YAML class names contain missing entries.")
    if spec.get("nc") is not None:
        _require(int(spec["nc"]) == len(names), "Dataset YAML nc does not match names length.")
    spec["names"] = names
    spec["_yaml_path"] = str(yaml_path)
    return spec


def _resolve_split_images(spec: Mapping[str, Any], split: str) -> list[Path]:
    split_value = spec.get(split)
    _require(split_value is not None, f"Dataset YAML has no {split!r} split entry.")
    yaml_path = Path(spec["_yaml_path"])
    root_value = spec.get("path", yaml_path.parent)
    root = Path(root_value)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    parts = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for part in parts:
        path = Path(str(part))
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            images.extend(p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    item = Path(line)
                    images.append(item if item.is_absolute() else (path.parent / item).resolve())
        elif path.is_file():
            images.append(path)
        else:
            raise ExperimentGateError(f"Split path is missing or unsupported: {path}")
    images = sorted(set(p.resolve() for p in images))
    _require(bool(images), f"No images found for split={split!r}.")
    return images


def _label_path_for_image(image: Path) -> Path:
    # Standard YOLO layout: .../images/subset/file.jpg -> .../labels/subset/file.txt
    parts = list(image.parts)
    lowered = [piece.lower() for piece in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _read_yolo_labels(label_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not label_path.exists():
        return []
    rows = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        _require(len(fields) == 5, f"Expected YOLO detection label of 5 fields at {label_path}:{line_no}.")
        cls = int(float(fields[0]))
        cx, cy, width, height = map(float, fields[1:])
        _require(width > 0 and height > 0, f"Non-positive box at {label_path}:{line_no}.")
        rows.append((cls, (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)))
    return rows


def _xyxy_norm_to_pixels(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    return (float(box[0]) * width, float(box[1]) * height, float(box[2]) * width, float(box[3]) * height)


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def matched_box_summary(model: Any, data_yaml: str | Path, split: str = "test",
                        image_size: int = 640, confidence: float = 0.25,
                        match_iou: float = 0.50, max_images: int | None = None,
                        nms_iou: float = COMMON_NMS_IOU,
                        max_det: int = COMMON_MAX_DETECTIONS) -> dict[str, Any]:
    """Class-aware greedy one-to-one match at ``match_iou``; summarize coverage.

    Predictions are globally ordered by descending confidence per image. Each
    prediction can match one still-unmatched ground truth of the same class.
    Coverage is matched GT / all GT, and prediction precision is matches / all
    retained predictions. This summary is descriptive and is not an AP metric.
    """
    _require_common_image_size(image_size)
    _require(0 <= confidence <= 1 and 0 < match_iou <= 1, "Confidence and IoU thresholds must be in valid ranges.")
    spec = load_data_spec(data_yaml)
    images = _resolve_split_images(spec, split)
    if max_images is not None:
        images = images[:max_images]
    matched_ious: list[float] = []
    gt_total = pred_total = 0
    per_class: dict[str, dict[str, int]] = {name: {"ground_truth": 0, "predictions": 0, "matched": 0} for name in spec["names"]}
    for image_path in images:
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
        truths = _read_yolo_labels(_label_path_for_image(image_path))
        gt_total += len(truths)
        for cls, _ in truths:
            if 0 <= cls < len(spec["names"]):
                per_class[spec["names"][cls]]["ground_truth"] += 1
        result = model.predict(source=str(image_path), imgsz=image_size, device=0,
                               batch=COMMON_EVAL_BATCH, half=False, rect=False,
                               conf=confidence, iou=nms_iou, max_det=max_det, nms=True,
                               verbose=False, save=False)[0]
        boxes = getattr(result, "boxes", None)
        predictions = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            predictions = sorted([(float(c), int(k), tuple(map(float, box))) for box, k, c in zip(xyxy, classes, confs)], reverse=True)
        pred_total += len(predictions)
        for _, cls, _ in predictions:
            if 0 <= cls < len(spec["names"]):
                per_class[spec["names"][cls]]["predictions"] += 1
        used: set[int] = set()
        for score, pred_cls, pred_box in predictions:
            choices = [(idx, _iou_xyxy(pred_box, _xyxy_norm_to_pixels(gt_box, width, height)))
                       for idx, (gt_cls, gt_box) in enumerate(truths) if idx not in used and gt_cls == pred_cls]
            if choices:
                gt_idx, iou = max(choices, key=lambda item: item[1])
                if iou >= match_iou:
                    used.add(gt_idx)
                    matched_ious.append(iou)
                    if 0 <= pred_cls < len(spec["names"]):
                        per_class[spec["names"][pred_cls]]["matched"] += 1
    matched = len(matched_ious)
    return {"split": split, "images": len(images), "confidence_threshold": confidence,
            "iou_match_threshold": match_iou, "nms_iou": nms_iou, "max_det": max_det,
            "inference": {"imgsz": image_size, "batch": COMMON_EVAL_BATCH, "half": False, "rect": False},
            "matching": "class-aware greedy confidence-ordered one-to-one per image",
            "ground_truth_count": gt_total, "prediction_count": pred_total, "matched_count": matched,
            "mean_matched_iou": sum(matched_ious) / matched if matched else None,
            "ground_truth_coverage": matched / gt_total if gt_total else None,
            "prediction_match_rate": matched / pred_total if pred_total else None,
            "per_class_counts": per_class}


def benchmark_inference(model: Any, image_paths: Sequence[str | Path], image_size: int = 640,
                        warmup: int = 20, repetitions: int = 100,
                        confidence: float = 0.25,
                        nms_iou: float = COMMON_NMS_IOU,
                        max_det: int = COMMON_MAX_DETECTIONS) -> dict[str, Any]:
    """Measure synchronized batch-1 end-to-end Ultralytics predict call latency."""
    _require(bool(image_paths), "At least one representative image is required for timing.")
    _require_common_image_size(image_size)
    _require(warmup >= 1 and repetitions >= 5, "Use at least one warm-up and five timed repetitions.")
    torch = _cuda()
    from itertools import cycle, islice
    paths = [str(Path(p)) for p in image_paths]
    _require(all(Path(p).is_file() for p in paths), "Every benchmark image must exist.")
    source_cycle = cycle(paths)
    torch.cuda.synchronize(0)
    for source in islice(source_cycle, warmup):
        model.predict(source=source, imgsz=image_size, device=0, batch=COMMON_EVAL_BATCH,
                      half=False, rect=False, conf=confidence, iou=nms_iou,
                      max_det=max_det, nms=True, verbose=False, save=False)
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    samples_ms = []
    for source in islice(source_cycle, repetitions):
        torch.cuda.synchronize(0)
        start = time.perf_counter()
        model.predict(source=source, imgsz=image_size, device=0, batch=COMMON_EVAL_BATCH,
                      half=False, rect=False, conf=confidence, iou=nms_iou,
                      max_det=max_det, nms=True, verbose=False, save=False)
        torch.cuda.synchronize(0)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    samples_ms.sort()
    mean_ms = sum(samples_ms) / len(samples_ms)
    return {"device": torch.cuda.get_device_name(0), "batch_size": 1, "image_size": image_size,
            "confidence_threshold": confidence, "nms_iou": nms_iou, "max_det": max_det,
            "half": False, "rect": False,
            "timing_scope": "synchronized end-to-end Ultralytics predict call (decode/preprocess/model/postprocess)",
            "warmup_iterations": warmup, "timed_iterations": repetitions, "latency_ms_mean": mean_ms,
            "latency_ms_p50": samples_ms[len(samples_ms) // 2],
            "latency_ms_p95": samples_ms[min(len(samples_ms) - 1, int(0.95 * len(samples_ms)))],
            "fps_from_mean_latency": 1000.0 / mean_ms if mean_ms else None,
            "peak_inference_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0))}


def reload_and_probe(checkpoint: str | Path, image_size: int = 640,
                     image_path: str | Path | None = None) -> dict[str, Any]:
    """Load persisted weights on CUDA and execute a small inference check."""
    _require_common_image_size(image_size)
    path = Path(checkpoint)
    _require(path.is_file(), f"Checkpoint is missing: {path}")
    torch = _cuda()
    import numpy as np
    from ultralytics import YOLO
    model = YOLO(str(path))
    source: Any = str(image_path) if image_path is not None else np.zeros((image_size, image_size, 3), dtype=np.uint8)
    if image_path is not None:
        _require(Path(image_path).is_file(), f"Reload probe image is missing: {image_path}")
    torch.cuda.synchronize(0)
    result = model.predict(source=source, imgsz=image_size, device=0, batch=COMMON_EVAL_BATCH,
                           half=False, rect=False, conf=COMMON_MAP_CONFIDENCE,
                           iou=COMMON_NMS_IOU, max_det=COMMON_MAX_DETECTIONS, nms=True,
                           verbose=False, save=False)
    torch.cuda.synchronize(0)
    return {"checkpoint": str(path.resolve()), "checkpoint_bytes": path.stat().st_size,
            "sha256": _sha256(path), "reload_inference": "passed", "result_count": len(result),
            "gpu": torch.cuda.get_device_name(0)}


def _train_one(model_id: str, stage: str, data_yaml: str | Path, output_dir: Path,
               ledger_path: Path, minutes: float, epochs: int, image_size: int,
               seed: int, allow_start: bool, require_saved_version: bool,
               batch: int = COMMON_TRAIN_BATCH, train_args: Mapping[str, Any] | None = None) -> dict[str, Any]:
    _require(model_id in MODEL_IDS, f"Model must be one of {MODEL_IDS}; got {model_id}.")
    _require(Path(data_yaml).is_file(), f"Dataset YAML not found: {data_yaml}")
    _require(0 < minutes <= (MAX_BASELINE_MINUTES if stage.startswith("baseline") else MAX_TUNING_MINUTES), "Run time cap exceeds its allowed maximum.")
    _require(0 < epochs <= (MAX_BASELINE_EPOCHS if stage.startswith("baseline") else MAX_TUNING_EPOCHS), "Epoch cap exceeds its allowed maximum.")
    _require_common_image_size(image_size)
    _require(batch == COMMON_TRAIN_BATCH,
             f"All model runs use the common explicit training batch={COMMON_TRAIN_BATCH}; do not change it without a measured pilot justification.")
    _assert_saved_version_if_required(require_saved_version)
    torch = _cuda()
    ledger = Path(ledger_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ledger_begin(ledger, stage, allow_start)
    torch.cuda.reset_peak_memory_stats(0)
    try:
        from ultralytics import YOLO
        model = YOLO(model_id)
        cap_state = _install_training_caps(model, epochs=epochs, minutes=minutes)
        args = dict(train_args or {})
        prohibited = {"data", "model", "epochs", "time", "imgsz", "device", "batch", "amp", "half", "project", "name", "exist_ok", "seed", "deterministic"}
        _require(not (prohibited & args.keys()), f"Do not override bounded/core training fields: {sorted(prohibited & args.keys())}")
        run_name = stage.replace("/", "-")
        result = model.train(data=str(data_yaml), epochs=epochs, time=minutes / 60.0,
                             imgsz=image_size, device=0, batch=batch, workers=2,
                             project=str(output_dir / "runs"), name=run_name, exist_ok=False,
                             save=True, plots=False, val=True, seed=seed,
                             deterministic=True, verbose=False, **args)
        training_elapsed = cap_state["last_checkpoint_elapsed_seconds"]
        _require(training_elapsed is not None, "Training ended without a checkpoint timestamp for actual run duration.")
        peak_training_memory = int(torch.cuda.max_memory_allocated(0))
        trainer = getattr(model, "trainer", None)
        save_dir_value = getattr(result, "save_dir", getattr(trainer, "save_dir", None))
        _require(save_dir_value is not None, "Could not determine Ultralytics output directory from the audited trainer API.")
        save_dir = Path(save_dir_value)
        best = save_dir / "weights" / "best.pt"
        latest = Path(cap_state["resumable_copy"]) if cap_state["resumable_copy"] else None
        _require(best.is_file(), f"Training finished without a persisted best checkpoint: {best}")
        loss_evidence = parse_loss_history(save_dir / "results.csv")
        _require(latest is not None and latest.is_file(), "Training did not preserve a pre-strip resumable checkpoint.")
        resume_evidence = _inspect_resumable_checkpoint(latest)
        val_image = _resolve_split_images(load_data_spec(data_yaml), "val")[0]
        reload_info = reload_and_probe(best, image_size=image_size, image_path=val_image)
        # Validation-only model selection. Held-out test is never read here.
        validation_model = YOLO(str(best))
        val_metrics = validation_model.val(data=str(data_yaml), split="val", imgsz=image_size, device=0,
                                           batch=COMMON_EVAL_BATCH, half=False, rect=False,
                                           conf=COMMON_MAP_CONFIDENCE, iou=COMMON_NMS_IOU,
                                           max_det=COMMON_MAX_DETECTIONS, nms=True, plots=False, verbose=False)
        spec = load_data_spec(data_yaml)
        summary = {"model": model_id, "stage": stage, "save_dir": str(save_dir.resolve()),
                   "best_checkpoint": str(best.resolve()), "latest_checkpoint": str(latest.resolve()),
                   "best_checkpoint_bytes": best.stat().st_size, "best_sha256": _sha256(best),
                   "latest_checkpoint_bytes": latest.stat().st_size,
                   "latest_checkpoint_sha256": _sha256(latest), "resume_evidence": resume_evidence,
                   "wall_cap_checkpointed_mid_epoch": cap_state["wall_cap_checkpointed_mid_epoch"],
                   "loss_evidence": loss_evidence,
                   "training_elapsed_minutes": training_elapsed / 60.0,
                   "peak_training_gpu_memory_bytes": peak_training_memory,
                   "validation": _extract_metric_summary(val_metrics, spec["names"]),
                   "reload_probe": reload_info, "heldout_test_evaluated": False,
                   "settings": {"epochs_cap": epochs, "wall_minutes_cap": minutes, "image_size": image_size,
                                "device": torch.cuda.get_device_name(0), "device_index": 0, "batch": batch,
                                "seed": seed, "deterministic": True,
                                "stopping": "trainer time limit plus on_train_batch_end wall callback and on_train_epoch_end epoch callback; wall limit can overrun by one batch",
                                "validation": {"imgsz": image_size, "batch": COMMON_EVAL_BATCH, "half": False,
                                               "rect": False, "conf": COMMON_MAP_CONFIDENCE,
                                               "iou": COMMON_NMS_IOU, "max_det": COMMON_MAX_DETECTIONS}}}
        _ledger_finish(ledger, stage, summary)
        return summary
    except Exception as exc:
        _ledger_finish(ledger, stage, error=f"{type(exc).__name__}: {exc}")
        raise


def run_baselines(data_yaml: str | Path, output_dir: str | Path, ledger_path: str | Path,
                  budget: BudgetPlan, expected_ultralytics_version: str,
                  allow_start: bool = False,
                  require_saved_version: bool = True, epochs: int = 20,
                  image_size: int = 640, seed: int = 2026,
                  batch: int = COMMON_TRAIN_BATCH) -> dict[str, Any]:
    """Run each required baseline at most once, skipping completed ledger stages."""
    budget_summary = budget.validate()
    _require(0 < epochs <= MAX_BASELINE_EPOCHS, f"Baseline epochs must be in 1..{MAX_BASELINE_EPOCHS}.")
    _require(Path(data_yaml).is_file(), f"Dataset YAML not found: {data_yaml}")
    _assert_saved_version_if_required(require_saved_version)
    environment = audit_environment(expected_ultralytics_version, model_ids=MODEL_IDS, image_size=image_size)
    _json_write(Path(output_dir) / "environment_audit.json", environment)
    ledger_stages = _ledger_load(Path(ledger_path))["stages"]
    _require(ledger_stages.get("pilot_yolov8n", {}).get("status") == "completed",
             "Complete the single-GPU pilot and checkpoint reload probe before baseline training.")
    results = {}
    for model_id in MODEL_IDS:
        stage = f"baseline_{Path(model_id).stem}"
        previous = _ledger_load(Path(ledger_path))["stages"].get(stage)
        if previous and previous.get("status") == "completed":
            results[model_id] = {"status": "skipped_completed", **previous.get("result", {})}
            continue
        results[model_id] = _train_one(model_id, stage, data_yaml, Path(output_dir), Path(ledger_path),
                                       budget.baseline_minutes_per_model, epochs, image_size, seed,
                                       allow_start, require_saved_version, batch=batch)
    return {"budget": budget_summary, "baselines": results}


def select_winner(validation_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Select by validation mAP50-95, with validation mAP50 as tie breaker."""
    candidates = []
    for model_id in MODEL_IDS:
        item = validation_results.get(model_id)
        _require(item is not None, f"Missing validation result for {model_id}.")
        validation = item.get("validation", item)
        _require(validation.get("map50_95") is not None and validation.get("map50") is not None,
                 f"Missing validation mAP fields for {model_id}.")
        candidates.append((float(validation["map50_95"]), float(validation["map50"]), model_id))
    winner = max(candidates)
    return {"model": winner[2], "selection_metric": "validation map50_95, then validation map50",
            "validation_map50_95": winner[0], "validation_map50": winner[1],
            "test_used_for_selection": False}


def run_tuning_trials(data_yaml: str | Path, output_dir: str | Path,
                      ledger_path: str | Path, budget: BudgetPlan,
                      baseline_results: Mapping[str, Mapping[str, Any]],
                      tuning_args: Mapping[str, Mapping[str, Any]],
                      expected_ultralytics_version: str,
                      allow_start: bool = False, require_saved_version: bool = True,
                      image_size: int = 640, seed: int = 2026,
                      batch: int = COMMON_TRAIN_BATCH) -> dict[str, Any]:
    """Run at most one equally budgeted, validation-selected tuning trial/model."""
    _require(budget.include_tuning, "Tuning is disabled in the explicit quota plan.")
    budget.validate()
    _assert_ultralytics_version(expected_ultralytics_version)
    ledger_stages = _ledger_load(Path(ledger_path))["stages"]
    _require(all(ledger_stages.get(f"baseline_{Path(model_id).stem}", {}).get("status") == "completed" for model_id in MODEL_IDS),
             "All three baseline runs must complete before optional tuning.")
    _require(set(tuning_args) == set(MODEL_IDS), "Provide exactly one tuning configuration for each of the three candidate models.")
    _require(all(bool(args) for args in tuning_args.values()), "Tuning arguments must be explicit; provide one non-empty config per model.")
    results = {}
    for model_id in MODEL_IDS:
        stage = f"tuning_{Path(model_id).stem}"
        previous = _ledger_load(Path(ledger_path))["stages"].get(stage)
        if previous and previous.get("status") == "completed":
            results[model_id] = {"status": "skipped_completed", **previous.get("result", {})}
            continue
        results[model_id] = _train_one(model_id, stage, data_yaml, Path(output_dir), Path(ledger_path),
                                       budget.tuning_minutes_per_model, MAX_TUNING_EPOCHS, image_size,
                                       seed, allow_start, require_saved_version, batch=batch,
                                       train_args=tuning_args[model_id])
    # Select validation-best checkpoint across baseline and tuned candidates;
    # one tuned run is not automatically preferred over an untuned baseline.
    selection = {}
    for model_id in MODEL_IDS:
        base = baseline_results[model_id]
        tuned = results[model_id]
        base_val = base.get("validation", base)
        tuned_val = tuned.get("validation", tuned)
        use_tuned = (float(tuned_val.get("map50_95", -1)) , float(tuned_val.get("map50", -1))) > (float(base_val.get("map50_95", -1)), float(base_val.get("map50", -1)))
        selection[model_id] = {**(tuned if use_tuned else base), "selected_stage": "tuning" if use_tuned else "baseline"}
    return {"tuning": results, "validation_selected_candidates": selection,
            "winner": select_winner(selection)}


def evaluate_all_on_test(data_yaml: str | Path,
                         checkpoints: Mapping[str, str | Path],
                         output_json: str | Path,
                         expected_ultralytics_version: str,
                         ledger_path: str | Path | None = None,
                         representative_images: Sequence[str | Path] | None = None,
                         image_size: int = 640, confidence: float = 0.25,
                         match_iou: float = 0.50, timing_warmup: int = 20,
                         timing_repetitions: int = 100,
                         allow_start: bool = False) -> dict[str, Any]:
    """Evaluate held-out test exactly across all three candidates, once per ledger.

    The caller should pass each candidate's validation-selected best checkpoint.
    This function does not select a winner from test results.
    """
    _require_common_image_size(image_size)
    _require(set(checkpoints) == set(MODEL_IDS), "Held-out test requires exactly the three model checkpoints.")
    _require(Path(data_yaml).is_file(), f"Dataset YAML not found: {data_yaml}")
    _assert_ultralytics_version(expected_ultralytics_version)
    out_path = Path(output_json)
    ledger_path = Path(ledger_path) if ledger_path is not None else out_path.with_name("run_ledger.json")
    stage = "final_test_all_three"
    ledger_stages = _ledger_load(ledger_path)["stages"]
    _require(all(ledger_stages.get(f"baseline_{Path(model_id).stem}", {}).get("status") == "completed" for model_id in MODEL_IDS),
             "All three baseline checkpoints and validation summaries must be completed before final test evaluation.")
    _ledger_begin(ledger_path, stage, allow_start)
    try:
        from ultralytics import YOLO
        torch = _cuda()
        spec = load_data_spec(data_yaml)
        images = list(representative_images or _resolve_split_images(spec, "test")[:5])
        output = {"schema": "project02-final-test-v1", "data_yaml": str(Path(data_yaml).resolve()),
                  "split": "test", "selection_used_test": False, "models": {}}
        for model_id in MODEL_IDS:
            checkpoint = Path(checkpoints[model_id])
            _require(checkpoint.is_file(), f"Checkpoint for {model_id} is missing: {checkpoint}")
            model = YOLO(str(checkpoint))
            torch.cuda.reset_peak_memory_stats(0)
            test_metrics = model.val(data=str(data_yaml), split="test", imgsz=image_size,
                                     device=0, batch=COMMON_EVAL_BATCH, half=False,
                                     rect=False, conf=COMMON_MAP_CONFIDENCE,
                                     iou=COMMON_NMS_IOU, max_det=COMMON_MAX_DETECTIONS, nms=True,
                                     plots=False, verbose=False)
            torch.cuda.synchronize(0)
            peak_test_eval_memory = int(torch.cuda.max_memory_allocated(0))
            metrics = _extract_metric_summary(test_metrics, spec["names"])
            torch.cuda.reset_peak_memory_stats(0)
            matching = matched_box_summary(model, data_yaml, split="test", image_size=image_size,
                                           confidence=confidence, match_iou=match_iou,
                                           nms_iou=COMMON_NMS_IOU,
                                           max_det=COMMON_MAX_DETECTIONS)
            torch.cuda.synchronize(0)
            peak_matched_memory = int(torch.cuda.max_memory_allocated(0))
            timing = benchmark_inference(model, images, image_size=image_size, warmup=timing_warmup,
                                         repetitions=timing_repetitions, confidence=confidence,
                                         nms_iou=COMMON_NMS_IOU, max_det=COMMON_MAX_DETECTIONS)
            output["models"][model_id] = {"checkpoint": str(checkpoint.resolve()),
                                          "checkpoint_bytes": checkpoint.stat().st_size,
                                          "checkpoint_sha256": _sha256(checkpoint),
                                          "test_metrics": metrics, "matched_box_summary": matching,
                                          "inference_timing": timing,
                                          "peak_test_eval_gpu_memory_bytes": peak_test_eval_memory,
                                          "peak_matched_summary_gpu_memory_bytes": peak_matched_memory,
                                          "architecture_note": ("YOLO26 evaluation explicitly selected the one-to-many NMS path (nms=True); its optional end-to-end nms=False head is not compared here."
                                                                if model_id == "yolo26n.pt" else
                                                                "YOLOv8/YOLO11 use the standard NMS prediction path with iou=0.7."),
                                          "inference_settings": {"imgsz": image_size, "batch": COMMON_EVAL_BATCH,
                                                                 "half": False, "rect": False,
                                                                 "conf_for_map": COMMON_MAP_CONFIDENCE,
                                                                 "nms_iou": COMMON_NMS_IOU,
                                                                 "max_det": COMMON_MAX_DETECTIONS}}
        output["environment"] = {"torch": _package_version("torch"), "ultralytics": _package_version("ultralytics"),
                                 "device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
        output["protocol"] = {"image_size": image_size, "confidence_threshold": confidence,
                              "matched_iou_threshold": match_iou, "timing_warmup": timing_warmup,
                              "timing_repetitions": timing_repetitions,
                              "timing_scope": "synchronized end-to-end batch-1 Ultralytics predict call",
                              "common_prediction_settings": {"imgsz": image_size, "batch": COMMON_EVAL_BATCH,
                                                              "half": False, "rect": False,
                                                              "map_conf": COMMON_MAP_CONFIDENCE,
                                                              "nms_iou": COMMON_NMS_IOU,
                                                              "max_det": COMMON_MAX_DETECTIONS},
                              "architecture_difference": "YOLO26 has an optional native end-to-end nms=False head; this comparison explicitly selects the common one-to-many NMS path for all models."}
        _json_write(out_path, output)
        _ledger_finish(ledger_path, stage, {"output_json": str(out_path.resolve()), "models": list(MODEL_IDS),
                                            "test_used_for_selection": False})
        return output
    except Exception as exc:
        _ledger_finish(ledger_path, stage, error=f"{type(exc).__name__}: {exc}")
        raise


def build_compact_package(winner_model: str, winner_checkpoint: str | Path,
                          candidate_results: Mapping[str, Mapping[str, Any]],
                          final_test_results: Mapping[str, Any],
                          environment: Mapping[str, Any],
                          provenance_manifest: Mapping[str, Any],
                          split_manifest: Mapping[str, Any],
                          companion_files: Sequence[str | Path],
                          output_zip: str | Path,
                          max_bytes: int = MAX_PACKAGE_BYTES) -> dict[str, Any]:
    """Create a checksummed compact export with the winning checkpoint only.

    All supplied results/manifests must be observed outputs from the cloud run;
    this function never generates metric values or fills in missing evidence.
    Companion app/source files are included exactly as passed by the coordinator.
    """
    _require(winner_model in MODEL_IDS, f"Winner must be one of {MODEL_IDS}.")
    _require(set(candidate_results) == set(MODEL_IDS), "Package needs observed results for all three candidates.")
    _require(set((final_test_results.get("models") or {}).keys()) == set(MODEL_IDS), "Package needs final held-out test outputs for all three candidates.")
    _require(final_test_results.get("selection_used_test") is False, "Final test manifest must explicitly state it was not used for selection.")
    checkpoint = Path(winner_checkpoint).resolve()
    _require(checkpoint.is_file(), f"Winning checkpoint is missing: {checkpoint}")
    selected_test_entry = (final_test_results.get("models") or {}).get(winner_model, {})
    _require(selected_test_entry.get("checkpoint_sha256") == _sha256(checkpoint),
             "Winning export checkpoint checksum does not match the checkpoint evaluated in the final test run.")
    _require(bool(companion_files), "Pass the application/inference companion source files to include in the package.")
    companions = [Path(p).resolve() for p in companion_files]
    _require(all(p.is_file() for p in companions), "One or more app companion files are missing.")
    target = Path(output_zip).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    members: dict[str, bytes] = {}
    model_member = "models/fine-tuned/best.pt"
    members[model_member] = checkpoint.read_bytes()
    payloads = {
        "results/candidates.json": candidate_results,
        "results/final_test.json": final_test_results,
        "environment.json": environment,
        "manifests/provenance.json": provenance_manifest,
        "manifests/split.json": split_manifest,
    }
    for name, value in payloads.items():
        members[name] = (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    for path in companions:
        # Package source under app/ while retaining its relative file name.
        member_name = f"app/{path.name}"
        _require(member_name not in members, f"Companion file name collision in package: {path.name}")
        members[member_name] = path.read_bytes()
    file_manifest = {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                     for name, content in sorted(members.items())}
    manifest = {"schema": PACKAGE_SCHEMA, "winner_model": winner_model,
                "winner_checkpoint_member": model_member,
                "selection_metric": candidate_results.get(winner_model, {}).get("selection_metric", "validation map50_95; see candidate results"),
                "test_used_for_selection": False, "max_package_bytes": max_bytes,
                "files": file_manifest}
    members["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temp = target.with_suffix(target.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)
    size = temp.stat().st_size
    if size >= max_bytes:
        temp.unlink(missing_ok=True)
        raise ExperimentGateError(f"Compact package is {size} bytes, not below the {max_bytes}-byte limit; do not download it.")
    with zipfile.ZipFile(temp, "r") as archive:
        bad_member = archive.testzip()
        _require(bad_member is None, f"Package integrity check failed for {bad_member}.")
    temp.replace(target)
    return {"path": str(target), "bytes": size, "sha256": _sha256(target),
            "member_count": len(members), "winner_model": winner_model,
            "winner_checkpoint_member": model_member, "under_200_mib": size < MAX_PACKAGE_BYTES}

