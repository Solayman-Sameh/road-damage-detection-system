"""Synthetic finalization integration check; no ML libraries or GPU required."""
import ast
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import types
import uuid
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
fixture = ROOT / "../../tmp/finalize-optuna-checks" / uuid.uuid4().hex
fixture.mkdir(parents=True)
notebook = json.loads((ROOT / "notebooks/instant-sprint02-project02-road-damage-detection.ipynb").read_text(encoding="utf-8"))
env = {"Path": Path, "hashlib": hashlib, "json": json, "os": __import__("os"), "np": np, "pd": pd}
for i in [8, 15]:
    functions = [node for node in ast.parse("".join(notebook["cells"][i]["source"])).body
                 if isinstance(node, ast.FunctionDef) and node.name in {
                     "digest", "file_sha", "atomic_json", "immutable_json", "completed_status"}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "notebook_helpers", "exec"), env)
source = ast.parse((ROOT / "cloud/finalize_optuna_cell.py").read_text())
source.body = [node for node in source.body if not isinstance(node, ast.Assign)]
exec(compile(source, "finalize_optuna_cell.py", "exec"), env)

root = fixture / "tuning"
root.mkdir()
original = fixture / "original"
reports = original / "reports"
reports.mkdir(parents=True)
original_eval = original / "evaluation"
original_eval.mkdir()
environment = {"packages": {"ultralytics": "8.4.191"}, "gpu": ["fake", "fake"]}
for name, value in [("data_identity.json", {"synthetic": True}), ("environment.json", environment)]:
    (original / name).write_text(json.dumps(value))
families = {"v8_small": "yolov8s", "v11_small": "yolo11s", "v26_medium": "yolo26m"}
sources, chosen, scores = {}, [], []
for i, (key, model) in enumerate(families.items()):
    sources[key] = [model, "unused-base.pt", "base-hash-" + key]
    for number in range(3):
        run = root / key / f"trial_{number:03d}"
        weights = run / "weights"
        weights.mkdir(parents=True)
        checkpoint = weights / "best.pt"
        checkpoint.write_bytes((key + str(number)).encode())
        checksum = env["file_sha"](checkpoint)
        history = pd.DataFrame({"epoch": [1, 2, 3], "train/box_loss": [1., .9, .8],
                                "val/box_loss": [1., .9, .8]})
        history.to_csv(run / "results.csv", index=False)
        (run / "complete.json").write_text(json.dumps({"losses_finite": True, "actual_epochs": 3,
            "planned_epochs": 3, "checkpoint_sha256": checksum}))
        (run / "training_spec.json").write_text(json.dumps({"train": {"epochs": 3, "cls_pw": .5},
            "pretrained_sha256": sources[key][2]}))
        row = {"model": model, "kind": "reference" if number == 0 else "tuned", "trial": number,
               "extra_epochs": 3, "map50_95": .45 + .02*i + .001*number, "map50": .75,
               "checkpoint": str(checkpoint), "checkpoint_sha256": checksum}
        scores.append(row)
        if number == 2:
            chosen.append(row)
    scores.append({"model": model, "kind": "original_v2", "extra_epochs": 0})
plan = {"data_id": "synthetic-data", "environment": environment, "sources": sources}
selection = {"plan_id": env["digest"](plan), "per_model": chosen, "overall": chosen[-1]}
for name, value in [("plan.json", plan), ("validation_selected.json", selection),
                    ("limits.json", {"total_trials_per_model": 3})]:
    (root / name).write_text(json.dumps(value))
pd.DataFrame(scores).to_csv(root / "validation_comparison.csv", index=False)
with sqlite3.connect(root / "optuna.db") as db:
    db.execute("CREATE TABLE synthetic(value INTEGER)")
    db.execute("INSERT INTO synthetic VALUES (7)")

calls = {"test": 0, "speed": 0, "probe": 0}
names = ["a", "b"]
def metrics(record, split):
    assert split == "test"
    calls["test"] += 1
    return {"map50": .75, "map50_95": .5, "perclass": [{"class": n, "ap50_95": .5} for n in names]}
def predictions(record, split):
    return [Path("synthetic-image.jpg")], [np.empty((0, 6))]
def summary(paths, boxes, confidence):
    result = {"tp": 2, "fp": 1, "fn": 1, "precision": .7, "recall": .7,
              "f1": 1. - abs(confidence-.35), "matched_iou_mean": .8}
    return result, [{"class": n, "fixed_precision": .7} for n in names], []
def speed(*args):
    calls["speed"] += 1
    return {"latency_mean_ms": 20., "fps": 50., "checkpoint_MiB": .01}
class FakeYOLO:
    def __init__(self, path):
        self.names = dict(enumerate(names))
    def predict(self, *args, **kwargs):
        calls["probe"] += 1
        return [object()]
env.update(DATA_ID="synthetic-data", ENVIRONMENT=environment, NAMES=names, OUTPUT=original,
           REPORTS=reports, EVALUATIONS=original_eval, TRIALS={k: {"epochs": 30} for k in families},
           SEED=2026, EVAL_SIZE=1024, FIXED_CONF=.25, MAP_CONF=.001, NMS_IOU=.7, MAX_DET=300,
           HPO_ROOT=str(root), dataset_yaml="synthetic.yaml", gc=gc, YOLO=FakeYOLO,
           torch=types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
           official_metrics=metrics, cached_predictions=predictions, fixed_summary=summary,
           benchmark=speed, split_paths=lambda split: [Path(f"image-{i}.jpg") for i in range(100)],
           display=lambda table: None)
archive = env["finalize_road_tuning"]()
assert env["EVALUATIONS"] == original_eval
assert calls == {"test": 3, "speed": 4, "probe": 1}, calls
contract = json.loads((root / "final/export/inference_contract.json").read_text())
assert contract["model"] == "yolo26m" and contract["confidence"] == .35
assert contract["checkpoint_sha256"] == chosen[-1]["checkpoint_sha256"]
with zipfile.ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    assert "best.pt" in zipped.namelist() and "optuna.db" in zipped.namelist()
    assert "tuning_evidence/v8_small/trial_000/results.csv" in zipped.namelist()
table = pd.read_csv(root / "final/test_comparison.csv")
assert table.actual_epochs.tolist() == [33, 33, 33]
env["finalize_road_tuning"]()
assert calls == {"test": 3, "speed": 4, "probe": 2}, calls
assert env["EVALUATIONS"] == original_eval
pd.DataFrame(scores[:-2]).to_csv(root / "validation_comparison.csv", index=False)
try:
    env["finalize_road_tuning"]()
except ValueError as error:
    assert "not all intended trials" in str(error)
else:
    raise AssertionError("Incomplete tuning was allowed to finalize")
print("PASS: selected winner, validation threshold, final test, package/DB evidence, cache reuse, "
      "restored notebook state, 33-epoch reporting and incomplete-study rejection. No GPU/model execution.")
