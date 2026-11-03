"""CPU-only checks for the Kaggle handoff. Does not load/train a YOLO model."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import types
import unittest
import uuid

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parents[1]
with contextlib.redirect_stdout(io.StringIO()):
    HELPERS = runpy.run_path(str(ROOT / "cloud/optuna_finetuning_cell.py"))
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HandoffChecks(unittest.TestCase):
    def study(self):
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial({"lr0": 1e-4, "cls_pw": 0.0})
        return study

    def test_total_limit_does_not_add_trials_on_rerun(self):
        study, seen = self.study(), []
        def evaluate(number, params):
            seen.append((number, params))
            return .4 + number / 100
        HELPERS["_hpo_run_study"](optuna, study, 4, evaluate)
        HELPERS["_hpo_run_study"](optuna, study, 4, evaluate)
        self.assertEqual(len(seen), 4)
        self.assertEqual(seen[0][1], {"lr0": 1e-4, "cls_pw": 0.0})
        self.assertEqual(len(study.trials), 4)

    def test_interrupted_trial_keeps_number_and_parameters(self):
        study, seen = self.study(), []
        def interrupted(number, params):
            seen.append((number, params))
            raise TimeoutError("bounded stop")
        with self.assertRaises(TimeoutError):
            HELPERS["_hpo_run_study"](optuna, study, 2, interrupted)
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.RUNNING)
        def finish(number, params):
            seen.append((number, params))
            return .5
        HELPERS["_hpo_run_study"](optuna, study, 2, finish)
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(len(study.trials), 2)

    def test_failure_stops_instead_of_launching_other_trials(self):
        study = self.study()
        def fail(number, params):
            raise RuntimeError("synthetic OOM")
        with self.assertRaisesRegex(RuntimeError, "synthetic OOM"):
            HELPERS["_hpo_run_study"](optuna, study, 4, fail)
        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.FAIL)

    def test_nonfinite_score_is_not_a_success(self):
        study = self.study()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            HELPERS["_hpo_run_study"](optuna, study, 4, lambda *_: float("nan"))
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.FAIL)

    def test_sqlite_restart_reuses_completed_trials(self):
        directory = ROOT / "../../tmp/hpo-handoff-checks" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        storage = "sqlite:///" + (directory.resolve() / "study.db").as_posix()
        study = optuna.create_study(study_name="resume", storage=storage, direction="maximize")
        study.enqueue_trial({"lr0": 1e-4, "cls_pw": 0.0})
        HELPERS["_hpo_run_study"](optuna, study, 2, lambda *_: .5)
        reloaded = optuna.create_study(study_name="resume", storage=storage,
                                       direction="maximize", load_if_exists=True)
        def must_not_run(*_):
            self.fail("A completed persistent study launched another trial")
        HELPERS["_hpo_run_study"](optuna, reloaded, 2, must_not_run)
        self.assertEqual(len(reloaded.trials), 2)

    def test_exhausted_budget_does_not_launch_a_process(self):
        directory = ROOT / "../../tmp/hpo-handoff-checks" / uuid.uuid4().hex
        (directory / "attempts").mkdir(parents=True)
        (directory / "attempts/previous.json").write_text(json.dumps(
            {"job": "previous", "started": 100.0, "finished": 100.0 + 5 * 3600}))
        with self.assertRaisesRegex(TimeoutError, "budget exhausted"):
            HELPERS["_hpo_launch"](["must-not-be-started"], directory, "new", 60)

    def test_changed_plan_is_rejected(self):
        directory = ROOT / "../../tmp/hpo-handoff-checks" / uuid.uuid4().hex
        path = directory / "plan.json"
        HELPERS["_hpo_json"](path, {"parent_sha": "first"}, immutable=True)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            HELPERS["_hpo_json"](path, {"parent_sha": "different"}, immutable=True)

    def test_recovery_hashes_match_actual_local_models(self):
        source = (ROOT / "cloud/recover_exact_kaggle_models_cell.py").read_text()
        assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "expected_hashes" for t in n.targets))
        expected = ast.literal_eval(assignment.value)
        local = {"v8_small": "yolov8n", "v11_small": "yolov11s", "v26_medium": "yolo26m"}
        for key, folder in local.items():
            self.assertEqual(HELPERS["_hpo_sha"](ROOT / "models/v2" / folder / "best.pt"), expected[key])

    def test_worker_and_notebooks_compile(self):
        for name in ["notebooks/instant-sprint02-project02-road-damage-detection.ipynb",
                     "cloud/Road_Damage_Optuna_Starter.ipynb"]:
            notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    code = "".join(cell["source"])
                    tree = ast.parse(code)
                    for node in tree.body:
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WORKER_SOURCE" for t in node.targets):
                            ast.parse(ast.literal_eval(node.value))
        ast.parse(HELPERS["HPO_VALIDATOR_SOURCE"])

    def test_validation_worker_uses_validation_only(self):
        directory = ROOT / "../../tmp/hpo-handoff-checks" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        checkpoint = directory / "fake-checkpoint.pt"
        checkpoint.write_bytes(b"synthetic checkpoint, no model")
        payload = {"checkpoint": str(checkpoint), "checkpoint_sha256": HELPERS["_hpo_sha"](checkpoint),
                   "names": ["a", "b"], "data": "synthetic.yaml", "project": str(directory)}
        payload_path = directory / "payload.json"
        payload_path.write_text(json.dumps(payload))
        calls = []
        class FakeYOLO:
            def __init__(self, path):
                self.names = {0: "a", 1: "b"}
            def val(self, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(box=types.SimpleNamespace(
                    map50=.7, map=.5, all_ap=np.ones((2, 10))*.5,
                    ap_class_index=[0, 1], p=[.6, .6], r=[.7, .7]))
        fake = types.ModuleType("ultralytics")
        fake.YOLO = FakeYOLO
        original_module, argv = sys.modules.get("ultralytics"), sys.argv
        try:
            sys.modules["ultralytics"] = fake
            sys.argv = ["validation_worker.py", str(payload_path)]
            exec(compile(HELPERS["HPO_VALIDATOR_SOURCE"], "validation_worker.py", "exec"), {})
        finally:
            sys.argv = argv
            if original_module is None:
                del sys.modules["ultralytics"]
            else:
                sys.modules["ultralytics"] = original_module
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["split"], "val")
        self.assertEqual(calls[0]["imgsz"], 1024)
        self.assertEqual(calls[0]["device"], 0)
        self.assertTrue(calls[0]["nms"])
        saved = json.loads((directory / "validation.json").read_text())
        self.assertEqual(saved["identity"], payload)
        self.assertEqual(len(saved["perclass"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
