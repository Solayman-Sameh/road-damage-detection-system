# Paste this entire file into one CODE cell at the END of the current notebook.
# Run the notebook's setup/data/recovery/validation cells first.
# Change only RUN_OPTUNA to True when ready to spend GPU time.
RUN_OPTUNA = False
HPO_TARGET_TRIALS = 3             # TOTAL per model, including the reference trial.
HPO_EPOCHS = 3                    # EXTRA epochs per trial, starting from v2 best.pt.
HPO_MAX_WALL_HOURS = 2.0          # Cumulative worker time, including validation.
HPO_MAX_TRIAL_MINUTES = 20        # Cumulative training time for each individual trial.
HPO_ROOT = "/kaggle/working/road_models_v3_tuning"
HPO_RESTORE_FROM = None           # Optional attached saved road_models_v3_tuning folder.

from pathlib import Path
import ast, gc, hashlib, json, math, os, shutil, signal, subprocess, sys, time


def _hpo_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _hpo_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _hpo_json(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Experiment identity changed: {path}. Use a new HPO_ROOT.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def _hpo_spent(root, job=None):
    # An unclosed attempt is charged up to the current time. Session loss cannot
    # silently reset the budget. A restored stale attempt is therefore conservative.
    total = 0.0
    for path in Path(root).glob("attempts/*.json"):
        record = json.loads(path.read_text())
        if job is None or record["job"] == job:
            total += max(0.0, record.get("finished", time.time()) - record["started"])
    return total


def _hpo_launch(command, root, job, job_cap_seconds):
    remaining = min(HPO_MAX_WALL_HOURS * 3600 - _hpo_spent(root),
                    job_cap_seconds - _hpo_spent(root, job))
    if remaining <= 0:
        raise TimeoutError("Tuning time budget exhausted. Saved trials/checkpoints are retained.")
    attempts = Path(root) / "attempts"
    attempts.mkdir(exist_ok=True)
    token = f"{time.time_ns()}-{os.getpid()}"
    path = attempts / (token + ".json")
    logfile = attempts / (token + ".log")
    record = {"job": job, "started": time.time(), "command": command,
              "log": str(logfile)}
    _hpo_json(path, record)
    process = None
    try:
        with logfile.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            next_update = time.monotonic() + 30
            deadline = time.monotonic() + remaining
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{job}: time cap reached. Rerun this cell to resume "
                                       "only after explicitly extending the exhausted budget.")
                if time.monotonic() >= next_update:
                    print(f"  {job}: running; log {logfile}", flush=True)
                    next_update = time.monotonic() + 30
                time.sleep(1)
            if process.returncode:
                tail = "\n".join(logfile.read_text(errors="replace").splitlines()[-45:])
                raise RuntimeError(f"{job} failed; settings were not changed.\n{tail}")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        record.update(finished=time.time(), exit_code=None if process is None else process.returncode)
        _hpo_json(path, record)


# Separate validation subprocess: GPU 0, FP32, batch 1, validation split ONLY.
HPO_VALIDATOR_SOURCE = r'''
from pathlib import Path
import hashlib, json, math, os, sys
import numpy as np
from ultralytics import YOLO
p = json.loads(Path(sys.argv[1]).read_text())
assert hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest() == p['checkpoint_sha256']
model = YOLO(p['checkpoint'])
assert list(model.names.values()) == p['names']
result = model.val(data=p['data'], split='val', imgsz=1024, batch=1, device=0,
    quantize=32, rect=False, conf=0.001, iou=0.7, max_det=300, nms=True,
    plots=False, save_json=False, project=p['project'], name='validator',
    exist_ok=True, verbose=False)
box = result.box
aps = np.asarray(box.all_ap)
assert np.isfinite([box.map50, box.map, *aps.ravel()]).all()
rows = [{'class':p['names'][int(c)], 'ap50':float(aps[j,0]),
         'ap50_95':float(aps[j].mean()), 'validator_precision':float(box.p[j]),
         'validator_recall':float(box.r[j])}
        for j,c in enumerate(box.ap_class_index)]
answer = {'identity':p, 'map50':float(box.map50), 'map50_95':float(box.map),
          'perclass':rows}
target = Path(p['project'])/'validation.json'
temporary = target.with_suffix('.tmp')
temporary.write_text(json.dumps(answer, indent=2, allow_nan=False))
os.replace(temporary, target)
'''


def _hpo_validate(checkpoint, directory, root, validator_path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint": str(checkpoint), "checkpoint_sha256": _hpo_sha(checkpoint),
                "names": NAMES, "data": str(dataset_yaml), "data_id": DATA_ID,
                "project": str(directory), "protocol": "val-1024-FP32-NMS-batch1",
                "validator_sha256": _hpo_sha(validator_path),
                "environment": ENVIRONMENT["packages"]}
    _hpo_json(directory / "validation_input.json", identity, immutable=True)
    cache = directory / "validation.json"
    if not cache.exists():
        _hpo_launch([sys.executable, str(validator_path), str(directory / "validation_input.json")],
                    root, "validation:" + str(directory.relative_to(root)), 600)
    saved = json.loads(cache.read_text())
    if saved["identity"] != identity or not all(math.isfinite(saved[k]) for k in ("map50", "map50_95")):
        raise ValueError(f"Invalid/stale validation cache: {cache}")
    return saved


def _hpo_run_study(optuna, study, target, evaluate):
    """Resume RUNNING trials before asking for new ones. The limit is a TOTAL."""
    state = optuna.trial.TrialState
    while True:
        running = study.get_trials(deepcopy=False, states=(state.RUNNING,))
        if len(running) > 1:
            raise RuntimeError("More than one running trial exists. Use only one tuning runner.")
        if running:
            trial = running[0]
            if set(trial.params) != {"lr0", "cls_pw"}:
                raise RuntimeError("A trial was interrupted while sampling parameters. "
                                   "Inspect the study before continuing; no new training was started.")
        else:
            attempted = study.get_trials(deepcopy=False,
                                         states=(state.COMPLETE, state.FAIL, state.PRUNED))
            if len(attempted) >= target:
                return
            # Reconstruct from history with a trial-specific seed, including after restart.
            study.sampler = optuna.samplers.TPESampler(seed=2026 + len(attempted),
                                                       n_startup_trials=2)
            live = study.ask(fixed_distributions={
                "lr0": optuna.distributions.FloatDistribution(1e-5, 3e-4, log=True),
                "cls_pw": optuna.distributions.CategoricalDistribution([0.0, 0.5]),
            })
            trial = next(t for t in study.get_trials(deepcopy=False) if t.number == live.number)
        try:
            score = evaluate(trial.number, dict(trial.params))
            if not math.isfinite(score):
                raise ValueError("Trial produced a non-finite validation score.")
        except (TimeoutError, KeyboardInterrupt):
            # Remain RUNNING so the same trial directory/optimizer checkpoint resumes.
            raise
        except Exception:
            study.tell(trial.number, state=state.FAIL)
            raise
        study.tell(trial.number, score)


def _hpo_main():
    import importlib.util
    required = ["COMMON", "TRIALS", "WORKER_SOURCE", "DATA_ID", "ENVIRONMENT",
                "dataset_yaml", "NAMES", "completed_status", "trial_dir"]
    missing = [key for key in required if key not in globals()]
    if missing:
        raise RuntimeError("Run setup, dataset, recovery, and model cells first. Missing: " + ", ".join(missing))
    if RUN_TRAINING or not REUSE_TRAINED_MODELS:
        raise RuntimeError("Keep REUSE_TRAINED_MODELS=True and RUN_TRAINING=False for original models.")
    if not isinstance(HPO_TARGET_TRIALS, int) or HPO_TARGET_TRIALS < 2:
        raise ValueError("Use at least two trials per model, including the reference.")
    if not isinstance(HPO_EPOCHS, int) or HPO_EPOCHS < 2:
        raise ValueError("Use at least two extra epochs per trial.")
    if not all(math.isfinite(x) and x > 0 for x in [HPO_MAX_WALL_HOURS, HPO_MAX_TRIAL_MINUTES]):
        raise ValueError("Time budgets must be positive finite numbers.")
    import torch, ultralytics
    if os.name != "posix" or torch.cuda.device_count() != 2:
        raise RuntimeError("This cell requires Kaggle Linux with GPU T4 x2.")
    if ultralytics.__version__ != "8.4.191":
        raise RuntimeError("Keep the notebook's audited ultralytics==8.4.191.")
    sources = {
        "v8_small": ("yolov8s", "/kaggle/input/datasets/solaymanabdelfattah/3x-v2-models/v2/yolov8n/best.pt",
                     "5a5b96a5204dffb8371ea621b6461bc3939ec0b11fc9f6ebfbc3af38d684a3ba"),
        "v11_small": ("yolo11s", "/kaggle/input/datasets/solaymanabdelfattah/3x-v2-models/v2/yolov11s/best.pt",
                      "aed9032517943b4a5ffe30a01a74c974bc8e7e625b6b7a6e6ede1f688e1a6c63"),
        "v26_medium": ("yolo26m", "/kaggle/input/datasets/solaymanabdelfattah/3x-v2-models/v2/yolo26m/best.pt",
                       "fab0e117d8871eeea47a45a98e33963a524b7751bae793c796258bf1a457c0c4"),
    }
    for key, (model, source, checksum) in sources.items():
        if not Path(source).is_file() or _hpo_sha(source) != checksum:
            raise ValueError(f"Missing or changed trained Kaggle input: {source}")
        if TRIALS[key]["model"] != model:
            raise ValueError(f"{key}: original notebook architecture changed.")
        completed_status(trial_dir(key, TRIALS[key]), TRIALS[key])
    if importlib.util.find_spec("optuna") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna==4.5.0"])
    import optuna
    import pandas as pd
    import fcntl
    root = Path(HPO_ROOT)
    if not root.is_absolute() or not root.resolve().is_relative_to(Path("/kaggle/working")):
        raise ValueError("HPO_ROOT must be below /kaggle/working.")
    if HPO_RESTORE_FROM is not None and not (root / "plan.json").exists():
        restore = Path(HPO_RESTORE_FROM).resolve()
        if not restore.is_relative_to(Path("/kaggle/input")) or not (restore / "plan.json").is_file():
            raise ValueError("Point HPO_RESTORE_FROM to your attached saved tuning output folder.")
        if root.exists():
            raise ValueError("Restore into a fresh HPO_ROOT; do not merge experiments.")
        shutil.copytree(restore, root, ignore=shutil.ignore_patterns(".runner.lock"))
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another tuning cell is already running in this output folder.")
        # Reuse the notebook's checked DDP worker, but with NEW trial directories
        # and your road-trained input as 'pretrained'. No baseline globals are changed.
        ast.parse(WORKER_SOURCE)
        train_worker = root / "training_worker.py"
        val_worker = root / "validation_worker.py"
        fixed = {**COMMON, "epochs": HPO_EPOCHS, "patience": HPO_EPOCHS + 1,
                 "imgsz": 1024, "batch": 8, "device": [0, 1], "optimizer": "AdamW",
                 "weight_decay": 0.0005, "lr0": 1e-4, "lrf": 0.1,
                 "warmup_epochs": 0.0, "warmup_bias_lr": 0.0,
                 "mosaic": 0.0, "close_mosaic": 0, "plots": False, "save_period": -1}
        plan = {"kind": "short warm-start fine-tuning search", "data_id": DATA_ID,
                "environment": ENVIRONMENT, "optuna": optuna.__version__, "sources": sources,
                "fixed_train": fixed, "search": {"lr0": [1e-5, 3e-4, "log"], "cls_pw": [0.0, 0.5]},
                "reference": {"lr0": 1e-4, "cls_pw": 0.0},
                "training_worker_sha256": hashlib.sha256(WORKER_SOURCE.encode()).hexdigest(),
                "validation_worker_sha256": hashlib.sha256(HPO_VALIDATOR_SOURCE.encode()).hexdigest()}
        # JSON roundtrip also normalizes tuples before comparing a restored plan.
        plan = json.loads(json.dumps(plan))
        _hpo_json(root / "plan.json", plan, immutable=True)
        _hpo_json(root / "limits.json", {"total_trials_per_model": HPO_TARGET_TRIALS,
                  "extra_epochs": HPO_EPOCHS, "max_wall_hours": HPO_MAX_WALL_HOURS,
                  "max_training_minutes_per_trial": HPO_MAX_TRIAL_MINUTES})
        train_worker.write_text(WORKER_SOURCE)
        val_worker.write_text(HPO_VALIDATOR_SOURCE)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Tuning: {len(sources)} models x {HPO_TARGET_TRIALS} total trials x "
              f"{HPO_EPOCHS} extra epochs. Both T4s train ONE trial at a time.", flush=True)
        print("Search: learning rate + class weighting; weight decay is fixed. "
              "Test and original model exports are untouched.", flush=True)
        storage = "sqlite:///" + str(root / "optuna.db")
        baseline_scores, studies = {}, {}

        def write_reports():
            rows, winners = [], []
            for key, study in studies.items():
                model, source, checksum = sources[key]
                base = baseline_scores[key]
                rows.append({"model": model, "kind": "original_v2", "trial": None,
                             "extra_epochs": 0, "map50_95": base["map50_95"],
                             "map50": base["map50"], "checkpoint": source,
                             "checkpoint_sha256": checksum})
                for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
                    directory = root / key / f"trial_{trial.number:03d}"
                    metrics = json.loads((directory / "evaluation/validation.json").read_text())
                    ck = directory / "weights/best.pt"
                    if _hpo_sha(ck) != metrics["identity"]["checkpoint_sha256"]:
                        raise ValueError(f"Completed tuning checkpoint changed: {ck}")
                    rows.append({"model": model, "kind": "reference" if trial.number == 0 else "tuned",
                                 "trial": trial.number, "extra_epochs": HPO_EPOCHS,
                                 **trial.params, "map50_95": metrics["map50_95"],
                                 "map50": metrics["map50"], "checkpoint": str(ck),
                                 "checkpoint_sha256": _hpo_sha(ck)})
                family = sorted([r for r in rows if r["model"] == model],
                                key=lambda r: (-r["map50_95"], -r["map50"], r["extra_epochs"]))
                winners.append(family[0])
                study.trials_dataframe().to_csv(root / key / "optuna_trials.csv", index=False)
            table = pd.DataFrame(rows)
            table.to_csv(root / "validation_comparison.csv", index=False)
            _hpo_json(root / "validation_selected.json", {
                "scope": "provisional validation selection; short search, no new test/export",
                "plan_id": _hpo_digest(plan), "per_model": winners,
                "overall": sorted(winners, key=lambda r: (-r["map50_95"], -r["map50"]))[0]})
            return table

        for key, (model, source, checksum) in sources.items():
            (root / key).mkdir(exist_ok=True)
            baseline_scores[key] = _hpo_validate(source, root / key / "baseline", root, val_worker)
            study = optuna.create_study(study_name=key, storage=storage, direction="maximize",
                                        load_if_exists=True)
            studies[key] = study
            if not study.trials:
                study.enqueue_trial({"lr0": 1e-4, "cls_pw": 0.0})

            def evaluate(number, params):
                run = root / key / f"trial_{number:03d}"
                run.mkdir(parents=True, exist_ok=True)
                cfg = {**fixed, **params, "project": str(run.parent), "name": run.name}
                payload = {"train": cfg, "pretrained": source, "pretrained_sha256": checksum,
                           "dataset_id": DATA_ID, "environment": ENVIRONMENT["packages"],
                           "worker_sha256": _hpo_sha(train_worker)}
                payload_path = run / "training_spec.json"
                _hpo_json(payload_path, payload, immutable=True)
                print(f"{key} trial {number + 1}/{HPO_TARGET_TRIALS}: {params}", flush=True)
                if not (run / "complete.json").exists():
                    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                               "--nproc_per_node=2", str(train_worker), str(payload_path)]
                    # Resume interrupted trials only from THEIR optimizer checkpoint.
                    # A final saved epoch can recover its metadata without more training.
                    if (run / "weights/last-resumable.pt").exists() and (run / "results.csv").exists():
                        if len(pd.read_csv(run / "results.csv")) == HPO_EPOCHS:
                            command.append("--recover-only")
                    elif (run / "results.csv").exists() or (run / "weights/best.pt").exists():
                        raise RuntimeError(f"{run}: partial output has no resumable state. "
                                           "No automatic restart was attempted.")
                    _hpo_launch(command, root, "train:" + str(run.relative_to(root)),
                                HPO_MAX_TRIAL_MINUTES * 60)
                completed_status(run, cfg)
                metrics = _hpo_validate(run / "weights/best.pt", run / "evaluation", root, val_worker)
                print(f"  validation mAP50-95: {metrics['map50_95']:.4f}", flush=True)
                return metrics["map50_95"]

            try:
                _hpo_run_study(optuna, study, HPO_TARGET_TRIALS, evaluate)
            finally:
                # Also retain a readable report if a later trial hits an error or cap.
                write_reports()
        table = write_reports()
        print(table[["model", "kind", "trial", "extra_epochs", "map50_95", "map50"]].to_string(index=False))
        print(f"Reports, study database and trial checkpoints: {root}")
        print("Save Version WITH OUTPUTS before stopping Kaggle. For another session, "
              "attach that output and set HPO_RESTORE_FROM to its tuning folder.")
        print("This is a small exploratory search. Confirm the best recipe against "
              "its reference with equal additional epochs before the final test/export.")


if RUN_OPTUNA:
    _hpo_main()
else:
    print("Tuning is OFF. Set RUN_OPTUNA=True in this cell when ready. "
          "Leave original RUN_TRAINING=False.")
