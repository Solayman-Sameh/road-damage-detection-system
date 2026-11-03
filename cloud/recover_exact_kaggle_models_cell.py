# Paste into one Kaggle CODE cell after section 6 (the recovery-helper cell).
# It can also be run at the end of the already-executed notebook.
# This copies/registers finished models. It never starts training.
from pathlib import Path

required = ["TRIALS", "trial_dir", "completed_status", "register_finished_download",
            "file_sha", "OUTPUT"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError("Run the notebook from the top through the recovery-helper "
                       "cell in section 6 first. Missing: " + ", ".join(missing))

REUSE_TRAINED_MODELS = True
REQUIRE_RECOVERED_MODELS = True
RUN_TRAINING = False
RUN_PILOT = False
RUN_CONTROLS = False
RUN_WEIGHTING_TRIAL = False

RECOVERED_MODEL_INPUTS = {
    "v8_small": Path("/kaggle/input/datasets/solaymanabdelfattah/3x-v2-models/v2/yolov8n/best.pt"),
    "v11_small": Path("/kaggle/input/datasets/solaymanabdelfattah/3x-v2-models/v2/yolov11s/best.pt"),
    "v26_medium": Path("/kaggle/input/datasets/solaymanabdelfattah/3x-v2-models/v2/yolo26m/best.pt"),
}
expected_hashes = {
    "v8_small": "5a5b96a5204dffb8371ea621b6461bc3939ec0b11fc9f6ebfbc3af38d684a3ba",
    "v11_small": "aed9032517943b4a5ffe30a01a74c974bc8e7e625b6b7a6e6ede1f688e1a6c63",
    "v26_medium": "fab0e117d8871eeea47a45a98e33963a524b7751bae793c796258bf1a457c0c4",
}

# Check every input before copying any of them. Folder names are not architectures.
for key, source in RECOVERED_MODEL_INPUTS.items():
    if not source.is_file():
        raise FileNotFoundError(f"Missing Kaggle input: {source}")
    if file_sha(source) != expected_hashes[key]:
        raise ValueError(f"{key}: this file differs from your audited trained model.")
    run = trial_dir(key, TRIALS[key])
    if (run / "complete.json").exists():
        completed_status(run, TRIALS[key])
        if file_sha(run / "weights/best.pt") != expected_hashes[key]:
            raise ValueError(f"{key}: the completed destination is not the audited input model.")
    elif (OUTPUT / "frozen_test_plan.json").exists():
        raise RuntimeError("This output has a frozen test plan but a missing model "
                           "completion record. Restore the full saved output or "
                           "use a fresh Kaggle session; do not delete the frozen plan.")

for key, source in RECOVERED_MODEL_INPUTS.items():
    run = trial_dir(key, TRIALS[key])
    if not (run / "complete.json").exists():
        register_finished_download(key, source)
    else:
        print(f"REUSE: {key} | verified completed model")
    print(f"  Notebook checkpoint: {run / 'weights/best.pt'}")

print("All three models are ready. Original training and pilot remain OFF.")
