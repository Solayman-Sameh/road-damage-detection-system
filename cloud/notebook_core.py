# %% Environment — install missing packages; keep Kaggle's Torch/CUDA stack.
from pathlib import Path
import sys, os, json, time, gc, shutil, subprocess, importlib.util, hashlib, math, signal
from collections import Counter
from importlib.metadata import version
from IPython.display import display
os.chdir('/kaggle/working')

required = {'ultralytics': 'ultralytics==8.4.191', 'pandas': 'pandas',
            'yaml': 'PyYAML', 'PIL': 'Pillow', 'numpy': 'numpy'}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if importlib.util.find_spec('ultralytics') is not None and version('ultralytics') != '8.4.191':
    missing.append(required['ultralytics'])
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
import numpy as np
import pandas as pd
import torch, torchvision, yaml, ultralytics
import importlib
torch._utils = importlib.import_module('torch._utils')
from PIL import Image
from ultralytics import YOLO
assert ultralytics.__version__ == '8.4.191', 'Use audited ultralytics 8.4.191; do not change Torch/CUDA.'
assert torch.cuda.device_count() == 2, 'Select GPU T4 x2 in Kaggle before running.'
print({p: version(p) for p in ['torch','torchvision','ultralytics','numpy','Pillow']})
print([torch.cuda.get_device_name(i) for i in range(2)])

# %% Settings — identical training budgets and inference settings.
import importlib
import torch
torch._utils = importlib.import_module('torch._utils')
assert hasattr(torch._utils, '_element_size'), 'PyTorch is incompletely initialized. Restart the kernel and rerun cells 1–3.'

MODELS = ['yolov8n', 'yolo11n', 'yolo26n']
RUNS = Path('/kaggle/working/road_models'); RUNS.mkdir(exist_ok=True)
IMG_SIZE, SEED, EPOCHS, BATCH = 640, 2026, 20, 16
for name in MODELS:
    checkpoint = Path('/kaggle/working')/f'{name}.pt'
    print(f'Loading {name}...', flush=True)
    probe = YOLO(str(checkpoint))  # Reuse existing weights; download only if missing.
    del probe
config = dict(data=str(dataset_yaml), imgsz=IMG_SIZE, seed=SEED, epochs=EPOCHS,
              batch=BATCH, device=[0,1], workers=4, cache=False, amp=True,
              optimizer='AdamW', lr0=0.001, lrf=0.01, patience=20,
              deterministic=True, save=True, save_period=1, plots=True,
              project=str(RUNS), exist_ok=True, verbose=False)
(RUNS/'settings.json').write_text(json.dumps(config, indent=2))
print('3 models; 20 epochs and at most 2 hours each; dual GPU, batch 16.')

# %% Worker — preserve resumable checkpoints before final optimizer stripping.
worker = r'''
from pathlib import Path
import os, sys, json, shutil, torch
import importlib
torch._utils = importlib.import_module('torch._utils')
from ultralytics import YOLO
name = sys.argv[1]
cfg = json.loads(Path('/kaggle/working/road_models/settings.json').read_text())
if name == 'pilot': cfg.update(epochs=1, patience=1, plots=False)
run = Path(cfg['project'])/name
resume = run/'weights/last-resumable.pt'
rank = int(os.environ.get('RANK', '-1'))
def preserve(trainer):
    if rank in (-1,0) and trainer.last.exists():
        shutil.copy2(trainer.last, trainer.last.with_name('last-resumable.pt'))
if resume.exists():
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
    assert checkpoint.get('optimizer') is not None, 'Checkpoint cannot resume optimizer.'
    del checkpoint
base = 'yolov8n' if name == 'pilot' else name
model = YOLO(str(resume) if resume.exists() else f'/kaggle/working/{base}.pt')
model.add_callback('on_model_save', preserve)
model.train(**cfg, name=name, resume=str(resume) if resume.exists() else False)
if rank in (-1,0):
    assert (run/'weights/best.pt').is_file()
    (run/'complete.json').write_text(json.dumps({'model':name,'training_returned':True}))
'''
worker_path = RUNS/'train_worker.py'; worker_path.write_text(worker)

# %% Training launcher — logs stay visible; each model has a two-hour limit.
def train(name):
    run = RUNS/name
    if name != 'pilot': assert (RUNS/'pilot/reload-passed.json').is_file(), 'Run the short pilot first.'
    if (run/'complete.json').exists():
        print(name, 'already finished; reuse saved checkpoint.'); return
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=2', str(worker_path), name]
    history = run/'timing.jsonl'
    spent = sum(json.loads(line)['seconds'] for line in history.read_text().splitlines()) if history.exists() else 0
    remaining = (600 if name == 'pilot' else 7200) - spent
    assert remaining > 0, f'{name}: time budget exhausted; preserve checkpoint and review run.'
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=remaining)
        assert code == 0, f'{name} failed. Inspect the training log above.'
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if process.poll() is None: os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
        raise RuntimeError(f'{name} interrupted/capped; keep checkpoint and resume explicitly.')
    finally:
        run.mkdir(parents=True, exist_ok=True)
        with (run/'timing.jsonl').open('a') as log:
            log.write(json.dumps({'seconds':time.monotonic()-started,'exit_code':process.returncode})+'\n')

# %% Pilot — one epoch, maximum ten minutes; validates training and checkpoint reload.
train('pilot')
pilot_history = pd.read_csv(RUNS/'pilot/results.csv')
loss_cols = [c for c in pilot_history if 'loss' in c]
assert len(pilot_history) == 1 and loss_cols and np.isfinite(pilot_history[loss_cols].to_numpy()).all()
resume_check = torch.load(RUNS/'pilot/weights/last-resumable.pt', map_location='cpu', weights_only=False)
assert resume_check.get('optimizer') is not None
del resume_check
pilot = YOLO(str(RUNS/'pilot/weights/best.pt'))
pilot_image = Path(json.loads(dataset_yaml.read_text())['val']).read_text().splitlines()[0]
assert len(pilot.predict(pilot_image, imgsz=640, device=0, nms=True, verbose=False)) == 1
del pilot; gc.collect(); torch.cuda.empty_cache()
(RUNS/'pilot/reload-passed.json').write_text(json.dumps({'reload':'passed'}))
print('Pilot checkpoint reload passed.')

# %% Train YOLOv8n.
train('yolov8n')

# %% Train YOLO11n.
train('yolo11n')

# %% Train YOLO26n.
train('yolo26n')

# %% Validation — choose a winner before looking at the held-out test set.
checkpoints = {name: RUNS/name/'weights/best.pt' for name in MODELS}
assert all((RUNS/name/'complete.json').is_file() for name in MODELS), 'Finish all three runs first.'
validation = []
for name, checkpoint in checkpoints.items():
    model = YOLO(str(checkpoint))
    assert list(model.names.values()) == NAMES, f'{name}: class order differs.'
    metrics = model.val(data=str(dataset_yaml), split='val', imgsz=640, batch=1,
                        device=0, quantize=32, rect=False, conf=0.001, iou=0.7,
                        nms=True, plots=True, project=str(RUNS/'evaluation'), name=name+'_val')
    assert np.isfinite([metrics.box.map50, metrics.box.map]).all(), f'{name}: invalid validation metrics.'
    validation.append(dict(model=name, map50=float(metrics.box.map50), map50_95=float(metrics.box.map)))
    del model, metrics; gc.collect(); torch.cuda.empty_cache()
validation = pd.DataFrame(validation).sort_values('map50_95', ascending=False)
validation.to_csv(RUNS/'validation.csv', index=False)
winner = validation.iloc[0]['model']
(RUNS/'selection.json').write_text(json.dumps({'winner':winner,'rule':'highest validation mAP50:95; no test tuning'}))
display(validation)

# %% Export — retain all models, configurations, and comparison reports on Kaggle.
export = RUNS/'export'; export.mkdir(exist_ok=True)
for name, checkpoint in checkpoints.items():
    shutil.copy2(checkpoint, export/f'{name}-best.pt')
selected = YOLO(str(export/f'{winner}-best.pt'))
assert list(selected.names.values()) == NAMES
probe = selected.predict(str(test_paths()[0]), imgsz=640, device=0, quantize=32, nms=True, verbose=False)
assert len(probe) == 1
del selected, probe; gc.collect(); torch.cuda.empty_cache()
(export/'labels.json').write_text(json.dumps(NAMES, indent=2))
(export/'selected_model.json').write_text(json.dumps({'model':winner,'file':f'{winner}-best.pt'}))
print('Export reload passed. Use Save Version > Quick Save > save outputs; then stop the session.')
