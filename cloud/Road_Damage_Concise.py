# %% [markdown]
# ## 1. Environment
# Install only missing packages, then confirm the audited two-GPU runtime.

# %%
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

# %% [markdown]
# ## 2. Dataset
# Verify the saved SVRDD audit and prepare leakage-screened manifests for the attached data.

# %%
SOURCE = Path('/kaggle/input/notebooks/solaymanabdelfattah/instant-sprint02-project02-road-damage-detection')
if not (SOURCE/'prepared/image_manifest.jsonl').is_file():
    candidates = [p.parent.parent for p in Path('/kaggle/input').rglob('image_manifest.jsonl')
                  if (p.parent/'data_preparation_audit.json').is_file()]
    assert len(candidates) == 1, 'Add Input: attach version 1 of the verified SVRDD notebook output.'
    SOURCE = candidates[0]
NAMES = ['longitudinal crack', 'transverse crack', 'alligator crack', 'pothole',
         'manhole cover', 'longitudinal patch', 'transverse patch']
audit_file = SOURCE/'prepared/data_preparation_audit.json'
manifest_file = SOURCE/'prepared/image_manifest.jsonl'
assert audit_file.is_file() and manifest_file.is_file(), 'Attach version 1: Verified SVRDD data and leakage audit.'
audit = json.loads(audit_file.read_text()); checks = audit['checks']
assert audit['archive']['md5'] == '4b897aa7081e8d0fa07f3d330e86e936'
assert audit['archive']['bytes'] == 2_893_208_002
expected = {'decoded_images':8000, 'decode_errors':0, 'geometry_errors':0,
            'empty_labels':0, 'exact_duplicate_pairs':0, 'perceptual_candidates_dhash_le_3':183}
assert all(checks.get(k) == v for k,v in expected.items()), 'Saved audit differs.'
assert checks['class_names'] == NAMES
assert audit['status'] == 'failed' and audit['errors']
assert all('perceptual' in str(e).lower() or 'dhash' in str(e).lower() for e in audit['errors'])
roots = [p.parent for p in (SOURCE/'prepared/extracted').rglob('train.txt')
         if (p.parent/'val.txt').is_file() and (p.parent/'test.txt').is_file()]
assert len(roots) == 1; dataset_root = roots[0].resolve()
records = [json.loads(x) for x in manifest_file.read_text().splitlines() if x.strip()]
assert len(records) == 8000 and len({r['path'] for r in records}) == 8000
priority = {'train':0, 'val':1, 'test':2}; excluded = set(); edges = []
hashes = [int(r['dhash64'],16) for r in records]
for i,left in enumerate(records):
    for j in range(i+1,len(records)):
        right = records[j]
        if left['split'] != right['split'] and (hashes[i]^hashes[j]).bit_count() <= 3:
            loser = left if priority[left['split']] < priority[right['split']] else right
            excluded.add(loser['path']); edges.append([left['path'],right['path'],loser['path']])
assert len(edges) == 183 and len(excluded) == 109
out = Path('/kaggle/working/svrdd_screened'); out.mkdir(exist_ok=True)
paths = {}; counts = {}; objects = {}
for split, expected_count in [('train',5918),('val',973),('test',1000)]:
    images = [(dataset_root/r['path']).resolve() for r in records if r['split']==split and r['path'] not in excluded]
    assert len(images) == expected_count and all(p.is_relative_to(dataset_root) and p.is_file() for p in images)
    support = Counter()
    for image in images:
        parts = list(image.parts); parts[parts.index('images')] = 'labels'
        label = Path(*parts).with_suffix('.txt')
        for line_no, line in enumerate(label.read_text().splitlines(), 1):
            if not line.strip(): continue
            c,x,y,w,h = map(float,line.split())
            assert c.is_integer() and 0<=c<7 and all(map(math.isfinite,[x,y,w,h])), f'{label}:{line_no}: invalid class or nonfinite box'
            bounds = (x-w/2, y-h/2, x+w/2, y+h/2)
            assert w>0 and h>0 and min(bounds)>=-1e-6 and max(bounds)<=1+1e-6, f'{label}:{line_no}: invalid box {bounds}; size={(w,h)}'
            support[int(c)] += 1
    assert all(support[c]>0 for c in range(7))
    target = out/f'{split}.txt'; target.write_text('\n'.join(map(str,images))+'\n')
    paths[split] = str(target); counts[split] = len(images); objects[split] = dict(support)
assert [[objects[s].get(c,0) for c in range(7)] for s in priority] == [
    [3400,2520,1315,673,2468,3077,1945], [576,414,219,111,393,506,337], [616,436,182,124,419,500,295]]
dataset_yaml = out/'svrdd.yaml'
dataset_yaml.write_text(json.dumps({'path':str(dataset_root), **paths, 'names':NAMES}, indent=2))
(out/'split_audit.json').write_text(json.dumps({'source_md5':audit['archive']['md5'],
    'manifest_sha256':hashlib.sha256(manifest_file.read_bytes()).hexdigest(), 'images':counts,
    'class_objects':objects, 'candidate_pairs':edges, 'excluded':sorted(excluded),
    'policy':'exclude possible leakage; no duplicate claim; source/frame separation unverified'}, indent=2))
print('Ready:', dataset_yaml, counts, '; all seven classes retained.')

# %% [markdown]
# ## 3. Settings
# Set the shared model, resolution, seed, and bounded training configuration.

# %%
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

# %% [markdown]
# ## 4. Worker
# Define the resumable per-model training worker and checkpoint preservation callback.

# %%
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

# %% [markdown]
# ## 5. Training launcher
# Run each bounded training job, preserve its checkpoint, and record its duration.

# %%
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

# %% [markdown]
# ## 6. Pilot
# Run the one-epoch pilot and confirm its saved checkpoint reloads for inference.

# %%
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

# %% [markdown]
# ## 7. Train YOLOv8n
# Train the YOLOv8 nano baseline within the shared experiment budget.

# %%
train('yolov8n')

# %% [markdown]
# ## 8. Train YOLO11n
# Train the YOLO11 nano candidate with the same settings and budget.

# %%
train('yolo11n')

# %% [markdown]
# ## 9. Train YOLO26n
# Train the YOLO26 nano candidate with the same settings and budget.

# %%
train('yolo26n')

# %% [markdown]
# ## 10. Validation
# Compare all completed models on validation data and select the winner before test evaluation.

# %%
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

# %% [markdown]
# ## 11. Test manifest and matching
# Define the fixed test paths, ground-truth loader, matching rule, and training evidence checks.

# %%
MAP_CONF, FIXED_CONF, NMS_IOU, MAX_DET = 0.001, 0.25, 0.7, 300
def test_paths():
    cfg = yaml.safe_load(Path(dataset_yaml).read_text())
    path = Path(cfg['test'])
    path = path if path.is_absolute() else Path(cfg.get('path', '.')) / path
    if path.suffix.lower() != '.txt': raise ValueError('Expected the frozen test manifest text file')
    return [Path(x.strip()).resolve() for x in path.read_text().splitlines() if x.strip()]

def ground_truth(image):
    parts = list(image.parts); low = [x.lower() for x in parts]
    if 'images' not in low: raise ValueError(f'No images/ component: {image}')
    parts[len(low)-1-low[::-1].index('images')] = 'labels'
    label = Path(*parts).with_suffix('.txt')
    if not label.is_file(): raise FileNotFoundError(label)
    with Image.open(image) as im: width, height = im.size
    rows = []
    for line in label.read_text().splitlines():
        if line.strip():
            c,cx,cy,w,h = map(float,line.split())
            rows.append([int(c),(cx-w/2)*width,(cy-h/2)*height,
                         (cx+w/2)*width,(cy+h/2)*height])
    return np.asarray(rows,dtype=float).reshape(-1,5)

def match_boxes(gt, pred):
    counts = np.zeros((len(NAMES),3),dtype=int); used=set(); matched=[]
    for k in np.argsort(-pred[:,1],kind='stable') if len(pred) else []:
        cls=int(pred[k,0]); ids=np.array([j for j in range(len(gt))
                            if j not in used and int(gt[j,0])==cls],dtype=int)
        if len(ids):
            b=pred[k,2:6]; g=gt[ids,1:5]
            wh=np.maximum(0,np.minimum(g[:,2:],b[2:])-np.maximum(g[:,:2],b[:2]))
            inter=wh[:,0]*wh[:,1]; ag=(g[:,2]-g[:,0])*(g[:,3]-g[:,1])
            ab=(b[2]-b[0])*(b[3]-b[1]); union=ag+ab-inter
            ious=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
            j=int(np.argmax(ious))
            if ious[j]>=.5: used.add(int(ids[j])); counts[cls,0]+=1; matched.append(float(ious[j])); continue
        counts[cls,1]+=1
    for j,cls in enumerate(gt[:,0].astype(int)):
        if j not in used: counts[cls,2]+=1
    return counts,matched

def train_evidence(name):
    run=RUNS/name; df=pd.read_csv(run/'results.csv'); df.columns=[x.strip() for x in df.columns]
    cols=[x for x in df if x.lower().startswith(('train/','val/')) and 'loss' in x.lower()]
    if not len(df) or not cols or not np.isfinite(df[cols].to_numpy(float)).all():
        raise ValueError(f'{name}: missing/nonfinite train or validation losses')
    records=[json.loads(x) for x in (run/'timing.jsonl').read_text().splitlines() if x.strip()]
    seconds=[r['seconds'] for r in records]
    if not seconds or not np.isfinite(seconds).all(): raise ValueError(f'{name}: timing evidence invalid')
    return {'actual_epochs':len(df),'losses_finite':True,'training_seconds':sum(seconds),
            'last_losses':df[cols].iloc[-1].to_dict()}

# %% [markdown]
# ## 12. GPU prediction and timing
# Set one shared prediction and synchronized timing boundary for all candidates.

# %%
def predict_boxes(model,image,confidence):
    result=model.predict(str(image),imgsz=640,device=0,batch=1,quantize=32,rect=False,
        conf=confidence,iou=NMS_IOU,max_det=MAX_DET,nms=True,verbose=False,save=False)[0]
    boxes=result.boxes.data.detach().cpu().numpy() if result.boxes is not None else np.empty((0,6))
    pred=np.column_stack((boxes[:,5],boxes[:,4],boxes[:,:4])) if len(boxes) else np.empty((0,6))
    del result,boxes
    return pred

def timed_call(model,image):
    torch.cuda.synchronize(0); before=torch.cuda.memory_allocated(0)
    torch.cuda.reset_peak_memory_stats(0); start=time.perf_counter()
    boxes=predict_boxes(model,image,FIXED_CONF); del boxes; torch.cuda.synchronize(0)
    elapsed=(time.perf_counter()-start)*1000
    alloc=torch.cuda.max_memory_allocated(0); reserved=torch.cuda.max_memory_reserved(0)
    return elapsed,alloc,reserved,max(0,alloc-before)

# %% [markdown]
# ## 13. Held-out evaluation
# Define independent test metrics, per-class matching, and model timing measurements.

# %%
def accuracy_scores(model,name,out):
    val=model.val(data=str(dataset_yaml),split='test',imgsz=640,batch=1,device=0,
        quantize=32,rect=False,conf=MAP_CONF,iou=NMS_IOU,max_det=MAX_DET,nms=True,
        plots=True,save_json=False,verbose=False,project=str(out),name=name+'_test',exist_ok=True)
    box=val.box; ids=np.asarray(box.ap_class_index,dtype=int); aps=np.asarray(box.all_ap)
    assert np.isfinite([box.map50,box.map,*aps.ravel(),*box.p,*box.r,*box.f1]).all()
    per={c:{'validator_ap50':0.,'validator_ap50_95':0.,'validator_precision':0.,
            'validator_recall':0.,'validator_f1':0.} for c in range(len(NAMES))}
    per.update({int(c):{'validator_ap50':float(aps[j,0]),'validator_ap50_95':float(aps[j].mean()),
        'validator_precision':float(box.p[j]),'validator_recall':float(box.r[j]),
        'validator_f1':float(box.f1[j])} for j,c in enumerate(ids)})
    return val,{'map50':float(box.map50),'map50_95':float(box.map)},per

def fixed_scores(model,name,paths,validator):
    totals=np.zeros((len(NAMES),3),dtype=int); ious=[]; n_gt=n_pred=matched=0
    for i,image in enumerate(paths,1):
        gt=ground_truth(image); pred=predict_boxes(model,image,FIXED_CONF)
        counts,mi=match_boxes(gt,pred); totals+=counts; ious.extend(mi)
        n_gt+=len(gt); n_pred+=len(pred); matched+=len(mi)
        if i % 100 == 0 or i == len(paths): print(f'{name}: checked {i}/{len(paths)} test images', flush=True)
    rows=[]
    for c,label in enumerate(NAMES):
        tp,fp,fn=map(int,totals[c]); den=2*tp+fp+fn
        rows.append({'model':name,'class':label,'tp':tp,'fp':fp,'fn':fn,
            'fixed_precision':tp/(tp+fp) if tp+fp else None,
            'fixed_recall':tp/(tp+fn) if tp+fn else None,'fixed_f1':2*tp/den if den else None,
            **validator.get(c,{})})
    precision=matched/n_pred if n_pred else None; recall=matched/n_gt if n_gt else None
    total={'tp':matched,'fp':n_pred-matched,'fn':n_gt-matched,'precision':precision,'recall':recall,
        'f1':2*matched/(2*matched+n_pred-matched+n_gt-matched) if n_pred+n_gt else None}
    return rows,total,n_gt,n_pred,matched,float(np.mean(ious)) if ious else None

def benchmark(model,sample):
    for image in sample[:20]: predict_boxes(model,image,FIXED_CONF)
    torch.cuda.synchronize(0); resident_a=torch.cuda.memory_allocated(0)
    resident_r=torch.cuda.memory_reserved(0); calls=[timed_call(model,image) for image in sample]
    times=[x[0] for x in calls]; avg=float(np.mean(times))
    return {'latency_mean_ms':avg,'latency_p50_ms':float(np.percentile(times,50)),
        'latency_p95_ms':float(np.percentile(times,95)),'fps':1000/avg,
        'resident_allocated_bytes':resident_a,'resident_reserved_bytes':resident_r,
        'peak_allocated_bytes':max(x[1] for x in calls),'peak_reserved_bytes':max(x[2] for x in calls),
        'incremental_allocated_bytes':max(x[3] for x in calls),'per_call_peak_bytes':[list(map(int,x[1:])) for x in calls]}

# %% [markdown]
# ## 14. Evaluation worker
# Evaluate one checkpoint and record its metrics, losses, timing, and memory evidence.

# %%
def evaluate_model(name,paths,sample,out):
    ck=checkpoints[name]; assert (RUNS/name/'complete.json').is_file()
    model=YOLO(str(ck)); model.model.to('cuda:0'); torch.cuda.synchronize(0)
    assert list(model.names.values())==NAMES, f'{name}: class map mismatch'
    val,maps,validator=accuracy_scores(model,name,out)
    rows,total,ng,npred,matched,mean_iou=fixed_scores(model,name,paths,validator)
    parameter_count=sum(p.numel() for p in model.model.parameters())
    del model,val; gc.collect(); torch.cuda.empty_cache()
    load_start=time.perf_counter(); model=YOLO(str(ck)); model.model.to('cuda:0'); torch.cuda.synchronize(0)
    load_ms=(time.perf_counter()-load_start)*1000
    first_call=timed_call(model,sample[0]); measure=benchmark(model,sample); train=train_evidence(name)
    backend_end2end=bool(getattr(model.predictor.model,'end2end',False))
    head_end2end=bool(getattr(model.model.model[-1],'end2end',False))
    assert not backend_end2end and not head_end2end, 'Common NMS comparison path changed; reconcile before reporting.'
    row={'model':name,'head_mode':'one-to-many + NMS' if name=='yolo26n' else 'NMS',**maps,**total,
        'matched_iou_mean':mean_iou,'gt_coverage':matched/ng if ng else None,
        'prediction_match_rate':matched/npred if npred else None,**measure,
        'parameter_count':parameter_count,'checkpoint_load_ms':load_ms,'first_prediction_ms':first_call[0],
        'checkpoint_bytes':Path(ck).stat().st_size,**train}
    details={'test_ground_truth':ng,'test_predictions':npred,'test_matches':matched,
        'timing_sample':[str(x) for x in sample],'warmup_calls':20,'timed_calls':100,
        'validator_per_class_operating_point':'Ultralytics max-F1 confidence',
        'runtime_backend_end2end':backend_end2end,'runtime_head_end2end':head_end2end,
        'timing_boundary':'warm-cache decode, preprocess, forward, NMS, CPU boxes; excludes render, model load, network',
        'device':torch.cuda.get_device_name(0),'precision':'FP32','first_prediction_ms':first_call[0],
        'memory_increment':'per-call peak allocated minus allocated before that call; not persistent per-image memory',
        'per_call_peak_bytes':measure['per_call_peak_bytes']}
    del model; gc.collect(); torch.cuda.empty_cache()
    return row,rows,details

# %% [markdown]
# ## 15. Evaluation orchestration
# Freeze the validation-selected winner and evaluate all three models once on held-out test data.

# %%
def evaluate_final_test():
    assert set(checkpoints)==set(MODELS) and winner in MODELS
    paths=test_paths(); assert len(paths)>=100
    out=RUNS/'evaluation'; out.mkdir(parents=True,exist_ok=True); manifest=out/'timing_sample.json'
    if manifest.exists(): sample=[Path(x) for x in json.loads(manifest.read_text())]
    else:
        sample=[Path(x) for x in np.random.default_rng(2026).choice([str(x) for x in paths],100,replace=False)]
        manifest.write_text(json.dumps([str(x) for x in sample],indent=2))
    assert len(sample)==100 and all(x in paths for x in sample)
    comparison=[]; perclass=[]; details={}
    for name in MODELS:
        print(f'Evaluating {name}...', flush=True)
        row,rows,detail=evaluate_model(name,paths,sample,out)
        comparison.append(row); perclass.extend(rows); details[name]=detail
    comparison=pd.DataFrame(comparison); perclass=pd.DataFrame(perclass)
    comparison.to_csv(out/'comparison.csv',index=False); perclass.to_csv(out/'perclass.csv',index=False)
    settings={'imgsz':640,'batch':1,'quantize':32,'rect':False,'map_conf':MAP_CONF,
        'fixed_conf':FIXED_CONF,'nms_iou':NMS_IOU,'nms':True,'max_det':MAX_DET,'warmup':20,'timed':100,'seed':2026}
    (out/'details.json').write_text(json.dumps({'winner':winner,'selection':'validation only',
        'settings':settings,'models':details},indent=2,allow_nan=False))
    return comparison,perclass

# %% [markdown]
# ## 16. Run held-out evaluation
# Run the final evaluation after all three training runs and validation selection are complete.

# %%
comparison, perclass = evaluate_final_test()
for source,label in [('resident_allocated_bytes','resident_MiB'),('peak_allocated_bytes','peak_MiB'),
                     ('incremental_allocated_bytes','extra_MiB')]: comparison[label]=comparison[source]/2**20
display(comparison[['model','map50','map50_95','precision','recall','f1','fp','fn','matched_iou_mean']])
display(comparison[['model','latency_mean_ms','latency_p95_ms','fps','resident_MiB','peak_MiB',
                    'extra_MiB','first_prediction_ms']])
display(perclass[['model','class','validator_ap50_95','fixed_precision','fixed_recall','fixed_f1','fp','fn']])

# %% [markdown]
# ## 17. Export
# Reload and retain the model checkpoints and compact project outputs in Kaggle storage.

# %%
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

