"""Inspect user-downloaded checkpoints and validate on reconstructed screened validation images.

Does not train, modify checkpoints, or evaluate the held-out test split.
"""
from pathlib import Path
import gc
import hashlib
import io
import json
import os
import platform
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
torch.set_num_threads(2)
os.environ['YOLO_CONFIG_DIR'] = str(Path(__file__).resolve().parents[1]/'research/downloaded_checkpoint_audit_2026-10-29/runtime_config')
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research' / 'downloaded_checkpoint_audit_2026-10-29'
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ['longitudinal crack', 'transverse crack', 'alligator crack', 'pothole',
         'manhole cover', 'longitudinal patch', 'transverse patch']


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect_checkpoints():
    records = []
    for path in sorted((ROOT/'models').rglob('*.pt')):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model = ck.get('ema') if ck.get('ema') is not None else ck.get('model')
        args = ck.get('train_args', {})
        history = ck.get('train_results') or {}
        identity = Path(args.get('model', '')).stem
        assert list(model.names.values()) == NAMES, path
        assert all(torch.isfinite(x).all().item() for x in model.state_dict().values()
                   if x.is_floating_point()), path
        record = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                      model=identity, parameters=sum(p.numel() for p in model.parameters()),
                      names=model.names, checkpoint_epoch=ck.get('epoch'),
                      history_epochs=len(history.get('epoch', [])),
                      optimizer_present=ck.get('optimizer') is not None,
                      checkpoint_version=ck.get('version'), date=ck.get('date'),
                      train_args=args, stored_metrics=ck.get('train_metrics'), history=history,
                      finite_weights=True)
        records.append(record)
        if history:
            pd.DataFrame(history).to_csv(OUT/(path.parent.parent.name+'_'+identity+'_'+path.stem+'_history.csv'), index=False)
        print('IDENTITY', str(path.relative_to(ROOT)), identity, record['history_epochs'], flush=True)
        del model, ck
        gc.collect()
    save('checkpoints.json', records)
    # The periodic checkpoint stores EMA alongside optimizer state; compare EMA with inference best.pt.
    best = next(r for r in records if '/v2/' in r['path'].replace('\\','/') and r['model']=='yolo11s' and Path(r['path']).stem=='best')
    periodic = next(r for r in records if Path(r['path']).name=='epoch29.pt')
    a = torch.load(best['path'], map_location='cpu', weights_only=False)
    b = torch.load(periodic['path'], map_location='cpu', weights_only=False)
    ma = a['model'].state_dict(); mb = b['ema'].state_dict()
    equal = ma.keys()==mb.keys() and all(torch.equal(ma[k], mb[k]) for k in ma)
    save('yolo11_checkpoint_pair.json', dict(best_sha256=best['sha256'], periodic_sha256=periodic['sha256'],
         best_equals_periodic_ema=equal, periodic_zero_based_epoch=b['epoch'],
         optimizer_present=b.get('optimizer') is not None))
    del a,b,ma,mb
    gc.collect()
    return records


def prepare_validation():
    archive = ROOT/'data/raw/svrdd/SVRDD_YOLO.zip'
    destination = OUT/'validation_data'
    destination.mkdir(exist_ok=True)
    hash_cache = OUT/'val_test_dhashes.json'
    with zipfile.ZipFile(archive) as z:
        splits = {s:[x.strip().replace('\\','/') for x in z.read('SVRDD_YOLO/'+s+'.txt').decode().splitlines() if x.strip()]
                  for s in ['val','test']}
        assert all(len(v)==1000 and len(set(v))==1000 for v in splits.values())
        if hash_cache.exists():
            hashes = json.loads(hash_cache.read_text())
        else:
            hashes = {}
            for split, paths in splits.items():
                hashes[split] = []
                for i,path in enumerate(paths):
                    with Image.open(io.BytesIO(z.read('SVRDD_YOLO/'+path))) as im:
                        a = np.asarray(im.convert('L').resize((9,8)))
                        value = 0
                        for bit in (a[:,:-1]>a[:,1:]).ravel(): value = (value<<1)|int(bit)
                    hashes[split].append(value)
                    if (i+1)%250==0: print('HASH', split, i+1, flush=True)
            save(hash_cache.name, hashes)
        excluded = {i for i,h in enumerate(hashes['val']) if any((h^t).bit_count()<=3 for t in hashes['test'])}
        paths = [p for i,p in enumerate(splits['val']) if i not in excluded]
        assert len(paths)==973, f'Screened validation differs: {len(paths)}'
        support = np.zeros(7,dtype=int)
        for path in paths:
            label = str(Path(path.replace('/images/','/labels/')).with_suffix('.txt')).replace('\\','/')
            for member in [path,label]:
                target = destination/member
                assert target.resolve().is_relative_to(destination.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists(): target.write_bytes(z.read('SVRDD_YOLO/'+member))
            for row in (destination/label).read_text().splitlines():
                if row.strip(): support[int(float(row.split()[0]))]+=1
        assert support.tolist()==[576,414,219,111,393,506,337], support
        manifest = destination/'val.txt'
        manifest.write_text('\n'.join(str((destination/p).resolve()) for p in paths)+'\n')
        yaml_path = destination/'svrdd_val.json'
        yaml_path = yaml_path.with_suffix('.yaml')
        yaml_path.write_text(json.dumps(dict(path=str(destination.resolve()), train=str(manifest.resolve()),
                                           val=str(manifest.resolve()), names=NAMES)))
        save('validation_split.json',dict(images=len(paths), raw_objects=support.tolist(),
              excluded=[splits['val'][i] for i in sorted(excluded)],
              screening='same dHash <=3 policy; val excluded against higher-priority test only',
              test_usage='images read only for screening hashes; no test inference or metrics',
              archive_bytes=archive.stat().st_size))
        return yaml_path


def evaluate(records, data):
    rows=[]; classes=[]
    torch.set_num_threads(2)
    environment=dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
        python=platform.python_version(),gpu=torch.cuda.get_device_name(0),
        note='Local re-evaluation. Runtime differs from Kaggle training; all five weights evaluated with this same runtime.',
        protocol=dict(split='val', images=973,imgsz=1024,batch=1,quantize=32,rect=False,nms=True,
                      conf=0.001,iou=0.7,max_det=300,workers=0))
    save('evaluation_environment.json',environment)
    for record in records:
        path=Path(record['path'])
        if path.name!='best.pt':continue
        key=path.parent.parent.name+'_'+record['model']
        saved_path=OUT/(key+'_validation.json')
        if saved_path.exists():
            saved=json.loads(saved_path.read_text());assert saved['sha256']==record['sha256']
        else:
            print('VALIDATING',key,flush=True)
            model=YOLO(str(path))
            result=model.val(data=str(data),split='val',imgsz=1024,batch=1,device=0,
                quantize=32,rect=False,nms=True,conf=.001,iou=.7,max_det=300,
                workers=0,plots=False,project=str(OUT/'validator'),name=key,exist_ok=True,verbose=False)
            box=result.box
            assert np.isfinite([box.map50,box.map]).all()
            saved=dict(sha256=record['sha256'],row=dict(trial=key,model=record['model'],
                train_imgsz=record['train_args']['imgsz'],train_epochs=record['history_epochs'],
                eval_imgsz=1024,map50=float(box.map50),map50_95=float(box.map),
                precision=float(box.mp),recall=float(box.mr),checkpoint_MB=record['bytes']/1e6),
                perclass=[dict(trial=key,**{'class':NAMES[int(c)]},ap50=float(box.all_ap[j,0]),
                               ap50_95=float(box.all_ap[j].mean())) for j,c in enumerate(box.ap_class_index)])
            save(saved_path.name,saved)
            del model,result,box
            gc.collect();torch.cuda.empty_cache()
        rows.append(saved['row']);classes.extend(saved['perclass'])
        pd.DataFrame(rows).to_csv(OUT/'common_1024_validation.csv',index=False)
        pd.DataFrame(classes).to_csv(OUT/'common_1024_perclass.csv',index=False)
        print('RESULT',json.dumps(saved['row']),flush=True)
    print('ALL VALIDATION FINISHED',flush=True)


if __name__=='__main__':
    records=inspect_checkpoints()
    data=prepare_validation()
    evaluate(records,data)
