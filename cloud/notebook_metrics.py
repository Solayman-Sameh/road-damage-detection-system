# %% 1. Test manifest, labels, one-to-one matching, and saved-run evidence
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

# %% 2. Shared GPU prediction and synchronized timing boundary
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

# %% 3. Independent held-out accuracy, matched-box, and timing measurements
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

# %% Per-checkpoint evaluation worker
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

# %% Orchestrate once; use validation's selected winner, never test for tuning
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

# %% Run once after all models finish and validation selection is frozen
comparison, perclass = evaluate_final_test()
for source,label in [('resident_allocated_bytes','resident_MiB'),('peak_allocated_bytes','peak_MiB'),
                     ('incremental_allocated_bytes','extra_MiB')]: comparison[label]=comparison[source]/2**20
display(comparison[['model','map50','map50_95','precision','recall','f1','fp','fn','matched_iou_mean']])
display(comparison[['model','latency_mean_ms','latency_p95_ms','fps','resident_MiB','peak_MiB',
                    'extra_MiB','first_prediction_ms']])
display(perclass[['model','class','validator_ap50_95','fixed_precision','fixed_recall','fixed_f1','fp','fn']])
