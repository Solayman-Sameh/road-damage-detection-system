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
