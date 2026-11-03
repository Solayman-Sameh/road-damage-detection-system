# Replace the CODE cell in section 3 with this whole cell.
# Run after the library setup cell; enable Kaggle Internet.
# Download -> verify -> extract/audit -> reconstruct the original screened splits.
from pathlib import Path
from collections import Counter
import hashlib, json, math, os, time, urllib.request
from tqdm.auto import tqdm

SOURCE = Path('/kaggle/working/svrdd_source')
SOURCE.mkdir(parents=True, exist_ok=True)
_ARCHIVE = SOURCE / 'SVRDD_YOLO.zip'
_PARTIAL = SOURCE / 'SVRDD_YOLO.zip.part'
_URL = 'https://zenodo.org/api/records/10100129/files/SVRDD_YOLO.zip/content'
_BYTES = 2_893_208_002
_MD5 = '4b897aa7081e8d0fa07f3d330e86e936'

def _verify_svrdd_zip(path):
    if not path.is_file() or path.stat().st_size != _BYTES:
        return False
    checksum = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            checksum.update(block)
    return checksum.hexdigest() == _MD5

if _ARCHIVE.exists():
    if not _verify_svrdd_zip(_ARCHIVE):
        raise RuntimeError(f'Existing archive differs from the original release: {_ARCHIVE}. Move it aside before retrying.')
    print('Reusing checksum-verified SVRDD archive:', _ARCHIVE)
else:
    print('Downloading the original SVRDD YOLO archive from Zenodo (~2.9 GB).')
    for attempt in range(1, 5):
        try:
            offset = _PARTIAL.stat().st_size if _PARTIAL.exists() else 0
            if offset > _BYTES:
                raise ValueError('Partial archive is larger than the expected release; move it aside.')
            if offset < _BYTES:
                headers = {'User-Agent': 'SVRDD-research-notebook/1.0', 'Accept-Encoding': 'identity'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                request = urllib.request.Request(_URL, headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    status = response.status
                    if status == 206:
                        content_range = response.headers.get('Content-Range', '')
                        if not content_range.startswith(f'bytes {offset}-') or not content_range.endswith(f'/{_BYTES}'):
                            raise ValueError(f'Unexpected download range: {content_range}')
                    elif status == 200:
                        offset = 0  # Server ignored Range: safely restart the partial download.
                    else:
                        raise RuntimeError(f'Unexpected HTTP status: {status}')
                    with _PARTIAL.open('ab' if offset else 'wb') as target, tqdm(
                            total=_BYTES, initial=offset, unit='B', unit_scale=True,
                            desc='SVRDD download') as progress:
                        received = offset
                        while True:
                            block = response.read(4 * 1024**2)
                            if not block:
                                break
                            received += len(block)
                            if received > _BYTES:
                                raise ValueError('Download exceeded the original archive size.')
                            target.write(block)
                            progress.update(len(block))
            if not _verify_svrdd_zip(_PARTIAL):
                raise RuntimeError('Archive is incomplete or its checksum differs. Partial download retained for inspection/retry.')
            os.replace(_PARTIAL, _ARCHIVE)
            break
        except Exception as error:
            print(f'Download attempt {attempt}/4: {error}')
            if attempt == 4:
                raise
            time.sleep(3 * attempt)
    print('Original archive size and MD5 verified.')

# Existing project auditor; kept inside a function so its helper names stay local.
def _prepare_original_svrdd(archive_path, output_dir):
    """Paste as one Kaggle cell; then call prepare_svrdd(archive_path, output_dir)."""
    import hashlib,json,math,os,shutil,stat,tempfile,zipfile
    from pathlib import Path,PurePosixPath
    from collections import Counter,defaultdict
    from datetime import datetime,timezone
    MD5="4b897aa7081e8d0fa07f3d330e86e936"; SIZE=2893208002
    NAMES=["longitudinal crack","transverse crack","alligator crack","pothole","manhole cover","longitudinal patch","transverse patch"]
    BOX=[4665,3404,1728,918,3339,4128,2622]
    SBOX={"train":[3461,2550,1324,679,2517,3109,1980],"val":[588,418,222,115,403,519,347],"test":[616,436,182,124,419,500,295]}
    COUNTS={"train":6000,"val":1000,"test":1000}; EXTS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
    def prepare_svrdd(archive_path,output_dir):
    	"""Return YAML and audit paths; write audit and raise on every unresolved gate."""
    	a=Path(archive_path).resolve(); out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    	ap=out/"data_preparation_audit.json"; yp=out/"svrdd.yaml"
    	report={"source":"SVRDD original YOLO release","record":"https://zenodo.org/records/10100129","license":"CC BY 4.0; respect Baidu Maps imagery terms","status":"running","checks":{},"errors":[],"source_group_guarantee":False,"source_group_note":"No verified frame/source groups; district and filename tokens are clues only."}
    	try:
    		h=hashlib.sha256(); m=hashlib.md5(); size=0
    		with a.open("rb") as f:
    			for b in iter(lambda:f.read(8*1024*1024),b""): size+=len(b); h.update(b); m.update(b)
    		h,m=h.hexdigest(),m.hexdigest(); report["archive"]={"path":str(a),"bytes":size,"md5":m,"sha256":h}
    		if size!=SIZE or m!=MD5: raise ValueError("not the exact publisher ZIP (size/MD5 mismatch)")
    		with zipfile.ZipFile(a) as z:
    			infos=z.infolist(); names=[]; seen=set(); total=0
    			for i in infos:
    				raw=i.filename.replace("\\","/"); p=PurePosixPath(raw); mode=i.external_attr>>16
    				if not raw or p.is_absolute() or ".." in p.parts or (p.parts and p.parts[0].endswith(":")) or stat.S_ISLNK(mode): raise ValueError("unsafe ZIP member: "+raw)
    				n=p.as_posix()
    				if n in seen: raise ValueError("duplicate ZIP member: "+n)
    				seen.add(n); names.append(n); total+=i.file_size
    				if i.file_size>2*1024**3 or total>20*1024**3 or (i.compress_size and i.file_size/i.compress_size>1000): raise ValueError("ZIP expansion limit exceeded")
    			bad=z.testzip()
    			if bad: raise ValueError("ZIP CRC failure: "+bad)
    			imgs=[n for n in names if PurePosixPath(n).suffix.lower() in EXTS and "/images/" in "/"+n.lower()]
    			labs=[n for n in names if "/labels/" in "/"+n.lower() and n.lower().endswith(".txt")]
    			if len(imgs)!=8000 or len(labs)!=8000: raise ValueError("expected 8,000 images and 8,000 labels")
    			report["checks"]["zip"]={"members":len(names),"images":len(imgs),"labels":len(labs),"uncompressed_bytes":total,"crc":"passed"}
    			ex=out/"extracted"; marker=ex/".archive_sha256"
    			if ex.exists():
    				if not marker.is_file() or marker.read_text().strip()!=h: raise ValueError("existing extraction has no matching hash marker")
    			else:
    				stage=Path(tempfile.mkdtemp(prefix=".svrdd-",dir=out))
    				try:
    					base=stage.resolve()
    					for i,n in zip(infos,names):
    						dst=(stage/Path(*PurePosixPath(n).parts)).resolve()
    						if not dst.is_relative_to(base): raise ValueError("ZIP path escapes extraction root")
    						if i.is_dir(): dst.mkdir(parents=True,exist_ok=True)
    						else:
    							dst.parent.mkdir(parents=True,exist_ok=True)
    							with z.open(i) as src,dst.open("xb") as d: shutil.copyfileobj(src,d,1024*1024)
    					(stage/".archive_sha256").write_text(h+"\n"); os.replace(stage,ex)
    				except Exception: shutil.rmtree(stage,ignore_errors=True); raise
    		roots=[p.parent for p in ex.rglob("train.txt") if all((p.parent/(s+".txt")).is_file() for s in ("val","test"))]
    		if len(roots)!=1: raise ValueError("expected one source root with train/val/test.txt")
    		root=roots[0]; by=defaultdict(list)
    		for p in root.rglob("*"):
    			if p.is_file() and p.suffix.lower() in EXTS and "images" in p.parts: by[p.relative_to(root).as_posix().lower()].append(p)
    		splits={}; owner={}
    		for s,want in COUNTS.items():
    			paths=[]; local=set()
    			for line in (root/(s+".txt")).read_text(encoding="utf-8-sig").splitlines():
    				v=line.strip().replace("\\","/")
    				while v.startswith("./"): v=v[2:]
    				p=PurePosixPath(v)
    				if not v or p.is_absolute() or ".." in p.parts or (p.parts and p.parts[0].endswith(":")): raise ValueError("unsafe split path")
    				match=by.get(p.as_posix().lower(),[])
    				if len(match)!=1: raise ValueError("missing/ambiguous source image: "+v)
    				im=match[0]; k=im.relative_to(root).as_posix().lower()
    				if k in local or k in owner: raise ValueError("duplicate/cross-split manifest image: "+k)
    				local.add(k); owner[k]=s; paths.append(im)
    			if len(paths)!=want: raise ValueError("source split count mismatch: "+s)
    			splits[s]=paths
    		if len(owner)!=8000 or len(by)!=8000: raise ValueError("source manifests do not cover exactly 8,000 images")
    		from PIL import Image,ImageFile
    		ImageFile.LOAD_TRUNCATED_IMAGES=False
    		classes=Counter(); sc={s:Counter() for s in splits}; exact={}; pixel={}; rows=[]; geom=[]; badimg=[]; exact_pairs=[]; empty=0
    		for s,paths in splits.items():
    			for im in paths:
    				rel=im.relative_to(root).as_posix(); parts=list(im.parts); parts[parts.index("images")]="labels"; lab=Path(*parts).with_suffix(".txt")
    				if not lab.is_file(): geom.append(rel+":missing label"); continue
    				lines=lab.read_text(encoding="utf-8-sig").splitlines()
    				if not "".join(lines).strip(): empty+=1
    				for j,line in enumerate(lines,1):
    					if not line.strip(): continue
    					f=line.split()
    					try: v=[float(x) for x in f]
    					except Exception: geom.append(f"{rel}:row{j}:non-numeric"); continue
    					if len(v)!=5 or not v[0].is_integer() or not 0<=v[0]<7 or not all(math.isfinite(x) for x in v[1:]): geom.append(f"{rel}:row{j}:invalid"); continue
    					c,x,y,w,h=v[0],*v[1:]; box=(x-w/2,y-h/2,x+w/2,y+h/2)
    					if w<=0 or h<=0 or min(box)<-1e-6 or max(box)>1+1e-6: geom.append(f"{rel}:row{j}:geometry"); continue
    					classes[int(c)]+=1; sc[s][int(c)]+=1
    				try:
    					with Image.open(im) as img:
    						img.load(); raw=im.read_bytes(); rgb=img.convert("RGB"); b=hashlib.sha256(raw).hexdigest(); px=hashlib.sha256(rgb.tobytes()).hexdigest(); width,height=img.size
    						q=list(img.convert("L").resize((9,8)).getdata()); dh=0
    						for y in range(8):
    							for x in range(8): dh=(dh<<1)|(q[y*9+x]>q[y*9+x+1])
    						rec=(rel,s,b,px,dh,width,height); rows.append(rec)
    						for key,table,kind in ((b,exact,"byte"),(px,pixel,"pixel")):
    							for prev in table.get(key,[]):
    								if prev[1]!=s: exact_pairs.append({"kind":kind,"left":prev[0],"right":rel})
    							table.setdefault(key,[]).append(rec)
    				except Exception as e: badimg.append(rel+":"+str(e))
    		actual=[classes[i] for i in range(7)]; split_actual={s:[sc[s][i] for i in range(7)] for s in splits}
    		manifest=out/"image_manifest.jsonl"
    		manifest.write_text("".join(json.dumps({"path":r[0],"absolute_path":str((root/r[0]).resolve()),"split":r[1],"sha256":r[2],"decoded_rgb_sha256":r[3],"dhash64":f"{r[4]:016x}","width":r[5],"height":r[6]})+"\n" for r in rows))
    		report["checks"].update({"source_split_images":{s:len(v) for s,v in splits.items()},"decoded_images":len(rows),"decode_errors":len(badimg),"decode_error_examples":badimg[:10],"empty_labels":empty,"geometry_errors":len(geom),"geometry_error_examples":geom[:10],"class_names":NAMES,"boxes_by_class":actual,"split_boxes_by_class":split_actual,"exact_duplicate_pairs":len(exact_pairs),"exact_duplicate_examples":exact_pairs[:10],"image_manifest":str(manifest)})
    		if exact_pairs: raise ValueError(f"exact or decoded-pixel cross-split duplicate pairs={len(exact_pairs)}; examples={exact_pairs[:10]}")
    		if geom or empty: raise ValueError(f"label geometry/errors={len(geom)}, empty={empty}; examples={geom[:4]}")
    		if badimg or len(rows)!=8000: raise ValueError(f"decode errors={len(badimg)}, decoded={len(rows)}; examples={badimg[:4]}")
    		if actual!=BOX or split_actual!=SBOX: raise ValueError("class mapping/counts differ from inspected source evidence")
    		near=[]; near_count=0
    		for i,a0 in enumerate(rows):
    			for j in range(i+1,len(rows)):
    				b0=rows[j]
    				if a0[1]!=b0[1] and (a0[4]^b0[4]).bit_count()<=3:
    					near_count+=1
    					if len(near)<20: near.append([a0[0],b0[0]])
    		report["checks"].update({"perceptual_candidates_dhash_le_3":near_count,"perceptual_candidate_examples":near[:10]})
    		if near_count: raise ValueError(f"unresolved perceptual cross-split candidates={near_count}, sample={near[:10]}")
    		md=out/"manifests"; md.mkdir(parents=True,exist_ok=True); mp={}
    		for s,paths in splits.items():
    			f=md/(s+".txt"); f.write_text("\n".join(str(p.resolve()) for p in paths)+"\n"); mp[s]=str(f)
    		yp.write_text(json.dumps({"path":str(root.resolve()),**mp,"names":NAMES},indent=2)+"\n")
    		report["checks"].update({"split_overlap":0,"source_split_assignment":"preserved","source_group_guarantee":False})
    		report["outputs"]={"yaml":str(yp),"manifests":mp,"image_manifest":str(manifest)}; report["status"]="passed"
    		report["finished_utc"]=datetime.now(timezone.utc).isoformat(); ap.write_text(json.dumps(report,indent=2)+"\n")
    		print(json.dumps({"status":"passed","yaml":str(yp),"audit":str(ap),"summary":report["checks"]},separators=(",",":")))
    		return yp,ap
    	except Exception as e:
    		report["status"]="failed"; report["errors"].append(f"{type(e).__name__}: {e}")
    		report["finished_utc"]=datetime.now(timezone.utc).isoformat(); ap.write_text(json.dumps(report,indent=2)+"\n")
    		print("Initial audit saved:", ap)
    		raise RuntimeError(f"SVRDD data gate failed; audit: {ap}; {e}") from e

    return prepare_svrdd(archive_path, output_dir)

_prepared = SOURCE / 'prepared'
_audit_path = _prepared / 'data_preparation_audit.json'
_manifest_path = _prepared / 'image_manifest.jsonl'
_extraction_marker = _prepared / 'extracted' / '.archive_sha256'
_reuse = False
if _audit_path.is_file() and _manifest_path.is_file() and _extraction_marker.is_file():
    saved = json.loads(_audit_path.read_text())
    _reuse = (saved.get('archive', {}).get('md5') == _MD5
              and saved.get('archive', {}).get('bytes') == _BYTES
              and saved.get('checks', {}).get('decoded_images') == 8000
              and _extraction_marker.read_text().strip() == saved.get('archive', {}).get('sha256'))
if _reuse:
    print('Reusing prepared SVRDD files; checking screened splits again below.')
else:
    print('Extracting and auditing 8,000 images and their labels. This can take several minutes.')
    try:
        _prepare_original_svrdd(_ARCHIVE, _prepared)
    except RuntimeError:
        # Original data has known cross-split similarity candidates. The following
        # unchanged screening code excludes them using the original fixed policy.
        if not _audit_path.is_file():
            raise
        initial = json.loads(_audit_path.read_text())
        errors = initial.get('errors', [])
        if not (initial.get('status') == 'failed' and errors
                and all('perceptual' in str(e).lower() for e in errors)
                and initial.get('checks', {}).get('perceptual_candidates_dhash_le_3') == 183):
            raise
        print('Found the 183 known similarity candidates; applying the original exclusion policy.')

# The following screening/checks are copied from section 3 of the Downloads notebook.
NAMES = ['longitudinal crack', 'transverse crack', 'alligator crack', 'pothole',
         'manhole cover', 'longitudinal patch', 'transverse patch']
audit_file = SOURCE/'prepared/data_preparation_audit.json'
manifest_file = SOURCE/'prepared/image_manifest.jsonl'
assert audit_file.is_file() and manifest_file.is_file(), 'Prepared SVRDD audit/manifest missing.'
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
        for line in Path(*parts).with_suffix('.txt').read_text().splitlines():
            if not line.strip(): continue
            c,x,y,w,h = map(float,line.split())
            assert c.is_integer() and 0<=c<7 and all(map(math.isfinite,[x,y,w,h]))
            bounds = (x-w/2, y-h/2, x+w/2, y+h/2)
            assert w > 0 and h > 0 and min(bounds) >= -1e-6 and max(bounds) <= 1+1e-6, f"Invalid box in {image}: {bounds}"
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
