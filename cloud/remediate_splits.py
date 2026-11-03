"""Conservatively resolve SVRDD cross-split leakage candidates in Kaggle.
No candidate is called a duplicate or waived. For every cross-split exact or
dHash<=3 candidate, remove the lower-priority image copy (test > val > train).
This preserves test, protects val from training exposure, and records candidate
edges separately from exclusions. Pass mode explicitly to make the policy clear.
"""
import hashlib,json,math
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
NAMES=["longitudinal crack","transverse crack","alligator crack","pothole","manhole cover","longitudinal patch","transverse patch"]
SPLITS=("train","val","test");PRIORITY={"train":0,"val":1,"test":2}
BOX=[4665,3404,1728,918,3339,4128,2622]
SBOX={"train":[3461,2550,1324,679,2517,3109,1980],"val":[588,418,222,115,403,519,347],"test":[616,436,182,124,419,500,295]}
def _sha(path):
	h=hashlib.sha256()
	with Path(path).open("rb")as f:
		for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
	return h.hexdigest()
def _load_manifest(path):
	rows=[json.loads(x)for x in Path(path).read_text(encoding="utf-8").splitlines()if x.strip()]
	if len(rows)!=8000:raise ValueError(f"expected 8,000 audited images, got {len(rows)}")
	seen=set()
	for r in rows:
		if r.get("split")not in SPLITS or not isinstance(r.get("path"),str)or r["path"]in seen:raise ValueError("invalid or duplicate manifest row")
		seen.add(r["path"])
		for k in("sha256","decoded_rgb_sha256","dhash64"):
			if k not in r:raise ValueError(f"image manifest lacks {k}")
	return rows
def _candidate_edges(rows,distance=3):
	edges={}
	def add(a,b,signal,d=None):
		left,right=sorted((a["path"],b["path"]));key=(left,right)
		e=edges.setdefault(key,{"left":left,"right":right,"signals":[],"hamming_distance":None})
		if signal not in e["signals"]:e["signals"].append(signal)
		if d is not None:e["hamming_distance"]=d
	for field,signal in(("sha256","byte_exact"),("decoded_rgb_sha256","decoded_pixel_exact")):
		groups=defaultdict(list)
		for r in rows:groups[str(r[field])].append(r)
		for group in groups.values():
			for i,a in enumerate(group):
				for b in group[i+1:]:
					if a["split"]!=b["split"]:add(a,b,signal)
	hashes=[(r,int(str(r["dhash64"]),16))for r in rows]
	for i,(a,ha)in enumerate(hashes):
		for b,hb in hashes[i+1:]:
			if a["split"]!=b["split"]:
				d=(ha^hb).bit_count()
				if d<=distance:add(a,b,"perceptual_candidate",d)
	return sorted(edges.values(),key=lambda e:(e["left"],e["right"]))
def remediate_splits(image_manifest_path,original_audit_path,extracted_root,output_dir,mode):
	"""Freeze clean manifests and a new YAML; leave original failed audit untouched."""
	manifest=Path(image_manifest_path).resolve();source_audit=Path(original_audit_path).resolve()
	root=Path(extracted_root).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
	audit_path=out/"split_remediation_audit.json";yaml_path=out/"svrdd_remediated.yaml"
	report={"status":"running","mode":mode,"policy":"All unresolved cross-split exact/pixel/dHash<=3 candidates are treated as possible leakage; lower-priority copy excluded. No duplicate is asserted and no candidate is waived.","priority":["test","val","train"],"source_group_guarantee":False,"group_note":"No source/frame groups inferred.","original_audit":str(source_audit),"image_manifest":str(manifest),"errors":[]}
	def fail(e):
		report["status"]="failed";report["errors"].append(f"{type(e).__name__}: {e}");report["finished_utc"]=datetime.now(timezone.utc).isoformat()
		audit_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
		print(json.dumps({"status":"failed","audit":str(audit_path),"error":str(e),"summary":report.get("summary",{})},separators=(",",":")))
	try:
		if mode!="conservative_exclusion_all_candidates":raise ValueError("mode must explicitly be conservative_exclusion_all_candidates")
		source=json.loads(source_audit.read_text(encoding="utf-8"));checks=source.get("checks",{})
		report["original_audit_sha256"]=_sha(source_audit);report["image_manifest_sha256"]=_sha(manifest)
		if source.get("status")!="failed":raise ValueError("expected the original data audit to retain its failed leakage-review status")
		if source.get("archive",{}).get("bytes")!=2893208002 or source.get("archive",{}).get("md5")!="4b897aa7081e8d0fa07f3d330e86e936":raise ValueError("original audit does not confirm the exact SVRDD archive")
		exact=checks.get("exact_duplicate_pairs")
		if exact is None:
			exact=sum(len(checks.get(k,[]))if isinstance(checks.get(k,[]),list)else checks.get(k,0)for k in("byte_exact_cross_split_pairs","decoded_pixel_exact_cross_split_pairs","byte_exact_cross_split_duplicates","decoded_pixel_exact_cross_split_duplicates"))
		observed={"decoded_images":checks.get("decoded_images",checks.get("images_fully_decoded")),"decode_errors":checks.get("decode_errors",checks.get("image_decode_error_count")),"geometry_errors":checks.get("geometry_errors",checks.get("label_geometry_error_count")),"empty_labels":checks.get("empty_labels",checks.get("empty_label_files")),"exact_duplicate_pairs":exact}
		for key,want in(("decoded_images",8000),("decode_errors",0),("geometry_errors",0),("empty_labels",0),("exact_duplicate_pairs",0)):
			if observed[key]!=want:raise ValueError(f"original audit gate {key} expected {want}, observed {observed[key]}")
		if not source.get("errors")or any("perceptual" not in str(e).lower()and "dhash" not in str(e).lower()for e in source["errors"]):raise ValueError("original audit has failures beyond unresolved perceptual candidates")
		rows=_load_manifest(manifest);before={s:sum(r["split"]==s for r in rows)for s in SPLITS}
		if before!={"train":6000,"val":1000,"test":1000}:raise ValueError(f"source split counts changed: {before}")
		for r in rows:
			rel=Path(r["path"])
			if rel.is_absolute()or ".." in rel.parts:raise ValueError(f"unsafe image path in manifest: {r['path']}")
			im=(root/rel).resolve()
			if not im.is_relative_to(root)or not im.is_file():raise ValueError(f"manifest path missing or outside extracted root: {r['path']}")
		edges=_candidate_edges(rows,3)
		kinds=Counter(signal for edge in edges for signal in edge["signals"])
		near_count=sum("perceptual_candidate" in e["signals"]for e in edges)
		report["candidate_counts"]={"unique_pairs":len(edges),"byte_exact":kinds["byte_exact"],"decoded_pixel_exact":kinds["decoded_pixel_exact"],"dhash_le_3":near_count,"dhash_threshold":3}
		original_near=checks.get("perceptual_candidates_dhash_le_3",checks.get("perceptual_candidate_count"))
		if near_count!=original_near:raise ValueError(f"recomputed dHash candidate count {near_count} differs from original audit {original_near}")
		if kinds["byte_exact"]or kinds["decoded_pixel_exact"]:raise ValueError("exact cross-split edges disagree with original audit; stop for audit review")
		by={r["path"]:r for r in rows};removed={};
		for edge in edges:
			a,b=by[edge["left"]],by[edge["right"]]
			loser,winner=(a,b)if PRIORITY[a["split"]]<PRIORITY[b["split"]]else(b,a)
			edge.update({"left_split":a["split"],"right_split":b["split"],"excluded_path":loser["path"],"excluded_split":loser["split"],"action":"exclude_possible_leakage_copy_without_duplicate_claim"})
			entry=removed.setdefault(loser["path"],{"path":loser["path"],"split":loser["split"],"reasons":[]})
			entry["reasons"].append({"candidate_with":winner["path"],"candidate_split":winner["split"],"signals":edge["signals"],"hamming_distance":edge["hamming_distance"],"decision":"possible_leakage_excluded_without_duplicate_claim"})
		retained={s:[]for s in SPLITS}
		for r in rows:
			if r["path"]not in removed:retained[r["split"]].append(r)
		if len(retained["test"])!=1000 or{r["path"]for r in retained["test"]}!={r["path"]for r in rows if r["split"]=="test"}:raise ValueError("test set changed; remediation must preserve it exactly")
		for e in edges:
			if e["left"]not in removed and e["right"]not in removed:raise ValueError("cross-split candidate survived exclusion")
		cand_path=out/"all_cross_split_candidates.json";cand_path.write_text(json.dumps(edges,indent=2)+"\n",encoding="utf-8")
		excl_path=out/"split_exclusions.json";excl_path.write_text(json.dumps(sorted(removed.values(),key=lambda x:x["path"]),indent=2)+"\n",encoding="utf-8")
		report["outputs"]={"all_candidates":str(cand_path),"exclusions":str(excl_path)}
		report["summary"]={"images_before":before,"images_after":{s:len(retained[s])for s in SPLITS},"excluded_images":len(removed),"test_unchanged":len(retained["test"])==1000}
		images={s:Counter()for s in SPLITS};objects={s:Counter()for s in SPLITS};before_obj={s:Counter()for s in SPLITS};before_images={s:Counter()for s in SPLITS}
		for r in rows:
			im=root/r["path"];p=list(im.parts);p[p.index("images")]="labels";lab=Path(*p).with_suffix(".txt")
			if not lab.is_file():raise ValueError(f"label missing: {lab}")
			ids=[]
			for n,line in enumerate(lab.read_text(encoding="utf-8-sig").splitlines(),1):
				if not line.strip():continue
				f=line.split()
				try:vals=[float(x)for x in f]
				except Exception:raise ValueError(f"invalid label {lab}:{n}")
				if len(vals)!=5 or not vals[0].is_integer()or not 0<=vals[0]<7 or not all(math.isfinite(x)for x in vals[1:]):raise ValueError(f"invalid label {lab}:{n}")
				c,x,y,w,h=vals;box=(x-w/2,y-h/2,x+w/2,y+h/2)
				if w<=0 or h<=0 or min(box)<-1e-6 or max(box)>1+1e-6:raise ValueError(f"geometry error {lab}:{n}")
				ids.append(int(c))
			if not ids:raise ValueError(f"unexpected empty label: {lab}")
			before_obj[r["split"]].update(ids);before_images[r["split"]].update(set(ids))
			if r["path"]not in removed:objects[r["split"]].update(ids);images[r["split"]].update(set(ids))
		all_counts=[sum(before_obj[s][i]for s in SPLITS)for i in range(7)]
		split_counts={s:[before_obj[s][i]for i in range(7)]for s in SPLITS}
		if all_counts!=BOX or split_counts!={"train":SBOX["train"],"val":SBOX["val"],"test":SBOX["test"]}:raise ValueError("source labels/class map differ from inspected evidence")
		missing={s:[NAMES[i]for i in range(7)if images[s][i]==0]for s in SPLITS}
		if any(missing.values()):raise ValueError(f"retained split lacks class support: {missing}")
		mp={};md=out/"manifests";md.mkdir(parents=True,exist_ok=True)
		for s in SPLITS:
			f=md/(s+".txt");f.write_text("\n".join(str((root/r["path"]).resolve())for r in retained[s])+"\n",encoding="utf-8");mp[s]=str(f.resolve())
		yaml_path.write_text(json.dumps({"path":str(root),**mp,"names":NAMES},indent=2)+"\n",encoding="utf-8")
		report["summary"]={"images_before":before,"images_after":{s:len(retained[s])for s in SPLITS},"excluded_images":len(removed),"test_unchanged":len(retained["test"])==1000,"class_image_support_before":{s:dict(before_images[s])for s in SPLITS},"class_image_support_after":{s:dict(images[s])for s in SPLITS},"class_object_rows_before":{s:dict(before_obj[s])for s in SPLITS},"class_object_rows_after":{s:dict(objects[s])for s in SPLITS},"missing_class_support":missing}
		report["outputs"]={"yaml":str(yaml_path),"manifests":mp,"all_candidates":str(cand_path),"exclusions":str(excl_path)}
		report["status"]="passed";report["finished_utc"]=datetime.now(timezone.utc).isoformat();audit_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
		print(json.dumps({"status":"passed","audit":str(audit_path),"candidate_counts":report["candidate_counts"],"summary":report["summary"]},separators=(",",":")))
		return yaml_path,audit_path
	except Exception as e:
		fail(e);raise RuntimeError(f"Split remediation failed; audit: {audit_path}; {e}")from e
