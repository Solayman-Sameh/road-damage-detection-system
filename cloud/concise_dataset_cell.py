# Kaggle cell 2: use attached audited SVRDD snapshot; no download/extraction.
import os, json, hashlib, math
from pathlib import Path
from collections import Counter

EXPECTED_MD5 = "4b897aa7081e8d0fa07f3d330e86e936"; EXPECTED_BYTES = 2_893_208_002
NAMES = ["longitudinal crack", "transverse crack", "alligator crack", "pothole", "manhole cover", "longitudinal patch", "transverse patch"]
SOURCE_BOXES = [4665,3404,1728,918,3339,4128,2622]
SOURCE_SPLIT_BOXES = {"train":[3461,2550,1324,679,2517,3109,1980], "val":[588,418,222,115,403,519,347], "test":[616,436,182,124,419,500,295]}

def files_below(root, wanted):
    for here, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if wanted <= set(files): yield Path(here)

mounts = list(Path("/kaggle/input").iterdir()) if Path("/kaggle/input").is_dir() else []
pairs = []
for mount in mounts:
    audits = list(files_below(mount, {"data_preparation_audit.json"})); manifests = list(files_below(mount, {"image_manifest.jsonl"}))
    for audit_dir in audits:
        pairs.extend((audit_dir/"data_preparation_audit.json", m/"image_manifest.jsonl", mount) for m in manifests if m==audit_dir or audit_dir in m.parents or m in audit_dir.parents)
if len(pairs) != 1:
    raise RuntimeError(f"Expected one attached version-1 audit+image manifest; found {len(pairs)}. Attach the private saved-output snapshot containing both, then rerun this cell.")
audit_path, image_manifest, mount = pairs[0]; audit = json.loads(audit_path.read_text(encoding="utf-8")); checks = audit.get("checks", {}); archive = audit.get("archive", {})
labels = checks.get("labels_and_classes", {}); image_audit = checks.get("image_decode_and_geometry", {}); duplicate_audit = checks.get("duplicates", {})
assert archive.get("bytes") == EXPECTED_BYTES and archive.get("md5") == EXPECTED_MD5, "Saved audit does not verify the original SVRDD archive."
assert audit.get("status") == "failed", "Expected the original audit with its unresolved near-duplicate gate."
boxes = checks.get("boxes_by_class", labels.get("object_rows_by_class")); split_boxes = checks.get("split_boxes_by_class", labels.get("split_object_rows_by_class"))
ordered = lambda values: [values[str(i)] if str(i) in values else values[i] for i in range(7)] if isinstance(values, dict) else values
if isinstance(split_boxes, dict): split_boxes = {s:ordered(split_boxes[s]) for s in ("train","val","test")}
exact_count = checks.get("exact_duplicate_pairs", len(duplicate_audit.get("byte_exact_cross_split_pairs", [])) + len(duplicate_audit.get("decoded_pixel_exact_cross_split_pairs", [])))
near_count = checks.get("perceptual_candidates_dhash_le_3", duplicate_audit.get("perceptual_candidate_count"))
gates = {"decoded images":(checks.get("decoded_images",image_audit.get("images_fully_decoded")),8000), "decode errors":(checks.get("decode_errors",image_audit.get("image_decode_error_count")),0), "geometry errors":(checks.get("geometry_errors",labels.get("label_geometry_error_count")),0), "empty labels":(checks.get("empty_labels",labels.get("empty_label_files")),0), "exact pairs":(exact_count,0)}
assert all(found == expected for found, expected in gates.values()), f"Saved audit gates differ: {gates}"
assert ordered(boxes) == SOURCE_BOXES and split_boxes == SOURCE_SPLIT_BOXES, "Saved class/split counts differ from inspected source evidence."
errors = audit.get("errors", [])
assert audit["status"] == "failed" and errors and all("perceptual" in str(e).lower() or "dhash" in str(e).lower() for e in errors), "Saved audit must have failed only on the perceptual gate."
assert near_count == 183, f"Saved audit reports {near_count} perceptual candidates; expected 183."

roots = [p for p in files_below(mount, {"train.txt", "val.txt", "test.txt"}) if any(p.glob("*/images")) and any(p.glob("*/labels"))]
if len(roots) != 1: raise RuntimeError(f"Found {len(roots)} extracted SVRDD roots. Attach version-1 saved outputs containing train/val/test.txt and district images/labels; this cell will not re-extract or download the ZIP.")
dataset_root = roots[0].resolve(); rows = [json.loads(line) for line in image_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 8000, f"Expected 8,000 image audit rows, got {len(rows)}"
seen = set(); split_priority = {"train":0, "val":1, "test":2}; records = []
for row in rows:
    rel = Path(row["path"]); assert row["split"] in split_priority and not rel.is_absolute() and ".." not in rel.parts and row["path"] not in seen
    seen.add(row["path"]); image = (dataset_root / rel).resolve(); assert image.is_relative_to(dataset_root) and image.is_file(), f"Image missing: {rel}"
    records.append({**row, "absolute_path":str(image)})

# Recompute every cross-split dHash<=3 pair from the saved, fully audited image manifest.
edges = []
for i, left in enumerate(records):
    a = int(left["dhash64"], 16) if isinstance(left["dhash64"], str) else int(left["dhash64"])
    for right in records[i+1:]:
        if left["split"] == right["split"]: continue
        b = int(right["dhash64"], 16) if isinstance(right["dhash64"], str) else int(right["dhash64"]); distance = (a ^ b).bit_count()
        if distance <= 3:
            loser, winner = (left, right) if split_priority[left["split"]] < split_priority[right["split"]] else (right, left)
            edges.append({"left":left["path"], "right":right["path"], "left_split":left["split"], "right_split":right["split"], "hamming_distance":distance,
                          "excluded_path":loser["path"], "excluded_split":loser["split"], "policy":"possible leakage excluded; duplicate not asserted"})
assert len(edges) == 183, f"Recomputed {len(edges)} candidates; saved audit reports 183. Stop and reconcile before training."
excluded = {}
for edge in edges: excluded.setdefault(edge["excluded_path"], []).append(edge)
remaining = {s:[r for r in records if r["split"] == s and r["path"] not in excluded] for s in split_priority}; counts = {s:len(v) for s,v in remaining.items()}
assert counts == {"train":5918, "val":973, "test":1000}, f"Unexpected filtered split counts: {counts}"
assert all(e["left"] in excluded or e["right"] in excluded for e in edges), "A candidate still crosses retained splits."

# Recount class support from retained labels; test stays byte-for-byte assigned to test.
support = {s:Counter() for s in split_priority}; objects = {s:Counter() for s in split_priority}
for split, images in remaining.items():
    for row in images:
        parts = list(Path(row["absolute_path"]).parts); parts[parts.index("images")] = "labels"; label = Path(*parts).with_suffix(".txt")
        assert label.is_file(), f"Missing label: {label}"
        ids = []
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip(): continue
            fields = line.split(); assert len(fields) == 5
            c,x,y,w,h = map(float,fields); assert c.is_integer() and 0 <= c < 7 and all(map(math.isfinite,(x,y,w,h))) and w > 0 and h > 0
            assert 0 <= x-w/2 and x+w/2 <= 1 and 0 <= y-h/2 and y+h/2 <= 1
            ids.append(int(c))
        assert ids, f"Unexpected empty retained label: {label}"
        objects[split].update(ids); support[split].update(set(ids))
missing = {s:[NAMES[i] for i in range(7) if support[s][i] == 0] for s in split_priority}
assert not any(missing.values()), f"Filtered split lost class support: {missing}"

out = Path("/kaggle/working/svrdd_screened"); (out/"manifests").mkdir(parents=True,exist_ok=True); manifest_paths = {}
for split, images in remaining.items():
    target = out/"manifests"/f"{split}.txt"
    target.write_text("\n".join(r["absolute_path"] for r in images)+"\n",encoding="utf-8"); manifest_paths[split] = str(target.resolve())
dataset_yaml = out/"svrdd.yaml"; dataset_yaml.write_text(json.dumps({"path":str(dataset_root),**manifest_paths,"names":NAMES},indent=2)+"\n",encoding="utf-8")
(out/"candidate_edges.json").write_text(json.dumps(edges,indent=2)+"\n",encoding="utf-8"); (out/"excluded_images.json").write_text(json.dumps(excluded,indent=2)+"\n",encoding="utf-8")
summary = {"attached_snapshot":str(mount),"audit":str(audit_path),"image_manifest":str(image_manifest),"image_manifest_sha256":hashlib.sha256(image_manifest.read_bytes()).hexdigest(),"dataset_root":str(dataset_root),"source_md5":archive["md5"],"dhash_candidate_pairs":len(edges),"excluded_images":len(excluded),"split_images":counts,"class_object_counts_after":{s:{NAMES[i]:objects[s][i] for i in range(7)} for s in split_priority},"class_support_after":{s:{NAMES[i]:support[s][i] for i in range(7)} for s in split_priority},"test_preserved":counts["test"]==1000,"duplicate_claim":"none; all candidates treated as possible leakage"}
(out/"dataset_cell_audit.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":"ready","dataset_yaml":str(dataset_yaml),"dhash_candidates":len(edges),"excluded_images":len(excluded),"split_images":counts,"all_class_support_retained":True},separators=(",",":")))
