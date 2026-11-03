"""Prepare the original SVRDD YOLO archive for a reproducible Kaggle run.

The public API is ``prepare(archive_path, output_dir) -> (dataset_yaml, audit_json)``.
This module intentionally uses only the Python standard library and Pillow. The
full audit is designed to run in Kaggle after the archive is already available;
it is not intended to be run against a local copy during source preparation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SOURCE = {
    "title": "SVRDD",
    "record": "https://zenodo.org/records/10100129",
    "doi": "10.5281/zenodo.10100129",
    "archive_name": "SVRDD_YOLO.zip",
    "expected_bytes": 2_893_208_002,
    "expected_md5": "4b897aa7081e8d0fa07f3d330e86e936",
    "expected_images": 8_000,
    "expected_split_counts": {"train": 6_000, "val": 1_000, "test": 1_000},
    "expected_archive_entries": 16_003,
    "expected_box_counts": [4_665, 3_404, 1_728, 918, 3_339, 4_128, 2_622],
    "expected_split_box_counts": {
        "train": [3_461, 2_550, 1_324, 679, 2_517, 3_109, 1_980],
        "val": [588, 418, 222, 115, 403, 519, 347],
        "test": [616, 436, 182, 124, 419, 500, 295],
    },
    "license": "CC BY 4.0; respect Baidu Maps imagery terms",
    "class_names": [
        "longitudinal crack",
        "transverse crack",
        "alligator crack",
        "pothole",
        "manhole cover",
        "longitudinal patch",
        "transverse patch",
    ],
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PHASH_REVIEW_DISTANCE = 3
MAX_ARCHIVE_UNCOMPRESSED = 20 * 1024**3
MAX_MEMBER_UNCOMPRESSED = 2 * 1024**3


class PreparationError(RuntimeError):
    """Raised after writing the audit when a preparation gate fails."""

    def __init__(self, message: str, audit_path: Path | None = None):
        super().__init__(message)
        self.audit_path = audit_path


def _sha256_and_md5(path: Path) -> tuple[str, str, int]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            sha256.update(block)
            md5.update(block)
    return sha256.hexdigest(), md5.hexdigest(), size


def _safe_member_name(info: zipfile.ZipInfo) -> str:
    raw = info.filename.replace("\\", "/")
    path = PurePosixPath(raw)
    mode = info.external_attr >> 16
    if path.is_absolute() or not raw or any(part in {"", ".", ".."} for part in path.parts):
        raise PreparationError(f"Unsafe ZIP member path: {info.filename!r}")
    if path.parts and path.parts[0].endswith(":"):
        raise PreparationError(f"Drive-qualified ZIP member path: {info.filename!r}")
    if stat.S_ISLNK(mode):
        raise PreparationError(f"Symbolic link in ZIP is not allowed: {info.filename!r}")
    return path.as_posix()


def _audit_zip(archive_path: Path) -> tuple[list[tuple[zipfile.ZipInfo, str]], dict[str, Any]]:
    members: list[tuple[zipfile.ZipInfo, str]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        total = sum(info.file_size for info in infos)
        if total > MAX_ARCHIVE_UNCOMPRESSED:
            raise PreparationError(f"ZIP expands to {total:,} bytes, above the safety limit")
        for info in infos:
            name = _safe_member_name(info)
            if name in seen:
                raise PreparationError(f"Duplicate normalized ZIP member: {name}")
            seen.add(name)
            if info.file_size > MAX_MEMBER_UNCOMPRESSED:
                raise PreparationError(f"ZIP member exceeds the safety limit: {name}")
            if info.file_size and info.compress_size == 0:
                raise PreparationError(f"Invalid ZIP compression size for {name}")
            if info.compress_size and info.file_size / info.compress_size > 1_000:
                raise PreparationError(f"Suspicious ZIP compression ratio for {name}")
            members.append((info, name))
        bad_member = archive.testzip()
        if bad_member:
            raise PreparationError(f"ZIP CRC check failed at member: {bad_member}")
    return members, {"member_count": len(members), "uncompressed_bytes": total, "crc_check": "passed"}


def _extract_safely(archive_path: Path, members: list[tuple[zipfile.ZipInfo, str]],
                    extraction_root: Path, archive_sha256: str) -> None:
    marker = extraction_root / ".svrdd_archive_sha256"
    if extraction_root.exists():
        if marker.is_file() and marker.read_text(encoding="ascii").strip() == archive_sha256:
            return
        raise PreparationError(
            f"Refusing to replace existing extraction directory without a matching marker: {extraction_root}"
        )

    extraction_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".svrdd-extract-", dir=extraction_root.parent))
    try:
        base = staging.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for info, name in members:
                target = (staging / Path(*PurePosixPath(name).parts)).resolve()
                if not target.is_relative_to(base):
                    raise PreparationError(f"ZIP member escapes extraction directory: {name}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        (staging / ".svrdd_archive_sha256").write_text(archive_sha256 + "\n", encoding="ascii")
        os.replace(staging, extraction_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _find_dataset_root(extraction_root: Path) -> tuple[Path, dict[str, Path]]:
    candidates = []
    for path in extraction_root.rglob("train.txt"):
        if (path.parent / "val.txt").is_file() and (path.parent / "test.txt").is_file():
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise PreparationError(f"Expected one root with train.txt/val.txt/test.txt; found {len(candidates)}")
    root = candidates[0]
    metadata = {split: root / f"{split}.txt" for split in ("train", "val", "test")}
    return root, metadata


def _normalize_metadata_path(line: str) -> str:
    value = line.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PreparationError(f"Unsafe/empty path in split metadata: {line!r}")
    if path.parts and path.parts[0].endswith(":"):
        raise PreparationError(f"Drive-qualified path in split metadata: {line!r}")
    return path.as_posix()


def _load_splits(root: Path, metadata: dict[str, Path]) -> dict[str, list[Path]]:
    by_suffix: dict[str, list[Path]] = defaultdict(list)
    for image in root.rglob("*"):
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES and "images" in image.parts:
            rel = image.relative_to(root).as_posix()
            by_suffix[rel.lower()].append(image)
    splits: dict[str, list[Path]] = {}
    for split, manifest in metadata.items():
        paths = []
        seen = set()
        for line_no, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            rel = _normalize_metadata_path(line)
            # The published lists contain District/images/name; match only within
            # the source dataset root so a repackaging wrapper is harmless.
            matches = by_suffix.get(rel.lower(), [])
            if not matches:
                raise PreparationError(f"Manifest image missing from archive: {split}:{line_no}: {rel}")
            if len(matches) != 1:
                raise PreparationError(f"Ambiguous manifest image path: {split}:{line_no}: {rel}")
            image = matches[0]
            key = image.relative_to(root).as_posix().lower()
            if key in seen:
                raise PreparationError(f"Duplicate image inside {split} manifest: {rel}")
            seen.add(key)
            paths.append(image)
        splits[split] = paths
    ownership: dict[str, str] = {}
    overlap = []
    for split, images in splits.items():
        for image in images:
            key = image.relative_to(root).as_posix().lower()
            if key in ownership:
                overlap.append({"image": key, "splits": [ownership[key], split]})
            else:
                ownership[key] = split
    if overlap:
        raise PreparationError(f"Source split metadata overlaps across splits ({len(overlap)} images)")
    all_images = {p for paths in splits.values() for p in paths}
    extracted_images = {p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and "images" in p.parts}
    missing = sorted(str(p.relative_to(root)) for p in extracted_images - all_images)
    if missing:
        raise PreparationError(f"{len(missing)} extracted images are absent from source manifests; examples: {missing[:5]}")
    return splits


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise PreparationError(f"Image path lacks an images directory: {image}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _parse_label_file(label_path: Path) -> tuple[list[int], list[str]]:
    classes: list[int] = []
    errors: list[str] = []
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        return classes, [f"cannot_read_label:{exc}"]
    for row_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"row_{row_no}_field_count_{len(fields)}")
            continue
        try:
            raw_class = float(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"row_{row_no}_non_numeric")
            continue
        if not math.isfinite(raw_class) or not raw_class.is_integer():
            errors.append(f"row_{row_no}_invalid_class_id")
            continue
        class_id = int(raw_class)
        if not 0 <= class_id < len(SOURCE["class_names"]):
            errors.append(f"row_{row_no}_class_out_of_range_{class_id}")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"row_{row_no}_non_finite_box")
            continue
        x, y, width, height = values
        if width <= 0 or height <= 0:
            errors.append(f"row_{row_no}_non_positive_size")
            continue
        bounds = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
        if min(bounds) < -1e-6 or max(bounds) > 1 + 1e-6:
            errors.append(f"row_{row_no}_box_out_of_normalized_image_bounds")
            continue
        classes.append(class_id)
    return classes, errors


def _dhash(image: Any) -> int:
    # 64-bit difference hash, computed after a full decode. This is a candidate
    # finder for human review, not a semantic duplicate classifier.
    gray = image.convert("L").resize((9, 8))
    pixels = list(gray.getdata())
    value = 0
    for y in range(8):
        offset = y * 9
        for x in range(8):
            value = (value << 1) | (pixels[offset + x] > pixels[offset + x + 1])
    return value


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _duplicate_review(records: list[dict[str, Any]], decisions_path: Path) -> dict[str, Any]:
    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pixel_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        exact_groups[record["sha256"]].append(record)
        pixel_groups[record["decoded_rgb_sha256"]].append(record)
    exact_pairs = []
    for group in exact_groups.values():
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                if left["split"] != right["split"]:
                    exact_pairs.append(_pair_record(left, right, "byte_exact", 0))
    decoded_pairs = []
    for group in pixel_groups.values():
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                if left["split"] != right["split"]:
                    decoded_pairs.append(_pair_record(left, right, "decoded_pixel_exact", 0))

    candidates = []
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            if left["split"] == right["split"]:
                continue
            distance = _hamming(left["dhash64"], right["dhash64"])
            if distance <= PHASH_REVIEW_DISTANCE:
                candidates.append(_pair_record(left, right, "perceptual_candidate", distance))

    decisions: dict[str, str] = {}
    if decisions_path.is_file():
        try:
            raw = json.loads(decisions_path.read_text(encoding="utf-8"))
            decisions = raw.get("decisions", {}) if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparationError(f"Invalid duplicate review decisions file: {exc}") from exc
    unresolved = []
    accepted_distinct = 0
    for pair in candidates:
        key = pair["pair_id"]
        decision = decisions.get(key)
        if decision == "distinct":
            accepted_distinct += 1
        elif decision == "duplicate":
            pair["review_decision"] = "duplicate"
            unresolved.append(pair)
        else:
            pair["review_decision"] = "unresolved"
            unresolved.append(pair)
    return {
        "exact_cross_split_pairs": exact_pairs,
        "decoded_pixel_exact_cross_split_pairs": decoded_pairs,
        "perceptual_candidate_threshold_hamming": PHASH_REVIEW_DISTANCE,
        "perceptual_candidate_count": len(candidates),
        "perceptual_candidates": candidates,
        "accepted_distinct_perceptual_candidates": accepted_distinct,
        "unresolved_or_confirmed_cross_split_perceptual_pairs": unresolved,
        "review_decisions_path": str(decisions_path),
        "decision_format": {"decisions": {"<pair_id>": "distinct or duplicate"}},
    }


def _pair_record(left: dict[str, Any], right: dict[str, Any], kind: str, distance: int) -> dict[str, Any]:
    a, b = sorted((left["relative_path"], right["relative_path"]))
    identity = hashlib.sha256(f"{a}\n{b}\n{kind}".encode("utf-8")).hexdigest()[:20]
    return {"pair_id": identity, "left": a, "right": b, "kind": kind, "hamming_distance": distance}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(archive_path: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Audit/extract SVRDD, freeze its published splits, and write a YOLO YAML.

    Args:
        archive_path: Path to an already-present SVRDD YOLO ZIP archive.
        output_dir: Writable directory for extraction, manifests, YAML and audit.

    Returns:
        ``(dataset_yaml_path, audit_json_path)`` on a passing audit.

    Raises:
        PreparationError: after saving the audit if an integrity, geometry,
            image-decode, class-map or cross-split leakage review gate fails.
    """
    archive_path = Path(archive_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "data_preparation_audit.json"
    yaml_path = output_dir / "svrdd.yaml"
    audit: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "archive": {"path": str(archive_path)},
        "status": "running",
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    def fail(message: str) -> None:
        audit["errors"].append(message)

    try:
        if not archive_path.is_file():
            raise PreparationError(f"Archive does not exist: {archive_path}")
        archive_sha, archive_md5, archive_size = _sha256_and_md5(archive_path)
        exact_release_match = archive_size == SOURCE["expected_bytes"] and archive_md5 == SOURCE["expected_md5"]
        audit["archive"].update({
            "bytes": archive_size,
            "md5": archive_md5,
            "sha256": archive_sha,
            "publisher_size_match": archive_size == SOURCE["expected_bytes"],
            "publisher_md5_match": archive_md5 == SOURCE["expected_md5"],
            "exact_publisher_archive_match": exact_release_match,
        })
        if not exact_release_match:
            audit["warnings"].append(
                "Archive does not match the exact publisher ZIP size+MD5. Structural, split, label and image checks are applied; "
                "a repackaged mirror's external provenance still requires confirmation."
            )

        members, zip_summary = _audit_zip(archive_path)
        audit["checks"]["zip"] = zip_summary
        name_list = [name for _, name in members]
        archive_images = [name for name in name_list if PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES and "/images/" in f"/{name.lower()}"]
        archive_labels = [name for name in name_list if "/labels/" in f"/{name.lower()}" and name.lower().endswith(".txt")]
        audit["checks"]["archive_inventory"] = {
            "image_files": len(archive_images),
            "yolo_label_files": len(archive_labels),
            "member_count": len(name_list),
            "original_release_member_count": SOURCE["expected_archive_entries"],
            "member_count_matches_original": len(name_list) == SOURCE["expected_archive_entries"],
            "split_metadata_files": [name for name in name_list if PurePosixPath(name).name.lower() in {"train.txt", "val.txt", "test.txt"}],
            "expected_image_count_match": len(archive_images) == SOURCE["expected_images"],
            "expected_label_file_count_match": len(archive_labels) == SOURCE["expected_images"],
        }
        if len(archive_images) != SOURCE["expected_images"]:
            fail(f"Expected {SOURCE['expected_images']} archive images; found {len(archive_images)}")
        if len(archive_labels) != SOURCE["expected_images"]:
            fail(f"Expected {SOURCE['expected_images']} YOLO label files; found {len(archive_labels)}")

        extraction_root = output_dir / "extracted"
        _extract_safely(archive_path, members, extraction_root, archive_sha)
        root, metadata = _find_dataset_root(extraction_root)
        splits = _load_splits(root, metadata)
        split_counts = {name: len(paths) for name, paths in splits.items()}
        audit["checks"]["source_split_manifests"] = {
            "counts": split_counts,
            "expected_counts_match": split_counts == SOURCE["expected_split_counts"],
            "split_sources": {name: str(path) for name, path in metadata.items()},
            "overlap": "none",
            "boundary_policy": "The source's train/val/test assignment is retained; no random repartitioning or cross-boundary deduplication is performed.",
        }
        if split_counts != SOURCE["expected_split_counts"]:
            fail(f"Source split counts differ from expected counts: {split_counts}")

        all_images = [(split, image) for split, paths in splits.items() for image in paths]
        image_records = []
        geometry_errors = []
        decode_errors = []
        class_counts: Counter[int] = Counter()
        split_class_counts: dict[str, Counter[int]] = {split: Counter() for split in splits}
        empty_labels = 0
        try:
            from PIL import Image, ImageFile
        except ImportError as exc:
            raise PreparationError("Pillow is required in the Kaggle runtime to decode and inspect images") from exc
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        for split, image_path in all_images:
            relative = image_path.relative_to(root).as_posix()
            label_path = _label_path(image_path)
            if not label_path.is_file():
                geometry_errors.append({"image": relative, "error": "label_file_missing"})
                continue
            classes, label_errors = _parse_label_file(label_path)
            if not label_path.read_text(encoding="utf-8-sig").strip():
                empty_labels += 1
            for class_id in classes:
                class_counts[class_id] += 1
                split_class_counts[split][class_id] += 1
            if label_errors:
                geometry_errors.append({"image": relative, "errors": label_errors[:20], "error_count": len(label_errors)})
            try:
                with Image.open(image_path) as img:
                    img.load()
                    width, height = img.size
                    if width <= 0 or height <= 0:
                        raise ValueError(f"invalid image dimensions: {img.size}")
                    rgb = img.convert("RGB")
                    pixels_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                    perceptual = _dhash(img)
                    image_records.append({
                        "relative_path": relative,
                        "split": split,
                        "bytes": image_path.stat().st_size,
                        "width": width,
                        "height": height,
                        "sha256": _file_sha256(image_path),
                        "decoded_rgb_sha256": pixels_sha,
                        "dhash64": perceptual,
                        "district": image_path.parent.parent.name,
                        "filename": image_path.name,
                    })
            except Exception as exc:
                decode_errors.append({"image": relative, "error": f"{type(exc).__name__}: {exc}"})

        audit["checks"]["labels_and_classes"] = {
            "class_map": SOURCE["class_names"],
            "class_map_evidence": "Prompted source order; independently cross-checked against paired SVRDD YOLO/XML sample annotations in research/evidence/alternatives and archive-wide observed IDs 0..6.",
            "class_ids_observed": sorted(class_counts),
            "class_id_set_valid": sorted(class_counts) == list(range(len(SOURCE["class_names"]))),
            "object_rows_by_class": {str(k): class_counts[k] for k in range(len(SOURCE["class_names"]))},
            "split_object_rows_by_class": {split: {str(k): counts[k] for k in range(len(SOURCE["class_names"]))} for split, counts in split_class_counts.items()},
            "source_evidence_box_counts": {str(k): SOURCE["expected_box_counts"][k] for k in range(len(SOURCE["class_names"]))},
            "source_evidence_split_box_counts": {
                split: {str(k): counts[k] for k, counts in enumerate(SOURCE["expected_split_box_counts"][split])}
                for split in splits
            },
            "box_counts_match_source_evidence": (
                [class_counts[k] for k in range(len(SOURCE["class_names"]))] == SOURCE["expected_box_counts"]
                and all(
                    [split_class_counts[split][k] for k in range(len(SOURCE["class_names"]))]
                    == SOURCE["expected_split_box_counts"][split]
                    for split in splits
                )
            ),
            "empty_label_files": empty_labels,
            "label_geometry_errors": geometry_errors[:100],
            "label_geometry_error_count": len(geometry_errors),
        }
        if sorted(class_counts) != list(range(len(SOURCE["class_names"]))):
            fail(f"Class-map validation failed; observed IDs {sorted(class_counts)}")
        if not audit["checks"]["labels_and_classes"]["box_counts_match_source_evidence"]:
            fail("Per-class/split object counts do not match the independently inspected SVRDD source evidence")
        if geometry_errors:
            fail(f"Found {len(geometry_errors)} missing/malformed/out-of-bounds labels; geometry must be resolved")
        if empty_labels:
            audit["warnings"].append(f"Found {empty_labels} empty label files; investigate before treating any as intentional negatives")

        audit["checks"]["image_decode_and_geometry"] = {
            "images_expected": len(all_images),
            "images_fully_decoded": len(image_records),
            "image_decode_errors": decode_errors[:100],
            "image_decode_error_count": len(decode_errors),
            "dimension_summary": _dimension_summary(image_records),
            "normalized_box_geometry": "Every parsed box has finite center/size, positive width/height, and corners within normalized [0,1] (1e-6 tolerance).",
        }
        if decode_errors:
            fail(f"Found {len(decode_errors)} images that failed full decode or dimension checks")
        if len(image_records) != len(all_images):
            fail(f"Only {len(image_records)} of {len(all_images)} manifest images passed complete image audit")

        review_path = output_dir / "DUPLICATE_REVIEW_DECISIONS.json"
        duplicates = _duplicate_review(image_records, review_path)
        audit["checks"]["duplicates"] = {
            "byte_exact_cross_split_pairs": duplicates["exact_cross_split_pairs"],
            "decoded_pixel_exact_cross_split_pairs": duplicates["decoded_pixel_exact_cross_split_pairs"],
            "perceptual_candidate_threshold_hamming": duplicates["perceptual_candidate_threshold_hamming"],
            "perceptual_candidate_count": duplicates["perceptual_candidate_count"],
            "perceptual_candidates": duplicates["perceptual_candidates"],
            "accepted_distinct_perceptual_candidates": duplicates["accepted_distinct_perceptual_candidates"],
            "unresolved_or_confirmed_cross_split_perceptual_pairs": duplicates["unresolved_or_confirmed_cross_split_perceptual_pairs"],
            "review_decisions_path": str(review_path),
            "review_decisions_format": duplicates["decision_format"],
            "policy": "Any byte/pixel exact cross-split pair fails. Perceptual candidates require explicit distinct/duplicate decisions; unresolved or confirmed cross-split candidates fail. No automatic reassignment is made.",
        }
        if duplicates["exact_cross_split_pairs"] or duplicates["decoded_pixel_exact_cross_split_pairs"]:
            fail("Exact or decoded-pixel-identical image found across published split boundaries")
        if duplicates["unresolved_or_confirmed_cross_split_perceptual_pairs"]:
            fail(
                f"{len(duplicates['unresolved_or_confirmed_cross_split_perceptual_pairs'])} cross-split perceptual candidates remain unresolved or confirmed duplicate; review the audit and decisions file"
            )

        audit["checks"]["source_group_leakage"] = {
            "status": "not_proven",
            "districts": sorted({record["district"] for record in image_records}),
            "filename_tokens_preserved": True,
            "group_guarantee": False,
            "note": "The archive has no verified source/frame group manifest. Districts and filename tokens are retained as audit clues only; no grouping guarantee or inferred frame group is asserted.",
        }

        if audit["errors"]:
            audit["status"] = "failed"
            audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(audit_path, audit)
            raise PreparationError("SVRDD preparation failed: " + "; ".join(audit["errors"]), audit_path)

        manifest_dir = output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_paths = {}
        for split, paths in splits.items():
            target = manifest_dir / f"{split}.txt"
            target.write_text("\n".join(str(path.resolve()) for path in paths) + "\n", encoding="utf-8")
            manifest_paths[split] = str(target.resolve())
        yaml_doc = {
            "path": str(root.resolve()),
            "train": manifest_paths["train"],
            "val": manifest_paths["val"],
            "test": manifest_paths["test"],
            "names": SOURCE["class_names"],
        }
        # JSON is a strict subset of YAML; writing it this way avoids a PyYAML dependency.
        yaml_path.write_text(json.dumps(yaml_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        audit["outputs"] = {
            "dataset_yaml": str(yaml_path),
            "manifests": manifest_paths,
            "extracted_dataset_root": str(root.resolve()),
        }
        audit["status"] = "passed"
        audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(audit_path, audit)
        return yaml_path, audit_path
    except PreparationError as exc:
        if not audit_path.exists() or audit.get("status") == "running":
            audit["status"] = "failed"
            audit["errors"].append(str(exc))
            audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(audit_path, audit)
        if exc.audit_path is None:
            exc.audit_path = audit_path
        raise
    except Exception as exc:
        audit["status"] = "failed"
        audit["errors"].append(f"Unexpected {type(exc).__name__}: {exc}")
        audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(audit_path, audit)
        raise PreparationError(f"SVRDD preparation failed: {type(exc).__name__}: {exc}", audit_path) from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dimension_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    sizes = Counter(f"{row['width']}x{row['height']}" for row in records)
    return {"unique_dimensions": len(sizes), "most_common": sizes.most_common(12)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_path", help="Path to an already-present SVRDD YOLO ZIP")
    parser.add_argument("output_dir", help="Writable preparation output directory")
    args = parser.parse_args()
    try:
        yaml_file, audit_file = prepare(args.archive_path, args.output_dir)
    except PreparationError as exc:
        print(f"FAILED: {exc}")
        if exc.audit_path:
            print(f"Audit: {exc.audit_path}")
        raise SystemExit(2)
    print(json.dumps({"dataset_yaml": str(yaml_file), "audit": str(audit_file)}, indent=2))
