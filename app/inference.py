"""Verified v3 inference shared by image and video processing."""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock

# Dependencies are installed at setup time, never while a visitor processes a file.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
from PIL import ImageOps

EXPECTED_NAMES = [
    "longitudinal crack", "transverse crack", "alligator crack", "pothole",
    "manhole cover", "longitudinal patch", "transverse patch",
]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models/v3/yolo26m/best.pt"


@dataclass
class RoadDetector:
    model: object
    contract: dict
    device: str
    lock: RLock = field(default_factory=RLock)

    def annotate(self, bgr):
        settings = self.contract
        with self.lock:
            result = self.model.predict(
                bgr, imgsz=settings["imgsz"], conf=settings["confidence"],
                iou=settings["iou"], max_det=settings["max_det"],
                quantize=settings["quantize"], rect=settings["rect"], nms=settings["nms"],
                device=self.device, batch=1, verbose=False, save=False,
            )[0]
            count = len(result.boxes) if result.boxes is not None else 0
            annotated = result.plot()
        return annotated, count


def load_detector(weights=DEFAULT_WEIGHTS, contract_path=None, device=None):
    """Load only the supplied verified checkpoint; never download fallback weights."""
    weights = Path(weights).resolve()
    contract_path = Path(contract_path) if contract_path else weights.with_name("inference_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not weights.is_file():
        raise FileNotFoundError(f"Model missing: {weights}")
    with weights.open("rb") as stream:
        checksum = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != contract["checkpoint_sha256"]:
        raise ValueError("Model checksum differs from the exported contract.")
    expected = {"names": EXPECTED_NAMES, "imgsz": 1024, "confidence": 0.35,
                "quantize": 32, "rect": False, "nms": True, "iou": 0.7, "max_det": 300}
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported inference contract.")
    import torch
    import ultralytics
    from ultralytics import YOLO
    if ultralytics.__version__ != contract["ultralytics"]:
        raise RuntimeError(f"Use ultralytics=={contract['ultralytics']} for this export.")
    torch.set_num_threads(max(1, min(4, int(os.environ.get("ROAD_CPU_THREADS", "2")))))
    device = device or os.environ.get("ROAD_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights), task="detect")
    if [model.names[i] for i in range(len(model.names))] != EXPECTED_NAMES:
        raise ValueError("Checkpoint class order differs from the export.")
    return RoadDetector(model, contract, device)


def detect_image(detector, image):
    """Decoded PIL image -> RGB annotated array and detection count."""
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    annotated, count = detector.annotate(np.ascontiguousarray(rgb[:, :, ::-1]))
    return np.ascontiguousarray(annotated[:, :, ::-1]), count
