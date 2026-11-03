"""Bounded decoding and silent H.264 export, one video frame at a time."""
from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import subprocess
import tempfile
import warnings
import uuid
import weakref

import cv2
import imageio_ffmpeg
from PIL import Image, ImageOps, UnidentifiedImageError

# Preserve Pillow's normal decoder before Ultralytics installs optional-format hooks.
_PIL_OPEN = Image.open

IMAGE_LIMIT = 15 * 1024 * 1024
VIDEO_LIMIT = 25 * 1024 * 1024
MAX_PIXELS = 30_000_000
MAX_VIDEO_SECONDS = 30
MAX_VIDEO_FRAMES = 900
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_TYPES = {".mp4", ".avi", ".mov"}


class MediaError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def media_kind(name):
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_TYPES:
        return "image"
    if suffix in VIDEO_TYPES:
        return "video"
    raise MediaError("unsupported")


def open_image(data):
    if len(data) > IMAGE_LIMIT:
        raise MediaError("image_size")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with _PIL_OPEN(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise MediaError("unsupported")
                if image.width * image.height > MAX_PIXELS:
                    raise MediaError("image_pixels")
                image.load()
                return ImageOps.exif_transpose(image).copy()
    except (UnidentifiedImageError, OSError, ModuleNotFoundError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise MediaError("corrupt") from None


def _remove_workspace(path, parent):
    if not path.exists():
        return
    if path.resolve().parent != parent or not path.name.startswith("road-video-"):
        raise ValueError("Unexpected video temporary directory.")
    # Only known, directly owned files: never recursively delete an arbitrary tree.
    for item in path.iterdir():
        if item.name not in {"input.mp4", "input.mov", "input.avi", "annotated.mp4", "encoder.log"}:
            raise ValueError("Unexpected file in video temporary directory.")
        item.unlink()
    path.rmdir()


class MediaWorkspace:
    def __init__(self, parent=None):
        parent = Path(parent or tempfile.gettempdir()).resolve()
        path = parent / ("road-video-" + uuid.uuid4().hex)
        path.mkdir()
        self.name = str(path)
        self._finalizer = weakref.finalize(self, _remove_workspace, path, parent)

    def cleanup(self):
        self._finalizer()


@dataclass
class VideoOutput:
    directory: object
    path: Path
    frames: int
    detections: int

    def cleanup(self):
        self.directory.cleanup()


def process_video(data, suffix, detector, progress=None, temp_parent=None):
    """Stream every frame through inference. Original audio is never included."""
    if suffix.lower() not in VIDEO_TYPES:
        raise MediaError("unsupported")
    if len(data) > VIDEO_LIMIT:
        raise MediaError("video_size")
    directory = MediaWorkspace(temp_parent)
    folder = Path(directory.name)
    source, target, log_path = folder / ("input" + suffix.lower()), folder / "annotated.mp4", folder / "encoder.log"
    capture = encoder = None
    success = False
    try:
        source.write_bytes(data)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise MediaError("corrupt")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not all(math.isfinite(x) and x > 0 for x in (fps, total, width, height)) or fps > 240:
            raise MediaError("corrupt")
        if total > MAX_VIDEO_FRAMES or total / fps > MAX_VIDEO_SECONDS + 0.01:
            raise MediaError("video_length")
        if width * height > 3840 * 2160:
            raise MediaError("video_pixels")
        total = int(total)
        with log_path.open("wb") as log:
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                       "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                       "-preset", "veryfast", "-crf", "23", "-threads", "2",
                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", str(target)]
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            frames = detections = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames += 1
                if frames > MAX_VIDEO_FRAMES or frames / fps > MAX_VIDEO_SECONDS + 0.01:
                    raise MediaError("video_length")
                if frame.shape[:2] != (height, width):
                    raise MediaError("corrupt")
                annotated, count = detector.annotate(frame)
                detections += count
                encoder.stdin.write(annotated.tobytes())
                if progress:
                    progress(min(frames / total, 0.99))
            encoder.stdin.close()
            if encoder.wait(timeout=60) != 0:
                raise MediaError("encoding")
        if frames == 0 or abs(frames - total) > 1 or not target.is_file():
            raise MediaError("corrupt")
        if progress:
            progress(1.0)
        success = True
        return VideoOutput(directory, target, frames, detections)
    except (BrokenPipeError, subprocess.TimeoutExpired, FileNotFoundError):
        raise MediaError("encoding") from None
    finally:
        if capture is not None:
            capture.release()
        if encoder is not None and encoder.poll() is None:
            try:
                if encoder.stdin and not encoder.stdin.closed:
                    encoder.stdin.close()
            except BrokenPipeError:
                pass
            encoder.kill()
            encoder.wait()
        if success:
            source.unlink(missing_ok=True)
            log_path.unlink(missing_ok=True)
        else:
            directory.cleanup()
