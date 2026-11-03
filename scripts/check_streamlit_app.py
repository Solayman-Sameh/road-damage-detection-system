"""Exercise the real model, silent codec pipeline, limits and Streamlit state."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
import uuid

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
os.environ["ROAD_DEVICE"] = "cpu"
from inference import load_detector, detect_image
from media import MediaError, open_image, process_video
from streamlit.testing.v1 import AppTest

QA = ROOT / "../../tmp/road-ui-qa" / uuid.uuid4().hex
QA.mkdir(parents=True)


class AppChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = load_detector(device="cpu")
        cls.example = (ROOT / "app/examples/scene_01.jpg").read_bytes()
        cls.source = QA / "source-with-audio.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
                        "-loop", "1", "-framerate", "2", "-i", str(ROOT / "app/examples/scene_01.jpg"),
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
                        "-t", "1.5", "-vf", "scale=320:320", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.source)],
                       check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def test_real_image_and_contract(self):
        start = time.perf_counter()
        rgb, count = detect_image(self.detector, open_image(self.example))
        self.assertEqual(rgb.shape, (1024, 1024, 3))
        self.assertGreater(count, 0)
        self.assertEqual(self.detector.contract["confidence"], .35)
        print("CPU warm image seconds:", round(time.perf_counter()-start, 3), flush=True)

    def test_real_video_no_audio_all_frames_and_cleanup(self):
        progress = []
        start = time.perf_counter()
        result = process_video(self.source.read_bytes(), ".mp4", self.detector,
                               progress.append, temp_parent=QA)
        folder = result.path.parent
        self.assertEqual(result.frames, 3)
        self.assertGreater(result.detections, 0)
        self.assertEqual(progress[-1], 1.)
        info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(result.path)],
                              capture_output=True, text=True).stderr
        self.assertIn("h264", info)
        self.assertNotIn("Audio:", info)
        self.assertEqual(sorted(x.name for x in folder.iterdir()), ["annotated.mp4"])
        capture = cv2.VideoCapture(str(result.path))
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
        capture.release()
        print("CPU video seconds:", round(time.perf_counter()-start, 3), flush=True)
        result.cleanup()
        self.assertFalse(folder.exists())

    def test_bad_images_and_video_limits_rejected(self):
        with self.assertRaisesRegex(MediaError, "corrupt"):
            open_image(b"invalid image")
        before = set(QA.iterdir())
        with self.assertRaises(MediaError):
            process_video(b"invalid video", ".mp4", self.detector, temp_parent=QA)
        self.assertEqual(before, set(QA.iterdir()))
        too_long = QA / "long.avi"
        writer = cv2.VideoWriter(str(too_long), cv2.VideoWriter_fourcc(*"MJPG"), 1., (64, 64))
        for _ in range(31):
            writer.write(np.zeros((64, 64, 3), np.uint8))
        writer.release()
        with self.assertRaisesRegex(MediaError, "video_length"):
            process_video(too_long.read_bytes(), ".avi", self.detector, temp_parent=QA)

    def test_streamlit_example_detect_language_and_clear(self):
        app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=45).run()
        self.assertFalse(app.exception, list(app.exception))
        app.button(key="example_scene_01").click().run()
        self.assertFalse(app.exception, list(app.exception))
        self.assertIsNone(app.session_state["road_result"])

        next(b for b in app.button if b.label == "Detect").click().run()
        self.assertFalse(app.exception, list(app.exception))
        result = app.session_state["road_result"]
        self.assertGreater(result["count"], 0)
        original = sha256(result["image"].tobytes()).hexdigest()
        app.toggle(key="road_arabic").set_value(True).run()
        self.assertFalse(app.exception, list(app.exception))
        self.assertEqual(app.title[0].value, "كشف أضرار الطرق")
        self.assertEqual(sha256(app.session_state["road_result"]["image"].tobytes()).hexdigest(), original)
        app.toggle(key="road_arabic").set_value(False).run()
        self.assertEqual(app.title[0].value, "Road Damage Detection")
        self.assertEqual(sha256(app.session_state["road_result"]["image"].tobytes()).hexdigest(), original)
        app.toggle(key="road_arabic").set_value(True).run()
        next(b for b in app.button if b.label == "مسح").click().run()
        self.assertIsNone(app.session_state["road_input"])
        self.assertIsNone(app.session_state["road_result"])

    def test_failure_retry_is_immediately_available(self):
        os.environ["ROAD_MODEL_PATH"] = str(QA / "missing-best.pt")
        try:
            app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=45).run()
            app.button(key="example_scene_01").click().run()
            next(b for b in app.button if b.label == "Detect").click().run()
            self.assertFalse(app.exception, list(app.exception))
            self.assertEqual(app.session_state["road_error"], "model")
            self.assertTrue(app.button(key="road_retry"))
        finally:
            del os.environ["ROAD_MODEL_PATH"]
        app.button(key="road_retry").click().run()
        self.assertFalse(app.exception, list(app.exception))
        self.assertIsNone(app.session_state["road_error"])
        self.assertGreater(app.session_state["road_result"]["count"], 0)



if __name__ == "__main__":
    unittest.main(verbosity=2)
