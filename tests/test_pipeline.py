"""Small smoke tests against the shipped synthetic assets (no movie media required)."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.cover_replacer import locate_cover, process_image, process_video, read_image


DEMO = Path(__file__).resolve().parents[1] / "examples" / "synthetic"


class CoverReplacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = read_image(DEMO / "scene_before.png")
        cls.cover = read_image(DEMO / "cover_reference.png")
        cls.replacement = read_image(DEMO / "replacement_art.png")

    def test_detects_cover(self):
        match = locate_cover(self.scene, self.cover)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match.good_matches, 10)
        self.assertGreaterEqual(match.inliers, 6)

    def test_overlay_changes_scene(self):
        result, match = process_image(self.scene, self.cover, self.replacement)
        self.assertIsNotNone(match)
        self.assertGreater(int(np.count_nonzero(cv2.absdiff(self.scene, result))), 5000)

    def test_video_demo_contains_frames(self):
        cap = cv2.VideoCapture(str(DEMO / "output_clip.mp4"))
        self.assertTrue(cap.isOpened())
        self.assertGreaterEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
        cap.release()


if __name__ == "__main__":
    unittest.main()
