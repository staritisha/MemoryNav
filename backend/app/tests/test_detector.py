"""
MemoryNav — Detector Unit Test
backend/tests/test_detector.py

Static-image smoke test for Module 1's YOLOv8 wrapper. Deliberately
avoids the webcam (scripts/day1_yolo_test.py already covers that) so
this is deterministic and CI-friendly.

Uses the sample image bundled inside the `ultralytics` package itself
(ultralytics.utils.ASSETS / "bus.jpg") rather than a custom fixture
file or a network download — it ships with `pip install ultralytics`,
so there's nothing extra to commit or fetch, and Ultralytics' own test
suite relies on it the same way. It reliably contains multiple people
and a bus, so a working detector should never return zero detections
on it.

Run:
    pytest backend/tests/test_detector.py -v
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from app.perception.detector import Detection, Detector

try:
    from ultralytics.utils import ASSETS

    TEST_IMAGE_PATH: Optional[Path] = ASSETS / "bus.jpg"
except ImportError:
    TEST_IMAGE_PATH = None


@pytest.fixture(scope="module")
def detector() -> Detector:
    """One Detector instance shared across this file's tests — loading
    the model is the expensive part, so it only happens once per run."""
    return Detector()


@pytest.fixture(scope="module")
def test_image() -> np.ndarray:
    if TEST_IMAGE_PATH is None or not TEST_IMAGE_PATH.exists():
        pytest.skip(
            "ultralytics-bundled sample image not found — check that "
            "`ultralytics` is installed correctly."
        )
    frame = cv2.imread(str(TEST_IMAGE_PATH))
    if frame is None:
        pytest.skip(f"cv2 failed to load {TEST_IMAGE_PATH}")
    return frame


def test_detector_finds_objects_in_static_image(detector, test_image):
    """bus.jpg contains multiple people and a bus — a working detector
    must return at least one detection on it."""
    detections = detector.detect(test_image)
    assert len(detections) > 0, "Detector returned no detections on a known non-empty image."


def test_detections_are_well_formed(detector, test_image):
    """Every detection should carry sane, usable values — catches
    silent unit/format regressions even when detections aren't empty."""
    detections = detector.detect(test_image)
    h, w = test_image.shape[:2]

    for d in detections:
        assert isinstance(d, Detection)
        assert isinstance(d.class_name, str) and d.class_name
        assert 0.0 <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h


def test_detector_finds_a_person(detector, test_image):
    """bus.jpg is the canonical Ultralytics sample with several
    pedestrians in it — a sanity check that class IDs are mapped to the
    right names, not just that *something* with high confidence came
    back."""
    detections = detector.detect(test_image)
    class_names = {d.class_name for d in detections}
    assert "person" in class_names


def test_empty_frame_returns_no_detections(detector):
    """detect() should fail soft (empty list) on bad input, not raise —
    matters once this is wired into a live capture loop where an
    occasional dropped/empty frame shouldn't crash the pipeline."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detector.detect(empty) == []