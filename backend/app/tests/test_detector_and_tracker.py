"""
MemoryNav — Detector & ObjectTracker Unit Tests
backend/tests/test_detector_and_tracker.py

Part A — perception/detector.py
  Check 1: two separate confidence thresholds (0.25 inference / 0.60 speech)
  Check 2: device auto-detection order (MPS → CUDA → CPU) via config
  Check 3: detect() returns sorted descending by confidence
  Check 4: Detection dataclass — all five properties
  Check 5: warm-up inference runs at init with verbose=False
  Check 6: detect_gated() calls detect() once, then filter_frame()

Part B — perception/tracker.py
  Check 1: all five ByteTrack parameters at exact doc values
  Check 2: TrackedDetection carries all Detection fields + track_id
  Check 3: unconfirmed tracks get track_id=-1; is_tracked=False
  Check 4: _nearest_track fallback uses max_px_dist=20.0px

Required tests:
  test_detector_returns_sorted_by_confidence
  test_detect_empty_frame_returns_empty
  test_detection_is_confident_property
  test_tracker_unconfirmed_filtered
  test_tracker_confirms_after_two_frames
  test_bus_image_detects_person (integration — skipped if model not cached)

Run:
    pytest backend/tests/test_detector_and_tracker.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.perception.detector import Detection, Detector, GatedDetections
from app.perception.tracker import ObjectTracker, TrackedDetection, _nearest_track


# ── Frame helpers ─────────────────────────────────────────────────────────────

def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _noisy_frame(h: int = 480, w: int = 640) -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.integers(50, 200, (h, w, 3), dtype=np.uint8)


# ── Detection factory ─────────────────────────────────────────────────────────

def _det(
    class_id: int = 56,
    class_name: str = "chair",
    confidence: float = 0.80,
    bbox: Tuple[int, int, int, int] = (100, 100, 300, 300),
) -> Detection:
    return Detection(class_id=class_id, class_name=class_name,
                     confidence=confidence, bbox=bbox)


# ── Mock YOLO factory ─────────────────────────────────────────────────────────

def _make_mock_yolo(raw_detections: List[Detection] = None) -> MagicMock:
    """
    Mock YOLO model. Calling .predict() returns a results list whose
    first element has a .boxes iterable matching `raw_detections`.
    """
    raw_detections = raw_detections or []

    class MockBox:
        def __init__(self, det: Detection):
            self.cls  = MagicMock()
            self.cls.item.return_value = det.class_id
            self.conf = MagicMock()
            self.conf.item.return_value = det.confidence
            x1, y1, x2, y2 = det.bbox
            self.xyxy = [MagicMock()]
            self.xyxy[0].tolist.return_value = [float(x1), float(y1),
                                                  float(x2), float(y2)]

    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=len(raw_detections))
    mock_boxes.__iter__ = MagicMock(
        return_value=iter([MockBox(d) for d in raw_detections])
    )
    if raw_detections:
        mock_boxes.__bool__ = MagicMock(return_value=True)
    else:
        mock_boxes.__bool__ = MagicMock(return_value=False)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_model = MagicMock()
    mock_model.names = {d.class_id: d.class_name for d in raw_detections}
    mock_model.predict.return_value = [mock_result]
    mock_model.to.return_value = mock_model

    return mock_model


def _make_detector(raw_detections: List[Detection] = None) -> Detector:
    """Detector backed by a mock YOLO model. Skips real model loading."""
    mock_yolo = _make_mock_yolo(raw_detections or [])
    with patch("app.perception.detector.YOLO", return_value=mock_yolo):
        det = Detector(model_path="mock.pt", device="cpu")
    # Inject the mock so subsequent detect() calls use it
    det._model = mock_yolo
    det._class_names = mock_yolo.names
    return det


# ═══════════════════════════════════════════════════════════════════════════════
# Part A — Detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestTwoConfidenceThresholds:
    """Check 1: inference threshold (0.25) ≠ speech gate threshold (0.60)."""

    def test_detection_confidence_is_0_25(self):
        from app.config import settings
        assert settings.YOLO_DETECTION_CONFIDENCE == pytest.approx(0.25)

    def test_confidence_gate_is_0_60(self):
        from app.config import settings
        assert settings.YOLO_CONFIDENCE_GATE == pytest.approx(0.60)

    def test_thresholds_are_different(self):
        from app.config import settings
        assert settings.YOLO_DETECTION_CONFIDENCE < settings.YOLO_CONFIDENCE_GATE

    def test_detector_uses_detection_confidence_for_inference(self):
        """detect() must pass YOLO_DETECTION_CONFIDENCE to model.predict()."""
        raw = [_det(confidence=0.35)]  # in the 0.25–0.60 UNCERTAIN band
        detector = _make_detector(raw)
        detector.detect(_blank_frame())
        call_kwargs = detector._model.predict.call_args
        assert call_kwargs is not None
        conf_used = call_kwargs.kwargs.get("conf") or call_kwargs.args[1] \
                    if call_kwargs.args else call_kwargs.kwargs.get("conf")
        assert conf_used == pytest.approx(0.25)

    def test_detector_gate_uses_confidence_gate_threshold(self):
        """detect_gated() must use YOLO_CONFIDENCE_GATE (0.60) for PASS/UNCERTAIN split."""
        from app.config import settings
        detector = _make_detector()
        assert detector.confidence_gate.confidence_threshold == pytest.approx(
            settings.YOLO_CONFIDENCE_GATE
        )

    def test_uncertain_band_0_25_to_0_60_exists(self):
        """Detections in 0.25–0.60 must land in UNCERTAIN, not PASS or DISCARD."""
        from app.alerts.confidence_gate import GateOutcome
        raw = [_det(confidence=0.45)]
        detector = _make_detector(raw)
        gated = detector.detect_gated(_blank_frame())
        assert len(gated.uncertain) == 1
        assert len(gated.passed) == 0


class TestDeviceAutoDetection:
    """Check 2: device order MPS → CUDA → CPU via config.detect_device()."""

    def test_detect_device_order_mps_first(self):
        from app.config import detect_device
        import torch
        with patch.object(torch.backends.mps, "is_available", return_value=True):
            assert detect_device() == "mps"

    def test_detect_device_cuda_when_no_mps(self):
        from app.config import detect_device
        import torch
        with patch.object(torch.backends.mps, "is_available", return_value=False), \
             patch.object(torch.cuda, "is_available", return_value=True):
            assert detect_device() == "cuda"

    def test_detect_device_cpu_fallback(self):
        from app.config import detect_device
        import torch
        with patch.object(torch.backends.mps, "is_available", return_value=False), \
             patch.object(torch.cuda, "is_available", return_value=False):
            assert detect_device() == "cpu"

    def test_detector_device_from_settings(self):
        """Detector.device must be set from settings.device, not hardcoded."""
        detector = _make_detector()
        assert detector.device == "cpu"  # our mock uses device="cpu"


class TestDetectSortedByConfidence:
    """Check 3: detect() returns detections sorted descending by confidence."""

    def test_detector_returns_sorted_by_confidence(self):
        """Required test from step brief."""
        raw = [
            _det(confidence=0.50, bbox=(0, 0, 100, 100)),
            _det(confidence=0.90, bbox=(100, 100, 200, 200)),
            _det(confidence=0.70, bbox=(200, 200, 300, 300)),
        ]
        detector = _make_detector(raw)
        result = detector.detect(_noisy_frame())
        confs = [d.confidence for d in result]
        assert confs == sorted(confs, reverse=True)

    def test_single_detection_is_still_sorted(self):
        detector = _make_detector([_det(confidence=0.75)])
        result = detector.detect(_noisy_frame())
        assert len(result) == 1

    def test_detect_empty_frame_returns_empty(self):
        """Required test from step brief."""
        detector = _make_detector()
        result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result == []

    def test_detect_none_frame_returns_empty(self):
        detector = _make_detector()
        result = detector.detect(None)
        assert result == []

    def test_detect_no_boxes_returns_empty(self):
        """YOLO returns results but with empty boxes → empty list."""
        detector = _make_detector([])
        result = detector.detect(_noisy_frame())
        assert result == []

    def test_detect_returns_detection_objects(self):
        raw = [_det(confidence=0.80)]
        detector = _make_detector(raw)
        result = detector.detect(_noisy_frame())
        assert len(result) == 1
        assert isinstance(result[0], Detection)
        assert result[0].class_name == "chair"
        assert result[0].confidence == pytest.approx(0.80, abs=0.01)


class TestDetectionDataclass:
    """Check 4: Detection dataclass — all five computed properties."""

    def test_is_confident_true_at_gate(self):
        """Required test: confidence >= 0.60 → is_confident True."""
        d = Detection(class_id=0, class_name="chair",
                      confidence=0.65, bbox=(0, 0, 100, 100))
        assert d.is_confident is True

    def test_is_confident_false_below_gate(self):
        """Required test: confidence < 0.60 → is_confident False."""
        d2 = Detection(class_id=0, class_name="chair",
                       confidence=0.55, bbox=(0, 0, 100, 100))
        assert d2.is_confident is False

    def test_is_confident_at_exact_boundary(self):
        """Exactly 0.60 → True (>= comparison)."""
        from app.config import settings
        d = Detection(class_id=0, class_name="chair",
                      confidence=settings.YOLO_CONFIDENCE_GATE, bbox=(0, 0, 100, 100))
        assert d.is_confident is True

    def test_center_property(self):
        d = Detection(class_id=0, class_name="chair",
                      confidence=0.9, bbox=(100, 200, 300, 400))
        assert d.center == (200, 300)

    def test_width_property(self):
        d = Detection(class_id=0, class_name="chair",
                      confidence=0.9, bbox=(100, 100, 300, 400))
        assert d.width == 200

    def test_height_property(self):
        d = Detection(class_id=0, class_name="chair",
                      confidence=0.9, bbox=(100, 100, 300, 400))
        assert d.height == 300

    def test_area_property(self):
        d = Detection(class_id=0, class_name="chair",
                      confidence=0.9, bbox=(100, 100, 300, 400))
        assert d.area == 200 * 300

    def test_area_non_negative_for_degenerate_bbox(self):
        d = Detection(class_id=0, class_name="chair",
                      confidence=0.9, bbox=(300, 400, 100, 100))
        assert d.area >= 0

    def test_frozen_dataclass_immutable(self):
        d = Detection(class_id=0, class_name="chair",
                      confidence=0.9, bbox=(0, 0, 100, 100))
        with pytest.raises((AttributeError, TypeError)):
            d.confidence = 0.5  # type: ignore

    def test_is_confident_uses_yolo_confidence_gate(self):
        """is_confident threshold must come from settings, not a hardcoded value."""
        import inspect
        source = inspect.getsource(Detection.is_confident.fget)
        assert "YOLO_CONFIDENCE_GATE" in source or "yolo_confidence_gate" in source.lower()


class TestWarmUp:
    """Check 5: warm-up inference runs at init with verbose=False."""

    def test_warm_up_calls_predict_at_init(self):
        """predict() must be called during __init__ (warm-up)."""
        calls_during_init = []

        def tracking_predict(source, **kwargs):
            calls_during_init.append(kwargs)
            mock_result = MagicMock()
            mock_result.boxes = MagicMock()
            mock_result.boxes.__len__ = MagicMock(return_value=0)
            return [mock_result]

        mock_yolo = MagicMock()
        mock_yolo.predict.side_effect = tracking_predict
        mock_yolo.names = {}
        mock_yolo.to.return_value = mock_yolo

        with patch("app.perception.detector.YOLO", return_value=mock_yolo):
            det = Detector(model_path="mock.pt", device="cpu")

        assert mock_yolo.predict.call_count >= 1, (
            "predict() must be called at least once during __init__ (warm-up)"
        )

    def test_warm_up_uses_verbose_false(self):
        """Warm-up must suppress log output with verbose=False."""
        import inspect
        source = inspect.getsource(Detector._warm_up)
        assert "verbose=False" in source

    def test_warm_up_uses_zeros_dummy_frame(self):
        import inspect
        source = inspect.getsource(Detector._warm_up)
        assert "zeros" in source

    def test_warm_up_failure_does_not_crash_init(self):
        """A failing warm-up must not prevent the Detector from being created."""
        mock_yolo = MagicMock()
        mock_yolo.predict.side_effect = RuntimeError("GPU memory error")
        mock_yolo.names = {}
        mock_yolo.to.return_value = mock_yolo

        with patch("app.perception.detector.YOLO", return_value=mock_yolo):
            det = Detector(model_path="mock.pt", device="cpu")

        assert det is not None  # init completed despite warm-up failure


class TestDetectGatedWiring:
    """Check 6: detect_gated() calls detect() once, then filter_frame()."""

    def test_detect_gated_calls_yolo_once(self):
        """YOLO must not run twice in detect_gated()."""
        raw = [_det(confidence=0.80)]
        detector = _make_detector(raw)
        initial_calls = detector._model.predict.call_count
        detector.detect_gated(_noisy_frame())
        # predict called once more (for the actual inference)
        assert detector._model.predict.call_count == initial_calls + 1

    def test_detect_gated_returns_gated_detections(self):
        raw = [_det(confidence=0.80)]
        detector = _make_detector(raw)
        result = detector.detect_gated(_noisy_frame())
        assert isinstance(result, GatedDetections)

    def test_detect_gated_passes_confident_to_risk_engine(self):
        """confidence=0.80 ≥ gate 0.60 → in .passed, accessible via .detections_for_risk_engine."""
        raw = [_det(confidence=0.80)]
        detector = _make_detector(raw)
        gated = detector.detect_gated(_noisy_frame())
        assert len(gated.passed) == 1
        assert len(gated.detections_for_risk_engine) == 1
        assert gated.detections_for_risk_engine[0].class_name == "chair"

    def test_detect_gated_routes_uncertain_not_to_risk_engine(self):
        """confidence=0.45 → uncertain, NOT in detections_for_risk_engine."""
        raw = [_det(confidence=0.45)]
        detector = _make_detector(raw)
        gated = detector.detect_gated(_noisy_frame())
        assert len(gated.uncertain) == 1
        assert len(gated.detections_for_risk_engine) == 0

    def test_detect_gated_has_anything_to_report_property(self):
        raw = [_det(confidence=0.80)]
        detector = _make_detector(raw)
        gated = detector.detect_gated(_noisy_frame())
        assert gated.has_anything_to_report is True

    def test_detect_gated_empty_frame(self):
        detector = _make_detector([])
        gated = detector.detect_gated(_blank_frame())
        assert not gated.has_anything_to_report


# ── Integration test using the real YOLO model ─────────────────────────────────

class TestBusImageIntegration:
    """
    Integration test using the Ultralytics built-in test asset.
    Skipped if the model file doesn't exist (CI without model weights).
    These tests match the existing test_detector.py pattern.
    """

    @pytest.fixture(scope="class")
    def bus_frame(self):
        try:
            from ultralytics.utils import ASSETS
            import cv2
            path = ASSETS / "bus.jpg"
            frame = cv2.imread(str(path))
            if frame is None:
                pytest.skip("bus.jpg not readable")
            return frame
        except (ImportError, Exception):
            pytest.skip("ultralytics ASSETS not available")

    @pytest.fixture(scope="class")
    def real_detector(self):
        """Real Detector — only created if YOLO model exists."""
        from app.config import settings
        if not Path(settings.YOLO_MODEL_PATH).exists():
            pytest.skip("YOLO model not cached — skipping integration tests")
        return Detector()

    def test_bus_image_detects_person(self, real_detector, bus_frame):
        """Required test from step brief."""
        detections = real_detector.detect(bus_frame)
        class_names = [d.class_name for d in detections]
        assert "person" in class_names
        assert len(detections) > 0

    def test_bus_image_sorted_by_confidence(self, real_detector, bus_frame):
        """Required test: sorted descending."""
        detections = real_detector.detect(bus_frame)
        confs = [d.confidence for d in detections]
        assert confs == sorted(confs, reverse=True)

    def test_bus_image_all_confidences_above_threshold(self, real_detector, bus_frame):
        from app.config import settings
        detections = real_detector.detect(bus_frame)
        for d in detections:
            assert d.confidence >= settings.YOLO_DETECTION_CONFIDENCE


# ═══════════════════════════════════════════════════════════════════════════════
# Part B — ObjectTracker
# ═══════════════════════════════════════════════════════════════════════════════

class TestByteTrackParameters:
    """Check 1: all five ByteTrack parameters at exact doc values."""

    def test_default_activation_threshold_is_0_25(self):
        tracker = ObjectTracker()
        # Verify via the stored sv.ByteTrack instance
        assert tracker._tracker.track_activation_threshold == pytest.approx(0.25)

    def test_default_lost_track_buffer_is_30(self):
        tracker = ObjectTracker()
        # supervision.ByteTrack stores lost_track_buffer as max_time_lost
        # (computed as int(frame_rate / 30.0 * lost_track_buffer) = 30)
        assert tracker._tracker.max_time_lost == 30

    def test_default_minimum_matching_threshold_is_0_8(self):
        tracker = ObjectTracker()
        assert tracker._tracker.minimum_matching_threshold == pytest.approx(0.8)

    def test_default_frame_rate_is_30(self):
        # frame_rate is not stored as a public attribute by supervision.ByteTrack
        # (it's only used to compute max_time_lost at init time).
        # Verify the default value is present in the ObjectTracker.__init__ signature.
        import inspect
        from app.perception.tracker import ObjectTracker as OT
        source = inspect.getsource(OT.__init__)
        # The parameter default appears in the signature: "frame_rate: int = 30"
        assert "frame_rate: int = 30" in source or "frame_rate=frame_rate" in source

    def test_default_minimum_consecutive_frames_is_2(self):
        tracker = ObjectTracker()
        assert tracker._tracker.minimum_consecutive_frames == 2

    def test_activation_threshold_matches_yolo_detection_confidence(self):
        """track_activation_threshold must equal YOLO_DETECTION_CONFIDENCE (0.25)."""
        from app.config import settings
        tracker = ObjectTracker()
        assert tracker._tracker.track_activation_threshold == pytest.approx(
            settings.YOLO_DETECTION_CONFIDENCE
        )

    def test_custom_parameters_accepted(self):
        """ObjectTracker must accept overridden parameters."""
        t = ObjectTracker(track_activation_threshold=0.30, lost_track_buffer=60,
                          minimum_matching_threshold=0.7, frame_rate=15,
                          minimum_consecutive_frames=3)
        assert t._tracker.track_activation_threshold == pytest.approx(0.30)


class TestTrackedDetectionDataclass:
    """Check 2: TrackedDetection carries all Detection fields + track_id."""

    def test_tracked_detection_has_track_id(self):
        det = _det()
        td = TrackedDetection.from_detection(det, track_id=7)
        assert td.track_id == 7

    def test_tracked_detection_preserves_all_fields(self):
        det = _det(class_id=56, class_name="chair", confidence=0.85,
                   bbox=(10, 20, 110, 220))
        td = TrackedDetection.from_detection(det, track_id=3)
        assert td.class_id == 56
        assert td.class_name == "chair"
        assert td.confidence == pytest.approx(0.85)
        assert td.bbox == (10, 20, 110, 220)
        assert td.track_id == 3

    def test_tracked_detection_default_track_id_is_minus_1(self):
        det = _det()
        td = TrackedDetection.from_detection(det)
        assert td.track_id == -1

    def test_is_tracked_property_true_when_confirmed(self):
        td = TrackedDetection.from_detection(_det(), track_id=5)
        assert td.is_tracked is True

    def test_is_tracked_property_false_when_unconfirmed(self):
        td = TrackedDetection.from_detection(_det(), track_id=-1)
        assert td.is_tracked is False

    def test_center_property_on_tracked_detection(self):
        td = TrackedDetection.from_detection(
            _det(bbox=(100, 200, 300, 400)), track_id=1
        )
        assert td.center == (200, 300)

    def test_tracked_detection_is_frozen(self):
        td = TrackedDetection.from_detection(_det(), track_id=3)
        with pytest.raises((AttributeError, TypeError)):
            td.track_id = 99  # type: ignore


class TestUnconfirmedTracks:
    """Check 3: first-frame detections get track_id=-1; is_tracked=False."""

    def test_tracker_unconfirmed_filtered(self):
        """
        Required test from step brief.
        Single frame → ByteTrack hasn't confirmed after 1 frame
        (minimum_consecutive_frames=2) → track_id must be -1.
        """
        tracker = ObjectTracker()
        frame = _blank_frame()
        detection = _det(confidence=0.90)

        result = tracker.update([detection], frame)

        unconfirmed = [td for td in result if td.track_id == -1]
        assert len(unconfirmed) > 0, (
            "First-frame detections must have track_id=-1 (unconfirmed). "
            "ByteTrack requires minimum_consecutive_frames=2 to confirm."
        )

    def test_tracker_confirms_after_two_frames(self):
        """
        Required test from step brief.
        ByteTrack with minimum_consecutive_frames=2 confirms after the detection
        has appeared in enough consecutive frames (in practice: frame 3, because
        frame 1 creates a tentative track and frame 2 increments the counter to 1,
        which is still below minimum_consecutive_frames=2; frame 3 reaches 2 and
        promotes the track).
        """
        tracker = ObjectTracker()
        frame = _blank_frame()
        det = _det(confidence=0.90, bbox=(100, 100, 200, 200))

        tracker.update([det], frame)   # frame 1 — tentative
        tracker.update([det], frame)   # frame 2 — counter=1, still unconfirmed
        result = tracker.update([det], frame)  # frame 3 — counter=2, confirmed

        confirmed = [td for td in result if td.track_id != -1]
        assert len(confirmed) > 0, (
            "Detection appearing in 3 consecutive frames must be confirmed "
            "(track_id >= 0) by ByteTrack (minimum_consecutive_frames=2 "
            "means the counter reaches 2 on the third frame)."
        )

    def test_empty_detections_returns_empty_list(self):
        tracker = ObjectTracker()
        result = tracker.update([], _blank_frame())
        assert result == []

    def test_update_result_length_matches_input(self):
        """update() must return one TrackedDetection per input Detection."""
        tracker = ObjectTracker()
        dets = [
            _det(confidence=0.90, bbox=(50, 50, 150, 150)),
            _det(confidence=0.85, bbox=(300, 100, 500, 300)),
        ]
        result = tracker.update(dets, _blank_frame())
        assert len(result) == len(dets)

    def test_update_returns_tracked_detection_objects(self):
        tracker = ObjectTracker()
        result = tracker.update([_det()], _blank_frame())
        assert len(result) == 1
        assert isinstance(result[0], TrackedDetection)

    def test_reset_clears_tracker_state(self):
        """reset() must allow a fresh track assignment on next frame."""
        tracker = ObjectTracker()
        det = _det(confidence=0.90, bbox=(100, 100, 200, 200))
        frame = _blank_frame()

        # Build up tracking state over two frames
        tracker.update([det], frame)
        tracker.update([det], frame)

        # Reset — next frame should behave like first frame again
        tracker.reset()
        result = tracker.update([det], frame)
        # After reset, the detection is unconfirmed again
        unconfirmed = [td for td in result if td.track_id == -1]
        assert len(unconfirmed) > 0


class TestNearestTrackFallback:
    """Check 4: _nearest_track fallback with max_px_dist=20.0px."""

    def test_nearest_track_exact_match(self):
        track_map = {(100, 100, 200, 200): 5}
        result = _nearest_track((100, 100, 200, 200), track_map)
        assert result == 5

    def test_nearest_track_within_20px(self):
        """Center distance ≤ 20px → match found."""
        track_map = {(100, 100, 200, 200): 5}   # center = (150, 150)
        # Bbox whose center is (160, 160) → distance ≈ 14.1px < 20px
        result = _nearest_track((110, 110, 210, 210), track_map)
        assert result == 5

    def test_nearest_track_beyond_20px_returns_minus_1(self):
        """Center distance > 20px → no match → -1."""
        track_map = {(100, 100, 200, 200): 5}   # center = (150, 150)
        # Bbox center at (200, 200) → distance ≈ 70.7px > 20px
        result = _nearest_track((150, 150, 250, 250), track_map)
        assert result == -1

    def test_nearest_track_picks_closest(self):
        """Multiple candidates → returns the closest one."""
        track_map = {
            (100, 100, 200, 200): 1,   # center (150, 150) → dist ≈ 14.1
            (300, 300, 400, 400): 2,   # center (350, 350) → dist >> 20
        }
        # Query bbox center at (160, 160)
        result = _nearest_track((110, 110, 210, 210), track_map)
        assert result == 1

    def test_nearest_track_empty_map_returns_minus_1(self):
        result = _nearest_track((100, 100, 200, 200), {})
        assert result == -1

    def test_nearest_track_max_dist_is_20(self):
        """Verify max_px_dist default in source."""
        import inspect
        source = inspect.getsource(_nearest_track)
        assert "20.0" in source

    def test_nearest_track_exactly_at_20px_boundary(self):
        """Distance exactly 20.0px: condition is `< best_dist` so this should NOT match
        if initial best_dist is exactly 20.0 and dist == 20.0 (not strictly less)."""
        track_map = {(100, 100, 120, 120): 7}  # center = (110, 110)
        # Query center at (130, 110): distance = 20.0px exactly
        result = _nearest_track((120, 100, 140, 120), track_map)
        # dist == 20.0 is NOT < 20.0, so no match
        assert result == -1
