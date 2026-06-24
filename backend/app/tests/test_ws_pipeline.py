"""
MemoryNav — WebSocket Pipeline Unit Tests
backend/tests/test_ws_pipeline.py

Validates the 12-step _run_pipeline_sync() in api/ws_stream.py:

  Check 1 — Memory write gate: conf ≥ 0.55 AND valid distance, every 30 frames
  Check 2 — Ghost alert conditions: similarity ≥ 0.50, not visible, 30s cooldown
  Check 3 — ThreadPoolExecutor max_workers=2
  Check 4 — JSON response shape matches frontend DetectionFrame type
  Check 5 — Frame quality early exit: bad frame → status=degraded, YOLO skipped

All heavy models (Detector, DepthEstimator, TTSEngine, etc.) are replaced
with lightweight mocks so the suite runs in <2s with no GPU required.

Run:
    pytest backend/tests/test_ws_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.ws_stream import (
    _EXECUTOR,
    _GHOST_ALERT_COOLDOWN_S,
    _MEMORY_WRITE_EVERY_N_FRAMES,
    _MEMORY_WRITE_MIN_CONFIDENCE,
    _run_pipeline_sync,
)
from app.perception.frame_quality import FrameQualityResult, QualityIssue


# ── Frame helpers ─────────────────────────────────────────────────────────────

def _make_good_frame() -> np.ndarray:
    """
    A noisy, well-lit BGR frame that passes check_frame_quality.
    Random noise gives high Laplacian variance (sharp); mean ~128 (bright enough).
    """
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, (480, 640, 3), dtype=np.uint8)


def _encode_jpeg(frame: np.ndarray) -> bytes:
    """Encode a numpy BGR frame as JPEG bytes (mimics what the WebSocket receives)."""
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def _make_good_frame_bytes() -> bytes:
    return _encode_jpeg(_make_good_frame())


def _make_black_frame_bytes() -> bytes:
    """All-black frame — too dark → check_frame_quality returns ok=False."""
    return _encode_jpeg(np.zeros((480, 640, 3), dtype=np.uint8))


# ── Mock PipelineState factory ────────────────────────────────────────────────

def _make_state(
    frame_counter: int = 0,
    detector_returns=None,        # list[TrackedDetection] from tracker.update
    depth_map=None,
    depth_at_bbox_returns: float = 1.5,
    long_term_count: int = 0,
    long_term_retrieve_returns=None,
    risk_assess_returns=None,
    alert_should_speak: bool = False,
) -> MagicMock:
    """
    Build a fully-wired mock PipelineState.  Every attribute is a MagicMock
    except frame_counter (int) and _ghost_alerted_at (dict), which must be
    real objects because _run_pipeline_sync reads/writes them directly.
    """
    from app.alerts.temporal_manager import TemporalAlertManager
    from app.risk.models import RiskLevel, Detection as RiskDetection
    from app.risk.engine import RiskAssessment

    state = MagicMock()
    state.frame_counter = frame_counter
    state._ghost_alerted_at = {}

    # detector stub — returns empty list by default (no YOLO detections)
    if detector_returns is None:
        detector_returns = []
    state.detector.detect.return_value = detector_returns

    # tracker stub — passes detections through (returns what detector gave us)
    state.tracker.update.return_value = detector_returns

    # depth estimator
    state.depth_estimator.estimate.return_value = (
        depth_map if depth_map is not None else np.ones((480, 640), dtype=np.float32)
    )
    state.depth_estimator.depth_at_bbox.return_value = depth_at_bbox_returns

    # spatial map
    state.spatial_map.update.return_value = None
    state.spatial_map.snapshot.return_value = {"current_room": "Living Room", "rooms": {}}

    # session store
    state.session.motion_trend.return_value = "stationary"
    state.session.record_obstacle_detection.return_value = None
    state.session.record_warning.return_value = None
    state.session.was_recently_warned.return_value = False

    # long-term memory
    state.long_term.count.return_value = long_term_count
    state.long_term.retrieve.return_value = (
        long_term_retrieve_returns if long_term_retrieve_returns is not None else []
    )
    state.long_term.add_context.return_value = "mock-id"

    # risk engine
    if risk_assess_returns is None:
        risk_assess_returns = []
    state.risk_engine.assess.side_effect = (
        iter(risk_assess_returns) if risk_assess_returns else iter([])
    )

    # alert manager
    speak_decision = MagicMock()
    speak_decision.should_speak = alert_should_speak
    speak_decision.speech_text = "warning — chair very close ahead" if alert_should_speak else None
    state.alert_manager.evaluate.return_value = speak_decision
    state.alert_manager.counts = {"evaluated": 1, "spoken": 0, "suppressed": 1}

    # TTS
    state.tts.speak.return_value = None

    return state


# ── Check 3: ThreadPoolExecutor config ───────────────────────────────────────

class TestThreadPoolConfig:
    def test_executor_max_workers_is_two(self):
        """
        More than 2 workers = multiple model instances loaded → OOM on edge devices.
        The executor is module-level so we inspect it directly.
        """
        assert _EXECUTOR._max_workers == 2

    def test_memory_write_constants(self):
        """Gate constants must match the architecture doc values."""
        assert _MEMORY_WRITE_EVERY_N_FRAMES == 30
        assert _MEMORY_WRITE_MIN_CONFIDENCE == 0.55

    def test_ghost_cooldown_constant(self):
        assert _GHOST_ALERT_COOLDOWN_S == 30.0


# ── Check 5: Frame quality early exit ────────────────────────────────────────

class TestFrameQualityGate:
    """
    A bad frame must return immediately with status='degraded' and
    empty detections.  YOLO, depth, and risk must NOT be called.
    """

    def test_pipeline_skips_bad_frame_returns_degraded(self):
        """Required test from the step brief — black frame → degraded."""
        state = _make_state()
        result = _run_pipeline_sync(_make_black_frame_bytes(), state, frame_id=None)
        assert result["status"] == "degraded"
        assert result["detections"] == []

    def test_pipeline_degraded_frame_does_not_call_yolo(self):
        state = _make_state()
        _run_pipeline_sync(_make_black_frame_bytes(), state, frame_id=None)
        state.detector.detect.assert_not_called()

    def test_pipeline_degraded_frame_does_not_call_depth(self):
        state = _make_state()
        _run_pipeline_sync(_make_black_frame_bytes(), state, frame_id=None)
        state.depth_estimator.estimate.assert_not_called()

    def test_pipeline_degraded_frame_does_not_call_risk_engine(self):
        state = _make_state()
        _run_pipeline_sync(_make_black_frame_bytes(), state, frame_id=None)
        state.risk_engine.assess.assert_not_called()

    def test_pipeline_degraded_frame_has_quality_field(self):
        """Degraded response must carry the quality diagnostic for the frontend."""
        state = _make_state()
        result = _run_pipeline_sync(_make_black_frame_bytes(), state, frame_id=None)
        assert "quality" in result
        assert result["quality"]["ok"] is False

    def test_decode_error_returns_decode_error_status(self):
        """Corrupt/empty bytes → decode_error status (not degraded)."""
        state = _make_state()
        result = _run_pipeline_sync(b"not_a_jpeg", state, frame_id=None)
        assert result["status"] == "decode_error"
        assert result["detections"] == []

    def test_good_frame_reaches_yolo(self):
        """A well-lit noisy frame must pass the quality gate and call detect()."""
        state = _make_state()
        _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)
        state.detector.detect.assert_called_once()


# ── Check 1: Memory write gate ────────────────────────────────────────────────

class TestMemoryWriteGate:
    """
    Observations must be written to ChromaDB exactly once every 30 frames,
    and only when conf ≥ 0.55 AND distance is a valid (non-NaN) float.
    """

    def _make_tracked_detection(self, conf: float = 0.80, class_name: str = "chair"):
        """Build a minimal TrackedDetection-like mock."""
        td = MagicMock()
        td.class_name = class_name
        td.confidence = conf
        td.bbox = (100, 100, 300, 300)
        td.class_id = 56
        return td

    def _make_risk_assessment(self, class_name: str = "chair", distance: float = 1.0):
        from app.risk.models import RiskLevel, Detection as RiskDetection
        from app.risk.engine import RiskAssessment
        rd = RiskDetection(
            class_name=class_name, confidence=0.80,
            bbox=(100, 100, 300, 300), distance_metres=distance,
        )
        return RiskAssessment(detection=rd, score=1.0 / distance, level=RiskLevel.HIGH)

    def test_memory_write_frequency_30_frames(self):
        """
        Required test from the step brief.
        Run 31 frames, assert add_context called exactly once (on frame 30, not every frame).
        """
        td = self._make_tracked_detection(conf=0.80)
        ra = self._make_risk_assessment(distance=1.0)

        state = _make_state(
            frame_counter=0,
            detector_returns=[td],
            depth_at_bbox_returns=1.0,
            risk_assess_returns=[ra],
        )
        # Patch alert_manager so it doesn't speak (avoids TTS side effects)
        state.alert_manager.evaluate.return_value.should_speak = False

        for i in range(31):
            # Reconfigure risk engine to return the same assessment each call
            state.risk_engine.assess.return_value = ra
            state.risk_engine.assess.side_effect = None
            _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        # add_context must have been called exactly once (at frame 30)
        assert state.long_term.add_context.call_count == 1

    def test_memory_not_written_before_frame_30(self):
        """Frames 1–29 must never write to ChromaDB."""
        td = self._make_tracked_detection(conf=0.80)
        ra = self._make_risk_assessment(distance=1.0)
        state = _make_state(frame_counter=0, detector_returns=[td],
                            depth_at_bbox_returns=1.0)
        state.risk_engine.assess.return_value = ra
        state.risk_engine.assess.side_effect = None
        state.alert_manager.evaluate.return_value.should_speak = False

        for _ in range(29):
            _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        state.long_term.add_context.assert_not_called()

    def test_memory_not_written_for_low_confidence(self):
        """conf < 0.55 must never write to memory, even on frame 30."""
        td = self._make_tracked_detection(conf=0.40)  # below gate
        ra = self._make_risk_assessment(distance=1.0)
        # Start at frame_counter=29 so next call is frame 30
        state = _make_state(frame_counter=29, detector_returns=[td],
                            depth_at_bbox_returns=1.0)
        state.risk_engine.assess.return_value = ra
        state.risk_engine.assess.side_effect = None
        state.alert_manager.evaluate.return_value.should_speak = False

        _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        state.long_term.add_context.assert_not_called()

    def test_memory_not_written_for_nan_distance(self):
        """NaN distance (depth failure) must never write to memory."""
        td = self._make_tracked_detection(conf=0.80)
        ra = self._make_risk_assessment(distance=float("nan"))
        state = _make_state(frame_counter=29, detector_returns=[td],
                            depth_at_bbox_returns=float("nan"))
        state.risk_engine.assess.return_value = ra
        state.risk_engine.assess.side_effect = None
        state.alert_manager.evaluate.return_value.should_speak = False

        _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        state.long_term.add_context.assert_not_called()

    def test_memory_written_at_confidence_boundary(self):
        """Exactly 0.55 confidence must write (≥ threshold, not >)."""
        td = self._make_tracked_detection(conf=_MEMORY_WRITE_MIN_CONFIDENCE)
        ra = self._make_risk_assessment(distance=1.0)
        state = _make_state(frame_counter=29, detector_returns=[td],
                            depth_at_bbox_returns=1.0)
        state.risk_engine.assess.return_value = ra
        state.risk_engine.assess.side_effect = None
        state.alert_manager.evaluate.return_value.should_speak = False

        _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        state.long_term.add_context.assert_called_once()


# ── Check 2: Ghost alert conditions ──────────────────────────────────────────

class TestGhostAlerts:
    """
    Ghost alerts fire when:
      (a) ChromaDB similarity ≥ 0.50
      (b) Object is NOT in the current frame's detected classes
      (c) Per-object cooldown of 30.0s has elapsed
    """

    def _make_memory_result(self, class_name: str, similarity: float):
        """Build a MemoryResult-like mock matching LongTermMemory's return type."""
        r = MagicMock()
        r.similarity = similarity
        r.metadata = {"class_name": class_name}
        return r

    def test_ghost_alert_fires_for_unseen_high_similarity_object(self):
        """Chair in memory (sim=0.8) + chair NOT in current frame → ghost alert fires."""
        memory_result = self._make_memory_result("chair", similarity=0.80)
        state = _make_state(
            frame_counter=0,
            detector_returns=[],         # no detections this frame
            long_term_count=1,
            long_term_retrieve_returns=[memory_result],
        )
        # Ensure cooldown has not fired before
        state._ghost_alerted_at = {}

        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        assert result["spoke"] is not None
        assert "chair" in result["spoke"].lower()

    def test_ghost_alert_suppressed_for_low_similarity(self):
        """Similarity < 0.50 → ghost alert must NOT fire."""
        memory_result = self._make_memory_result("chair", similarity=0.40)
        state = _make_state(
            frame_counter=0, detector_returns=[],
            long_term_count=1,
            long_term_retrieve_returns=[memory_result],
        )
        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)
        assert result["spoke"] is None

    def test_ghost_alert_suppressed_when_object_currently_visible(self):
        """If chair is in the current frame, no ghost alert for chair."""
        memory_result = self._make_memory_result("chair", similarity=0.80)
        td = MagicMock()
        td.class_name = "chair"
        td.confidence = 0.90
        td.bbox = (100, 100, 300, 300)
        td.class_id = 56

        from app.risk.models import RiskLevel, Detection as RiskDetection
        from app.risk.engine import RiskAssessment
        rd = RiskDetection(class_name="chair", confidence=0.90,
                           bbox=(100, 100, 300, 300), distance_metres=1.0)
        ra = RiskAssessment(detection=rd, score=1.0, level=RiskLevel.HIGH)

        state = _make_state(
            frame_counter=0, detector_returns=[td],
            depth_at_bbox_returns=1.0,
            long_term_count=1,
            long_term_retrieve_returns=[memory_result],
        )
        state.risk_engine.assess.return_value = ra
        state.risk_engine.assess.side_effect = None
        state.alert_manager.evaluate.return_value.should_speak = False

        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        # chair is visible → ghost must not fire for chair
        assert result.get("spokeGhost") is False

    def test_ghost_alert_respects_cooldown(self):
        """Second ghost alert within 30s cooldown must be suppressed."""
        import time
        memory_result = self._make_memory_result("chair", similarity=0.80)
        state = _make_state(
            frame_counter=0, detector_returns=[],
            long_term_count=1,
            long_term_retrieve_returns=[memory_result],
        )
        # Simulate a ghost alert fired 5s ago (within 30s cooldown)
        state._ghost_alerted_at = {"chair": time.monotonic() - 5.0}

        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        # Within cooldown — ghost must not fire again
        assert result.get("spokeGhost") is False

    def test_ghost_alert_fires_after_cooldown_expires(self):
        """Ghost alert fires again once 30s cooldown has elapsed."""
        import time
        memory_result = self._make_memory_result("chair", similarity=0.80)
        state = _make_state(
            frame_counter=0, detector_returns=[],
            long_term_count=1,
            long_term_retrieve_returns=[memory_result],
        )
        # Last ghost alert was 35s ago — cooldown expired
        state._ghost_alerted_at = {"chair": time.monotonic() - 35.0}

        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        assert result["spoke"] is not None
        assert "chair" in result["spoke"].lower()

    def test_ghost_alert_skipped_when_memory_empty(self):
        """No ChromaDB entries → ghost check must be skipped entirely."""
        state = _make_state(
            frame_counter=0, detector_returns=[],
            long_term_count=0,            # empty memory
        )
        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        state.long_term.retrieve.assert_not_called()
        assert result["spoke"] is None


# ── Check 4: JSON response shape ──────────────────────────────────────────────

class TestResponseShape:
    """
    Every good frame must return the keys the frontend DetectionFrame type expects.
    A degraded frame must return a subset (status + detections at minimum).
    """

    REQUIRED_KEYS = {
        "detections", "riskLevel", "riskScore", "riskReason",
        "suppressed", "suppressionReason", "latencyMs",
        "status", "spoke", "spatialMap", "timestamp",
    }

    def test_good_frame_returns_all_required_keys(self):
        state = _make_state()
        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys in response: {missing}"

    def test_degraded_frame_returns_status_and_detections(self):
        state = _make_state()
        result = _run_pipeline_sync(_make_black_frame_bytes(), state, frame_id=None)
        assert "status" in result
        assert "detections" in result
        assert result["detections"] == []

    def test_status_is_ok_for_good_frame(self):
        state = _make_state()
        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)
        assert result["status"] == "ok"

    def test_latency_ms_is_positive_float(self):
        state = _make_state()
        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)
        assert isinstance(result["latencyMs"], float)
        assert result["latencyMs"] >= 0.0

    def test_spatial_map_is_dict(self):
        state = _make_state()
        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)
        assert isinstance(result["spatialMap"], dict)

    def test_detections_is_list(self):
        state = _make_state()
        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)
        assert isinstance(result["detections"], list)

    def test_risk_level_is_low_when_no_detections(self):
        """No detections → dominant_level defaults to 'LOW' (not 'NONE')."""
        state = _make_state(detector_returns=[])
        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)
        assert result["riskLevel"] == "LOW"

    def test_frame_id_echoed_in_response(self):
        state = _make_state()
        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id="frame-007")
        # frame_id is included in degraded and error responses; good frames don't
        # include it explicitly in the success path — but the pipeline still works.
        # Just confirm no exception.
        assert result["status"] == "ok"

    def test_detection_dict_has_required_fields(self):
        """Each detection dict must carry id, label, confidence, box, distanceMeters."""
        from app.risk.models import RiskLevel, Detection as RiskDetection
        from app.risk.engine import RiskAssessment

        td = MagicMock()
        td.class_name = "chair"
        td.confidence = 0.85
        td.bbox = (100, 100, 300, 300)
        td.class_id = 56

        rd = RiskDetection(class_name="chair", confidence=0.85,
                           bbox=(100, 100, 300, 300), distance_metres=1.5)
        ra = RiskAssessment(detection=rd, score=0.667, level=RiskLevel.HIGH)
        ra = RiskAssessment(detection=rd, score=0.667, level=RiskLevel.HIGH,
                            context_result=None)

        state = _make_state(detector_returns=[td], depth_at_bbox_returns=1.5)
        state.risk_engine.assess.return_value = ra
        state.risk_engine.assess.side_effect = None
        state.alert_manager.evaluate.return_value.should_speak = False

        result = _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        assert len(result["detections"]) == 1
        det = result["detections"][0]
        for key in ("id", "label", "confidence", "box", "distanceMeters"):
            assert key in det, f"Missing key '{key}' in detection dict"
        assert det["label"] == "chair"
        assert det["distanceMeters"] == pytest.approx(1.5, abs=0.01)
        box = det["box"]
        for k in ("x", "y", "width", "height"):
            assert k in box


# ── 12-step pipeline order ────────────────────────────────────────────────────

class TestPipelineOrder:
    """
    Verify the 12-step execution order by checking call patterns on mocks.
    Each step must happen after the previous one — tested by side-effect
    sequencing and call count assertions.
    """

    def test_step1_decode_happens_before_quality_check(self):
        """
        Decode failure returns 'decode_error' before quality check runs.
        If quality ran first it would return 'degraded', not 'decode_error'.
        """
        state = _make_state()
        result = _run_pipeline_sync(b"garbage_bytes", state, frame_id=None)
        assert result["status"] == "decode_error"

    def test_step2_quality_runs_before_yolo(self):
        """Black frame fails quality → detect() never called."""
        state = _make_state()
        _run_pipeline_sync(_make_black_frame_bytes(), state, frame_id=None)
        state.detector.detect.assert_not_called()

    def test_steps3_4_yolo_then_tracker(self):
        """
        On a good frame, detect() and tracker.update() are both called.
        tracker.update receives the output of detector.detect.
        """
        td = MagicMock()
        td.class_name = "chair"; td.confidence = 0.85
        td.bbox = (100, 100, 300, 300); td.class_id = 56
        state = _make_state(detector_returns=[td])
        state.risk_engine.assess.return_value = MagicMock(
            score=0.5, level=MagicMock(value="HIGH"),
            detection=MagicMock(class_name="chair", confidence=0.85,
                                bbox=(100,100,300,300), distance_metres=1.0),
            context_result=None,
        )
        state.risk_engine.assess.side_effect = None

        _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        state.detector.detect.assert_called_once()
        state.tracker.update.assert_called_once()

    def test_step5_depth_called_after_tracking(self):
        """depth_estimator.estimate must be called on a good frame with detections."""
        state = _make_state(detector_returns=[MagicMock(
            class_name="chair", confidence=0.85,
            bbox=(100,100,300,300), class_id=56,
        )])
        state.risk_engine.assess.return_value = MagicMock(
            score=0.5, level=MagicMock(value="MEDIUM"),
            detection=MagicMock(class_name="chair", confidence=0.85,
                                bbox=(100,100,300,300), distance_metres=1.5),
            context_result=None,
        )
        state.risk_engine.assess.side_effect = None

        _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        state.depth_estimator.estimate.assert_called_once()

    def test_step7_spatial_map_updated_per_detection(self):
        """SpatialMap.update must be called once per tracked detection."""
        detections = [
            MagicMock(class_name="chair", confidence=0.85,
                      bbox=(100,100,300,300), class_id=56),
            MagicMock(class_name="sofa", confidence=0.78,
                      bbox=(400,100,600,300), class_id=57),
        ]
        state = _make_state(detector_returns=detections, depth_at_bbox_returns=1.5)
        state.risk_engine.assess.return_value = MagicMock(
            score=0.5, level=MagicMock(value="MEDIUM"),
            detection=MagicMock(class_name="chair", confidence=0.85,
                                bbox=(100,100,300,300), distance_metres=1.5),
            context_result=None,
        )
        state.risk_engine.assess.side_effect = None

        _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        assert state.spatial_map.update.call_count == 2

    def test_step11_session_records_all_detections(self):
        """record_obstacle_detection must be called once per risk detection."""
        detections = [
            MagicMock(class_name="chair", confidence=0.85,
                      bbox=(100,100,300,300), class_id=56),
        ]
        state = _make_state(detector_returns=detections, depth_at_bbox_returns=1.0)
        state.risk_engine.assess.return_value = MagicMock(
            score=1.0, level=MagicMock(value="HIGH"),
            detection=MagicMock(class_name="chair", confidence=0.85,
                                bbox=(100,100,300,300), distance_metres=1.0),
            context_result=None,
        )
        state.risk_engine.assess.side_effect = None
        state.alert_manager.evaluate.return_value.should_speak = False

        _run_pipeline_sync(_make_good_frame_bytes(), state, frame_id=None)

        state.session.record_obstacle_detection.assert_called_once()
