"""
MemoryNav — TemporalAlertManager & TTSEngine Unit Tests
backend/tests/test_tts_and_alert_manager.py

Covers the three required tests from Step 7 plus full structural
validation of both modules.

Part A — alerts/temporal_manager.py
  Required: test_medium_risk_window_elapsed_still_suppressed
  Required: test_suppression_rate_ten_identical
  Check 1:  all three speak rules (a/b/c) verified
  Check 2:  bucket boundaries (0.70 / 2.00)
  Check 3:  de-escalation always suppressed
  Check 4:  exact speech phrases for all level/bucket combos
  Check 5:  ablation counters + suppression_rate formula
  Check:    reset() vs reset_counters() — separate concerns

Part B — voice/tts.py
  Required: test_high_risk_interrupts_tts_queue
  Check 1:  worker thread is daemon=True
  Check 2:  interrupt clears queue; non-interrupt appends
  Check 3:  dispatch latency warning logged above 200ms
  Check 4:  shutdown sentinel drains cleanly

Note: suppression_rate returns a fraction in [0.0, 1.0], NOT a
percentage — the step brief example (== 90.0) reflects a different
convention; tests here match the actual implementation (== 0.9).

Run:
    pytest backend/tests/test_tts_and_alert_manager.py -v
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.alerts.temporal_manager import (
    DistanceBucket,
    SuppressReason,
    TemporalAlertManager,
    _build_speech,
    _is_escalation,
    bucket_for,
)
from app.risk.engine import RiskAssessment, compute_risk_score
from app.risk.models import Detection, RiskLevel


# ── Assessment helpers ────────────────────────────────────────────────────────

DIST_HIGH_CLOSE = 0.50   # score=2.0  → HIGH, CLOSE bucket
DIST_MED_NEAR   = 1.90   # score≈0.53 → MEDIUM, NEAR bucket  (< 2.0m)
DIST_LOW_FAR    = 3.00   # score≈0.33 → LOW, FAR bucket


def _assessment(class_name: str, distance: float, conf: float = 0.90) -> RiskAssessment:
    det   = Detection(class_name=class_name, confidence=conf,
                      bbox=(100, 100, 300, 300), distance_metres=distance)
    score = compute_risk_score(distance)
    level = RiskLevel.from_score(score)
    return RiskAssessment(detection=det, score=score, level=level)


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


@pytest.fixture
def manager():
    return TemporalAlertManager(suppression_window_s=4.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Part A — TemporalAlertManager
# ═══════════════════════════════════════════════════════════════════════════════

class TestThreeSpeakRules:
    """
    Check 1: all three speak rules present and distinct.
    (a) First evaluation always speaks regardless of risk level.
    (b) Bucket escalation speaks within window.
    (c) Window elapsed + HIGH speaks; MEDIUM does not.
    """

    def test_rule_a_first_evaluation_always_speaks(self, manager, clock):
        d = manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        assert d.should_speak
        assert d.reason == SuppressReason.FIRST_EVALUATION

    def test_rule_a_first_evaluation_low_risk_also_speaks(self, manager, clock):
        """Even LOW risk speaks on first evaluation — safety over silence."""
        d = manager.evaluate(_assessment("table", DIST_LOW_FAR))
        assert d.should_speak

    def test_rule_b_bucket_escalation_within_window(self, manager, clock):
        manager.evaluate(_assessment("chair", DIST_LOW_FAR))    # FAR
        clock.advance(0.5)
        d = manager.evaluate(_assessment("chair", DIST_MED_NEAR))  # NEAR → escalation
        assert d.should_speak
        assert d.reason == SuppressReason.BUCKET_ESCALATED

    def test_rule_c_window_elapsed_high_speaks(self, manager, clock):
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        clock.advance(4.1)
        d = manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        assert d.should_speak
        assert d.reason == SuppressReason.WINDOW_ELAPSED_HIGH

    def test_rule_c_does_not_fire_for_medium(self, manager, clock):
        """
        Required test: MEDIUM risk + window elapsed → still suppressed.
        Rule (c) explicitly requires HIGH — this is the critical safety
        property: medium alerts don't wake up after the timeout.
        """
        manager.evaluate(_assessment("chair", DIST_MED_NEAR))
        clock.advance(5.0)   # well past 4.0s window
        d = manager.evaluate(_assessment("chair", DIST_MED_NEAR))
        assert d.suppressed
        assert d.reason == SuppressReason.LOW_RISK_NO_OVERRIDE

    def test_rule_c_does_not_fire_for_low(self, manager, clock):
        manager.evaluate(_assessment("table", DIST_LOW_FAR))
        clock.advance(10.0)
        d = manager.evaluate(_assessment("table", DIST_LOW_FAR))
        assert d.suppressed
        assert d.reason == SuppressReason.LOW_RISK_NO_OVERRIDE

    def test_suppress_same_object_same_bucket_within_window(self, manager, clock):
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        clock.advance(1.0)
        d = manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        assert d.suppressed
        assert d.reason == SuppressReason.SAME_OBJECT_SAME_BUCKET_IN_WINDOW

    def test_medium_risk_window_elapsed_still_suppressed(self, manager, clock):
        """
        Required test from step brief — explicit alias for readability.
        Identical to test_rule_c_does_not_fire_for_medium.
        """
        manager.evaluate(_assessment("chair", DIST_MED_NEAR))
        clock.advance(5.0)
        d = manager.evaluate(_assessment("chair", DIST_MED_NEAR))
        assert d.suppressed is True


class TestBucketBoundaries:
    """Check 2: exact 0.70 / 2.00 boundary values."""

    @pytest.mark.parametrize("distance, expected", [
        (0.00, DistanceBucket.CLOSE),
        (0.69, DistanceBucket.CLOSE),
        (0.70, DistanceBucket.NEAR),    # exact boundary — NEAR not CLOSE
        (1.99, DistanceBucket.NEAR),
        (2.00, DistanceBucket.FAR),     # exact boundary — FAR not NEAR
        (9.99, DistanceBucket.FAR),
    ])
    def test_bucket_boundary(self, distance, expected):
        assert bucket_for(distance) == expected

    def test_close_threshold_is_0_70(self):
        from app.alerts.temporal_manager import _CLOSE_THRESHOLD
        assert _CLOSE_THRESHOLD == pytest.approx(0.70)

    def test_near_threshold_is_2_00(self):
        from app.alerts.temporal_manager import _NEAR_THRESHOLD
        assert _NEAR_THRESHOLD == pytest.approx(2.00)


class TestDeEscalation:
    """Check 3: de-escalation (moving away) is always suppressed."""

    @pytest.mark.parametrize("from_b, to_b", [
        (DistanceBucket.CLOSE, DistanceBucket.NEAR),   # moving away
        (DistanceBucket.NEAR,  DistanceBucket.FAR),
        (DistanceBucket.CLOSE, DistanceBucket.FAR),    # double jump away
        (DistanceBucket.CLOSE, DistanceBucket.CLOSE),  # same — not escalation
        (DistanceBucket.NEAR,  DistanceBucket.NEAR),
        (DistanceBucket.FAR,   DistanceBucket.FAR),
    ])
    def test_is_not_escalation(self, from_b, to_b):
        assert not _is_escalation(from_b, to_b)

    @pytest.mark.parametrize("from_b, to_b", [
        (DistanceBucket.FAR,  DistanceBucket.NEAR),
        (DistanceBucket.FAR,  DistanceBucket.CLOSE),
        (DistanceBucket.NEAR, DistanceBucket.CLOSE),
    ])
    def test_is_escalation(self, from_b, to_b):
        assert _is_escalation(from_b, to_b)

    def test_deescalation_suppressed_in_pipeline(self, manager, clock):
        """End-to-end: close → near (object moving away) must suppress."""
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))   # CLOSE
        clock.advance(0.1)
        d = manager.evaluate(_assessment("chair", DIST_MED_NEAR))  # NEAR — de-escalation
        assert d.suppressed

    def test_only_escalation_triggers_bucket_escalated_reason(self, manager, clock):
        manager.evaluate(_assessment("chair", DIST_MED_NEAR))     # NEAR
        clock.advance(0.1)
        d = manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))  # CLOSE — escalation
        assert d.should_speak
        assert d.reason == SuppressReason.BUCKET_ESCALATED

        clock.advance(0.1)
        d2 = manager.evaluate(_assessment("chair", DIST_MED_NEAR))  # back to NEAR — suppress
        assert d2.suppressed


class TestSpeechPhrases:
    """Check 4: exact speech strings for every level+bucket combination."""

    def _assess_at(self, cls: str, distance: float) -> RiskAssessment:
        det = Detection(class_name=cls, confidence=0.90,
                        bbox=(100, 100, 300, 300), distance_metres=distance)
        score = compute_risk_score(distance)
        level = RiskLevel.from_score(score)
        return RiskAssessment(detection=det, score=score, level=level)

    def test_high_close_phrase(self):
        a = self._assess_at("chair", 0.50)   # HIGH, CLOSE
        assert _build_speech(a) == "warning — chair very close ahead"

    def test_high_near_phrase(self):
        # Need HIGH risk but NEAR bucket: score > 0.55 AND 0.70 <= d < 2.00
        # At d=0.85: score = 1/0.85 ≈ 1.18 → HIGH; bucket = NEAR
        a = self._assess_at("chair", 0.85)
        assert a.level == RiskLevel.HIGH
        assert bucket_for(0.85) == DistanceBucket.NEAR
        assert _build_speech(a) == "warning — chair getting close"

    def test_medium_near_phrase(self):
        a = self._assess_at("table", DIST_MED_NEAR)   # MEDIUM, NEAR
        assert a.level == RiskLevel.MEDIUM
        assert bucket_for(DIST_MED_NEAR) == DistanceBucket.NEAR
        assert _build_speech(a) == "table ahead, getting closer"

    def test_medium_far_phrase(self):
        # MEDIUM + FAR: 0.35 < score <= 0.55, d >= 2.0
        # At d=2.5: score = 0.4 → MEDIUM; bucket = FAR
        a = self._assess_at("table", 2.5)
        assert a.level == RiskLevel.MEDIUM
        assert bucket_for(2.5) == DistanceBucket.FAR
        assert _build_speech(a) == "table ahead"

    def test_low_far_phrase(self):
        a = self._assess_at("table", DIST_LOW_FAR)    # LOW, FAR
        assert _build_speech(a) == "table in the distance"

    def test_class_name_in_phrase(self):
        """Class name must appear verbatim in all spoken phrases."""
        for cls in ("chair", "sofa", "person", "bicycle"):
            a = self._assess_at(cls, 0.5)
            assert cls in _build_speech(a)

    def test_speak_decision_contains_speech_text(self, manager, clock):
        d = manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        assert d.should_speak
        assert d.speech_text == "warning — chair very close ahead"

    def test_suppressed_decision_has_no_speech_text(self, manager, clock):
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        clock.advance(0.5)
        d = manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        assert d.suppressed
        assert d.speech_text is None


class TestAblationCounters:
    """Check 5: suppression_rate and counter semantics."""

    def test_suppression_rate_ten_identical(self, manager, clock):
        """
        Required test from step brief.
        10 identical HIGH-risk chair detections → exactly 1 spoken, 9 suppressed.
        suppression_rate = 9/10 = 0.9 (fraction, not percentage).
        """
        for _ in range(10):
            manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
            clock.advance(0.3)   # 0.3s each, total 3.0s < 4.0s window

        assert manager.counts["spoken"]     == 1
        assert manager.counts["suppressed"] == 9
        assert manager.counts["evaluated"]  == 10
        assert manager.suppression_rate     == pytest.approx(0.9)

    def test_suppression_rate_zero_when_no_evaluations(self, manager):
        assert manager.suppression_rate == 0.0

    def test_suppression_rate_formula_is_fraction(self, manager, clock):
        """suppression_rate is suppressed/evaluated in [0.0, 1.0]."""
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))  # speak
        clock.advance(0.5)
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))  # suppress
        clock.advance(0.5)
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))  # suppress
        assert manager.suppression_rate == pytest.approx(2 / 3)
        assert 0.0 <= manager.suppression_rate <= 1.0

    def test_reset_counters_zeroes_metrics_keeps_state(self, manager, clock):
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        manager.reset_counters()
        assert manager.counts == {"evaluated": 0, "spoken": 0, "suppressed": 0}
        # Tracking state preserved — chair still in window
        clock.advance(0.5)
        d = manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        assert d.suppressed   # state survived reset_counters()

    def test_reset_clears_state_and_allows_first_evaluation(self, manager, clock):
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        manager.reset()
        d = manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        assert d.should_speak
        assert d.reason == SuppressReason.FIRST_EVALUATION

    def test_reset_does_not_clear_counters(self, manager, clock):
        """reset() clears STATE; counters survive until reset_counters() is called."""
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        manager.reset()
        # counters still show 1 evaluated, 1 spoken
        assert manager.counts["evaluated"] == 1
        assert manager.counts["spoken"] == 1

    def test_class_warn_counts_tracks_per_class(self, manager, clock):
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        clock.advance(4.1)
        manager.evaluate(_assessment("chair", DIST_HIGH_CLOSE))
        manager.evaluate(_assessment("sofa",  DIST_HIGH_CLOSE))
        wc = manager.class_warn_counts()
        assert wc["chair"] == 2
        assert wc["sofa"]  == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Part B — TTSEngine
# ═══════════════════════════════════════════════════════════════════════════════

# pyttsx3.init() may fail in headless CI or on systems without a native
# speech driver.  All TTS tests patch the engine so they run anywhere.

@pytest.fixture
def tts():
    """TTSEngine with pyttsx3 replaced by a no-op mock."""
    mock_engine = MagicMock()
    mock_engine.say.return_value = None
    mock_engine.runAndWait.return_value = None
    mock_engine.stop.return_value = None

    with patch("pyttsx3.init", return_value=mock_engine):
        from app.voice.tts import TTSEngine
        engine = TTSEngine(rate_wpm=175)
        yield engine
        engine.shutdown()


class TestWorkerThreadArchitecture:
    """Check 1: dedicated daemon worker thread + queue."""

    def test_worker_thread_is_daemon(self, tts):
        assert tts._worker.daemon is True

    def test_worker_thread_is_alive_after_init(self, tts):
        assert tts._worker.is_alive()

    def test_queue_is_queue_instance(self, tts):
        assert isinstance(tts._queue, queue.Queue)

    def test_worker_thread_name(self, tts):
        assert tts._worker.name == "tts-worker"

    def test_speak_returns_immediately(self, tts):
        """speak() must be non-blocking — returns before utterance finishes."""
        start = time.perf_counter()
        tts.speak("this is a long sentence that would take time to say aloud")
        elapsed_ms = (time.perf_counter() - start) * 1000
        # speak() just enqueues — must complete in well under 200ms
        assert elapsed_ms < 200, f"speak() blocked for {elapsed_ms:.1f}ms"

    def test_shutdown_stops_worker(self, tts):
        tts.shutdown()
        tts._worker.join(timeout=1.0)
        assert not tts._worker.is_alive()


class TestInterruptBehavior:
    """Check 2: interrupt=True clears queue; interrupt=False appends."""

    def test_high_risk_interrupts_tts_queue(self):
        """
        Required test from step brief.
        speak(interrupt=False) then speak(interrupt=True):
        queue must contain only the HIGH-alert text.

        Uses a permanently-blocking runAndWait so the worker never
        drains the queue during the assertion window.
        """
        import importlib
        tts_mod = importlib.import_module("app.voice.tts")

        mock_engine = MagicMock()
        worker_started = threading.Event()
        hold_worker    = threading.Event()

        def blocking_run():
            worker_started.set()
            hold_worker.wait(timeout=5.0)  # block until we release

        mock_engine.runAndWait.side_effect = blocking_run

        with patch("pyttsx3.init", return_value=mock_engine):
            engine = tts_mod.TTSEngine(rate_wpm=175)
            try:
                # First speak — worker picks it up and blocks in runAndWait
                engine.speak("low priority first item", interrupt=False)
                worker_started.wait(timeout=1.0)  # worker is now blocked

                # Queue a second low-priority item while worker is busy
                engine.speak("medium priority item", interrupt=False)
                # Queue now has "medium priority item" pending

                # HIGH-risk interrupt: clears "medium priority item", enqueues HIGH
                engine.speak("warning — chair very close ahead", interrupt=True)

                # Queue must have exactly the HIGH alert (medium was cleared)
                assert engine._queue.qsize() == 1
                item = engine._queue.get_nowait()
                assert item.text == "warning — chair very close ahead"
            finally:
                hold_worker.set()
                engine.shutdown()

    def test_non_interrupt_appends_to_queue(self, tts):
        """interrupt=False must not clear existing queue items."""
        # Block the worker
        block = threading.Event()

        import importlib
        tts_mod = importlib.import_module("app.voice.tts")
        mock_engine = MagicMock()
        mock_engine.runAndWait.side_effect = lambda: block.wait(timeout=3.0)

        with patch("pyttsx3.init", return_value=mock_engine):
            engine = tts_mod.TTSEngine(rate_wpm=175)
            try:
                engine.speak("first item",  interrupt=False)
                # Wait for worker to pick up first item
                time.sleep(0.05)
                engine.speak("second item", interrupt=False)
                engine.speak("third item",  interrupt=False)

                # Queue should have second and third items (first is being processed)
                assert engine._queue.qsize() == 2
            finally:
                block.set()
                engine.shutdown()

    def test_empty_text_not_queued(self, tts):
        tts.speak("")
        tts.speak("   ")
        assert tts._queue.qsize() == 0

    def test_interrupt_clears_queue(self, tts):
        """_clear_queue must drain all pending items."""
        # Add items directly to queue, bypass worker
        from app.voice.tts import SpeechRequest
        tts._queue.put(SpeechRequest(text="item1", enqueued_at=time.perf_counter()))
        tts._queue.put(SpeechRequest(text="item2", enqueued_at=time.perf_counter()))
        assert tts._queue.qsize() == 2
        tts._clear_queue()
        assert tts._queue.qsize() == 0

    def test_stop_clears_queue(self, tts):
        """stop() must drain queue without speaking anything new."""
        from app.voice.tts import SpeechRequest
        tts._queue.put(SpeechRequest(text="pending item", enqueued_at=time.perf_counter()))
        tts.stop()
        assert tts._queue.qsize() == 0


class TestDispatchLatencyWarning:
    """Check 3: WARNING logged when dispatch exceeds 200ms target."""

    def test_dispatch_latency_constant_is_200(self):
        from app.voice.tts import _DISPATCH_LATENCY_TARGET_MS
        assert _DISPATCH_LATENCY_TARGET_MS == 200

    def test_high_latency_logs_warning(self, caplog):
        """
        Simulate a request that sat in queue > 200ms before being processed.
        The worker must emit a WARNING-level log with 'dispatch latency'.
        """
        import logging
        import importlib
        tts_mod = importlib.import_module("app.voice.tts")
        mock_engine = MagicMock()
        hold = threading.Event()

        def slow_run():
            hold.wait(timeout=3.0)

        mock_engine.runAndWait.side_effect = slow_run

        with patch("pyttsx3.init", return_value=mock_engine):
            engine = tts_mod.TTSEngine(rate_wpm=175)
            try:
                # Inject a request that was "enqueued" 300ms ago
                old_request = tts_mod.SpeechRequest(
                    text="delayed utterance",
                    enqueued_at=time.perf_counter() - 0.300,
                )
                with caplog.at_level(logging.WARNING, logger="app.voice.tts"):
                    engine._queue.put(old_request)
                    time.sleep(0.05)  # give worker time to pick it up and log
            finally:
                hold.set()
                engine.shutdown()

        latency_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "dispatch latency" in r.message.lower()
        ]
        assert latency_warnings, (
            "Expected a dispatch latency WARNING but none was logged. "
            f"All records: {[(r.levelname, r.message) for r in caplog.records]}"
        )

    def test_normal_latency_no_warning(self, tts, caplog):
        """speak() with fresh request must not log a latency warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger="app.voice.tts"):
            tts.speak("quick test")
            time.sleep(0.05)  # give worker time to process

        latency_warnings = [r for r in caplog.records
                            if "dispatch latency" in r.message.lower()]
        # A fresh request should not trigger the latency warning
        assert not latency_warnings


class TestTTSAvailability:
    """Graceful degradation when pyttsx3 has no native driver."""

    def test_unavailable_tts_does_not_raise_on_speak(self):
        """speak() must not raise even when the engine failed to init."""
        with patch("pyttsx3.init", side_effect=Exception("no driver")):
            from app.voice.tts import TTSEngine
            engine = TTSEngine(rate_wpm=175)
            assert engine.available is False
            try:
                engine.speak("test text")   # must not raise
            except Exception as exc:
                pytest.fail(f"speak() raised on unavailable engine: {exc}")
            finally:
                engine.shutdown()

    def test_available_flag_true_when_init_succeeds(self, tts):
        assert tts.available is True
