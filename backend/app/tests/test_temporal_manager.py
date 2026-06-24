"""
backend/tests/test_temporal_manager.py
----------------------------------------
Phase 4 / Roadmap item 19 — Alert Manager test suite

Gate criterion:
    "Assert second warning suppressed within window; fires after window"
    (same chair detected 10x in 4s -> voice fires exactly once)

These tests exercise app.alerts.temporal_manager.TemporalAlertManager
against the suppression rules from the architecture doc, reproduced in
temporal_manager.py's own docstring:

    Speak only if ANY of:
        (a) object class is different from the last warned object
        (b) distance bucket changed (escalation: far -> near -> close)
        (c) suppression_window has elapsed AND risk is HIGH

Time is mocked via monkeypatch on time.monotonic so the suite runs in
milliseconds instead of sleeping for real seconds (the smoke test in
temporal_manager.py's __main__ block uses real time.sleep(); this file
swaps that for a deterministic fake clock, which is the only change
needed to make the same assertions reliable and fast under pytest).

Run with:
    pytest backend/tests/test_temporal_manager.py -v
"""

from __future__ import annotations

import time

import pytest

from app.alerts.temporal_manager import (
    DistanceBucket,
    SuppressReason,
    TemporalAlertManager,
    bucket_for,
)
from app.risk.engine import compute_risk_score, RiskAssessment
from app.risk.models import Detection, RiskLevel


# ──────────────────────────────────────────────────────────────────────────
# Test data helpers
# ──────────────────────────────────────────────────────────────────────────

# Representative distances for each risk/bucket tier (mirrors the
# thresholds in temporal_manager.py: close < 0.7m, near < 2.0m, far >= 2.0m)
# Risk thresholds (from config.py): HIGH > 0.55, MEDIUM 0.35-0.55, LOW <= 0.35
DIST_CLOSE_HIGH = 0.5   # HIGH risk,   CLOSE bucket  (score=2.0)
DIST_NEAR_MED   = 1.9   # MEDIUM risk, NEAR bucket   (score≈0.53, in 0.35–0.55 range, d < 2.0m)
DIST_FAR_LOW    = 3.0   # LOW risk,    FAR bucket    (score≈0.33)


def make_assessment(
    class_name: str,
    distance_metres: float,
    confidence: float = 0.90,
) -> RiskAssessment:
    """Build a real RiskAssessment the same way the perception pipeline does."""
    detection = Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=(100, 100, 300, 300),
        distance_metres=distance_metres,
    )
    score = compute_risk_score(distance_metres)
    level = RiskLevel.from_score(score)
    return RiskAssessment(detection=detection, score=score, level=level)


class FakeClock:
    """
    Deterministic stand-in for time.monotonic().

    TemporalAlertManager calls time.monotonic() via `import time`, so
    patching the attribute on the shared time module affects it no
    matter how the manager imported time.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


@pytest.fixture
def manager() -> TemporalAlertManager:
    return TemporalAlertManager(suppression_window_s=4.0)


# ──────────────────────────────────────────────────────────────────────────
# Core gate — same object + same bucket: suppress within window, speak after
# ──────────────────────────────────────────────────────────────────────────

class TestSuppressionWindow:
    """Item 19 gate: 'same chair detected 10x in 4s -> voice fires exactly once.'"""

    def test_second_identical_warning_is_suppressed_within_window(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        first = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        assert first.should_speak
        assert first.reason == SuppressReason.FIRST_EVALUATION

        clock.advance(1.0)  # well inside the 4s window
        second = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))

        assert not second.should_speak
        assert second.reason == SuppressReason.SAME_OBJECT_SAME_BUCKET_IN_WINDOW
        assert second.elapsed_s == pytest.approx(1.0)

    def test_repeated_identical_detections_speak_exactly_once(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        """Same chair detected 10x in under 4s -> only the first one speaks."""
        decisions = []
        for _ in range(10):
            decisions.append(manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH)))
            clock.advance(0.3)  # 10 * 0.3s = 3.0s total, stays under the window

        spoken = [d for d in decisions if d.should_speak]
        assert len(spoken) == 1
        assert spoken[0].reason == SuppressReason.FIRST_EVALUATION
        assert manager.counts["spoken"] == 1
        assert manager.counts["suppressed"] == 9

    def test_warning_fires_again_after_window_elapses_with_high_risk(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))  # FIRST_EVALUATION

        clock.advance(4.1)  # past the 4s window
        decision = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))

        assert decision.should_speak
        assert decision.reason == SuppressReason.WINDOW_ELAPSED_HIGH
        assert decision.elapsed_s == pytest.approx(4.1)
        assert "chair" in decision.speech_text

    def test_window_boundary_is_inclusive(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        """elapsed >= suppression_window_s, so exactly 4.0s should already fire."""
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        clock.advance(4.0)
        decision = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))

        assert decision.should_speak
        assert decision.reason == SuppressReason.WINDOW_ELAPSED_HIGH

    def test_window_elapsed_but_medium_risk_stays_suppressed(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        """Rule (c) requires HIGH risk; MEDIUM/LOW never get a timeout override."""
        manager.evaluate(make_assessment("table", DIST_NEAR_MED))  # MEDIUM
        clock.advance(10.0)
        decision = manager.evaluate(make_assessment("table", DIST_NEAR_MED))

        assert not decision.should_speak
        assert decision.reason == SuppressReason.LOW_RISK_NO_OVERRIDE


# ──────────────────────────────────────────────────────────────────────────
# (a) New object always speaks
# ──────────────────────────────────────────────────────────────────────────

class TestNewObjectClass:
    def test_first_ever_evaluation_always_speaks(
        self, manager: TemporalAlertManager
    ) -> None:
        decision = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        assert decision.should_speak
        assert decision.reason == SuppressReason.FIRST_EVALUATION
        assert decision.elapsed_s is None

    def test_different_object_class_speaks_even_within_window(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        clock.advance(0.1)  # well inside window
        decision = manager.evaluate(make_assessment("sofa", DIST_CLOSE_HIGH))

        assert decision.should_speak
        # sofa has never been tracked before, so this is a fresh FIRST_EVALUATION
        assert decision.reason == SuppressReason.FIRST_EVALUATION
        assert "sofa" in decision.speech_text

        # the chair's own suppression window is untouched by the sofa warning
        chair_decision = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        assert not chair_decision.should_speak
        assert chair_decision.reason == SuppressReason.SAME_OBJECT_SAME_BUCKET_IN_WINDOW


# ──────────────────────────────────────────────────────────────────────────
# (b) Distance bucket escalation
# ──────────────────────────────────────────────────────────────────────────

class TestBucketEscalation:
    def test_far_to_near_escalation_speaks(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        manager.evaluate(make_assessment("chair", 3.0))   # FAR
        clock.advance(0.1)
        decision = manager.evaluate(make_assessment("chair", 1.0))  # NEAR

        assert decision.should_speak
        assert decision.reason == SuppressReason.BUCKET_ESCALATED

    def test_near_to_close_escalation_speaks(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        manager.evaluate(make_assessment("chair", 1.0))   # NEAR
        clock.advance(0.1)
        decision = manager.evaluate(make_assessment("chair", 0.4))  # CLOSE

        assert decision.should_speak
        assert decision.reason == SuppressReason.BUCKET_ESCALATED

    def test_far_to_close_double_jump_escalation_speaks(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        manager.evaluate(make_assessment("chair", 3.0))   # FAR
        clock.advance(0.1)
        decision = manager.evaluate(make_assessment("chair", 0.4))  # CLOSE (skips NEAR)

        assert decision.should_speak
        assert decision.reason == SuppressReason.BUCKET_ESCALATED

    @pytest.mark.parametrize(
        "from_dist, to_dist",
        [
            pytest.param(0.4, 1.0, id="close_to_near"),
            pytest.param(1.0, 3.0, id="near_to_far"),
            pytest.param(0.4, 3.0, id="close_to_far"),
        ],
    )
    def test_deescalation_is_suppressed(
        self,
        manager: TemporalAlertManager,
        clock: FakeClock,
        from_dist: float,
        to_dist: float,
    ) -> None:
        """Moving further away must never trigger speech, even on bucket change."""
        manager.evaluate(make_assessment("chair", from_dist))
        clock.advance(0.1)  # inside window, so escalation is the only possible trigger
        decision = manager.evaluate(make_assessment("chair", to_dist))

        assert not decision.should_speak


# ──────────────────────────────────────────────────────────────────────────
# bucket_for() boundary values
# ──────────────────────────────────────────────────────────────────────────

class TestBucketForBoundaries:
    @pytest.mark.parametrize(
        "distance_metres, expected_bucket",
        [
            (0.0, DistanceBucket.CLOSE),
            (0.69, DistanceBucket.CLOSE),
            (0.70, DistanceBucket.NEAR),
            (1.99, DistanceBucket.NEAR),
            (2.00, DistanceBucket.FAR),
            (9.99, DistanceBucket.FAR),
        ],
    )
    def test_bucket_boundaries(
        self, distance_metres: float, expected_bucket: DistanceBucket
    ) -> None:
        assert bucket_for(distance_metres) == expected_bucket


# ──────────────────────────────────────────────────────────────────────────
# Ablation study metrics (doc Table 6.3 — Alert System metrics)
# ──────────────────────────────────────────────────────────────────────────

class TestAblationMetrics:
    def test_suppression_rate_and_counts(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))   # speak
        clock.advance(0.5)
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))   # suppress
        clock.advance(0.5)
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))   # suppress

        assert manager.counts == {"evaluated": 3, "spoken": 1, "suppressed": 2}
        assert manager.suppression_rate == pytest.approx(2 / 3)

    def test_suppression_rate_is_zero_with_no_evaluations(
        self, manager: TemporalAlertManager
    ) -> None:
        assert manager.suppression_rate == 0.0

    def test_reset_counters_does_not_clear_tracking_state(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        manager.reset_counters()

        assert manager.counts == {"evaluated": 0, "spoken": 0, "suppressed": 0}

        # tracking state (last_warned_at, bucket) survives reset_counters()
        clock.advance(0.5)
        decision = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        assert not decision.should_speak

    def test_class_warn_counts(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        clock.advance(4.1)
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))   # window elapsed -> speak
        manager.evaluate(make_assessment("sofa", DIST_CLOSE_HIGH))

        assert manager.class_warn_counts() == {"chair": 2, "sofa": 1}


# ──────────────────────────────────────────────────────────────────────────
# State inspection helpers
# ──────────────────────────────────────────────────────────────────────────

class TestStateInspection:
    def test_last_warned_class_tracks_most_recent_speaker(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        assert manager.last_warned_class() is None

        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        assert manager.last_warned_class() == "chair"

        clock.advance(0.1)
        manager.evaluate(make_assessment("sofa", DIST_CLOSE_HIGH))
        assert manager.last_warned_class() == "sofa"

    def test_seconds_since_last_warning(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        assert manager.seconds_since_last_warning("chair") is None

        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        clock.advance(2.5)

        assert manager.seconds_since_last_warning("chair") == pytest.approx(2.5)

    def test_reset_clears_tracking_state(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        manager.reset()

        assert manager.last_warned_class() is None

        # chair is now treated as brand-new again
        decision = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        assert decision.should_speak
        assert decision.reason == SuppressReason.FIRST_EVALUATION


# ──────────────────────────────────────────────────────────────────────────
# Speech text phrasing
# ──────────────────────────────────────────────────────────────────────────

class TestSpeechText:
    def test_high_risk_close_bucket_uses_urgent_phrasing(
        self, manager: TemporalAlertManager
    ) -> None:
        decision = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        assert decision.speech_text == "warning — chair very close ahead"

    def test_medium_risk_near_bucket_phrasing(
        self, manager: TemporalAlertManager
    ) -> None:
        decision = manager.evaluate(make_assessment("table", DIST_NEAR_MED))
        assert decision.speech_text == "table ahead, getting closer"

    def test_suppressed_decisions_have_no_speech_text(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        clock.advance(0.1)
        decision = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))

        assert decision.speech_text is None


# ──────────────────────────────────────────────────────────────────────────
# SpeakDecision.suppressed convenience property
# ──────────────────────────────────────────────────────────────────────────

class TestSpeakDecisionProperty:
    def test_suppressed_property_mirrors_should_speak(
        self, manager: TemporalAlertManager, clock: FakeClock
    ) -> None:
        spoken = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        assert spoken.suppressed is False

        clock.advance(0.1)
        suppressed = manager.evaluate(make_assessment("chair", DIST_CLOSE_HIGH))
        assert suppressed.suppressed is True


# ──────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ──────────────────────────────────────────────────────────────────────────

class TestEvaluateBatch:
    def test_evaluate_batch_preserves_order_and_applies_same_rules(
        self, manager: TemporalAlertManager
    ) -> None:
        assessments = [
            make_assessment("chair", DIST_CLOSE_HIGH),
            make_assessment("sofa", DIST_FAR_LOW),
        ]
        decisions = manager.evaluate_batch(assessments)

        assert len(decisions) == 2
        assert all(d.should_speak for d in decisions)  # both are first-time
        assert decisions[0].assessment.detection.class_name == "chair"
        assert decisions[1].assessment.detection.class_name == "sofa"