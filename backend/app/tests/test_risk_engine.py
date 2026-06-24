"""
MemoryNav — Risk Engine Unit Tests
backend/tests/test_risk_engine.py

Verifies the core Risk Engine formula and threshold classification
described in the architecture doc (Module 2):

    Risk Score = (1 / distance_metres) x motion_factor x user_context_weight
    HIGH   if score > 0.55  (RISK_HIGH_THRESHOLD in config.py)
    MEDIUM if 0.35 < score <= 0.55  (RISK_MEDIUM_THRESHOLD in config.py)
    LOW    if score <= 0.35

Run:
    pytest backend/tests/test_risk_engine.py -v
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.risk.engine import (
    MotionState,
    assess_all,
    assess_risk,
    compute_risk_score,
    motion_factor_for,
)
from app.risk.models import Detection, RiskLevel


def _detection_at(distance_metres: float, class_name: str = "obstacle") -> Detection:
    """Minimal helper — bbox/confidence don't affect risk math, only distance does."""
    return Detection(
        class_name=class_name,
        confidence=0.9,
        bbox=(0, 0, 100, 100),
        distance_metres=distance_metres,
    )


# --------------------------------------------------------------------------- #
# Core ask: score thresholds at neutral motion/context (motion_factor and
# user_context_weight default to 1.0 until Short-Term/User Preference
# Memory exist — see engine.py docstring) — i.e. the formula's pure
# distance-only behavior.
# --------------------------------------------------------------------------- #
def test_close_obstacle_score_above_high_threshold():
    """0.5m obstacle: 1/0.5 = 2.0, comfortably above the 0.55 HIGH threshold."""
    score = compute_risk_score(distance_metres=0.5)
    assert score > 0.55  # RISK_HIGH_THRESHOLD


def test_far_obstacle_score_below_medium_threshold():
    """3m obstacle: 1/3 = 0.33, below the 0.35 MEDIUM/LOW threshold."""
    score = compute_risk_score(distance_metres=3.0)
    assert score < 0.35  # RISK_MEDIUM_THRESHOLD


def test_close_obstacle_classified_high():
    assessment = assess_risk(_detection_at(0.5))
    assert assessment.level == RiskLevel.HIGH
    assert assessment.action == "interrupt_immediately"


def test_far_obstacle_classified_low():
    assessment = assess_risk(_detection_at(3.0))
    assert assessment.level == RiskLevel.LOW
    assert assessment.action == "log_only"


# --------------------------------------------------------------------------- #
# Boundary / monotonicity sanity checks
# --------------------------------------------------------------------------- #
def test_score_increases_as_distance_decreases():
    far = compute_risk_score(distance_metres=3.0)
    near = compute_risk_score(distance_metres=0.5)
    assert near > far


def test_medium_band_between_thresholds():
    """Distance chosen so 1/d ~= 0.45, strictly between the 0.35 and 0.55
    thresholds -> should land in MEDIUM, not HIGH or LOW."""
    assessment = assess_risk(_detection_at(2.22))
    assert assessment.level == RiskLevel.MEDIUM
    assert assessment.action == "queue"


# --------------------------------------------------------------------------- #
# Edge cases the engine explicitly guards against
# --------------------------------------------------------------------------- #
def test_zero_distance_does_not_raise_and_is_high_risk():
    """Should floor at _MIN_DISTANCE_METRES, not divide by zero."""
    score = compute_risk_score(distance_metres=0.0)
    assert math.isfinite(score)
    assert score > 0.55  # RISK_HIGH_THRESHOLD


def test_nan_distance_is_treated_as_maximally_risky():
    """An unknown distance (e.g. DepthEstimator.depth_at_bbox on an
    invalid bbox) must never look like LOW risk."""
    score = compute_risk_score(distance_metres=float("nan"))
    assert score == float("inf")

    assessment = assess_risk(_detection_at(float("nan")))
    assert assessment.level == RiskLevel.HIGH


# --------------------------------------------------------------------------- #
# Doc-fidelity regression check: the two worked examples from the
# architecture doc, reproduced with their stated motion/context inputs.
# --------------------------------------------------------------------------- #
def test_doc_example_chair_is_high():
    """Chair at 0.5m, stationary, user has a bad knee -> doc says score 0.9."""
    chair = _detection_at(0.5, class_name="chair")
    result = assess_risk(
        chair,
        motion_factor=motion_factor_for(MotionState.STATIONARY),
        user_context_weight=0.45,
    )
    assert result.score == pytest.approx(0.9, abs=0.01)
    assert result.level == RiskLevel.HIGH


def test_doc_example_table_is_low():
    """Table at 3m, stationary, no context -> doc says score 0.1."""
    table = _detection_at(3.0, class_name="table")
    result = assess_risk(
        table,
        motion_factor=motion_factor_for(MotionState.STATIONARY),
        user_context_weight=0.30,
    )
    assert result.score == pytest.approx(0.1, abs=0.01)
    assert result.level == RiskLevel.LOW


# --------------------------------------------------------------------------- #
# Batch scoring
# --------------------------------------------------------------------------- #
def test_assess_all_sorts_by_descending_score():
    detections = [_detection_at(3.0), _detection_at(0.5), _detection_at(1.5)]
    results = assess_all(detections)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].detection.distance_metres == 0.5