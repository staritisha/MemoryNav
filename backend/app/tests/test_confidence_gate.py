"""
MemoryNav — Confidence Gate Unit Tests
backend/tests/test_confidence_gate.py

Verifies the three properties the architecture doc mandates for
the confidence gate (Module 5 — Voice Interface):

  Check 1 — Direction band:
      LEFT   → bbox centre_x < 0.425
      CENTRE → 0.425 ≤ centre_x ≤ 0.575  (no direction phrase spoken)
      RIGHT  → centre_x > 0.575

  Check 2 — Uncertain speech must NEVER contain the class name:
      "something may be ahead on your left, I am not certain"  ✅
      "chair may be ahead on your left, I am not certain"      ❌

  Check 3 — Ablation counters increment correctly for every outcome:
      pass_count      ← confidence ≥ 0.60
      uncertain_count ← 0.25 ≤ confidence < 0.60
      discard_count   ← confidence < 0.25

Run:
    pytest backend/tests/test_confidence_gate.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.alerts.confidence_gate import (
    CENTRE_BAND,
    CONFIDENCE_THRESHOLD,
    DISCARD_THRESHOLD,
    ConfidenceGate,
    Direction,
    GateOutcome,
    direction_from_bbox,
)
from app.risk.models import Detection

# ── Helpers ───────────────────────────────────────────────────────────────────

FRAME_W = 640  # pixels — standard MemoryNav capture width


def _det(
    conf: float,
    class_name: str = "chair",
    bbox_centre_x_frac: float = 0.50,  # fraction of FRAME_W
) -> Detection:
    """
    Build a minimal Detection with a bbox whose horizontal centre lands
    at exactly `bbox_centre_x_frac * FRAME_W`.  The vertical extent is
    fixed; it doesn't affect any gate logic.
    """
    cx = int(bbox_centre_x_frac * FRAME_W)
    half_w = 50
    return Detection(
        class_name=class_name,
        confidence=conf,
        bbox=(cx - half_w, 100, cx + half_w, 400),
        distance_metres=1.0,
    )


@pytest.fixture
def gate() -> ConfidenceGate:
    """Fresh gate for each test — counters start at zero."""
    return ConfidenceGate()


# ── Check 1: direction band (0.425 / 0.575 boundary) ────────────────────────

class TestDirectionBand:
    """Verify the exact band values and that CENTRE never produces a phrase."""

    def test_band_edges_match_doc(self):
        """CENTRE_BAND=0.15 → left_edge=0.425, right_edge=0.575."""
        left_edge  = 0.5 - CENTRE_BAND / 2
        right_edge = 0.5 + CENTRE_BAND / 2
        assert left_edge  == pytest.approx(0.425)
        assert right_edge == pytest.approx(0.575)

    def test_centre_x_just_left_of_band_is_left(self):
        """0.424 < 0.425 → LEFT."""
        det = _det(conf=0.45, bbox_centre_x_frac=0.424)
        result = gate_result(det)
        assert result.direction == Direction.LEFT

    def test_centre_x_at_left_band_edge_is_centre(self):
        """0.425 is the inclusive left boundary → CENTRE."""
        det = _det(conf=0.45, bbox_centre_x_frac=0.425)
        result = gate_result(det)
        assert result.direction == Direction.CENTRE

    def test_centre_x_at_midpoint_is_centre(self):
        """0.50 is squarely inside the band."""
        det = _det(conf=0.45, bbox_centre_x_frac=0.50)
        result = gate_result(det)
        assert result.direction == Direction.CENTRE

    def test_centre_x_at_right_band_edge_is_centre(self):
        """0.575 is the inclusive right boundary → CENTRE."""
        det = _det(conf=0.45, bbox_centre_x_frac=0.575)
        result = gate_result(det)
        assert result.direction == Direction.CENTRE

    def test_centre_x_just_right_of_band_is_right(self):
        """centre_x meaningfully past 0.575 → RIGHT.
        Use 0.60 to survive integer pixel round-trip (0.576×640=368.64 → 368 → 0.575 ≈ boundary)."""
        det = _det(conf=0.45, bbox_centre_x_frac=0.60)
        result = gate_result(det)
        assert result.direction == Direction.RIGHT

    def test_far_left_is_left(self):
        det = _det(conf=0.45, bbox_centre_x_frac=0.10)
        result = gate_result(det)
        assert result.direction == Direction.LEFT

    def test_far_right_is_right(self):
        det = _det(conf=0.45, bbox_centre_x_frac=0.90)
        result = gate_result(det)
        assert result.direction == Direction.RIGHT

    # direction_from_bbox standalone coverage
    def test_direction_from_bbox_left(self):
        bbox = (0, 100, 200, 400)          # centre_x = 100 → 100/640 = 0.156
        assert direction_from_bbox(bbox, FRAME_W) == Direction.LEFT

    def test_direction_from_bbox_right(self):
        bbox = (440, 100, 640, 400)        # centre_x = 540 → 540/640 = 0.844
        assert direction_from_bbox(bbox, FRAME_W) == Direction.RIGHT

    def test_direction_from_bbox_centre(self):
        bbox = (270, 100, 370, 400)        # centre_x = 320 → 320/640 = 0.500
        assert direction_from_bbox(bbox, FRAME_W) == Direction.CENTRE


def gate_result(det: Detection):
    """Helper: evaluate a single detection with FRAME_W."""
    return ConfidenceGate().evaluate(det, frame_width=FRAME_W)


# ── Check 2: class name NEVER spoken for UNCERTAIN detections ────────────────

class TestUncertainNeverSpeaksClassName:
    """
    Step 3 — required test:
        assert "chair" not in result.uncertain[0].speech_text
    """

    def test_uncertain_never_speaks_class_name(self, gate: ConfidenceGate):
        """Primary required test from the step brief."""
        passed, uncertain, discarded = gate.filter_frame(
            [_det(conf=0.45, class_name="chair")],
            frame_width=FRAME_W,
        )
        assert len(uncertain) == 1
        assert "chair" not in uncertain[0].speech_text

    def test_uncertain_speech_matches_exact_doc_phrasing_no_direction(self, gate: ConfidenceGate):
        """Centre bbox → no direction appended."""
        result = gate.evaluate(_det(conf=0.45, bbox_centre_x_frac=0.50), frame_width=FRAME_W)
        assert result.speech_text == "something may be ahead, I am not certain"

    def test_uncertain_speech_matches_exact_doc_phrasing_left(self, gate: ConfidenceGate):
        result = gate.evaluate(_det(conf=0.45, bbox_centre_x_frac=0.20), frame_width=FRAME_W)
        assert result.speech_text == "something may be ahead on your left, I am not certain"

    def test_uncertain_speech_matches_exact_doc_phrasing_right(self, gate: ConfidenceGate):
        result = gate.evaluate(_det(conf=0.45, bbox_centre_x_frac=0.80), frame_width=FRAME_W)
        assert result.speech_text == "something may be ahead on your right, I am not certain"

    @pytest.mark.parametrize("class_name", [
        "chair", "table", "sofa", "person", "dog", "bottle", "laptop",
    ])
    def test_no_class_name_in_uncertain_speech_for_any_class(
        self, gate: ConfidenceGate, class_name: str
    ):
        """Paranoid sweep — no YOLO class name should ever leak through."""
        result = gate.evaluate(
            _det(conf=0.50, class_name=class_name, bbox_centre_x_frac=0.20),
            frame_width=FRAME_W,
        )
        assert result.outcome == GateOutcome.UNCERTAIN
        assert class_name not in result.speech_text

    def test_pass_produces_no_speech_text(self, gate: ConfidenceGate):
        """PASS must return speech_text=None — caller owns the full alert phrasing."""
        result = gate.evaluate(_det(conf=0.90, class_name="chair"), frame_width=FRAME_W)
        assert result.outcome == GateOutcome.PASS
        assert result.speech_text is None

    def test_discard_produces_no_speech_text(self, gate: ConfidenceGate):
        result = gate.evaluate(_det(conf=0.10, class_name="chair"), frame_width=FRAME_W)
        assert result.outcome == GateOutcome.DISCARD
        assert result.speech_text is None


# ── Check 2 (supplementary): centre band produces no direction in speech ──────

class TestCentreBandNoDirection:
    """
    Step 3 — required test:
        assert "left" not in result.uncertain[0].speech_text
        assert "right" not in result.uncertain[0].speech_text
    """

    def test_centre_band_no_direction(self, gate: ConfidenceGate):
        """Primary required test from the step brief."""
        passed, uncertain, discarded = gate.filter_frame(
            [_det(conf=0.45, bbox_centre_x_frac=0.50)],
            frame_width=FRAME_W,
        )
        assert len(uncertain) == 1
        assert "left"  not in uncertain[0].speech_text
        assert "right" not in uncertain[0].speech_text

    def test_centre_band_left_edge_no_direction(self, gate: ConfidenceGate):
        """Exactly 0.425 — still centre, still no direction phrase."""
        result = gate.evaluate(_det(conf=0.45, bbox_centre_x_frac=0.425), frame_width=FRAME_W)
        assert result.direction == Direction.CENTRE
        assert "left"  not in result.speech_text
        assert "right" not in result.speech_text

    def test_centre_band_right_edge_no_direction(self, gate: ConfidenceGate):
        """Exactly 0.575 — still centre, still no direction phrase."""
        result = gate.evaluate(_det(conf=0.45, bbox_centre_x_frac=0.575), frame_width=FRAME_W)
        assert result.direction == Direction.CENTRE
        assert "left"  not in result.speech_text
        assert "right" not in result.speech_text

    def test_no_frame_width_no_direction(self, gate: ConfidenceGate):
        """When frame_width is None direction is skipped entirely."""
        result = gate.evaluate(_det(conf=0.45), frame_width=None)
        assert result.direction is None
        assert "left"  not in result.speech_text
        assert "right" not in result.speech_text


# ── Check 3: ablation counters ────────────────────────────────────────────────

class TestAblationCounters:
    """
    Verify pass / uncertain / discard counters increment correctly and
    that the suppression formula holds:
        suppression_rate = (uncertain + discard) / total × 100
    """

    def test_pass_count_increments(self, gate: ConfidenceGate):
        gate.evaluate(_det(conf=0.90))
        gate.evaluate(_det(conf=0.60))   # boundary — exactly at threshold = PASS
        assert gate.counts["pass"] == 2
        assert gate.counts["uncertain"] == 0
        assert gate.counts["discard"] == 0

    def test_uncertain_count_increments(self, gate: ConfidenceGate):
        gate.evaluate(_det(conf=0.59))   # just under threshold
        gate.evaluate(_det(conf=0.25))   # boundary — exactly at discard = UNCERTAIN
        assert gate.counts["pass"] == 0
        assert gate.counts["uncertain"] == 2
        assert gate.counts["discard"] == 0

    def test_discard_count_increments(self, gate: ConfidenceGate):
        gate.evaluate(_det(conf=0.24))   # just under discard threshold
        gate.evaluate(_det(conf=0.10))
        assert gate.counts["pass"] == 0
        assert gate.counts["uncertain"] == 0
        assert gate.counts["discard"] == 2

    def test_filter_frame_counters_accumulate_across_calls(self, gate: ConfidenceGate):
        """filter_frame delegates to evaluate() — counters must accumulate."""
        detections = [
            _det(conf=0.85),  # PASS
            _det(conf=0.48),  # UNCERTAIN
            _det(conf=0.20),  # DISCARD
            _det(conf=0.72),  # PASS
            _det(conf=0.30),  # UNCERTAIN
        ]
        gate.filter_frame(detections, frame_width=FRAME_W)
        assert gate.counts["pass"]      == 2
        assert gate.counts["uncertain"] == 2
        assert gate.counts["discard"]   == 1

    def test_suppression_rate_formula(self, gate: ConfidenceGate):
        """
        Total = pass + uncertain + discard
        Suppression rate = (uncertain + discard) / total
        """
        detections = [
            _det(conf=0.85),  # PASS
            _det(conf=0.48),  # UNCERTAIN
            _det(conf=0.20),  # DISCARD
        ]
        gate.filter_frame(detections, frame_width=FRAME_W)

        p = gate.counts["pass"]
        u = gate.counts["uncertain"]
        d = gate.counts["discard"]
        total = p + u + d

        assert total == 3
        suppression_rate = (u + d) / total * 100
        assert suppression_rate == pytest.approx(200 / 3)  # ≈ 66.7 %

    def test_reset_counts_zeroes_all_counters(self, gate: ConfidenceGate):
        gate.evaluate(_det(conf=0.90))
        gate.evaluate(_det(conf=0.45))
        gate.evaluate(_det(conf=0.10))
        gate.reset_counts()
        assert gate.counts == {"pass": 0, "uncertain": 0, "discard": 0}

    def test_total_from_stats_summary(self, gate: ConfidenceGate):
        gate.evaluate(_det(conf=0.90))
        gate.evaluate(_det(conf=0.45))
        gate.evaluate(_det(conf=0.10))
        summary = gate.stats_summary()
        assert "total=3" in summary
        assert "pass=1" in summary
        assert "uncertain=1" in summary
        assert "discard=1" in summary

    def test_stats_summary_before_any_evaluations(self, gate: ConfidenceGate):
        assert "no detections" in gate.stats_summary()


# ── Threshold boundary values ─────────────────────────────────────────────────

class TestThresholdBoundaries:
    """Exact threshold boundaries must be respected — off-by-one here
    means real detections get silenced or hallucinated."""

    def test_confidence_exactly_at_threshold_is_pass(self, gate: ConfidenceGate):
        """0.60 exactly → PASS (>= threshold)."""
        result = gate.evaluate(_det(conf=CONFIDENCE_THRESHOLD))
        assert result.outcome == GateOutcome.PASS

    def test_confidence_just_below_threshold_is_uncertain(self, gate: ConfidenceGate):
        result = gate.evaluate(_det(conf=CONFIDENCE_THRESHOLD - 0.001))
        assert result.outcome == GateOutcome.UNCERTAIN

    def test_confidence_exactly_at_discard_threshold_is_uncertain(self, gate: ConfidenceGate):
        """0.25 exactly → UNCERTAIN (>= discard threshold but < confidence threshold)."""
        result = gate.evaluate(_det(conf=DISCARD_THRESHOLD))
        assert result.outcome == GateOutcome.UNCERTAIN

    def test_confidence_just_below_discard_threshold_is_discard(self, gate: ConfidenceGate):
        result = gate.evaluate(_det(conf=DISCARD_THRESHOLD - 0.001))
        assert result.outcome == GateOutcome.DISCARD

    def test_zero_confidence_is_discard(self, gate: ConfidenceGate):
        result = gate.evaluate(_det(conf=0.0))
        assert result.outcome == GateOutcome.DISCARD
        assert result.speech_text is None

    def test_full_confidence_is_pass(self, gate: ConfidenceGate):
        result = gate.evaluate(_det(conf=1.0))
        assert result.outcome == GateOutcome.PASS


# ── Constructor guard ─────────────────────────────────────────────────────────

def test_invalid_threshold_order_raises():
    """discard_threshold >= confidence_threshold must raise ValueError."""
    with pytest.raises(ValueError, match="discard_threshold"):
        ConfidenceGate(confidence_threshold=0.50, discard_threshold=0.50)

    with pytest.raises(ValueError, match="discard_threshold"):
        ConfidenceGate(confidence_threshold=0.50, discard_threshold=0.60)
