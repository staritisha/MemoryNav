"""
backend/app/alerts/confidence_gate.py
--------------------------------------
Confidence Gate — sits between the Perception Layer and the Risk Engine.

Architecture doc spec (Module 5 — Voice Interface):
    "If YOLO confidence < 60%: say 'something may be ahead, I am not certain.'
     Never hallucinate."

Where this fits in the pipeline
--------------------------------
    YOLO detection
        ↓
    [ confidence_gate.py ]   ← YOU ARE HERE
        ↓ PASS              ↓ UNCERTAIN
    Risk Engine         vague speech only
    Alert Manager       class name NEVER spoken
    Voice Output        path guidance still given (left/right from bbox)

Decisions encoded here
-----------------------
1.  PASS  (confidence >= 0.60)
    The detection is trustworthy enough to name the object in speech.
    Downstream pipeline runs normally — Risk Engine scores it,
    Alert Manager suppresses duplicates, Voice speaks the class name.

2.  UNCERTAIN  (confidence < 0.60)
    The system cannot confidently name what it sees.
    → Speak a vague warning: "something may be ahead, I am not certain."
    → Optionally append a direction hint from the bbox position:
        left half of frame  → "on your left"
        right half of frame → "on your right"
        centre              → (no direction added — directly ahead is implied)
    → NEVER speak the class name — that is hallucination.
    → NEVER pass this detection to the Risk Engine with a named class.
      The detection is returned as UNCERTAIN and the caller decides
      whether to voice it (Alert Manager handles suppression).

3.  DISCARD  (confidence < discard_threshold, default 0.25)
    So noisy it is not worth voicing at all.  Dropped silently.
    Logged at DEBUG level for the ablation study's false-alert metrics.

Usage
------
    from app.alerts.confidence_gate import ConfidenceGate, GateOutcome
    from app.risk.models import Detection

    gate = ConfidenceGate()

    for detection in yolo_detections:
        result = gate.evaluate(detection, frame_width=640)

        if result.outcome == GateOutcome.PASS:
            # full pipeline: Risk Engine → Alert Manager → speak class name
            ...
        elif result.outcome == GateOutcome.UNCERTAIN:
            # speak result.speech_text only — do NOT name the object
            voice.speak(result.speech_text)
        else:  # DISCARD
            pass  # silent drop

    # Or filter a whole frame at once:
    passed, uncertain, discarded = gate.filter_frame(detections, frame_width=640)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.risk.models import Detection

logger = logging.getLogger(__name__)


# ── Thresholds ────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD: float = 0.60   # doc spec: < 0.60 → vague speech
DISCARD_THRESHOLD:    float = 0.25   # below this: not worth voicing at all

# Fraction of frame width that counts as "centre" — neither clearly left nor right.
# A bbox whose centre falls within (0.5 - CENTRE_BAND/2, 0.5 + CENTRE_BAND/2)
# of the frame gets no direction appended.
CENTRE_BAND: float = 0.15


# ── Enums / dataclasses ───────────────────────────────────────────────────────

class GateOutcome(str, Enum):
    PASS      = "pass"       # confidence >= threshold → name the object
    UNCERTAIN = "uncertain"  # low conf, not zero → vague speech only
    DISCARD   = "discard"    # too noisy → silent drop


class Direction(str, Enum):
    LEFT   = "left"
    RIGHT  = "right"
    CENTRE = "centre"   # directly ahead — no directional phrase appended


@dataclass(frozen=True)
class GateResult:
    """
    Output of ConfidenceGate.evaluate() for a single Detection.

    Attributes
    ----------
    detection   : The original Detection object (unchanged).
    outcome     : PASS / UNCERTAIN / DISCARD.
    speech_text : Ready-to-speak string.
                  PASS      → None  (caller builds the full alert from
                                     the class name + risk level).
                  UNCERTAIN → vague warning with optional direction.
                  DISCARD   → None  (speak nothing).
    direction   : LEFT / RIGHT / CENTRE derived from bbox position,
                  or None when frame_width was not provided.
    reason      : Short human-readable explanation for logs / ablation study.
    """
    detection:   Detection
    outcome:     GateOutcome
    speech_text: Optional[str]
    direction:   Optional[Direction]
    reason:      str

    @property
    def should_speak(self) -> bool:
        """True when there is something to say."""
        return self.speech_text is not None

    @property
    def passed(self) -> bool:
        return self.outcome == GateOutcome.PASS

    @property
    def uncertain(self) -> bool:
        return self.outcome == GateOutcome.UNCERTAIN

    @property
    def discarded(self) -> bool:
        return self.outcome == GateOutcome.DISCARD


# ── Speech text builders ──────────────────────────────────────────────────────

def _direction_phrase(direction: Direction) -> str:
    """
    Maps a Direction enum to the spoken phrase fragment.
    CENTRE returns empty string — "directly ahead" is implied and
    adding it makes uncertain alerts more verbose, not more useful.
    """
    return {
        Direction.LEFT:   " on your left",
        Direction.RIGHT:  " on your right",
        Direction.CENTRE: "",
    }[direction]


def _build_uncertain_speech(direction: Optional[Direction]) -> str:
    """
    Constructs the vague warning mandated by the architecture doc.
    Class name is intentionally absent — this is the anti-hallucination rule.

    Examples:
        "something may be ahead, I am not certain"
        "something may be ahead on your left, I am not certain"
        "something may be ahead on your right, I am not certain"
    """
    direction_phrase = (
        _direction_phrase(direction)
        if direction is not None
        else ""
    )
    return f"something may be ahead{direction_phrase}, I am not certain"


# ── Direction from bounding box ───────────────────────────────────────────────

def direction_from_bbox(
    bbox: tuple[int, int, int, int],
    frame_width: int,
) -> Direction:
    """
    Derive LEFT / RIGHT / CENTRE from the horizontal centre of a bbox.

    Args:
        bbox        : (x1, y1, x2, y2) in pixels — YOLO output format.
        frame_width : Width of the camera frame in pixels (e.g. 640).

    Returns:
        Direction enum value.

    Architecture doc spec (Module 5):
        "left half of frame = obstacle on left,
         right half of frame = obstacle on right"
    A centre band prevents jitter for objects sitting right on the
    midline from alternating left/right between frames.
    """
    x1, _, x2, _ = bbox
    bbox_centre_x = (x1 + x2) / 2.0
    relative_x    = bbox_centre_x / frame_width   # 0.0 (left) → 1.0 (right)

    left_edge   = 0.5 - CENTRE_BAND / 2   # 0.425
    right_edge  = 0.5 + CENTRE_BAND / 2   # 0.575

    if relative_x < left_edge:
        return Direction.LEFT
    if relative_x > right_edge:
        return Direction.RIGHT
    return Direction.CENTRE


# ── Main gate class ───────────────────────────────────────────────────────────

class ConfidenceGate:
    """
    Evaluates YOLO confidence and decides what the system is allowed to say.

    Parameters
    ----------
    confidence_threshold : float
        Detections at or above this value are PASS.  Default 0.60 (doc spec).
    discard_threshold    : float
        Detections below this value are DISCARD.  Default 0.25.

    Both thresholds are configurable so the ablation study can sweep them
    without touching call sites.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        discard_threshold:    float = DISCARD_THRESHOLD,
    ) -> None:
        if discard_threshold >= confidence_threshold:
            raise ValueError(
                f"discard_threshold ({discard_threshold}) must be "
                f"less than confidence_threshold ({confidence_threshold})"
            )
        self.confidence_threshold = confidence_threshold
        self.discard_threshold    = discard_threshold

        # Counters for ablation study metrics
        self._counts: dict[GateOutcome, int] = {
            GateOutcome.PASS:      0,
            GateOutcome.UNCERTAIN: 0,
            GateOutcome.DISCARD:   0,
        }

    # ── Primary API ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        detection: Detection,
        frame_width: Optional[int] = None,
    ) -> GateResult:
        """
        Evaluate one Detection and return a GateResult.

        Args:
            detection   : Detection from the Perception Layer.
            frame_width : Camera frame width in pixels.
                          Required for direction hints in UNCERTAIN speech.
                          Pass None to skip direction (speech will still work).

        Returns:
            GateResult — check .outcome / .speech_text / .should_speak.
        """
        conf = detection.confidence

        # ── DISCARD — too noisy to be useful ─────────────────────────────
        if conf < self.discard_threshold:
            result = GateResult(
                detection   = detection,
                outcome     = GateOutcome.DISCARD,
                speech_text = None,
                direction   = None,
                reason      = (
                    f"conf={conf:.2f} below discard threshold "
                    f"({self.discard_threshold:.2f}) — silent drop"
                ),
            )
            self._counts[GateOutcome.DISCARD] += 1
            logger.debug(
                "[ConfidenceGate] DISCARD  %s  conf=%.2f",
                detection.class_name, conf,
            )
            return result

        # ── UNCERTAIN — detectable but not nameable ───────────────────────
        if conf < self.confidence_threshold:
            direction = (
                direction_from_bbox(detection.bbox, frame_width)
                if frame_width is not None
                else None
            )
            speech = _build_uncertain_speech(direction)
            result = GateResult(
                detection   = detection,
                outcome     = GateOutcome.UNCERTAIN,
                speech_text = speech,
                direction   = direction,
                reason      = (
                    f"conf={conf:.2f} below threshold "
                    f"({self.confidence_threshold:.2f}) — vague speech only"
                ),
            )
            self._counts[GateOutcome.UNCERTAIN] += 1
            logger.info(
                "[ConfidenceGate] UNCERTAIN  %s  conf=%.2f  → \"%s\"",
                detection.class_name, conf, speech,
            )
            return result

        # ── PASS — confident enough to name the object ────────────────────
        direction = (
            direction_from_bbox(detection.bbox, frame_width)
            if frame_width is not None
            else None
        )
        result = GateResult(
            detection   = detection,
            outcome     = GateOutcome.PASS,
            speech_text = None,    # caller builds the full alert
            direction   = direction,
            reason      = f"conf={conf:.2f} >= threshold ({self.confidence_threshold:.2f})",
        )
        self._counts[GateOutcome.PASS] += 1
        logger.debug(
            "[ConfidenceGate] PASS  %s  conf=%.2f  direction=%s",
            detection.class_name, conf,
            direction.value if direction else "unknown",
        )
        return result

    # ── Batch API ─────────────────────────────────────────────────────────────

    def filter_frame(
        self,
        detections: list[Detection],
        frame_width: Optional[int] = None,
    ) -> tuple[list[GateResult], list[GateResult], list[GateResult]]:
        """
        Evaluate every Detection in a frame.

        Returns:
            (passed, uncertain, discarded)  — three separate lists.

            passed    → send to Risk Engine → Alert Manager → full named alert
            uncertain → speak .speech_text immediately (direction-aware)
            discarded → drop silently

        Example:
            passed, uncertain, discarded = gate.filter_frame(detections, 640)
            for r in uncertain:
                voice.speak(r.speech_text)
            for r in passed:
                risk_engine.assess(r.detection)
        """
        results = [self.evaluate(d, frame_width=frame_width) for d in detections]

        passed    = [r for r in results if r.outcome == GateOutcome.PASS]
        uncertain = [r for r in results if r.outcome == GateOutcome.UNCERTAIN]
        discarded = [r for r in results if r.outcome == GateOutcome.DISCARD]

        return passed, uncertain, discarded

    # ── Ablation study metrics ────────────────────────────────────────────────

    @property
    def counts(self) -> dict[str, int]:
        """
        Running totals since instantiation.
        Used by the ablation study to measure false-alert rate.

        false_alert_rate = uncertain / (pass + uncertain)
        discard_rate     = discarded / total
        """
        return {k.value: v for k, v in self._counts.items()}

    def reset_counts(self) -> None:
        """Reset counters between evaluation runs."""
        for k in self._counts:
            self._counts[k] = 0

    def stats_summary(self) -> str:
        """One-line summary for the ablation study log."""
        total = sum(self._counts.values())
        if total == 0:
            return "ConfidenceGate: no detections evaluated yet"
        p = self._counts[GateOutcome.PASS]
        u = self._counts[GateOutcome.UNCERTAIN]
        d = self._counts[GateOutcome.DISCARD]
        vague_rate = u / (p + u) if (p + u) > 0 else 0.0
        return (
            f"ConfidenceGate stats — total={total}  "
            f"pass={p}  uncertain={u}  discard={d}  "
            f"vague_rate={vague_rate:.1%}"
        )


# ── Convenience module-level function ─────────────────────────────────────────

_default_gate = ConfidenceGate()


def evaluate(
    detection: Detection,
    frame_width: Optional[int] = None,
) -> GateResult:
    """
    Module-level shortcut using the default gate (threshold=0.60).
    Import and call directly for simple single-detection use cases.

        from app.alerts.confidence_gate import evaluate
        result = evaluate(detection, frame_width=640)
    """
    return _default_gate.evaluate(detection, frame_width=frame_width)


# ── Smoke test:  python -m app.alerts.confidence_gate ────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    print("\n── MemoryNav Confidence Gate — smoke test ──\n")

    gate = ConfidenceGate()

    # Helper: fake Detection with bbox placed at a known horizontal position
    def make_det(conf: float, x1: int, x2: int, cls: str = "chair") -> Detection:
        return Detection(
            class_name=cls,
            confidence=conf,
            bbox=(x1, 100, x2, 400),
            distance_metres=1.0,
        )

    FRAME_W = 640

    # ── Test 1: PASS — high confidence, object on right ──────────────────
    print("TEST 1 — conf=0.91, bbox on right half → PASS")
    det = make_det(conf=0.91, x1=400, x2=580)
    r   = gate.evaluate(det, frame_width=FRAME_W)
    assert r.outcome      == GateOutcome.PASS,      f"Got {r.outcome}"
    assert r.speech_text  is None,                  "PASS must not produce speech_text"
    assert r.direction    == Direction.RIGHT,        f"Got {r.direction}"
    print(f"  outcome={r.outcome}  direction={r.direction}  reason={r.reason}")
    print("  ✅ PASS\n")

    # ── Test 2: UNCERTAIN — low confidence, object on left ───────────────
    print("TEST 2 — conf=0.45, bbox on left half → UNCERTAIN, vague speech")
    det = make_det(conf=0.45, x1=50, x2=200)
    r   = gate.evaluate(det, frame_width=FRAME_W)
    assert r.outcome     == GateOutcome.UNCERTAIN
    assert r.speech_text == "something may be ahead on your left, I am not certain"
    assert r.direction   == Direction.LEFT
    assert "chair" not in r.speech_text,  "Class name must NOT appear in uncertain speech"
    print(f"  outcome={r.outcome}  speech='{r.speech_text}'")
    print("  ✅ PASS\n")

    # ── Test 3: UNCERTAIN — centre bbox, no direction appended ───────────
    print("TEST 3 — conf=0.55, bbox at centre → UNCERTAIN, no direction")
    det = make_det(conf=0.55, x1=290, x2=350)
    r   = gate.evaluate(det, frame_width=FRAME_W)
    assert r.outcome    == GateOutcome.UNCERTAIN
    assert r.direction  == Direction.CENTRE
    assert r.speech_text == "something may be ahead, I am not certain"
    print(f"  outcome={r.outcome}  direction={r.direction}  speech='{r.speech_text}'")
    print("  ✅ PASS\n")

    # ── Test 4: UNCERTAIN — no frame_width (direction omitted) ───────────
    print("TEST 4 — conf=0.50, no frame_width → UNCERTAIN, no direction phrase")
    det = make_det(conf=0.50, x1=100, x2=200)
    r   = gate.evaluate(det, frame_width=None)
    assert r.outcome    == GateOutcome.UNCERTAIN
    assert r.direction  is None
    assert r.speech_text == "something may be ahead, I am not certain"
    print(f"  speech='{r.speech_text}'  direction={r.direction}")
    print("  ✅ PASS\n")

    # ── Test 5: DISCARD — very low confidence ────────────────────────────
    print("TEST 5 — conf=0.18 → DISCARD, no speech")
    det = make_det(conf=0.18, x1=100, x2=300)
    r   = gate.evaluate(det, frame_width=FRAME_W)
    assert r.outcome     == GateOutcome.DISCARD
    assert r.speech_text is None
    assert not r.should_speak
    print(f"  outcome={r.outcome}  speech={r.speech_text}")
    print("  ✅ PASS\n")

    # ── Test 6: filter_frame — mixed batch ───────────────────────────────
    print("TEST 6 — filter_frame with mixed confidence batch")
    detections = [
        make_det(conf=0.85, x1=400, x2=580, cls="sofa"),    # PASS
        make_det(conf=0.48, x1=50,  x2=180, cls="table"),   # UNCERTAIN
        make_det(conf=0.20, x1=100, x2=300, cls="person"),  # DISCARD
        make_det(conf=0.72, x1=270, x2=370, cls="chair"),   # PASS
    ]
    passed, uncertain, discarded = gate.filter_frame(detections, frame_width=FRAME_W)
    assert len(passed)    == 2
    assert len(uncertain) == 1
    assert len(discarded) == 1
    assert uncertain[0].speech_text == "something may be ahead on your left, I am not certain"
    print(f"  passed={len(passed)}  uncertain={len(uncertain)}  discarded={len(discarded)}")
    for u in uncertain:
        print(f"  → vague speech: '{u.speech_text}'")
    print("  ✅ PASS\n")

    # ── Test 7: stats / ablation counters ────────────────────────────────
    print("TEST 7 — ablation study counters")
    print(f"  {gate.stats_summary()}")
    assert gate.counts["pass"]      == 3   # tests 1 + 6×2
    assert gate.counts["uncertain"] == 4   # tests 2,3,4 + 6×1
    assert gate.counts["discard"]   == 2   # tests 5 + 6×1
    print("  ✅ PASS\n")

    # ── Test 8: exact architecture doc speech text ────────────────────────
    print("TEST 8 — exact speech from architecture doc")
    det = make_det(conf=0.59, x1=320, x2=380)   # just under threshold, centre
    r   = gate.evaluate(det, frame_width=FRAME_W)
    EXPECTED = "something may be ahead, I am not certain"
    assert r.speech_text == EXPECTED, f"Got: {r.speech_text!r}"
    print(f"  speech='{r.speech_text}'")
    print("  ✅ PASS — matches architecture doc exactly\n")

    print("── All 8 tests passed ──\n")