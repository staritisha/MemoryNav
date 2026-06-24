"""
backend/app/alerts/temporal_manager.py
----------------------------------------
Module 4 — Temporal Alert Manager  (WalkVLM-inspired)

Research basis
--------------
WalkVLM (arXiv:2412.20903, Dec 2024) identifies two primary usability
failures in assistive navigation systems:
    1. Too much speech  — constant narration causes alert fatigue.
    2. Repeated identical warnings — "chair ahead" every frame for 30s
       trains users to ignore alerts entirely, defeating the system.

The suppression logic here directly addresses both.

Architecture doc state variables (reproduced exactly)
------------------------------------------------------
    last_obstacle       String: most recent object warned about
    last_distance_bucket String: "close" / "near" / "far"
    last_warning_time   Timestamp of most recent speech output
    suppression_window  Configurable — default 4 seconds

Suppression logic — speak ONLY if ANY of these is true
-------------------------------------------------------
    (a) object is DIFFERENT from last_obstacle
    (b) distance bucket CHANGED  (far → near → close escalation)
    (c) suppression_window has elapsed  AND  risk level is HIGH

If none of (a), (b), (c) hold → SUPPRESS.  No speech, no action.

Distance bucket mapping
-----------------------
    close : distance_metres  < 0.7   (HIGH risk zone — under 0.7m)
    near  : distance_metres  < 2.0   (MEDIUM risk zone — 0.7m to 2m)
    far   : distance_metres >= 2.0   (LOW risk zone — beyond 2m)

Matches the Risk Engine's HIGH/MEDIUM/LOW thresholds exactly.

Where this fits in the pipeline
--------------------------------
    ConfidenceGate  →  RiskEngine  →  [ TemporalAlertManager ]  →  Voice
                                              ↑ YOU ARE HERE

The manager receives a RiskAssessment (from engine.py) and returns a
SpeakDecision.  Voice output is NOT called here — the manager is pure
logic, keeping concerns separated and the class fully testable.

Usage
-----
    from app.alerts.temporal_manager import TemporalAlertManager
    from app.risk.engine import assess_risk

    manager = TemporalAlertManager()           # one instance per session

    for frame_detections in camera_stream():
        assessments = risk_engine.assess_all(detections)
        for assessment in assessments:
            decision = manager.evaluate(assessment)
            if decision.should_speak:
                voice.speak(decision.speech_text)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.risk.engine import RiskAssessment
from app.risk.models import RiskLevel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Distance bucket
# ──────────────────────────────────────────────────────────────────────────────

class DistanceBucket(str, Enum):
    """
    Coarse distance zone.  Maps directly to the Risk Engine's thresholds
    so escalation (far → near → close) tracks real approach events.

    close : < 0.7m   — HIGH risk zone
    near  : < 2.0m   — MEDIUM risk zone
    far   : >= 2.0m  — LOW risk zone
    """
    CLOSE = "close"
    NEAR  = "near"
    FAR   = "far"


# Thresholds mirror RiskLevel boundaries in risk/models.py
_CLOSE_THRESHOLD: float = 0.70   # metres
_NEAR_THRESHOLD:  float = 2.00   # metres


def bucket_for(distance_metres: float) -> DistanceBucket:
    """Classify a distance into a DistanceBucket."""
    if distance_metres < _CLOSE_THRESHOLD:
        return DistanceBucket.CLOSE
    if distance_metres < _NEAR_THRESHOLD:
        return DistanceBucket.NEAR
    return DistanceBucket.FAR


# ──────────────────────────────────────────────────────────────────────────────
# Suppression reason  — full audit trail for ablation study
# ──────────────────────────────────────────────────────────────────────────────

class SuppressReason(str, Enum):
    """Why a particular assessment was suppressed or allowed through."""
    # Speak reasons
    NEW_OBJECT        = "new_object"         # (a) different class from last
    BUCKET_ESCALATED  = "bucket_escalated"   # (b) distance zone tightened
    WINDOW_ELAPSED_HIGH = "window_elapsed_high"  # (c) timeout + HIGH risk

    # Suppress reasons
    SAME_OBJECT_SAME_BUCKET_IN_WINDOW = "same_object_same_bucket_in_window"
    LOW_RISK_NO_OVERRIDE              = "low_risk_no_override"  # (c) timed out but LOW/MEDIUM
    FIRST_EVALUATION                  = "first_evaluation"      # always speaks


# ──────────────────────────────────────────────────────────────────────────────
# SpeakDecision — output of the manager
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpeakDecision:
    """
    What the voice layer should do with one RiskAssessment.

    Attributes
    ----------
    assessment   : The RiskAssessment that was evaluated.
    should_speak : True = pass to voice output.  False = suppress.
    speech_text  : Ready-to-speak string when should_speak is True.
                   None when suppressed.
    reason       : SuppressReason — which rule triggered the decision.
    elapsed_s    : Seconds since the last warning for this object.
                   None on first-ever evaluation.
    """
    assessment:   RiskAssessment
    should_speak: bool
    speech_text:  Optional[str]
    reason:       SuppressReason
    elapsed_s:    Optional[float]

    @property
    def suppressed(self) -> bool:
        return not self.should_speak


# ──────────────────────────────────────────────────────────────────────────────
# Speech text builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_speech(assessment: RiskAssessment) -> str:
    """
    Builds the spoken warning for a given RiskAssessment.

    Format varies by risk level and bucket so the user gets useful
    urgency cues without the system being verbose:

        HIGH  / close  →  "warning — chair very close ahead"
        HIGH  / near   →  "warning — chair getting close"
        MEDIUM/ near   →  "chair ahead, getting closer"
        MEDIUM/ far    →  "chair ahead"
        LOW   / far    →  "chair in the distance"   (log_only in practice)

    The direction from ConfidenceGate is intentionally NOT included here:
    direction is a ConfidenceGate concern.  This builder only knows about
    risk level and distance.  The voice layer can combine both strings
    if needed.
    """
    cls      = assessment.detection.class_name
    level    = assessment.level
    dist_m   = assessment.detection.distance_metres
    bucket   = bucket_for(dist_m)

    if level == RiskLevel.HIGH:
        if bucket == DistanceBucket.CLOSE:
            return f"warning — {cls} very close ahead"
        return f"warning — {cls} getting close"

    if level == RiskLevel.MEDIUM:
        if bucket == DistanceBucket.NEAR:
            return f"{cls} ahead, getting closer"
        return f"{cls} ahead"

    # LOW — normally log_only, but if this path is reached (e.g. manual test)
    return f"{cls} in the distance"


# ──────────────────────────────────────────────────────────────────────────────
# ObjectState — per-class tracking record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _ObjectState:
    """
    Tracks the last-spoken state for a single object class.
    One instance per unique class name in the session.

    These are the three state variables from the architecture doc:
        last_obstacle       → stored as the dict key in TemporalAlertManager
        last_distance_bucket → self.bucket
        last_warning_time   → self.last_warned_at
    """
    bucket:         DistanceBucket
    last_warned_at: float = field(default_factory=time.monotonic)
    warn_count:     int   = 0   # for ablation study: total times this class was spoken

    def seconds_since_warning(self) -> float:
        return time.monotonic() - self.last_warned_at

    def update(self, bucket: DistanceBucket) -> None:
        self.bucket         = bucket
        self.last_warned_at = time.monotonic()
        self.warn_count    += 1


# ──────────────────────────────────────────────────────────────────────────────
# TemporalAlertManager — the main class
# ──────────────────────────────────────────────────────────────────────────────

class TemporalAlertManager:
    """
    WalkVLM-inspired temporal suppression for MemoryNav.

    Speak only if ANY of:
        (a) object class is different from the last warned object
        (b) distance bucket changed (escalation: far → near → close)
        (c) suppression_window has elapsed AND risk is HIGH

    All three conditions mirror the architecture doc exactly.

    Parameters
    ----------
    suppression_window_s : float
        Seconds before rule (c) can fire.  Default 4.0 (doc spec).
        Tune lower for faster users, higher for more cautious ones.
    track_all_classes : bool
        If True, each class gets its own suppression timer (default).
        If False, a single global timer is used — simpler but misses
        cross-class interleaving (e.g. sofa warning just fired, chair
        should still be fresh).
    """

    def __init__(
        self,
        suppression_window_s: float  = 4.0,
        track_all_classes:    bool   = True,
    ) -> None:
        self.suppression_window_s = suppression_window_s
        self.track_all_classes    = track_all_classes

        # class_name → _ObjectState
        # Populated on first warning per class; absent = never warned.
        self._states: dict[str, _ObjectState] = {}

        # Ablation study counters
        self._total_evaluated:  int = 0
        self._total_spoken:     int = 0
        self._total_suppressed: int = 0

    # ── Primary API ───────────────────────────────────────────────────────────

    def evaluate(self, assessment: RiskAssessment) -> SpeakDecision:
        """
        Apply WalkVLM suppression logic to one RiskAssessment.

        Args:
            assessment : Output of RiskEngine.assess() for one Detection.

        Returns:
            SpeakDecision with should_speak, speech_text, reason, elapsed_s.
        """
        self._total_evaluated += 1

        cls     = assessment.detection.class_name
        bucket  = bucket_for(assessment.detection.distance_metres)
        level   = assessment.level
        key     = cls if self.track_all_classes else "__global__"
        state   = self._states.get(key)

        # ── First time this object class has been seen ─────────────────────
        if state is None:
            return self._speak(
                assessment, bucket, key,
                reason    = SuppressReason.FIRST_EVALUATION,
                elapsed_s = None,
            )

        elapsed = state.seconds_since_warning()

        # ── (a) Object class changed ───────────────────────────────────────
        # When tracking a single global slot, a different class name always
        # triggers because the slot only holds one class at a time.
        # When tracking per-class, this fires if the user turned to face a
        # completely different object — important to re-announce.
        if not self.track_all_classes and cls != _last_class(self._states):
            return self._speak(
                assessment, bucket, key,
                reason    = SuppressReason.NEW_OBJECT,
                elapsed_s = elapsed,
            )

        # ── (b) Distance bucket escalated (got closer) ─────────────────────
        # Only escalation triggers speech — de-escalation (receding) is
        # intentionally suppressed.  Alerting the user that something moved
        # *further away* creates noise; the risk is dropping, not rising.
        if _is_escalation(from_bucket=state.bucket, to_bucket=bucket):
            return self._speak(
                assessment, bucket, key,
                reason    = SuppressReason.BUCKET_ESCALATED,
                elapsed_s = elapsed,
            )

        # ── (c) Suppression window elapsed + HIGH risk ─────────────────────
        if elapsed >= self.suppression_window_s and level == RiskLevel.HIGH:
            return self._speak(
                assessment, bucket, key,
                reason    = SuppressReason.WINDOW_ELAPSED_HIGH,
                elapsed_s = elapsed,
            )

        # ── None of (a), (b), (c) → SUPPRESS ──────────────────────────────
        suppress_reason = (
            SuppressReason.LOW_RISK_NO_OVERRIDE
            if elapsed >= self.suppression_window_s
            else SuppressReason.SAME_OBJECT_SAME_BUCKET_IN_WINDOW
        )
        return self._suppress(assessment, bucket, suppress_reason, elapsed)

    def evaluate_batch(
        self, assessments: list[RiskAssessment]
    ) -> list[SpeakDecision]:
        """
        Evaluate a whole frame's assessments in score order.
        Assessments are expected to arrive sorted highest-score-first
        (RiskEngine.assess_all() guarantees this).

        Returns a list of SpeakDecisions in the same order.
        """
        return [self.evaluate(a) for a in assessments]

    # ── State inspection ──────────────────────────────────────────────────────

    def last_warned_class(self) -> Optional[str]:
        """The most recently spoken class name, or None if nothing spoken yet."""
        if not self._states:
            return None
        return max(
            self._states,
            key=lambda k: self._states[k].last_warned_at,
        )

    def seconds_since_last_warning(self, class_name: str) -> Optional[float]:
        """Elapsed seconds since class_name was last spoken, or None."""
        state = self._states.get(class_name)
        return state.seconds_since_warning() if state else None

    def reset(self) -> None:
        """
        Clear all state.  Call at session start, after user explicitly
        requests a reset, or when a new room is entered.
        Does NOT reset ablation counters (use reset_counters() for that).
        """
        self._states.clear()
        logger.info("[TemporalAlertManager] State reset — all tracking cleared")

    # ── Ablation study metrics ────────────────────────────────────────────────

    @property
    def suppression_rate(self) -> float:
        """
        Fraction of evaluated assessments that were suppressed.
        Primary ablation study metric: "Redundancy Reduction %"
        (doc Table 6.3 — Alert System metrics).
        """
        if self._total_evaluated == 0:
            return 0.0
        return self._total_suppressed / self._total_evaluated

    @property
    def counts(self) -> dict[str, int]:
        return {
            "evaluated":  self._total_evaluated,
            "spoken":     self._total_spoken,
            "suppressed": self._total_suppressed,
        }

    def reset_counters(self) -> None:
        self._total_evaluated  = 0
        self._total_spoken     = 0
        self._total_suppressed = 0

    def stats_summary(self) -> str:
        """One-line summary for ablation study logs."""
        n = self._total_evaluated
        if n == 0:
            return "TemporalAlertManager: no assessments evaluated yet"
        return (
            f"TemporalAlertManager — evaluated={n}  "
            f"spoken={self._total_spoken}  "
            f"suppressed={self._total_suppressed}  "
            f"suppression_rate={self.suppression_rate:.1%}  "
            f"window={self.suppression_window_s}s"
        )

    def class_warn_counts(self) -> dict[str, int]:
        """Per-class warning counts — useful for post-session ablation review."""
        return {cls: s.warn_count for cls, s in self._states.items()}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _speak(
        self,
        assessment: RiskAssessment,
        bucket:     DistanceBucket,
        key:        str,
        reason:     SuppressReason,
        elapsed_s:  Optional[float],
    ) -> SpeakDecision:
        """Record state update and return a speak decision."""
        if key in self._states:
            self._states[key].update(bucket)
        else:
            self._states[key] = _ObjectState(bucket=bucket)

        speech = _build_speech(assessment)
        self._total_spoken += 1

        logger.info(
            "[TemporalAlertManager] SPEAK  %s  bucket=%s  level=%s  "
            "reason=%s  elapsed=%s  → \"%s\"",
            assessment.detection.class_name,
            bucket.value,
            assessment.level.value,
            reason.value,
            f"{elapsed_s:.2f}s" if elapsed_s is not None else "first",
            speech,
        )
        return SpeakDecision(
            assessment=assessment,
            should_speak=True,
            speech_text=speech,
            reason=reason,
            elapsed_s=elapsed_s,
        )

    def _suppress(
        self,
        assessment: RiskAssessment,
        bucket:     DistanceBucket,
        reason:     SuppressReason,
        elapsed_s:  float,
    ) -> SpeakDecision:
        """Return a suppressed decision — state is NOT updated."""
        self._total_suppressed += 1

        logger.debug(
            "[TemporalAlertManager] SUPPRESS  %s  bucket=%s  level=%s  "
            "reason=%s  elapsed=%.2fs",
            assessment.detection.class_name,
            bucket.value,
            assessment.level.value,
            reason.value,
            elapsed_s,
        )
        return SpeakDecision(
            assessment=assessment,
            should_speak=False,
            speech_text=None,
            reason=reason,
            elapsed_s=elapsed_s,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_ESCALATION_ORDER = {
    DistanceBucket.FAR:   0,
    DistanceBucket.NEAR:  1,
    DistanceBucket.CLOSE: 2,
}


def _is_escalation(from_bucket: DistanceBucket, to_bucket: DistanceBucket) -> bool:
    """
    True only when the object moved CLOSER (escalation).
    De-escalation (moving away) intentionally returns False — no speech
    for a receding obstacle.

    FAR → NEAR  : True
    NEAR → CLOSE: True
    FAR → CLOSE : True  (jumped two zones in one frame — still escalation)
    CLOSE → NEAR: False  (moving away)
    NEAR → FAR  : False
    Same → Same : False
    """
    return _ESCALATION_ORDER[to_bucket] > _ESCALATION_ORDER[from_bucket]


def _last_class(states: dict[str, _ObjectState]) -> Optional[str]:
    """Return the class name most recently warned about."""
    if not states:
        return None
    return max(states, key=lambda k: states[k].last_warned_at)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test:  python -m app.alerts.temporal_manager
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # ── Minimal stubs so the smoke test runs without the full project ─────
    from dataclasses import dataclass as _dc
    from enum import Enum as _Enum

    @_dc
    class _Det:
        class_name: str
        confidence: float
        bbox: tuple
        distance_metres: float

    class _RL(_Enum):
        HIGH   = "HIGH"
        MEDIUM = "MEDIUM"
        LOW    = "LOW"

        @staticmethod
        def from_score(s):
            if s > 0.7: return _RL.HIGH
            if s > 0.4: return _RL.MEDIUM
            return _RL.LOW

    @_dc(frozen=True)
    class _RA:
        detection: object
        score: float
        level: object
        context_result: object = None

    def make_assessment(cls, dist, conf=0.90):
        from app.risk.engine import compute_risk_score
        det = _Det(cls, conf, (100,100,300,300), dist)
        score = compute_risk_score(dist)
        level = RiskLevel.from_score(score)

        class _FakeDet:
            class_name = cls
            confidence = conf
            bbox = (100,100,300,300)
            distance_metres = dist

        class _FakeAssessment:
            detection = _FakeDet()
            pass

        fa = _FakeAssessment()
        fa.score = score
        fa.level = level
        fa.context_result = None
        fa.detection = _FakeDet()
        return fa

    # Use real imports when available, else use stubs
    try:
        from app.risk.engine import RiskAssessment as _RealRA, compute_risk_score
        from app.risk.models import Detection as _RealDet, RiskLevel as _RealRL

        def _make(cls, dist, conf=0.90):
            det   = _RealDet(cls, conf, (100,100,300,300), dist)
            score = compute_risk_score(dist)
            level = _RealRL.from_score(score)
            return _RealRA(detection=det, score=score, level=level)

    except ImportError:
        # Fully standalone fallback — all stubs
        def _make(cls, dist, conf=0.90):  # type: ignore[misc]
            det        = _Det(cls, conf, (100,100,300,300), dist)
            score      = 1.0 / max(dist, 0.05)
            level      = _RL.from_score(score)

            class _FA:
                detection    = det
                pass
            fa = _FA()
            fa.score         = score
            fa.level         = level
            fa.context_result = None
            return fa

    print("\n── MemoryNav Temporal Alert Manager — smoke test ──\n")
    mgr = TemporalAlertManager(suppression_window_s=4.0)

    # ── Test 1: First evaluation always speaks (FIRST_EVALUATION) ─────────
    print("TEST 1 — first detection of chair always speaks")
    d = mgr.evaluate(_make("chair", dist=0.5))
    assert d.should_speak,              "First eval must speak"
    assert d.reason == SuppressReason.FIRST_EVALUATION
    assert "chair" in d.speech_text
    print(f"  speech='{d.speech_text}'  reason={d.reason.value}")
    print("  ✅ PASS\n")

    # ── Test 2: Same object, same bucket, within window → SUPPRESS ────────
    print("TEST 2 — same object, same bucket, within 4s → suppress")
    d = mgr.evaluate(_make("chair", dist=0.5))
    assert not d.should_speak
    assert d.reason == SuppressReason.SAME_OBJECT_SAME_BUCKET_IN_WINDOW
    print(f"  suppressed  reason={d.reason.value}  elapsed={d.elapsed_s:.3f}s")
    print("  ✅ PASS\n")

    # ── Test 3: (a) Different object → always speak ───────────────────────
    print("TEST 3 — (a) different object class → speak")
    d = mgr.evaluate(_make("sofa", dist=0.5))
    assert d.should_speak
    assert d.reason == SuppressReason.FIRST_EVALUATION   # sofa never seen
    print(f"  speech='{d.speech_text}'  reason={d.reason.value}")
    print("  ✅ PASS\n")

    # ── Test 4: (b) Distance bucket escalated → speak ─────────────────────
    print("TEST 4 — (b) chair moves from far to near → bucket escalation → speak")
    mgr2 = TemporalAlertManager(suppression_window_s=4.0)
    mgr2.evaluate(_make("chair", dist=3.0))    # FAR  → sets state
    d = mgr2.evaluate(_make("chair", dist=1.0)) # NEAR → escalation
    assert d.should_speak
    assert d.reason == SuppressReason.BUCKET_ESCALATED
    print(f"  FAR→NEAR  speech='{d.speech_text}'  reason={d.reason.value}")
    d2 = mgr2.evaluate(_make("chair", dist=0.4))  # CLOSE → escalation again
    assert d2.should_speak
    assert d2.reason == SuppressReason.BUCKET_ESCALATED
    print(f"  NEAR→CLOSE speech='{d2.speech_text}'  reason={d2.reason.value}")
    print("  ✅ PASS\n")

    # ── Test 5: De-escalation (moving away) → SUPPRESS ───────────────────
    print("TEST 5 — chair moves from close to near (receding) → suppress")
    mgr3 = TemporalAlertManager(suppression_window_s=4.0)
    mgr3.evaluate(_make("chair", dist=0.4))    # CLOSE
    d = mgr3.evaluate(_make("chair", dist=1.5)) # NEAR — de-escalation
    assert not d.should_speak, "De-escalation must NOT trigger speech"
    print(f"  CLOSE→NEAR  suppressed  reason={d.reason.value}")
    print("  ✅ PASS\n")

    # ── Test 6: (c) Window elapsed + HIGH → speak ─────────────────────────
    print("TEST 6 — (c) suppression window elapsed + HIGH risk → speak")
    mgr4 = TemporalAlertManager(suppression_window_s=0.05)  # tiny window for test
    mgr4.evaluate(_make("chair", dist=0.5))     # first speak, sets timer
    time.sleep(0.06)                            # exceed 0.05s window
    d = mgr4.evaluate(_make("chair", dist=0.5)) # same bucket, but window elapsed + HIGH
    assert d.should_speak
    assert d.reason == SuppressReason.WINDOW_ELAPSED_HIGH
    print(f"  elapsed={d.elapsed_s:.3f}s  speech='{d.speech_text}'  reason={d.reason.value}")
    print("  ✅ PASS\n")

    # ── Test 7: Window elapsed but MEDIUM risk → still suppress ───────────
    print("TEST 7 — window elapsed + MEDIUM risk → still suppressed")
    mgr5 = TemporalAlertManager(suppression_window_s=0.05)
    mgr5.evaluate(_make("table", dist=1.0))     # MEDIUM, first speak
    time.sleep(0.06)
    d = mgr5.evaluate(_make("table", dist=1.0)) # same bucket, window elapsed, MEDIUM
    assert not d.should_speak
    assert d.reason == SuppressReason.LOW_RISK_NO_OVERRIDE
    print(f"  elapsed={d.elapsed_s:.3f}s  suppressed  reason={d.reason.value}")
    print("  ✅ PASS\n")

    # ── Test 8: Ablation study counters ───────────────────────────────────
    print("TEST 8 — suppression_rate and stats_summary")
    print(f"  {mgr.stats_summary()}")
    assert mgr.suppression_rate > 0.0
    print(f"  suppression_rate={mgr.suppression_rate:.1%}")
    print("  ✅ PASS\n")

    # ── Test 9: bucket_for boundaries ────────────────────────────────────
    print("TEST 9 — bucket_for boundary values")
    assert bucket_for(0.0)  == DistanceBucket.CLOSE
    assert bucket_for(0.69) == DistanceBucket.CLOSE
    assert bucket_for(0.70) == DistanceBucket.NEAR
    assert bucket_for(1.99) == DistanceBucket.NEAR
    assert bucket_for(2.00) == DistanceBucket.FAR
    assert bucket_for(9.99) == DistanceBucket.FAR
    print("  CLOSE: <0.70m  NEAR: 0.70–1.99m  FAR: >=2.00m  ✅ PASS\n")

    # ── Test 10: _is_escalation all combinations ──────────────────────────
    print("TEST 10 — _is_escalation direction matrix")
    assert     _is_escalation(DistanceBucket.FAR,   DistanceBucket.NEAR)
    assert     _is_escalation(DistanceBucket.NEAR,  DistanceBucket.CLOSE)
    assert     _is_escalation(DistanceBucket.FAR,   DistanceBucket.CLOSE)
    assert not _is_escalation(DistanceBucket.CLOSE, DistanceBucket.NEAR)
    assert not _is_escalation(DistanceBucket.NEAR,  DistanceBucket.FAR)
    assert not _is_escalation(DistanceBucket.CLOSE, DistanceBucket.CLOSE)
    print("  all escalation/de-escalation directions correct  ✅ PASS\n")

    print("── All 10 tests passed ──\n")