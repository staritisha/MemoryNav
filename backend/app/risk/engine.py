"""
backend/app/risk/engine.py
--------------------------
Module 2 — Risk Engine  (Phase 3 UPDATE)

What changed from the Phase 2 version
--------------------------------------
The Phase 2 engine accepted user_context_weight as a plain float
argument defaulting to 1.0.  The caller was responsible for supplying
it — but nothing supplied it yet, so every assessment ran at 1.0.

This update wires that gap.  The engine now owns a ContextWeightResolver
that pulls the weight from two Phase 3 sources before scoring:

    1. preferences.py  (SQLite)
       Reads mobility_flags (bad_knee, uses_walker, low_vision …) and
       maps them to a base multiplier.

    2. long_term.py  (ChromaDB)
       Calls retrieve(query) with the detected class name and returns
       the top spatial memory hit ("loose rug near sofa" when sofa is
       detected).  If a hit exists AND its similarity is above a
       threshold, the weight is boosted — the environment itself
       contributes to risk.

Final weight:
    user_context_weight = clamp(mobility_weight + spatial_boost, 0.1, 3.0)

The full risk formula from the architecture doc is unchanged:
    Risk Score = (1 / distance_metres) × motion_factor × user_context_weight

Backwards compatibility
-----------------------
assess_risk() and assess_all() still accept an explicit
user_context_weight kwarg.  Pass it and the resolver is skipped
entirely — existing tests and the __main__ demo continue to work with
no changes.  To use the live resolver, omit the kwarg (or pass None).

Usage
-----
    # Resolved automatically from preferences + long_term (Phase 3):
    assessment = assess_risk(detection)

    # Or resolve once per frame and pass to every detection:
    engine = RiskEngine()
    weight = engine.resolve_context_weight("chair")
    results = assess_all(detections, user_context_weight=weight)

    # Legacy / test mode — explicit weight, no DB calls:
    assessment = assess_risk(detection, user_context_weight=0.45)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── lazy imports for Phase 3 modules (avoids hard crash if not installed yet)
try:
    from app.memory_modules.preferences import PreferencesStore
    _PREFS_AVAILABLE = True
except ImportError:
    _PREFS_AVAILABLE = False
    logger.warning(
        "preferences.py not found — mobility_weight will default to 1.0. "
        "Install Phase 3 modules to enable preference-based context weighting."
    )

try:
    from app.memory_modules.long_term import LongTermMemory
    _LONGTERM_AVAILABLE = True
except ImportError:
    _LONGTERM_AVAILABLE = False
    logger.warning(
        "long_term.py not found — spatial_boost will default to 0.0. "
        "Install Phase 3 modules to enable RAG-based context weighting."
    )

# ── keep the Detection / RiskLevel models import clean
from app.risk.models import Detection, RiskLevel


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_MIN_DISTANCE_METRES = 0.05   # floor to avoid div-by-zero / infinite score

# Weight bounds — prevent a single flag or bad DB value from producing nonsense
_WEIGHT_MIN = 0.1
_WEIGHT_MAX = 3.0

# Similarity threshold: only apply spatial boost when ChromaDB is this confident
_SPATIAL_SIMILARITY_THRESHOLD = 0.55

# How much the spatial memory boosts the weight when a relevant memory is found.
# Tuned to add ~30 % on top of whatever the mobility weight is.
_SPATIAL_BOOST_VALUE = 0.30

# ── Mobility flag → weight mapping (from preferences.py mobility_flags column)
# Values are additive contributions that stack with the base weight of 1.0.
# "bad_knee" alone → 1.0 + 0.45 = 1.45 (matches the architecture doc example:
#  chair at 0.5m, stationary, bad_knee → score ≈ 0.9, clamped to HIGH).
_MOBILITY_WEIGHT_MAP: dict[str, float] = {
    "bad_knee":       0.45,
    "uses_walker":    0.55,
    "low_vision":     0.35,
    "wheelchair":     0.60,
    "hearing_loss":   0.10,   # minor — doesn't affect obstacle risk directly
    "balance_issues": 0.50,
}


# ──────────────────────────────────────────────────────────────────────────────
# MotionState  (unchanged from Phase 2)
# ──────────────────────────────────────────────────────────────────────────────

class MotionState(str, Enum):
    """
    Vocabulary for motion_factor until real frame-to-frame tracking
    lands in Short-Term Memory (Module 3, Phase 4).
    """
    APPROACHING = "approaching"
    STATIONARY  = "stationary"
    RECEDING    = "receding"
    UNKNOWN     = "unknown"


_MOTION_FACTORS: dict[MotionState, float] = {
    MotionState.APPROACHING: 1.5,
    MotionState.STATIONARY:  1.0,
    MotionState.RECEDING:    0.5,
    MotionState.UNKNOWN:     1.0,
}


def motion_factor_for(state: MotionState) -> float:
    return _MOTION_FACTORS[state]


# ──────────────────────────────────────────────────────────────────────────────
# ContextWeightResolver  ← NEW in Phase 3
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextWeightResult:
    """
    Full breakdown of how user_context_weight was computed.
    Stored in RiskAssessment so the Alert Manager and logs can
    explain *why* something was rated HIGH — not just that it was.
    """
    final_weight:      float
    mobility_weight:   float           # base from SQLite preferences
    spatial_boost:     float           # additive from ChromaDB hit
    spatial_memory:    Optional[str]   # the retrieved memory text, if any
    spatial_similarity: Optional[float] # cosine similarity of that hit
    mobility_flags:    list[str]       # which flags were active


class ContextWeightResolver:
    """
    Pulls user_context_weight from:
      • PreferencesStore  (SQLite) — mobility_flags → mobility_weight
      • LongTermMemory    (ChromaDB) — retrieve(class_name) → spatial_boost

    Both sources are optional.  If neither is available the resolver
    returns 1.0 with a full audit trail so the caller can tell why.

    The resolver is stateless between calls — safe to call once per
    detection, or once per frame and reuse the result.
    """

    def __init__(
        self,
        prefs_store: Optional["PreferencesStore"] = None,
        long_term: Optional["LongTermMemory"] = None,
    ) -> None:
        # Accept injected instances (for testing) or build live ones
        self._prefs:     Optional["PreferencesStore"] = prefs_store
        self._long_term: Optional["LongTermMemory"]   = long_term
        self._prefs_loaded    = prefs_store is not None
        self._longterm_loaded = long_term is not None

    # ── lazy initialisation — only hit the DB once, on first call ──────────
    def _ensure_prefs(self) -> None:
        if not self._prefs_loaded and _PREFS_AVAILABLE:
            try:
                self._prefs = PreferencesStore()
                self._prefs_loaded = True
            except Exception as exc:
                logger.error("PreferencesStore init failed: %s", exc)

    def _ensure_long_term(self) -> None:
        if not self._longterm_loaded and _LONGTERM_AVAILABLE:
            try:
                self._long_term = LongTermMemory()
                self._longterm_loaded = True
            except Exception as exc:
                logger.error("LongTermMemory init failed: %s", exc)

    # ── public API ──────────────────────────────────────────────────────────
    def resolve(self, class_name: str) -> ContextWeightResult:
        """
        Compute user_context_weight for a detected object class.

        Args:
            class_name: YOLO class label, e.g. "chair", "sofa", "person"

        Returns:
            ContextWeightResult with final_weight and full audit trail.
        """
        mobility_weight, active_flags = self._get_mobility_weight()
        spatial_boost, memory_text, similarity = self._get_spatial_boost(class_name)

        final = _clamp(mobility_weight + spatial_boost, _WEIGHT_MIN, _WEIGHT_MAX)

        logger.debug(
            "[RiskEngine] %s → mobility=%.2f + spatial_boost=%.2f = %.2f  "
            "(flags=%s, memory=%r, sim=%s)",
            class_name, mobility_weight, spatial_boost, final,
            active_flags, memory_text,
            f"{similarity:.3f}" if similarity is not None else "n/a",
        )

        return ContextWeightResult(
            final_weight=final,
            mobility_weight=mobility_weight,
            spatial_boost=spatial_boost,
            spatial_memory=memory_text,
            spatial_similarity=similarity,
            mobility_flags=active_flags,
        )

    # ── private helpers ─────────────────────────────────────────────────────
    def _get_mobility_weight(self) -> tuple[float, list[str]]:
        """
        Read mobility_flags from SQLite preferences and sum their weights.
        Falls back to 1.0 (neutral) if the store is unavailable or empty.

        mobility_flags is stored in preferences.py as a comma-separated
        string: "bad_knee,uses_walker"
        """
        self._ensure_prefs()
        base = 1.0
        active: list[str] = []

        if self._prefs is None:
            return base, active

        try:
            flags_raw: str = self._prefs.get("mobility_flags", default="")
            if not flags_raw:
                return base, active

            flags = [f.strip().lower() for f in flags_raw.split(",") if f.strip()]
            for flag in flags:
                increment = _MOBILITY_WEIGHT_MAP.get(flag, 0.0)
                base += increment
                if increment > 0:
                    active.append(flag)
        except Exception as exc:
            logger.warning("Could not read mobility_flags from preferences: %s", exc)

        return base, active

    def _get_spatial_boost(
        self, class_name: str
    ) -> tuple[float, Optional[str], Optional[float]]:
        """
        Query ChromaDB for memories related to the detected class.
        Returns (boost, memory_text, similarity).

        A boost is applied only when:
          • long_term memory is available
          • a hit exists
          • similarity >= _SPATIAL_SIMILARITY_THRESHOLD

        Example: sofa detected → retrieve("sofa") →
          "loose rug near sofa in living room" at 0.81 similarity
          → boost applied → weight increased.
        """
        self._ensure_long_term()

        if self._long_term is None:
            return 0.0, None, None

        try:
            hits = self._long_term.retrieve(class_name, n_results=1)
        except Exception as exc:
            logger.warning("LongTermMemory.retrieve() failed: %s", exc)
            return 0.0, None, None

        if not hits:
            return 0.0, None, None

        top = hits[0]
        text       = top.get("text", "")
        similarity = top.get("similarity", 0.0)

        if similarity >= _SPATIAL_SIMILARITY_THRESHOLD:
            logger.info(
                "[RiskEngine] Spatial memory hit for '%s': %r (sim=%.3f) → +%.2f boost",
                class_name, text, similarity, _SPATIAL_BOOST_VALUE,
            )
            return _SPATIAL_BOOST_VALUE, text, similarity

        # Hit found but similarity too low — don't boost
        return 0.0, text, similarity


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ──────────────────────────────────────────────────────────────────────────────
# RiskAssessment  (extended from Phase 2)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskAssessment:
    """Full output of the Risk Engine for one Detection."""
    detection:      Detection
    score:          float
    level:          RiskLevel
    context_result: Optional[ContextWeightResult] = field(default=None)

    @property
    def action(self) -> str:
        return {
            RiskLevel.HIGH:   "interrupt_immediately",
            RiskLevel.MEDIUM: "queue",
            RiskLevel.LOW:    "log_only",
        }[self.level]

    @property
    def explanation(self) -> str:
        """
        Human-readable string for logs and the ablation study.
        Example:
          chair @0.50m | score=2.61 HIGH interrupt_immediately
          mobility=[bad_knee] spatial='loose rug near sofa' (sim=0.81)
        """
        cr = self.context_result
        parts = [
            f"{self.detection.class_name} @{self.detection.distance_metres:.2f}m",
            f"score={self.score:.2f}",
            f"{self.level.value}",
            self.action,
        ]
        if cr:
            if cr.mobility_flags:
                parts.append(f"mobility={cr.mobility_flags}")
            if cr.spatial_memory:
                parts.append(f"spatial={cr.spatial_memory!r} (sim={cr.spatial_similarity:.2f})")
        return "  |  ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Core scoring function  (unchanged formula, unchanged signature)
# ──────────────────────────────────────────────────────────────────────────────

def compute_risk_score(
    distance_metres: float,
    motion_factor: float = 1.0,
    user_context_weight: float = 1.0,
) -> float:
    """
    Risk Score = (1 / distance_metres) × motion_factor × user_context_weight

    NaN distance → float("inf")  (unknown distance = maximum risk).
    distance below _MIN_DISTANCE_METRES is floored (div-by-zero guard).
    """
    if math.isnan(distance_metres):
        return float("inf")
    safe_dist = max(distance_metres, _MIN_DISTANCE_METRES)
    return (1.0 / safe_dist) * motion_factor * user_context_weight


# ──────────────────────────────────────────────────────────────────────────────
# RiskEngine  ← NEW wrapper class for Phase 3
# ──────────────────────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Stateful wrapper that holds a single ContextWeightResolver instance
    across multiple assess_risk() calls in a frame loop.

    Prefer this over the module-level assess_risk() function when you
    process many detections per frame — it avoids re-initialising the
    DB connections on every call.

    Usage:
        engine = RiskEngine()                       # one instance per session
        results = engine.assess_all(detections)     # called once per frame
    """

    def __init__(
        self,
        prefs_store: Optional["PreferencesStore"] = None,
        long_term: Optional["LongTermMemory"] = None,
    ) -> None:
        self._resolver = ContextWeightResolver(
            prefs_store=prefs_store,
            long_term=long_term,
        )

    def resolve_context_weight(self, class_name: str) -> ContextWeightResult:
        """Expose the resolver for callers that want the weight separately."""
        return self._resolver.resolve(class_name)

    def assess(
        self,
        detection: Detection,
        motion_factor: float = 1.0,
        user_context_weight: Optional[float] = None,
    ) -> RiskAssessment:
        """
        Score one Detection.  If user_context_weight is None (default),
        the resolver pulls it from preferences + long_term.
        Pass an explicit float to skip resolution (legacy / test mode).
        """
        if user_context_weight is None:
            ctx = self._resolver.resolve(detection.class_name)
            weight = ctx.final_weight
        else:
            ctx = None
            weight = user_context_weight

        score = compute_risk_score(
            detection.distance_metres,
            motion_factor=motion_factor,
            user_context_weight=weight,
        )
        return RiskAssessment(
            detection=detection,
            score=score,
            level=RiskLevel.from_score(score),
            context_result=ctx,
        )

    def assess_all(
        self,
        detections: List[Detection],
        motion_factor: float = 1.0,
        user_context_weight: Optional[float] = None,
    ) -> List[RiskAssessment]:
        """
        Score a full frame's detections.  Returns list sorted by score
        descending — highest-risk obstacle first (what Alert Manager needs).

        user_context_weight is resolved once per unique class in the
        frame to avoid redundant DB calls when the same object appears
        in multiple bounding boxes.
        """
        # Pre-resolve weights by class to avoid duplicate DB hits per frame
        if user_context_weight is None:
            seen: dict[str, ContextWeightResult] = {}
            for d in detections:
                if d.class_name not in seen:
                    seen[d.class_name] = self._resolver.resolve(d.class_name)
        else:
            seen = {}  # explicit weight provided — resolver bypassed

        results: List[RiskAssessment] = []
        for d in detections:
            if user_context_weight is not None:
                w, ctx = user_context_weight, None
            else:
                ctx = seen[d.class_name]
                w = ctx.final_weight

            score = compute_risk_score(
                d.distance_metres,
                motion_factor=motion_factor,
                user_context_weight=w,
            )
            results.append(RiskAssessment(
                detection=d,
                score=score,
                level=RiskLevel.from_score(score),
                context_result=ctx,
            ))

        results.sort(key=lambda a: a.score, reverse=True)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience functions  (backwards-compatible with Phase 2 callers)
# ──────────────────────────────────────────────────────────────────────────────

# One shared engine for the process — callers that import the functions below
# get automatic Phase 3 resolution without changing a single line of their code.
_default_engine = RiskEngine()


def assess_risk(
    detection: Detection,
    motion_factor: float = 1.0,
    user_context_weight: Optional[float] = None,
) -> RiskAssessment:
    """
    Module-level entry point.  Phase 2 callers that passed an explicit
    user_context_weight float continue to work unchanged.  Omit it to
    get automatic resolution from preferences + long_term (Phase 3).
    """
    return _default_engine.assess(
        detection,
        motion_factor=motion_factor,
        user_context_weight=user_context_weight,
    )


def assess_all(
    detections: List[Detection],
    motion_factor: float = 1.0,
    user_context_weight: Optional[float] = None,
) -> List[RiskAssessment]:
    """Batch version of assess_risk. Returns sorted list, highest risk first."""
    return _default_engine.assess_all(
        detections,
        motion_factor=motion_factor,
        user_context_weight=user_context_weight,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test:  python -m app.risk.engine
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n── MemoryNav Risk Engine — Phase 3 smoke test ──\n")

    # ── Test 1: Legacy mode (explicit weight — no DB) ──────────────────────
    print("TEST 1 — Legacy / explicit weight (no DB calls)")
    chair = Detection(
        class_name="chair", confidence=0.95,
        bbox=(100, 100, 300, 400), distance_metres=0.5,
    )
    r = assess_risk(chair, motion_factor=1.0, user_context_weight=0.45)
    print(f"  {r.explanation}")
    assert r.level == RiskLevel.HIGH, f"Expected HIGH, got {r.level}"
    print("  ✅ PASS\n")

    # ── Test 2: Phase 3 auto-resolution (mocked sources) ──────────────────
    print("TEST 2 — Phase 3 resolver with mocked preferences + long_term")

    class MockPrefs:
        """Mimics preferences.py PreferencesStore.get()"""
        def get(self, key: str, default: str = "") -> str:
            return "bad_knee,uses_walker" if key == "mobility_flags" else default

    class MockLongTerm:
        """Mimics long_term.py LongTermMemory.retrieve()"""
        def retrieve(self, query: str, n_results: int = 1):
            if "sofa" in query.lower():
                return [{"text": "loose rug near sofa in living room",
                          "similarity": 0.81}]
            return []

    engine = RiskEngine(prefs_store=MockPrefs(), long_term=MockLongTerm())

    sofa = Detection(
        class_name="sofa", confidence=0.88,
        bbox=(50, 50, 400, 350), distance_metres=1.2,
    )
    r2 = engine.assess(sofa, motion_factor=1.0)
    print(f"  {r2.explanation}")
    assert r2.context_result is not None
    assert "bad_knee" in r2.context_result.mobility_flags
    assert r2.context_result.spatial_memory is not None
    print("  ✅ PASS — mobility flags AND spatial boost applied\n")

    # ── Test 3: Architecture doc example — table at 3m ────────────────────
    print("TEST 3 — Architecture doc example: table at 3m, no context")
    table = Detection(
        class_name="table", confidence=0.88,
        bbox=(50, 50, 250, 350), distance_metres=3.0,
    )
    r3 = assess_risk(table, motion_factor=1.0, user_context_weight=1.0)
    print(f"  {r3.explanation}")
    assert r3.level == RiskLevel.LOW
    print("  ✅ PASS\n")

    # ── Test 4: assess_all sorts by score ─────────────────────────────────
    print("TEST 4 — assess_all returns sorted by score descending")
    detections = [
        Detection("table",  0.80, (10,10,100,100), distance_metres=3.0),
        Detection("chair",  0.90, (10,10,100,100), distance_metres=0.5),
        Detection("person", 0.75, (10,10,100,100), distance_metres=1.5),
    ]
    all_r = assess_all(detections, user_context_weight=1.0)
    scores = [r.score for r in all_r]
    assert scores == sorted(scores, reverse=True), "Results not sorted"
    print(f"  Sorted: {[(r.detection.class_name, round(r.score,2)) for r in all_r]}")
    print("  ✅ PASS\n")

    print("── All tests passed ──\n")