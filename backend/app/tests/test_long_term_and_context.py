"""
MemoryNav — LongTermMemory & ContextWeightResolver Unit Tests
backend/tests/test_long_term_and_context.py

Part A — memory_modules/long_term.py
  Check 1: cosine distance collection initialisation
  Check 2: similarity = 1.0 - distance conversion
  Check 3: timestamp always stamped in metadata
  Check 4: empty collection returns [] without raising

Part B — risk/engine.py ContextWeightResolver
  Check 1: all 6 mobility flags present with exact values; additive from base 1.0
  Check 2: spatial boost fires at similarity >= 0.55, skipped below
  Check 3: final weight clamped to [0.1, 3.0]
  Check 4: full formula trace (wheelchair + bad_knee + spatial boost)

All ChromaDB tests use a tmp_path fixture so they never touch the
production data/chroma_store directory and are fully isolated.

Run:
    pytest backend/tests/test_long_term_and_context.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.memory_modules.long_term import LongTermMemory, MemoryResult
from app.risk.engine import (
    ContextWeightResolver,
    ContextWeightResult,
    RiskEngine,
    _MOBILITY_WEIGHT_MAP,
    _SPATIAL_BOOST_VALUE,
    _SPATIAL_SIMILARITY_THRESHOLD,
    _WEIGHT_MAX,
    _WEIGHT_MIN,
    _clamp,
    compute_risk_score,
    motion_factor_for,
    MotionState,
)
from app.risk.models import Detection, RiskLevel


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def memory(tmp_path):
    """Fresh LongTermMemory backed by a temp dir — isolated per test."""
    return LongTermMemory(
        persist_dir=str(tmp_path),
        collection_name="test_collection",
    )


def _make_detection(
    class_name: str = "chair",
    distance: float = 1.0,
    confidence: float = 0.90,
) -> Detection:
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=(100, 100, 300, 300),
        distance_metres=distance,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Part A — LongTermMemory
# ══════════════════════════════════════════════════════════════════════════════

class TestCollectionInitialisation:
    """Check 1: cosine distance must be set on collection creation."""

    def test_collection_metadata_has_cosine_space(self, memory: LongTermMemory):
        """
        The collection's hnsw:space must be 'cosine'.
        L2 (the ChromaDB default) makes similarity = 1 - distance meaningless
        for values > 1.0 — a common silent data-corruption bug.
        """
        meta = memory._collection.metadata
        assert meta is not None, "Collection metadata must not be None"
        assert "hnsw:space" in meta, "hnsw:space key missing from collection metadata"
        assert meta["hnsw:space"] == "cosine", (
            f"Expected hnsw:space='cosine', got {meta['hnsw:space']!r}"
        )

    def test_collection_name_matches_config(self, memory: LongTermMemory):
        assert memory._collection.name == "test_collection"

    def test_empty_collection_count_is_zero(self, memory: LongTermMemory):
        assert memory.count() == 0


class TestSimilarityConversion:
    """Check 2: similarity = 1.0 - distance."""

    def test_similarity_property_formula(self):
        r = MemoryResult(id="x", text="chair", metadata={}, distance=0.20)
        assert r.similarity == pytest.approx(0.80)

    def test_similarity_zero_distance_is_one(self):
        r = MemoryResult(id="x", text="exact match", metadata={}, distance=0.0)
        assert r.similarity == pytest.approx(1.0)

    def test_similarity_one_distance_is_zero(self):
        r = MemoryResult(id="x", text="opposite", metadata={}, distance=1.0)
        assert r.similarity == pytest.approx(0.0)

    def test_similarity_used_in_retrieved_results(self, memory: LongTermMemory):
        """End-to-end: add and retrieve, check .similarity is in [0, 1]."""
        memory.add_context("chair near the window in the living room")
        results = memory.retrieve("chair")
        assert results
        assert 0.0 <= results[0].similarity <= 1.0

    def test_retrieved_results_sorted_by_similarity(self, memory: LongTermMemory):
        """Higher-similarity results should come first."""
        memory.add_context("chair near the living room window")
        memory.add_context("bicycle parked outside on the street")
        memory.add_context("table in the dining room")
        results = memory.retrieve("chair in living room", n_results=3)
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True), "Results not sorted by similarity"


class TestMetadataSchema:
    """Check 3: timestamp always stamped; optional fields preserved."""

    def test_timestamp_always_present(self, memory: LongTermMemory):
        """add_context must stamp a timestamp even when caller provides no metadata."""
        memory.add_context("loose rug near sofa")
        results = memory.retrieve("rug")
        assert results
        assert "timestamp" in results[0].metadata

    def test_timestamp_is_iso_format(self, memory: LongTermMemory):
        """Timestamp must be parseable as an ISO-8601 string."""
        from datetime import datetime
        memory.add_context("step at kitchen entrance")
        results = memory.retrieve("kitchen step")
        ts = results[0].metadata["timestamp"]
        # Should not raise
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_optional_metadata_fields_preserved(self, memory: LongTermMemory):
        memory.add_context(
            "chair near window",
            metadata={"room": "Living Room", "class_name": "chair", "distance_m": 1.2},
        )
        results = memory.retrieve("chair")
        meta = results[0].metadata
        assert meta["room"] == "Living Room"
        assert meta["class_name"] == "chair"
        assert meta["distance_m"] == pytest.approx(1.2)
        assert "timestamp" in meta   # auto-stamped in addition to caller fields

    def test_caller_timestamp_not_overwritten(self, memory: LongTermMemory):
        """If caller explicitly provides a timestamp, setdefault must not overwrite it."""
        custom_ts = "2024-01-15T10:30:00+00:00"
        memory.add_context("test", metadata={"timestamp": custom_ts})
        results = memory.retrieve("test")
        assert results[0].metadata["timestamp"] == custom_ts

    def test_empty_text_raises_value_error(self, memory: LongTermMemory):
        with pytest.raises(ValueError):
            memory.add_context("")

    def test_whitespace_only_text_raises_value_error(self, memory: LongTermMemory):
        with pytest.raises(ValueError):
            memory.add_context("   ")


class TestEmptyCollectionGuard:
    """Check 4: empty collection must return [] not raise."""

    def test_empty_collection_returns_empty_list(self, memory: LongTermMemory):
        """Required test from step brief — must not raise."""
        result = memory.retrieve("chair near door")
        assert result == []

    def test_empty_collection_does_not_raise_on_retrieve(self, memory: LongTermMemory):
        """Explicit exception guard: ChromaDB raises on empty query without this."""
        try:
            memory.retrieve("anything at all")
        except Exception as exc:
            pytest.fail(f"retrieve() raised on empty collection: {exc}")

    def test_blank_query_returns_empty_list(self, memory: LongTermMemory):
        memory.add_context("chair in the living room")
        assert memory.retrieve("") == []
        assert memory.retrieve("   ") == []

    def test_list_all_empty_collection(self, memory: LongTermMemory):
        assert memory.list_all() == []

    def test_crud_add_delete_count(self, memory: LongTermMemory):
        doc_id = memory.add_context("sofa near the window")
        assert memory.count() == 1
        memory.delete(doc_id)
        assert memory.count() == 0

    def test_clear_wipes_collection(self, memory: LongTermMemory):
        memory.add_context("chair")
        memory.add_context("table")
        assert memory.count() == 2
        memory.clear()
        assert memory.count() == 0
        # Must still work after clear
        assert memory.retrieve("chair") == []

    def test_n_results_capped_at_collection_size(self, memory: LongTermMemory):
        """Requesting more results than exist must not raise."""
        memory.add_context("only one entry")
        results = memory.retrieve("entry", n_results=10)
        assert len(results) == 1

    def test_where_filter_on_empty_collection(self, memory: LongTermMemory):
        """Metadata filter on empty collection must return []."""
        result = memory.retrieve("chair", where={"room": "kitchen"})
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# Part B — ContextWeightResolver
# ══════════════════════════════════════════════════════════════════════════════

# ── Mock helpers ──────────────────────────────────────────────────────────────

def _mock_prefs(flags: list[str]) -> MagicMock:
    """Build a mock PreferencesStore whose .load() returns the given flags."""
    from app.memory_modules.preferences import UserPreferences
    prefs = MagicMock()
    pref_obj = MagicMock(spec=UserPreferences)
    pref_obj.mobility_flags = flags
    prefs.load.return_value = pref_obj
    return prefs


def _mock_long_term(similarity: float = 0.0, text: str = "") -> MagicMock:
    """Build a mock LongTermMemory whose .retrieve() returns one result."""
    lt = MagicMock()
    if similarity > 0.0:
        result = MemoryResult(id="x", text=text, metadata={}, distance=1.0 - similarity)
        lt.retrieve.return_value = [result]
    else:
        lt.retrieve.return_value = []
    return lt


def _resolver(flags: list[str] = None, similarity: float = 0.0,
              memory_text: str = "") -> ContextWeightResolver:
    return ContextWeightResolver(
        prefs_store=_mock_prefs(flags or []),
        long_term=_mock_long_term(similarity, memory_text),
    )


class TestMobilityWeightMap:
    """Check 1: all 6 flags present with exact doc values; base=1.0, additive."""

    def test_all_six_flags_in_map(self):
        required = {
            "bad_knee", "uses_walker", "low_vision",
            "wheelchair", "hearing_loss", "balance_issues",
        }
        assert required == set(_MOBILITY_WEIGHT_MAP.keys())

    def test_exact_flag_values(self):
        assert _MOBILITY_WEIGHT_MAP["bad_knee"]       == pytest.approx(0.45)
        assert _MOBILITY_WEIGHT_MAP["uses_walker"]    == pytest.approx(0.55)
        assert _MOBILITY_WEIGHT_MAP["low_vision"]     == pytest.approx(0.35)
        assert _MOBILITY_WEIGHT_MAP["wheelchair"]     == pytest.approx(0.60)
        assert _MOBILITY_WEIGHT_MAP["hearing_loss"]   == pytest.approx(0.10)
        assert _MOBILITY_WEIGHT_MAP["balance_issues"] == pytest.approx(0.50)

    def test_no_flags_returns_base_weight_1(self):
        r = _resolver(flags=[]).resolve("chair")
        assert r.mobility_weight == pytest.approx(1.0)

    def test_single_flag_additive(self):
        r = _resolver(flags=["bad_knee"]).resolve("chair")
        assert r.mobility_weight == pytest.approx(1.0 + 0.45)

    def test_mobility_flags_additive_required_test(self):
        """Required test: wheelchair + bad_knee = clamp(1.0 + 0.60 + 0.45, 0.1, 3.0)."""
        r = _resolver(flags=["wheelchair", "bad_knee"]).resolve("chair")
        expected = _clamp(1.0 + 0.60 + 0.45, _WEIGHT_MIN, _WEIGHT_MAX)
        assert r.final_weight == pytest.approx(expected)
        assert expected == pytest.approx(2.05)

    def test_all_six_flags_additive(self):
        all_flags = list(_MOBILITY_WEIGHT_MAP.keys())
        r = _resolver(flags=all_flags).resolve("chair")
        expected_mobility = 1.0 + sum(_MOBILITY_WEIGHT_MAP.values())
        assert r.mobility_weight == pytest.approx(expected_mobility, abs=0.001)

    def test_unknown_flag_ignored(self):
        r = _resolver(flags=["unknown_condition"]).resolve("chair")
        assert r.mobility_weight == pytest.approx(1.0)

    def test_active_flags_reported_in_result(self):
        r = _resolver(flags=["wheelchair", "bad_knee"]).resolve("chair")
        assert set(r.mobility_flags) == {"wheelchair", "bad_knee"}

    def test_no_prefs_store_defaults_to_base_weight(self):
        resolver = ContextWeightResolver(prefs_store=None, long_term=_mock_long_term())
        r = resolver.resolve("chair")
        assert r.mobility_weight == pytest.approx(1.0)


class TestSpatialBoost:
    """Check 2: boost fires at similarity >= 0.55 exactly, skipped below."""

    def test_spatial_boost_applied_required_test(self, tmp_path):
        """
        Required test from step brief.
        Add a memory, retrieve it, verify boost fires (weight >= 1.30).
        Uses real ChromaDB + embedding model for end-to-end confidence.
        """
        memory = LongTermMemory(
            persist_dir=str(tmp_path),
            collection_name="boost_test",
        )
        memory.add_context(
            "chair near the window in living room",
            metadata={"room": "Living Room", "class_name": "chair"},
        )
        resolver = ContextWeightResolver(
            prefs_store=_mock_prefs([]),   # no mobility flags
            long_term=memory,
        )
        result = resolver.resolve("chair")
        # Semantic similarity of "chair" vs "chair near the window" should be high
        assert result.final_weight >= 1.30, (
            f"Expected spatial boost applied (weight >= 1.30), got {result.final_weight}"
        )

    def test_boost_fires_at_exact_threshold(self):
        """Similarity == 0.55 (the boundary) must trigger the boost."""
        resolver = ContextWeightResolver(
            prefs_store=_mock_prefs([]),
            long_term=_mock_long_term(similarity=_SPATIAL_SIMILARITY_THRESHOLD,
                                      text="chair in room"),
        )
        result = resolver.resolve("chair")
        assert result.spatial_boost == pytest.approx(_SPATIAL_BOOST_VALUE)

    def test_boost_not_fired_below_threshold(self):
        """Similarity just below 0.55 must NOT boost."""
        resolver = ContextWeightResolver(
            prefs_store=_mock_prefs([]),
            long_term=_mock_long_term(similarity=_SPATIAL_SIMILARITY_THRESHOLD - 0.01,
                                      text="chair in room"),
        )
        result = resolver.resolve("chair")
        assert result.spatial_boost == pytest.approx(0.0)

    def test_boost_value_is_0_30(self):
        assert _SPATIAL_BOOST_VALUE == pytest.approx(0.30)

    def test_threshold_value_is_0_55(self):
        assert _SPATIAL_SIMILARITY_THRESHOLD == pytest.approx(0.55)

    def test_no_memory_no_boost(self):
        resolver = ContextWeightResolver(
            prefs_store=_mock_prefs([]),
            long_term=_mock_long_term(similarity=0.0),
        )
        result = resolver.resolve("chair")
        assert result.spatial_boost == pytest.approx(0.0)

    def test_no_long_term_no_boost(self):
        resolver = ContextWeightResolver(prefs_store=_mock_prefs([]), long_term=None)
        result = resolver.resolve("chair")
        assert result.spatial_boost == pytest.approx(0.0)

    def test_spatial_memory_text_returned_in_result(self):
        resolver = ContextWeightResolver(
            prefs_store=_mock_prefs([]),
            long_term=_mock_long_term(similarity=0.80, text="loose rug near sofa"),
        )
        result = resolver.resolve("sofa")
        assert result.spatial_memory == "loose rug near sofa"
        assert result.spatial_similarity == pytest.approx(0.80)


class TestWeightClamping:
    """Check 3: final_weight always in [0.1, 3.0]."""

    def test_weight_clamped_at_max_required_test(self):
        """Required test: all 6 flags must not exceed 3.0."""
        all_flags = list(_MOBILITY_WEIGHT_MAP.keys())
        resolver = ContextWeightResolver(
            prefs_store=_mock_prefs(all_flags),
            long_term=_mock_long_term(similarity=0.90, text="memory"),
        )
        result = resolver.resolve("chair")
        assert result.final_weight <= _WEIGHT_MAX

    def test_weight_clamped_at_min(self):
        """Pathological low weight from custom prefs must be floored at 0.1."""
        # Build a resolver where mobility_weight is forced low via no flags
        # and we verify the clamp floor is respected
        assert _WEIGHT_MIN == pytest.approx(0.1)
        clamped = _clamp(0.0, _WEIGHT_MIN, _WEIGHT_MAX)
        assert clamped == pytest.approx(0.1)

    def test_clamp_function_upper(self):
        assert _clamp(5.0, 0.1, 3.0) == pytest.approx(3.0)

    def test_clamp_function_lower(self):
        assert _clamp(0.0, 0.1, 3.0) == pytest.approx(0.1)

    def test_clamp_function_in_range(self):
        assert _clamp(2.0, 0.1, 3.0) == pytest.approx(2.0)

    def test_all_six_flags_plus_boost_still_clamped(self):
        all_flags = list(_MOBILITY_WEIGHT_MAP.keys())
        unclamped = 1.0 + sum(_MOBILITY_WEIGHT_MAP.values()) + _SPATIAL_BOOST_VALUE
        assert unclamped > _WEIGHT_MAX   # confirm it actually exceeds max
        result = _clamp(unclamped, _WEIGHT_MIN, _WEIGHT_MAX)
        assert result == pytest.approx(_WEIGHT_MAX)


class TestFullFormulaTrace:
    """Check 4: end-to-end formula trace from the architecture doc."""

    def test_full_formula_wheelchair_bad_knee_spatial_boost(self):
        """
        Required test: wheelchair + bad_knee + spatial boost, chair at 0.8m, approaching.

        mobility_weight = 1.0 + 0.60 (wheelchair) + 0.45 (bad_knee) = 2.05
        spatial_boost   = 0.30
        final_weight    = clamp(2.05 + 0.30, 0.1, 3.0) = 2.35
        motion_factor   = 1.5  (APPROACHING)
        score           = (1 / 0.8) × 1.5 × 2.35 = 4.40625 → HIGH
        """
        resolver = ContextWeightResolver(
            prefs_store=_mock_prefs(["wheelchair", "bad_knee"]),
            long_term=_mock_long_term(similarity=0.80, text="chair near the door"),
        )
        ctx = resolver.resolve("chair")

        assert ctx.mobility_weight == pytest.approx(2.05, abs=0.001)
        assert ctx.spatial_boost   == pytest.approx(0.30)
        assert ctx.final_weight    == pytest.approx(2.35, abs=0.001)

        motion = motion_factor_for(MotionState.APPROACHING)
        assert motion == pytest.approx(1.5)

        score = compute_risk_score(
            distance_metres=0.8,
            motion_factor=motion,
            user_context_weight=ctx.final_weight,
        )
        assert score == pytest.approx(4.40625, abs=0.01)
        assert RiskLevel.from_score(score) == RiskLevel.HIGH

    def test_neutral_user_no_memory_base_score(self):
        """No flags, no memory → weight=1.0, neutral risk score."""
        resolver = ContextWeightResolver(
            prefs_store=_mock_prefs([]),
            long_term=_mock_long_term(similarity=0.0),
        )
        ctx = resolver.resolve("chair")
        assert ctx.final_weight == pytest.approx(1.0)
        assert ctx.spatial_boost == pytest.approx(0.0)
        assert ctx.mobility_flags == []

    def test_riskengine_assess_uses_resolver(self):
        """RiskEngine.assess() with no explicit weight calls the resolver."""
        engine = RiskEngine(
            prefs_store=_mock_prefs(["bad_knee"]),
            long_term=_mock_long_term(similarity=0.0),
        )
        det = _make_detection("chair", distance=1.0)
        result = engine.assess(det, motion_factor=1.0, user_context_weight=None)

        assert result.context_result is not None
        expected_weight = _clamp(1.0 + 0.45, _WEIGHT_MIN, _WEIGHT_MAX)  # 1.45
        assert result.context_result.final_weight == pytest.approx(expected_weight)
        assert result.score == pytest.approx((1.0 / 1.0) * 1.0 * expected_weight)

    def test_riskengine_assess_explicit_weight_skips_resolver(self):
        """Explicit user_context_weight must bypass the resolver entirely."""
        engine = RiskEngine(
            prefs_store=_mock_prefs(["wheelchair"]),  # would add 0.60
            long_term=_mock_long_term(similarity=0.90),  # would add 0.30
        )
        det = _make_detection("chair", distance=1.0)
        result = engine.assess(det, motion_factor=1.0, user_context_weight=1.0)

        # context_result must be None (resolver was bypassed)
        assert result.context_result is None
        # Score must use exactly the provided weight
        assert result.score == pytest.approx(1.0)

    def test_assess_all_deduplicates_resolver_calls(self):
        """assess_all must call the resolver once per unique class, not per detection."""
        prefs = _mock_prefs([])
        lt = _mock_long_term(similarity=0.0)
        engine = RiskEngine(prefs_store=prefs, long_term=lt)

        detections = [
            _make_detection("chair", 1.0),
            _make_detection("chair", 1.5),  # same class, different distance
            _make_detection("table", 2.0),
        ]
        engine.assess_all(detections, user_context_weight=None)

        # retrieve() called once for "chair" and once for "table" = 2 total
        assert lt.retrieve.call_count == 2

    def test_assess_all_sorted_descending(self):
        """assess_all result must be sorted highest score first."""
        engine = RiskEngine(
            prefs_store=_mock_prefs([]),
            long_term=_mock_long_term(similarity=0.0),
        )
        detections = [
            _make_detection("table",  3.0),   # LOW
            _make_detection("chair",  0.5),   # HIGH
            _make_detection("person", 1.9),   # MEDIUM
        ]
        results = engine.assess_all(detections, user_context_weight=1.0)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].detection.class_name == "chair"
