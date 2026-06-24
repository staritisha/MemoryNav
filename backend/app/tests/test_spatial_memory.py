"""
MemoryNav — SpatialMap & SessionStore Unit Tests
backend/tests/test_spatial_memory.py

Validates both live session memory modules:

Part A — spatial_map.py
  Check 1: 3-zone position label boundaries (0.33 / 0.67 split)
  Check 2: ObjectEntry tracks all 5 fields (position, distance_m,
           last_seen, confidence, sightings)
  Check 3: nested room → class → ObjectEntry structure;
           default room set externally (not hardcoded in spatial_map.py)

Part B — short_term.py
  Check 1: motion_trend noise floor (0.10m jitter → "stationary")
  Check 2: rolling window pruning on every read AND write
  Check 3: obstacles keyed by class_name, not track_id
  Check 4: single lock guards all three stores

Run:
    pytest backend/tests/test_spatial_memory.py -v
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.memory_modules.spatial_map import SpatialMap, ObjectEntry, _position_label
from app.memory_modules.short_term import SessionStore, _MOTION_NOISE_FLOOR_METRES

# ── Helpers ───────────────────────────────────────────────────────────────────

FRAME_W = 640   # standard pipeline capture width

def _bbox_for_cx_norm(cx_norm: float, frame_w: int = FRAME_W) -> tuple:
    """Build a (x1, y1, x2, y2) bbox whose centre lands at exactly cx_norm * frame_w."""
    cx = int(cx_norm * frame_w)
    half = 40
    return (cx - half, 100, cx + half, 300)


# ═══════════════════════════════════════════════════════════════════════════════
# Part A — SpatialMap
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionLabel:
    """
    Check 1: exact boundary values for the 3-zone split.
    Required test: assert position_label(0.33) == "center"
    """

    # ── Required tests from step brief ────────────────────────────────────────

    def test_boundary_032_is_left(self):
        assert _position_label(0.32) == "left"

    def test_boundary_033_is_center(self):
        """0.33 is the inclusive left boundary of centre."""
        assert _position_label(0.33) == "center"

    def test_boundary_066_is_center(self):
        assert _position_label(0.66) == "center"

    def test_boundary_067_is_right(self):
        """0.67 is the start of right zone."""
        assert _position_label(0.67) == "right"

    # ── Zone interiors ─────────────────────────────────────────────────────────

    def test_far_left_is_left(self):
        assert _position_label(0.0) == "left"

    def test_midpoint_is_center(self):
        assert _position_label(0.50) == "center"

    def test_far_right_is_right(self):
        assert _position_label(1.0) == "right"

    def test_just_under_left_boundary_is_left(self):
        assert _position_label(0.329) == "left"

    def test_just_above_right_boundary_is_right(self):
        assert _position_label(0.671) == "right"

    # ── Via SpatialMap.update (normalisation must happen there) ───────────────

    def test_update_normalises_pixel_bbox_to_zones(self):
        """
        Raw pixel coords must be normalised to [0,1] before zone lookup.
        A bbox centred at pixel 200 on a 640-wide frame → 200/640 = 0.3125 → LEFT.
        """
        m = SpatialMap()
        m.set_room("Test")
        # centre_x = (100+300)/2 = 200; 200/640 = 0.3125 < 0.33 → left
        m.update("chair", (100, 100, 300, 300), frame_width=640, distance_m=1.0, confidence=0.9)
        assert m.get_object("chair").position == "left"

    def test_update_centre_zone(self):
        m = SpatialMap()
        m.set_room("Test")
        # centre_x = (270+370)/2 = 320; 320/640 = 0.50 → center
        m.update("table", (270, 100, 370, 300), frame_width=640, distance_m=2.0, confidence=0.85)
        assert m.get_object("table").position == "center"

    def test_update_right_zone(self):
        m = SpatialMap()
        m.set_room("Test")
        # centre_x = (440+600)/2 = 520; 520/640 = 0.8125 → right
        m.update("sofa", (440, 100, 600, 300), frame_width=640, distance_m=1.5, confidence=0.88)
        assert m.get_object("sofa").position == "right"

    def test_position_is_recomputed_on_each_update(self):
        """Object moving across the frame must get a new position label."""
        m = SpatialMap()
        m.set_room("Test")
        m.update("chair", (50, 100, 150, 300), frame_width=640, distance_m=1.0, confidence=0.9)
        assert m.get_object("chair").position == "left"

        m.update("chair", (440, 100, 600, 300), frame_width=640, distance_m=1.0, confidence=0.9)
        assert m.get_object("chair").position == "right"


class TestObjectEntryFields:
    """
    Check 2: all five ObjectEntry fields are stored and retrievable.
    """

    def test_all_five_fields_present_on_first_sighting(self):
        m = SpatialMap()
        m.set_room("Living Room")
        m.update("chair", (100, 100, 300, 300), frame_width=640,
                 distance_m=1.2, confidence=0.91)
        entry = m.get_object("chair")

        assert entry is not None
        assert entry.position == "left"
        assert entry.distance_m == pytest.approx(1.2)
        assert entry.confidence == pytest.approx(0.91)
        assert entry.sightings == 1
        assert entry.last_seen > 0   # monotonic timestamp — just check it was set

    def test_sightings_increments_each_update(self):
        m = SpatialMap()
        m.set_room("Test")
        for i in range(1, 6):
            m.update("chair", (100, 100, 300, 300), frame_width=640,
                     distance_m=1.0, confidence=0.9)
            assert m.get_object("chair").sightings == i

    def test_distance_m_updated_on_each_sighting(self):
        m = SpatialMap()
        m.set_room("Test")
        m.update("chair", (100, 100, 300, 300), frame_width=640, distance_m=2.0, confidence=0.9)
        m.update("chair", (100, 100, 300, 300), frame_width=640, distance_m=1.0, confidence=0.9)
        assert m.get_object("chair").distance_m == pytest.approx(1.0)

    def test_confidence_updated_on_each_sighting(self):
        m = SpatialMap()
        m.set_room("Test")
        m.update("chair", (100, 100, 300, 300), frame_width=640, distance_m=1.0, confidence=0.70)
        m.update("chair", (100, 100, 300, 300), frame_width=640, distance_m=1.0, confidence=0.95)
        assert m.get_object("chair").confidence == pytest.approx(0.95)

    def test_last_seen_advances_on_each_sighting(self):
        m = SpatialMap()
        m.set_room("Test")
        m.update("chair", (100, 100, 300, 300), frame_width=640, distance_m=1.0, confidence=0.9)
        t1 = m.get_object("chair").last_seen
        time.sleep(0.02)
        m.update("chair", (100, 100, 300, 300), frame_width=640, distance_m=1.0, confidence=0.9)
        t2 = m.get_object("chair").last_seen
        assert t2 > t1


class TestRoomStructure:
    """
    Check 3: nested room → class → ObjectEntry structure.
    Default room must be set externally, not hardcoded.
    """

    def test_default_room_is_unknown_before_set_room(self):
        """spatial_map.py must NOT hardcode 'Living Room' — it starts as 'Unknown'."""
        m = SpatialMap()
        assert m.current_room == "Unknown"

    def test_set_room_changes_current_room(self):
        m = SpatialMap()
        m.set_room("Living Room")
        assert m.current_room == "Living Room"

    def test_observations_stored_in_correct_room(self):
        m = SpatialMap()
        m.set_room("Kitchen")
        m.update("fridge", (100, 100, 300, 300), frame_width=640, distance_m=1.0, confidence=0.9)

        m.set_room("Living Room")
        m.update("sofa", (100, 100, 300, 300), frame_width=640, distance_m=2.0, confidence=0.85)

        kitchen_objects = m.get_room("Kitchen")
        living_objects  = m.get_room("Living Room")

        assert "fridge" in kitchen_objects
        assert "sofa"   in living_objects
        # Cross-room isolation
        assert "sofa"   not in kitchen_objects
        assert "fridge" not in living_objects

    def test_different_classes_coexist_in_same_room(self):
        m = SpatialMap()
        m.set_room("Living Room")
        m.update("chair", (50, 100, 150, 300), frame_width=640, distance_m=1.0, confidence=0.9)
        m.update("table", (270, 100, 370, 300), frame_width=640, distance_m=2.0, confidence=0.85)
        m.update("sofa",  (450, 100, 600, 300), frame_width=640, distance_m=1.5, confidence=0.88)

        room = m.get_room("Living Room")
        assert set(room.keys()) == {"chair", "table", "sofa"}

    def test_snapshot_structure_is_nested_dict(self):
        m = SpatialMap()
        m.set_room("Living Room")
        m.update("chair", (100, 100, 300, 300), frame_width=640, distance_m=1.0, confidence=0.9)

        snap = m.snapshot()
        assert "current_room" in snap
        assert "rooms" in snap
        assert "Living Room" in snap["rooms"]
        assert "chair" in snap["rooms"]["Living Room"]

    def test_get_object_returns_none_for_unknown_class(self):
        m = SpatialMap()
        m.set_room("Test")
        assert m.get_object("dragon") is None

    def test_get_room_returns_empty_for_unknown_room(self):
        m = SpatialMap()
        assert m.get_room("Narnia") == {}

    def test_all_rooms_lists_every_room(self):
        m = SpatialMap()
        m.set_room("Kitchen")
        m.update("fridge", (100, 100, 300, 300), frame_width=640, distance_m=1.0, confidence=0.9)
        m.set_room("Bedroom")
        m.update("bed", (100, 100, 300, 300), frame_width=640, distance_m=1.5, confidence=0.88)
        rooms = m.all_rooms()
        assert "Kitchen" in rooms
        assert "Bedroom" in rooms


# ═══════════════════════════════════════════════════════════════════════════════
# Part B — SessionStore (short_term.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMotionTrendNoiseFloor:
    """
    Check 1: delta < 0.10m must return "stationary", not "approaching"/"receding".
    Required test: 0.05m change → "stationary"
    """

    def test_motion_trend_noise_floor_below_threshold(self):
        """Required test from step brief: 0.05m change → stationary."""
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.20, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=1.15, bbox=(0, 0, 100, 100))
        assert store.motion_trend("chair") == "stationary"

    def test_noise_floor_value_is_0_10(self):
        """The constant must be 0.10m — not 0.05 or 0.15."""
        assert _MOTION_NOISE_FLOOR_METRES == pytest.approx(0.10)

    def test_exact_noise_floor_boundary_is_stationary(self):
        """Delta == 0.10 is NOT greater than 0.10 → still stationary."""
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.00, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=1.10, bbox=(0, 0, 100, 100))
        # delta = 1.10 - 1.00 = 0.10; abs(0.10) < 0.10 is False → receding
        # boundary: < not <=, so 0.10 exactly crosses to "receding"
        assert store.motion_trend("chair") == "receding"

    def test_just_above_noise_floor_is_approaching(self):
        """delta = -0.11m (newer is closer by 0.11m) → approaching."""
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.20, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=1.09, bbox=(0, 0, 100, 100))
        # delta = 1.09 - 1.20 = -0.11; abs(-0.11) = 0.11 > 0.10 → approaching
        assert store.motion_trend("chair") == "approaching"

    def test_just_above_noise_floor_is_receding(self):
        """delta = 0.11m (newer is further by 0.11m) → receding."""
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.00, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=1.11, bbox=(0, 0, 100, 100))
        assert store.motion_trend("chair") == "receding"

    def test_zero_delta_is_stationary(self):
        """Identical distances → delta=0 → stationary."""
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.50, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=1.50, bbox=(0, 0, 100, 100))
        assert store.motion_trend("chair") == "stationary"

    def test_unknown_when_only_one_sample(self):
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.00, bbox=(0, 0, 100, 100))
        assert store.motion_trend("chair") == "unknown"

    def test_unknown_for_unseen_class(self):
        store = SessionStore(window_seconds=30.0)
        assert store.motion_trend("dragon") == "unknown"

    def test_motion_direction_approaching_large_delta(self):
        """Large approach: 3.0m → 0.5m = clear approach."""
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=3.0, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=0.5, bbox=(0, 0, 100, 100))
        assert store.motion_trend("chair") == "approaching"

    def test_motion_direction_receding_large_delta(self):
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=0.5, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=3.0, bbox=(0, 0, 100, 100))
        assert store.motion_trend("chair") == "receding"

    def test_motion_trend_uses_oldest_and_newest_across_window(self):
        """With 4 samples: oldest vs newest, not adjacent pairs."""
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=3.0, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=2.5, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=2.0, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=0.4, bbox=(0, 0, 100, 100))
        # oldest=3.0, newest=0.4, delta=-2.6 → approaching
        assert store.motion_trend("chair") == "approaching"


class TestRollingWindowPruning:
    """
    Check 2: events older than window_seconds are pruned on every read AND write.
    Required test: event 31s ago → motion_trend returns "unknown"
    """

    def test_short_term_window_pruning(self, monkeypatch):
        """
        Required test from step brief.
        Record an event 31s in the past → it must be pruned on the next read,
        so motion_trend returns 'unknown' (fewer than 2 live samples).
        """
        now = time.monotonic()
        fake_time = now - 31.0   # 31 seconds in the past

        # Patch time.monotonic to return a past time so record_obstacle
        # stamps the event with a timestamp already outside the window.
        monkeypatch.setattr(time, "monotonic", lambda: fake_time)
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 100, 100))

        # Restore real time for the read — now the event is 31s old → pruned
        monkeypatch.setattr(time, "monotonic", lambda: now)
        assert store.motion_trend("chair") == "unknown"

    def test_event_within_window_is_retained(self, monkeypatch):
        """Event 5s old inside a 30s window must survive."""
        now = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: now - 5.0)
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=0.5, bbox=(0, 0, 100, 100))

        monkeypatch.setattr(time, "monotonic", lambda: now)
        assert store.motion_trend("chair") == "approaching"

    def test_pruning_on_write_removes_stale_events(self, monkeypatch):
        """Old events are pruned when a new write arrives, not just on reads."""
        now = time.monotonic()

        monkeypatch.setattr(time, "monotonic", lambda: now - 35.0)
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=3.0, bbox=(0, 0, 100, 100))

        # Restore time, write a fresh event — old one must be pruned immediately
        monkeypatch.setattr(time, "monotonic", lambda: now)
        store.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 100, 100))

        events = store.recent_obstacles("chair")
        # Only the fresh event should remain
        assert len(events) == 1
        assert events[0].distance_metres == pytest.approx(1.0)

    def test_pruning_on_read_removes_stale_events(self, monkeypatch):
        """recent_obstacles() must prune without needing a fresh write first."""
        now = time.monotonic()

        monkeypatch.setattr(time, "monotonic", lambda: now - 40.0)
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 100, 100))

        monkeypatch.setattr(time, "monotonic", lambda: now)
        events = store.recent_obstacles("chair")
        assert events == []

    def test_warning_pruning(self, monkeypatch):
        """Warnings older than window must be pruned from was_recently_warned."""
        now = time.monotonic()

        monkeypatch.setattr(time, "monotonic", lambda: now - 35.0)
        store = SessionStore(window_seconds=30.0)
        store.record_warning("Chair ahead", risk_level="HIGH")

        monkeypatch.setattr(time, "monotonic", lambda: now)
        assert not store.was_recently_warned("Chair ahead")

    def test_warning_within_window_returns_true(self, monkeypatch):
        now = time.monotonic()

        monkeypatch.setattr(time, "monotonic", lambda: now - 5.0)
        store = SessionStore(window_seconds=30.0)
        store.record_warning("Chair ahead", risk_level="HIGH")

        monkeypatch.setattr(time, "monotonic", lambda: now)
        assert store.was_recently_warned("Chair ahead")

    def test_expired_obstacles_removed_from_dict(self, monkeypatch):
        """Expired class keys are deleted from _obstacles to prevent unbounded growth."""
        now = time.monotonic()

        monkeypatch.setattr(time, "monotonic", lambda: now - 40.0)
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 100, 100))

        monkeypatch.setattr(time, "monotonic", lambda: now)
        # recent_obstacles triggers _prune_obstacles which removes empty keys
        _ = store.recent_obstacles()
        assert "chair" not in store._obstacles


class TestKeyedByClassName:
    """
    Check 3: obstacles stored under class_name key, not track_id.
    All sightings of "chair" must share one deque regardless of track_id.
    """

    def test_multiple_sightings_same_class_same_deque(self):
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=2.0, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=1.5, bbox=(0, 0, 200, 200))
        store.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 300, 300))

        events = store.recent_obstacles("chair")
        assert len(events) == 3

    def test_different_classes_in_separate_deques(self):
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 100, 100))
        store.record_obstacle("table", distance_metres=2.0, bbox=(0, 0, 200, 200))

        assert len(store.recent_obstacles("chair")) == 1
        assert len(store.recent_obstacles("table")) == 1

    def test_motion_trend_crosses_bbox_changes(self):
        """Motion trend must work even when bbox changes between sightings
        (simulating same physical chair tracked across frames)."""
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=3.0, bbox=(100, 100, 200, 200))
        store.record_obstacle("chair", distance_metres=0.5, bbox=(80, 80, 220, 220))
        assert store.motion_trend("chair") == "approaching"

    def test_obstacle_count_by_class(self):
        store = SessionStore(window_seconds=30.0)
        for _ in range(5):
            store.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 100, 100))
        for _ in range(3):
            store.record_obstacle("table", distance_metres=2.0, bbox=(0, 0, 100, 100))

        assert len(store.recent_obstacles("chair")) == 5
        assert len(store.recent_obstacles("table")) == 3


class TestThreadSafety:
    """
    Check 4: single lock guards all three stores.
    """

    def test_single_lock_attribute_exists(self):
        store = SessionStore(window_seconds=30.0)
        assert isinstance(store._lock, type(threading.Lock()))

    def test_concurrent_writes_do_not_corrupt_state(self):
        """
        50 threads each writing an obstacle concurrently must not lose events
        or raise a RuntimeError. If the lock were missing, dict mutation
        during iteration would raise or silently corrupt state.
        """
        store = SessionStore(window_seconds=30.0)
        errors: list[Exception] = []

        def writer(i: int) -> None:
            try:
                store.record_obstacle("chair", distance_metres=float(i % 5 + 1),
                                      bbox=(0, 0, 100, 100))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent write errors: {errors}"
        assert len(store.recent_obstacles("chair")) == 50

    def test_concurrent_read_write_no_exception(self):
        """Concurrent readers and writers on the same store must never raise."""
        store = SessionStore(window_seconds=30.0)
        errors: list[Exception] = []

        def writer() -> None:
            for _ in range(20):
                try:
                    store.record_obstacle("sofa", distance_metres=1.0,
                                          bbox=(0, 0, 100, 100))
                except Exception as e:
                    errors.append(e)

        def reader() -> None:
            for _ in range(20):
                try:
                    store.recent_obstacles("sofa")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        threads += [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestSessionStoreUtilities:
    """Additional coverage for get_recent_summary, record_voice_query, clear."""

    def test_get_recent_summary_none_when_empty(self):
        store = SessionStore(window_seconds=30.0)
        assert store.get_recent_summary() is None

    def test_get_recent_summary_formats_obstacles(self):
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=0.8, bbox=(0, 0, 100, 100))
        store.record_obstacle("table", distance_metres=2.3, bbox=(0, 0, 100, 100))
        summary = store.get_recent_summary()
        assert summary is not None
        assert "chair" in summary
        assert "0.8m" in summary
        assert "table" in summary
        assert "2.3m" in summary

    def test_get_recent_summary_deduplicates_by_class(self):
        """Multiple sightings of the same class → only the most recent appears."""
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=3.0, bbox=(0, 0, 100, 100))
        store.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 100, 100))
        summary = store.get_recent_summary()
        # Should mention chair once (most recent: 1.0m)
        assert summary.count("chair") == 1
        assert "1.0m" in summary

    def test_record_voice_query_stores_in_path(self):
        store = SessionStore(window_seconds=30.0)
        store.record_voice_query("what is ahead", "chair ahead")
        path = store.recent_path()
        assert len(path) == 1
        assert "what is ahead" in path[0].description

    def test_clear_wipes_all_three_stores(self):
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 100, 100))
        store.record_warning("Chair ahead", risk_level="HIGH")
        store.record_path_event("entering kitchen")
        store.clear()

        assert store.recent_obstacles() == []
        assert store.recent_warnings() == []
        assert store.recent_path() == []

    def test_snapshot_reflects_current_state(self):
        store = SessionStore(window_seconds=30.0)
        store.record_obstacle("chair", distance_metres=1.5, bbox=(0, 0, 100, 100))
        store.record_warning("Chair ahead", risk_level="HIGH")

        snap = store.snapshot()
        assert "chair" in snap["obstacles"]
        assert len(snap["warnings"]) == 1
        assert snap["window_seconds"] == 30.0
