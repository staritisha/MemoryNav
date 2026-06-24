"""
MemoryNav — Short-Term Session Memory
backend/app/memory_modules/short_term.py

Module 3 (Memory System): a plain in-memory, rolling-window record of
the current session — obstacles seen, warnings spoken, and path
context — covering the last settings.SHORT_TERM_MEMORY_WINDOW_SECONDS
(default 30s). Nothing here touches disk; restart the app and it's
gone, by design — this is "what's going on right now," not a durable
record. (For "where did I last see my keys" across sessions, see
long_term.py's ChromaDB store instead.)

This feeds the Risk Engine's motion_factor: comparing an obstacle's
distance now vs. a few seconds ago is what turns "stationary" into
"approaching." See motion_trend() below — it returns a plain string,
not app.risk.engine.MotionState, deliberately: this module doesn't
import app.risk, the same way app.risk.models doesn't import
app.perception. Whatever maps "approaching" -> a risk multiplier is
the Risk Engine's call, not this store's.

Usage:

    from app.memory_modules.short_term import SessionStore

    session = SessionStore()
    session.record_obstacle("chair", distance_metres=1.2, bbox=(100, 100, 300, 400))
    session.record_warning("Chair ahead, two meters", risk_level="MEDIUM")
    session.record_path_event("approaching kitchen doorway")

    trend = session.motion_trend("chair")        # "approaching" / "receding" / "stationary" / "unknown"
    recent = session.recent_obstacles("chair")    # events from the last 30s only

Dependencies: none beyond the standard library.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from app.config import settings

# Distance changes smaller than this are treated as depth-estimation
# jitter, not real motion. Not in config.py — an internal tuning
# constant for this module's heuristic, not a project-wide setting.
_MOTION_NOISE_FLOOR_METRES = 0.10


@dataclass(frozen=True)
class ObstacleEvent:
    timestamp: float  # time.monotonic() when recorded
    class_name: str
    distance_metres: float
    bbox: Tuple[int, int, int, int]

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.timestamp


@dataclass(frozen=True)
class WarningEvent:
    timestamp: float
    text: str
    risk_level: str  # plain string (e.g. "HIGH") — see module docstring on decoupling

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.timestamp


@dataclass(frozen=True)
class PathEvent:
    timestamp: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.timestamp


class SessionStore:
    """
    Rolling-window store for the current navigation session, backed by
    plain Python dicts/deques — no database, nothing persisted.

    Obstacles are tracked keyed by class_name only (e.g. all "chair"
    sightings share one history), not per-instance. Distinguishing two
    chairs in the same frame would need real multi-object tracking
    (SORT/ByteTrack-style bbox association across frames) — out of
    scope for a plain session store; this just records and prunes.

    Thread-safety: one lock guards all three stores. Writes happen from
    the capture loop; reads may happen concurrently from the Alert
    Manager or a future FastAPI debug endpoint.
    """

    def __init__(self, window_seconds: Optional[float] = None) -> None:
        self.window_seconds = (
            window_seconds if window_seconds is not None else settings.SHORT_TERM_MEMORY_WINDOW_SECONDS
        )
        self._lock = threading.Lock()
        self._obstacles: Dict[str, Deque[ObstacleEvent]] = {}
        self._warnings: Deque[WarningEvent] = deque()
        self._path: Deque[PathEvent] = deque()

    # ------------------------------------------------------------------- #
    # Pruning — every write and every read drops anything older than the
    # window, so callers never have to think about staleness themselves.
    # ------------------------------------------------------------------- #
    def _is_expired(self, timestamp: float) -> bool:
        return (time.monotonic() - timestamp) > self.window_seconds

    def _prune(self, events: Deque) -> None:
        while events and self._is_expired(events[0].timestamp):
            events.popleft()

    def _prune_obstacles(self) -> None:
        for events in self._obstacles.values():
            self._prune(events)
        empty_keys = [k for k, events in self._obstacles.items() if not events]
        for k in empty_keys:
            del self._obstacles[k]  # don't let the dict grow forever with empty entries

    # ------------------------------------------------------------------- #
    # Obstacles
    # ------------------------------------------------------------------- #
    def record_obstacle(
        self, class_name: str, distance_metres: float, bbox: Tuple[int, int, int, int]
    ) -> None:
        event = ObstacleEvent(
            timestamp=time.monotonic(),
            class_name=class_name,
            distance_metres=distance_metres,
            bbox=bbox,
        )
        with self._lock:
            self._obstacles.setdefault(class_name, deque()).append(event)
            self._prune(self._obstacles[class_name])

    def record_obstacle_detection(self, detection) -> None:
        """
        Convenience for recording straight from a risk-ready Detection
        — anything with .class_name, .distance_metres, .bbox (duck-typed,
        no import of app.risk needed here, same decoupling pattern
        app.risk.models uses for app.perception detections).
        """
        self.record_obstacle(
            class_name=detection.class_name,
            distance_metres=detection.distance_metres,
            bbox=detection.bbox,
        )

    def recent_obstacles(self, class_name: Optional[str] = None) -> List[ObstacleEvent]:
        """All obstacle events within the window, optionally filtered to one class_name."""
        with self._lock:
            self._prune_obstacles()
            if class_name is not None:
                return list(self._obstacles.get(class_name, []))
            return [event for events in self._obstacles.values() for event in events]

    def motion_trend(self, class_name: str, min_samples: int = 2) -> str:
        """
        Cheap proxy for whether an obstacle is approaching, receding, or
        holding steady, based on distance change across the stored
        window. Returns "approaching" / "receding" / "stationary" /
        "unknown" — "unknown" if there aren't enough samples yet to say.

        This compares the oldest vs. newest sample in the window, not a
        full regression — good enough for a multiplier lookup, not
        intended as a precise velocity estimate.
        """
        events = self.recent_obstacles(class_name)
        if len(events) < min_samples:
            return "unknown"

        events = sorted(events, key=lambda e: e.timestamp)
        delta = events[-1].distance_metres - events[0].distance_metres

        if abs(delta) < _MOTION_NOISE_FLOOR_METRES:
            return "stationary"
        return "receding" if delta > 0 else "approaching"

    # ------------------------------------------------------------------- #
    # Warnings
    # ------------------------------------------------------------------- #
    def record_warning(self, text: str, risk_level: str) -> None:
        event = WarningEvent(timestamp=time.monotonic(), text=text, risk_level=risk_level)
        with self._lock:
            self._warnings.append(event)
            self._prune(self._warnings)

    def recent_warnings(self) -> List[WarningEvent]:
        with self._lock:
            self._prune(self._warnings)
            return list(self._warnings)

    def was_recently_warned(self, text: str, within_seconds: Optional[float] = None) -> bool:
        """
        Convenience for an Alert Manager suppression check: has this
        exact text already been spoken within `within_seconds` (defaults
        to the full session window)? Exact-text match only — fuzzy
        "similar warning" matching belongs to the Alert Manager, not here.
        """
        cutoff = within_seconds if within_seconds is not None else self.window_seconds
        with self._lock:
            self._prune(self._warnings)
            return any(w.text == text and w.age_seconds <= cutoff for w in self._warnings)

    # ------------------------------------------------------------------- #
    # Path
    # ------------------------------------------------------------------- #
    def record_path_event(self, description: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        event = PathEvent(timestamp=time.monotonic(), description=description, metadata=metadata or {})
        with self._lock:
            self._path.append(event)
            self._prune(self._path)

    def recent_path(self) -> List[PathEvent]:
        with self._lock:
            self._prune(self._path)
            return list(self._path)

    # ------------------------------------------------------------------- #
    # Whole-session utilities
    # ------------------------------------------------------------------- #
    def snapshot(self) -> Dict[str, Any]:
        """
        Full session state as plain JSON-serializable dicts — handy for
        a FastAPI debug endpoint or for logging into the evaluation
        pipeline. Reports `age_seconds` per event rather than the raw
        monotonic timestamp, since the latter is meaningless outside
        this process.
        """
        with self._lock:
            self._prune_obstacles()
            self._prune(self._warnings)
            self._prune(self._path)
            return {
                "window_seconds": self.window_seconds,
                "obstacles": {
                    class_name: [
                        {
                            "class_name": e.class_name,
                            "distance_metres": e.distance_metres,
                            "bbox": e.bbox,
                            "age_seconds": round(e.age_seconds, 2),
                        }
                        for e in events
                    ]
                    for class_name, events in self._obstacles.items()
                },
                "warnings": [
                    {
                        "text": w.text,
                        "risk_level": w.risk_level,
                        "age_seconds": round(w.age_seconds, 2),
                    }
                    for w in self._warnings
                ],
                "path": [
                    {
                        "description": p.description,
                        "metadata": p.metadata,
                        "age_seconds": round(p.age_seconds, 2),
                    }
                    for p in self._path
                ],
            }

    def get_recent_summary(self) -> Optional[str]:
        """
        Return a plain-text summary of the most recent obstacles in the
        session window — e.g. "chair at 0.8m, table at 2.3m".

        Returns None when no obstacles have been recorded yet (or all
        have expired from the window).  Used by voice_router.py to
        inject short-term context into the obstacle/scene answer without
        coupling the router to the internal ObstacleEvent structure.
        """
        events = self.recent_obstacles()
        if not events:
            return None
        # Sort most-recent first, deduplicate class names to one entry each
        seen: Dict[str, ObstacleEvent] = {}
        for e in sorted(events, key=lambda ev: ev.timestamp, reverse=True):
            seen.setdefault(e.class_name, e)
        parts = [
            f"{e.class_name} at {e.distance_metres:.1f}m"
            for e in sorted(seen.values(), key=lambda ev: ev.distance_metres)
        ]
        return ", ".join(parts) if parts else None

    def record_voice_query(self, question: str, answer: str) -> None:
        """
        Log a voice interaction into the session path log so the
        temporal suppression layer and future queries within this session
        can see it.

        Used by voice_router.py to close the feedback loop without
        coupling it to the internal PathEvent structure.
        """
        self.record_path_event(
            description=f"voice: {question[:80]}",
            metadata={"answer": answer[:120]},
        )

    def clear(self) -> None:
        """Wipes all three stores — e.g. on explicit session reset."""
        with self._lock:
            self._obstacles.clear()
            self._warnings.clear()
            self._path.clear()


if __name__ == "__main__":
    # Quick manual check: `python -m app.memory_modules.short_term`
    session = SessionStore(window_seconds=2.0)  # short window to demo expiry quickly

    session.record_obstacle("chair", distance_metres=2.0, bbox=(0, 0, 100, 100))
    time.sleep(0.3)
    session.record_obstacle("chair", distance_metres=1.0, bbox=(0, 0, 120, 120))
    print("Trend after approaching chair:", session.motion_trend("chair"))  # expect "approaching"

    session.record_warning("Chair ahead, one meter", risk_level="HIGH")
    print("Was recently warned:", session.was_recently_warned("Chair ahead, one meter"))  # True
    print("Was warned (different text):", session.was_recently_warned("Table ahead"))  # False

    session.record_path_event("approaching kitchen doorway")

    import json

    print(json.dumps(session.snapshot(), indent=2))

    print("Waiting for window to expire...")
    time.sleep(2.2)
    print("Obstacles after expiry:", session.recent_obstacles())  # expect []
    print("Trend after expiry:", session.motion_trend("chair"))  # expect "unknown"