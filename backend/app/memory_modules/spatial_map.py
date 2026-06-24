"""
MemoryNav — Spatial Room Map
backend/app/memory_modules/spatial_map.py

Phase 2 item 5: a live, in-memory world model of what's been seen
and where during the current session.

Structure:
    {
      "Living Room": {
        "chair":  {"position": "left", "distance_m": 1.2, "last_seen": 1234.5,
                   "confidence": 0.91, "sightings": 14},
        "sofa":   {...},
      },
      "Kitchen": {...}
    }

Position is a coarse 3-zone label derived from the bbox center x:
    left   — bbox center in left third of frame
    center — bbox center in middle third of frame
    right  — bbox center in right third of frame

This is enough for the voice layer to say "chair to your left" and for
the demo scenario to confirm that first-pass observations survive into
the second pass via long-term memory.

Thread-safe: one RLock guards all writes. Reads return copies.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ObjectEntry:
    position:    str    # "left" | "center" | "right"
    distance_m:  float  # metres (nan if unknown)
    last_seen:   float  # time.monotonic()
    confidence:  float  # YOLO confidence of latest sighting
    sightings:   int    # total update count this session

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.last_seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position":   self.position,
            "distance_m": round(self.distance_m, 2) if self.distance_m == self.distance_m else None,
            "last_seen_s_ago": round(self.age_s, 1),
            "confidence": round(self.confidence, 3),
            "sightings":  self.sightings,
        }


def _position_label(cx_norm: float) -> str:
    """Normalised [0,1] bbox center x → 'left' / 'center' / 'right'."""
    if cx_norm < 0.33:
        return "left"
    if cx_norm < 0.67:
        return "center"
    return "right"


class SpatialMap:
    """
    Live world model: room → object → last known position + distance.

    One instance per pipeline session (created in init_pipeline).
    Persists for the duration of the session only — not written to disk.
    Use LongTermMemory for cross-session persistence.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # room_name → {class_name → ObjectEntry}
        self._map: Dict[str, Dict[str, ObjectEntry]] = {}
        self._current_room: str = "Unknown"

    # ── Room management ─────────────────────────────────────────────────

    def set_room(self, room_name: str) -> None:
        """Call when the user enters a new room."""
        with self._lock:
            self._current_room = room_name
            if room_name not in self._map:
                self._map[room_name] = {}

    @property
    def current_room(self) -> str:
        return self._current_room

    # ── Update ──────────────────────────────────────────────────────────

    def update(
        self,
        class_name: str,
        bbox: tuple,
        frame_width: int,
        distance_m: float,
        confidence: float,
    ) -> None:
        """
        Record or refresh one object observation in the current room.

        Args:
            class_name  : YOLO class label, e.g. "chair"
            bbox        : (x1, y1, x2, y2) pixel coords
            frame_width : frame width in pixels, for position label
            distance_m  : estimated distance in metres (nan if unknown)
            confidence  : YOLO detection confidence
        """
        x1, _, x2, _ = bbox
        cx_norm = ((x1 + x2) / 2) / max(frame_width, 1)
        position = _position_label(cx_norm)

        with self._lock:
            room = self._map.setdefault(self._current_room, {})
            if class_name in room:
                entry = room[class_name]
                room[class_name] = ObjectEntry(
                    position=position,
                    distance_m=distance_m,
                    last_seen=time.monotonic(),
                    confidence=confidence,
                    sightings=entry.sightings + 1,
                )
            else:
                room[class_name] = ObjectEntry(
                    position=position,
                    distance_m=distance_m,
                    last_seen=time.monotonic(),
                    confidence=confidence,
                    sightings=1,
                )

    # ── Queries ─────────────────────────────────────────────────────────

    def get_room(self, room_name: Optional[str] = None) -> Dict[str, ObjectEntry]:
        """Return all objects seen in a room (default: current room)."""
        with self._lock:
            room = room_name or self._current_room
            return dict(self._map.get(room, {}))

    def get_object(
        self, class_name: str, room_name: Optional[str] = None
    ) -> Optional[ObjectEntry]:
        """Return the latest entry for a specific object, or None."""
        with self._lock:
            room = room_name or self._current_room
            return self._map.get(room, {}).get(class_name)

    def all_rooms(self) -> list[str]:
        with self._lock:
            return list(self._map.keys())

    def snapshot(self) -> Dict[str, Any]:
        """Full map as a JSON-serializable dict."""
        with self._lock:
            return {
                "current_room": self._current_room,
                "rooms": {
                    room: {
                        obj: entry.to_dict()
                        for obj, entry in objects.items()
                    }
                    for room, objects in self._map.items()
                },
            }

    def summary_text(self, room_name: Optional[str] = None) -> str:
        """
        One-line summary for voice output or logs.
        Example: "Living Room: chair (left, 1.2m), sofa (center, 2.0m)"
        """
        room = room_name or self._current_room
        objects = self.get_room(room)
        if not objects:
            return f"{room}: nothing mapped yet"
        parts = []
        for name, entry in sorted(objects.items(), key=lambda x: x[1].distance_m):
            dist = f"{entry.distance_m:.1f}m" if entry.distance_m == entry.distance_m else "?"
            parts.append(f"{name} ({entry.position}, {dist})")
        return f"{room}: " + ", ".join(parts)


if __name__ == "__main__":
    # Smoke test
    m = SpatialMap()
    m.set_room("Kitchen")
    m.update("chair",       (100, 100, 200, 400), 640, 1.2,  0.91)
    m.update("refrigerator",(400, 50,  580, 400), 640, 2.5,  0.87)
    m.update("chair",       (110, 110, 210, 410), 640, 1.1,  0.93)  # second sighting

    chair = m.get_object("chair")
    assert chair is not None
    assert chair.sightings == 2
    assert chair.position == "left"

    m.set_room("Living Room")
    m.update("sofa", (200, 150, 500, 400), 640, 1.8, 0.88)

    print(m.summary_text("Kitchen"))
    print(m.summary_text("Living Room"))
    import json
    print(json.dumps(m.snapshot(), indent=2))
    print("SpatialMap smoke test passed.")
