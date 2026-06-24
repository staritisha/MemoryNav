"""
Lightweight spatial room map.
Maintains a dict: {room_label: {object: {"position": str, "last_seen_s": float}}}
Populated from live detections. Retrieved for voice context and memory writes.
"""
import time
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class SpatialObject:
    position: str        # e.g. "left", "center", "right"
    last_seen_s: float   # timestamp
    confidence: float
    distance_m: float


class SpatialMap:
    def __init__(self):
        # {room: {object_name: SpatialObject}}
        self._map: Dict[str, Dict[str, SpatialObject]] = {}
        self._current_room: str = "unknown"

    def set_room(self, room: str) -> None:
        self._current_room = room
        if room not in self._map:
            self._map[room] = {}

    def update(
        self,
        class_name: str,
        bbox: tuple,
        frame_width: int,
        distance_m: float,
        confidence: float,
    ) -> None:
        """Called every frame for each detection."""
        position = self._bbox_to_position(bbox, frame_width)
        room = self._current_room
        if room not in self._map:
            self._map[room] = {}
        self._map[room][class_name] = SpatialObject(
            position=position,
            last_seen_s=time.time(),
            confidence=confidence,
            distance_m=distance_m,
        )

    def get_room_summary(self, room: Optional[str] = None) -> str:
        """
        Returns human-readable summary for voice/memory.
        e.g. "Living Room: chair (left, 1.2m), couch (center, 2.1m)"
        """
        room = room or self._current_room
        if room not in self._map or not self._map[room]:
            return f"{room}: no objects recorded"
        items = []
        for obj, data in self._map[room].items():
            items.append(f"{obj} ({data.position}, {data.distance_m:.1f}m)")
        return f"{room}: {', '.join(items)}"

    def get_map(self) -> Dict:
        """Returns full map as plain dict for JSON serialization."""
        result = {}
        for room, objects in self._map.items():
            result[room] = {}
            for obj, data in objects.items():
                result[room][obj] = {
                    "position": data.position,
                    "distance_m": round(data.distance_m, 2),
                    "confidence": round(data.confidence, 3),
                    "last_seen_s": round(data.last_seen_s, 1),
                }
        return result

    def objects_in_room(self, room: Optional[str] = None) -> Dict[str, SpatialObject]:
        room = room or self._current_room
        return self._map.get(room, {})

    @staticmethod
    def _bbox_to_position(bbox: tuple, frame_width: int) -> str:
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2
        third = frame_width / 3
        if center_x < third:
            return "left"
        elif center_x < 2 * third:
            return "center"
        return "right"
