"""
MemoryNav — ByteTrack Object Tracker
backend/app/perception/tracker.py

Sits between YOLO detection and depth estimation in the pipeline.
Assigns a stable integer track_id to each detection across frames,
so the Risk Engine and Session Store can track the same physical
object instance over time rather than just its class name.

Before tracking:
    Frame N: chair (conf=0.91) — is this the same chair as last frame?
    Frame N: chair (conf=0.87) — unknown, treated as the same class

After tracking:
    Frame N: chair track_id=3 — definitively the same object
    Frame N: chair track_id=7 — a different chair

This enables:
    - Per-instance motion tracking (track_id=3 is approaching, track_id=7 is stationary)
    - Correct false-alert suppression (suppress per instance, not per class)
    - Accurate memory writes (don't write the same chair 30 times)

Interface:
    tracker = ObjectTracker()
    tracked = tracker.update(detections, frame)
    # tracked: list of TrackedDetection with all original fields + track_id

Uses supervision.ByteTrack (wraps the original ByteTrack algorithm).
supervision is a lightweight wrapper — no additional model weights needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import supervision as sv

from app.perception.detector import Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackedDetection:
    """
    A Detection enriched with a stable track_id from ByteTrack.
    Drop-in replacement for Detection everywhere downstream.
    """
    class_id:    int
    class_name:  str
    confidence:  float
    bbox:        Tuple[int, int, int, int]   # (x1, y1, x2, y2) pixel coords
    track_id:    int                          # -1 = not yet confirmed by tracker

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def is_tracked(self) -> bool:
        """True once ByteTrack has confirmed this as a real track."""
        return self.track_id >= 0

    @classmethod
    def from_detection(cls, det: Detection, track_id: int = -1) -> "TrackedDetection":
        return cls(
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=det.bbox,
            track_id=track_id,
        )


class ObjectTracker:
    """
    ByteTrack wrapper for MemoryNav.

    One instance per pipeline session — ByteTrack maintains internal
    state (Kalman filters, lost track buffer) across frames and must
    not be recreated per-frame.

    Parameters tuned for indoor navigation at ~10-30fps:
        track_activation_threshold : 0.25  — matches YOLO_DETECTION_CONFIDENCE
        lost_track_buffer          : 30    — hold lost tracks for 1s at 30fps
        minimum_matching_threshold : 0.8   — tight IoU to avoid ID switches
        frame_rate                 : 30    — expected camera fps
        minimum_consecutive_frames : 2     — confirm track after 2 frames
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 2,
    ) -> None:
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        logger.info(
            "ObjectTracker (ByteTrack) initialised — "
            "activation=%.2f  buffer=%d  match=%.2f  fps=%d  min_frames=%d",
            track_activation_threshold, lost_track_buffer,
            minimum_matching_threshold, frame_rate, minimum_consecutive_frames,
        )

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> List[TrackedDetection]:
        """
        Feed one frame's detections into ByteTrack and return
        TrackedDetection objects with stable track_ids.

        Args:
            detections : Output of Detector.detect() — List[Detection]
            frame      : The BGR frame these detections came from.
                         ByteTrack uses frame dimensions for coordinate
                         normalisation internally.

        Returns:
            List[TrackedDetection], same length as input detections that
            were matched to tracks. Unconfirmed detections (below
            minimum_consecutive_frames) are returned with track_id=-1.
        """
        if not detections:
            # Still need to call update so ByteTrack ages/prunes lost tracks
            empty = sv.Detections.empty()
            self._tracker.update_with_detections(empty)
            return []

        h, w = frame.shape[:2]

        # Build sv.Detections from our Detection dataclass list
        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confs = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=class_ids,
        )

        # ByteTrack updates in-place and returns matched detections with tracker_id set
        tracked_sv = self._tracker.update_with_detections(sv_dets)

        # Build a lookup: bbox → track_id from tracked output
        # ByteTrack may drop or reorder detections, so we match by bbox proximity
        track_map: dict[tuple, int] = {}
        if tracked_sv.tracker_id is not None:
            for i in range(len(tracked_sv)):
                bbox_key = tuple(tracked_sv.xyxy[i].astype(int).tolist())
                tid = int(tracked_sv.tracker_id[i])
                track_map[bbox_key] = tid

        result: List[TrackedDetection] = []
        for det in detections:
            bbox_key = tuple(det.bbox)
            track_id = track_map.get(bbox_key, -1)

            # Fallback: if bbox key not exact match (float rounding), find nearest
            if track_id == -1 and track_map:
                track_id = _nearest_track(det.bbox, track_map)

            result.append(TrackedDetection.from_detection(det, track_id=track_id))

        return result

    def reset(self) -> None:
        """Reset tracker state — call when entering a new room/scene."""
        self._tracker.reset()
        logger.info("ObjectTracker state reset.")


def _nearest_track(
    bbox: Tuple[int, int, int, int],
    track_map: dict[tuple, int],
    max_px_dist: float = 20.0,
) -> int:
    """
    Find the track_id whose bbox center is closest to this bbox's center.
    Returns -1 if no match within max_px_dist pixels.
    Used as a fallback when exact bbox key lookup fails due to float rounding.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    best_tid = -1
    best_dist = max_px_dist

    for (tx1, ty1, tx2, ty2), tid in track_map.items():
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_tid = tid

    return best_tid


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)

    # Smoke test: two chairs approaching from different positions
    tracker = ObjectTracker()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Simulate Detection objects
    @dataclass(frozen=True)
    class _FakeDet:
        class_id: int
        class_name: str
        confidence: float
        bbox: tuple

    def _det(x1, y1, x2, y2, name="chair", cid=56, conf=0.9):
        return _FakeDet(class_id=cid, class_name=name,
                        confidence=conf, bbox=(x1, y1, x2, y2))

    # Frame 1 — two chairs
    dets1 = [_det(100, 100, 200, 200), _det(300, 100, 400, 200)]
    r1 = tracker.update(dets1, dummy_frame)
    print(f"Frame 1: {[(d.class_name, d.track_id) for d in r1]}")

    # Frame 2 — chairs moved slightly (approaching)
    dets2 = [_det(105, 110, 205, 210), _det(295, 95, 395, 195)]
    r2 = tracker.update(dets2, dummy_frame)
    print(f"Frame 2: {[(d.class_name, d.track_id) for d in r2]}")

    # After min_consecutive_frames=2, should have confirmed track_ids
    confirmed = [d for d in r2 if d.track_id >= 0]
    print(f"Confirmed tracks: {len(confirmed)}/2")
    print("Smoke test complete.")
