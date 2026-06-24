"""
MemoryNav — Risk Engine Data Models
backend/app/risk/models.py

Module 2 (Risk Engine): shared data structures for distance-aware risk
scoring. Deliberately decoupled from app.perception — this module only
needs class name, confidence, bounding box, and distance, not anything
about how a detection was produced. That keeps the Risk Engine testable
in isolation (mock Detection instances, no model loading required) and
avoids a hard import dependency between the risk and perception
packages.

Usage:

    from app.risk.models import Detection, RiskLevel

    d = Detection(class_name="chair", confidence=0.91,
                  bbox=(120, 200, 340, 480), distance_metres=0.5)

    # Or convert straight from a Detector output once distance is known
    # (e.g. via DepthEstimator.depth_at_bbox):
    d = Detection.from_perception(perception_detection, distance_metres=0.5)

    level = RiskLevel.from_score(0.85)   # -> RiskLevel.HIGH
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from app.config import settings


class RiskLevel(str, Enum):
    """
    Three-tier risk classification (architecture doc, Module 2):

        HIGH    score > settings.RISK_HIGH_THRESHOLD               -> interrupt immediately
        MEDIUM  settings.RISK_MEDIUM_THRESHOLD < score <= HIGH      -> queue, don't interrupt speech
        LOW     score <= settings.RISK_MEDIUM_THRESHOLD             -> log only, no speech

    Inherits from str so it serializes cleanly to JSON (FastAPI
    responses, SQLite, ChromaDB metadata) without a custom encoder.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Maps a raw risk score to a level using the thresholds in config.py."""
        if score > settings.RISK_HIGH_THRESHOLD:
            return cls.HIGH
        if score > settings.RISK_MEDIUM_THRESHOLD:
            return cls.MEDIUM
        return cls.LOW


@dataclass(frozen=True)
class Detection:
    """
    A single object detection enriched with distance — the Risk
    Engine's unit of input. Mirrors app.perception.detector.Detection's
    core fields (class_name, confidence, bbox) plus distance_metres,
    which the perception-layer Detection doesn't carry on its own
    (distance comes from a separate DepthEstimator pass).
    """

    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords, top-left origin
    distance_metres: float

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @classmethod
    def from_perception(cls, detection, distance_metres: float) -> "Detection":
        """
        Builds a risk-ready Detection from a perception-layer detection
        — anything with .class_name, .confidence, .bbox (duck-typed
        rather than importing app.perception.detector.Detection, to
        keep this module dependency-free) — plus a distance value, e.g.
        from DepthEstimator.depth_at_bbox().
        """
        return cls(
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
            distance_metres=distance_metres,
        )


if __name__ == "__main__":
    # Quick manual check: `python -m app.risk.models`
    print(RiskLevel.from_score(0.9))  # RiskLevel.HIGH
    print(RiskLevel.from_score(0.5))  # RiskLevel.MEDIUM
    print(RiskLevel.from_score(0.1))  # RiskLevel.LOW

    d = Detection(
        class_name="chair", confidence=0.91, bbox=(120, 200, 340, 480), distance_metres=0.5
    )
    print(d, "| center:", d.center)