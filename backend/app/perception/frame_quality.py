"""
MemoryNav — Frame Quality Gate
backend/app/perception/frame_quality.py

Module 1 (Perception Layer): cheap pre-checks that run before the YOLO
detector touches a frame. Catches frames that are too dark, too
blown-out, or too blurry to trust — protecting the Risk Engine from
confident-looking garbage detections, and giving the Voice Interface
something honest to say ("having trouble seeing right now") instead of
silently skipping or hallucinating an obstacle.

Usage:

    from app.perception.frame_quality import check_frame_quality, is_frame_usable

    # Simple gate — exactly what's needed before calling the detector:
    if not is_frame_usable(frame):
        continue  # skip detector this frame

    # Richer result, when you also want *why* it failed (e.g. to drive
    # a spoken warning or a debug overlay):
    result = check_frame_quality(frame)
    if not result.ok:
        print(result.reason)   # e.g. "too_dark"

Dependencies: opencv-python, numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from app.config import settings


class QualityIssue(str, Enum):
    NONE = "none"
    EMPTY_FRAME = "empty_frame"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    TOO_BLURRY = "too_blurry"


@dataclass(frozen=True)
class FrameQualityResult:
    ok: bool
    issue: QualityIssue
    brightness: float       # mean pixel intensity, 0-255
    blur_variance: float    # Laplacian variance; 0.0 if not computed (failed earlier check)

    @property
    def reason(self) -> Optional[str]:
        return None if self.ok else self.issue.value


def _compute_brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity, 0-255. Cheap proxy for ambient light."""
    return float(gray.mean())


def _compute_blur_variance(gray: np.ndarray) -> float:
    """
    Variance of the Laplacian — a standard, cheap blur metric. Sharp
    images have lots of high-frequency edge content (-> high variance);
    blurry/motion-smeared images don't.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_frame_quality(
    frame: np.ndarray,
    min_brightness: Optional[float] = None,
    max_brightness: Optional[float] = None,
    min_blur_variance: Optional[float] = None,
) -> FrameQualityResult:
    """
    Runs brightness + blur checks on a single BGR frame (OpenCV format).

    Brightness thresholds default to settings.FRAME_MIN_BRIGHTNESS
    (backend/app/config.py). `max_brightness` (overexposure / glare,
    e.g. pointing the camera at a bright window) isn't in config.py —
    it's outside the architecture doc's spec, so it's hardcoded here
    with a generous default rather than added as a new global setting.
    Override per-call if you want it stricter.

    Blur threshold defaults to settings.FRAME_MIN_BLUR_VARIANCE.

    Checks brightness before blur: a Laplacian variance reading on a
    near-black frame is meaningless noise, so there's no point computing
    it once brightness has already failed — keeps the common-case
    rejection (dim room) cheap.
    """
    if frame is None or frame.size == 0:
        return FrameQualityResult(
            ok=False, issue=QualityIssue.EMPTY_FRAME, brightness=0.0, blur_variance=0.0
        )

    min_brightness = (
        settings.FRAME_MIN_BRIGHTNESS if min_brightness is None else min_brightness
    )
    min_blur_variance = (
        settings.FRAME_MIN_BLUR_VARIANCE if min_blur_variance is None else min_blur_variance
    )
    max_brightness = 250.0 if max_brightness is None else max_brightness

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = _compute_brightness(gray)

    if brightness < min_brightness:
        return FrameQualityResult(
            ok=False, issue=QualityIssue.TOO_DARK, brightness=brightness, blur_variance=0.0
        )

    if brightness > max_brightness:
        return FrameQualityResult(
            ok=False, issue=QualityIssue.TOO_BRIGHT, brightness=brightness, blur_variance=0.0
        )

    blur_variance = _compute_blur_variance(gray)
    if blur_variance < min_blur_variance:
        return FrameQualityResult(
            ok=False,
            issue=QualityIssue.TOO_BLURRY,
            brightness=brightness,
            blur_variance=blur_variance,
        )

    return FrameQualityResult(
        ok=True, issue=QualityIssue.NONE, brightness=brightness, blur_variance=blur_variance
    )


def is_frame_usable(frame: np.ndarray) -> bool:
    """
    Plain True/False gate, using config.py defaults end-to-end — the
    call this module exists for:

        if not is_frame_usable(frame):
            continue  # skip detector this frame
    """
    return check_frame_quality(frame).ok


if __name__ == "__main__":
    # Quick manual check: `python -m app.perception.frame_quality`
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    print("Black frame:        ", check_frame_quality(black))

    flat_gray = np.full((480, 640, 3), 128, dtype=np.uint8)
    print("Flat gray (blurry): ", check_frame_quality(flat_gray))

    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    print("Random noise (sharp):", check_frame_quality(noisy))

    blown_out = np.full((480, 640, 3), 255, dtype=np.uint8)
    print("Pure white (overexposed):", check_frame_quality(blown_out))