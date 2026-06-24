"""
MemoryNav — Depth-Anything Wrapper
backend/app/perception/depth.py

Module 1 (Perception Layer): monocular distance estimation — no LiDAR
needed. Thin wrapper around a HuggingFace Depth-Anything checkpoint
that loads onto the configured device and returns a raw depth map
array sized to match the input frame, so it lines up pixel-for-pixel
with Detector bounding boxes.

IMPORTANT — relative vs. metric depth:
Depth-Anything's *base* checkpoints (the "-hf" models, e.g. the
LiheYoung/depth-anything-small-hf that settings.DEPTH_MODEL_NAME points
to out of the box) output RELATIVE inverse depth — larger values mean
closer, but the numbers are NOT metres. The Risk Engine's formula
(Risk = 1 / distance_metres) needs real metric distance to be
physically meaningful. For genuine metre-scale output, point
DEPTH_MODEL_NAME at a metric checkpoint instead, e.g.:

    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

(fine-tuned on NYU Depth V2 — a strong match for this project's indoor
use case). This wrapper exposes `.is_metric` so downstream code (the
Risk Engine) can branch on which kind of model is loaded instead of
silently assuming meters.

Usage:

    from app.perception.depth import DepthEstimator

    depth_estimator = DepthEstimator()
    depth_map = depth_estimator.estimate(frame)   # frame: BGR np.ndarray (OpenCV)
    # depth_map.shape == frame.shape[:2]; larger value = closer

Dependencies: transformers, torch, pillow, numpy.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from app.config import settings

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Depth-Anything wrapper. Loads once, reused across frames.

    Like Detector, not guaranteed safe for concurrent .estimate() calls
    across threads — use one instance per worker/process.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or settings.DEPTH_MODEL_NAME
        self.device = device or settings.device
        self.is_metric = "metric" in self.model_name.lower()

        logger.info(
            "Loading Depth-Anything model '%s' on device '%s' (metric=%s)",
            self.model_name,
            self.device,
            self.is_metric,
        )
        if not self.is_metric:
            logger.warning(
                "Depth model '%s' outputs RELATIVE depth, not meters. "
                "Risk Engine distance math will be wrong unless it's aware "
                "of this. See depth.py module docstring for a metric "
                "checkpoint alternative.",
                self.model_name,
            )

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        self._warm_up()

    def _warm_up(self) -> None:
        """One dummy inference so the first real frame isn't slowed down
        by lazy MPS/CUDA kernel compilation."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.estimate(dummy)
            logger.info("DepthEstimator warm-up complete.")
        except Exception:
            logger.warning("DepthEstimator warm-up failed; continuing anyway.", exc_info=True)

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on a single BGR frame (OpenCV format).

        Returns a float32 array shaped (height, width) — same spatial
        size as the input frame, so depth_map[y, x] corresponds directly
        to pixel (x, y) in `frame` and to Detector bounding boxes.
        Larger values mean closer (inverse-depth convention); see the
        module docstring for the relative-vs-metric caveat.
        """
        if frame is None or frame.size == 0:
            logger.warning("estimate() called with an empty frame; returning an empty array.")
            return np.zeros((0, 0), dtype=np.float32)

        h, w = frame.shape[:2]
        rgb_frame = frame[:, :, ::-1]  # BGR (OpenCV) -> RGB (PIL/HF expect this)
        image = Image.fromarray(rgb_frame)

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, H', W'), lower-res than input

        # Upsample back to the original frame's resolution so it lines up
        # pixel-for-pixel with Detector's bounding boxes.
        resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        depth_map = resized.squeeze().detach().cpu().numpy().astype(np.float32)
        return depth_map

    def depth_at_bbox(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Convenience for the Risk Engine: median depth value inside a
        Detection's bounding box. Median rather than mean or
        center-pixel — robust against partial occlusion and background
        bleeding in near the box edges.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return float("nan")
        region = depth_map[y1:y2, x1:x2]
        return float(np.median(region))

    def visualize(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalizes a depth map to a uint8 grayscale image (0-255) for
        debugging/demo purposes — e.g. an overlay alongside Detector's
        annotate() output. Not used by the Risk Engine; display only.
        """
        if depth_map.size == 0:
            return depth_map.astype(np.uint8)
        d_min, d_max = float(depth_map.min()), float(depth_map.max())
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth_map, dtype=np.uint8)
        normalized = (depth_map - d_min) / (d_max - d_min) * 255.0
        return normalized.astype(np.uint8)


if __name__ == "__main__":
    # Quick manual check: `python -m app.perception.depth`
    logging.basicConfig(level=logging.INFO)
    depth_estimator = DepthEstimator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_map = depth_estimator.estimate(dummy_frame)
    print(
        f"DepthEstimator loaded OK on '{depth_estimator.device}'. "
        f"Output shape: {depth_map.shape}, dtype: {depth_map.dtype}, "
        f"metric: {depth_estimator.is_metric}"
    )