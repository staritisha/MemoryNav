"""
MemoryNav — YOLOv8-nano Detector Wrapper
backend/app/perception/detector.py

Module 1 (Perception Layer): real-time object detection. Thin wrapper
around Ultralytics YOLOv8-nano that loads onto the configured device
(MPS on Apple Silicon, CUDA, or CPU) and returns plain `Detection`
objects — no Ultralytics-specific types leak past this file, so
downstream modules (Risk Engine, Memory, Alert Manager) don't need to
know anything about the underlying detection library.

Changelog
---------
Roadmap item 20 (Phase 4): the detector now routes its raw detections
through the Confidence Gate (app.alerts.confidence_gate, item 17)
before anything reaches the Risk Engine. See detect_gated() below and
the updated pipeline diagram.

Updated pipeline
-----------------
    camera frame
        ↓
    Detector.detect()            ← unchanged, still returns ALL detections
        ↓
    ConfidenceGate.filter_frame()    ← NEW: detect_gated() does this for you
        ↓ PASS                  ↓ UNCERTAIN           ↓ DISCARD
    Risk Engine              vague speech only      silent drop
    Alert Manager            class name NEVER
    Voice (names object)     spoken

    NOTE — distance_metres gap:
    GateResult.detection below is still this module's own `Detection`
    (class_id, class_name, confidence, bbox — no distance). The Risk
    Engine's RiskAssessment needs distance_metres, which is added by
    whatever depth-estimation step sits between "PASS" and the Risk
    Engine (Phase 2 — Depth-Anything). That module isn't part of this
    file, so detect_gated() does NOT attempt to build a RiskAssessment
    itself; it stops at "these are the detections worth scoring."
    Confirm your depth step consumes GatedDetections.passed (or
    .detections_for_risk_engine) and produces app.risk.models.Detection
    objects with distance_metres populated before calling
    RiskEngine.assess() on them.

Usage
-----
    from app.perception.detector import Detector

    detector = Detector()

    # Raw, ungated detections (debug tooling, annotate(), unit tests):
    detections = detector.detect(frame)

    # Gated pipeline entry point — use this in the main loop:
    gated = detector.detect_gated(frame)
    for result in gated.uncertain:
        voice.speak(result.speech_text)          # vague warning only
    for detection in gated.detections_for_risk_engine:
        ...                                       # → depth → Risk Engine

Dependencies: ultralytics, numpy, torch (and opencv-python, only if you
use the optional `annotate()` debug helper).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from app.alerts.confidence_gate import ConfidenceGate, GateResult
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """A single object detection, decoupled from any specific detection library."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords, top-left origin

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def is_confident(self) -> bool:
        """
        True once confidence clears the voice-layer gate (Module 5,
        settings.YOLO_CONFIDENCE_GATE — default 60%).

        Kept for the annotate() debug view, which just needs a quick
        boolean for box color and doesn't need direction hints or
        PASS/UNCERTAIN/DISCARD granularity. For anything that decides
        what the system is allowed to *say*, use detect_gated() /
        ConfidenceGate instead of this property — that's the single
        source of truth for the pipeline now.
        """
        return self.confidence >= settings.YOLO_CONFIDENCE_GATE


@dataclass(frozen=True)
class GatedDetections:
    """
    One frame's detections, already split by the Confidence Gate.

    Attributes
    ----------
    passed    : Confident enough to name — forward to depth estimation
                then the Risk Engine. Alert Manager will speak the
                class name for these once risk is assessed.
    uncertain : Below the naming threshold. Speak result.speech_text
                directly (a vague, direction-hinted warning). These
                never reach the Risk Engine with a named class —
                that would be hallucination.
    discarded : Too noisy to be worth voicing at all. Already dropped;
                kept here only so callers can log ablation-study
                metrics if they want them.
    """

    passed:    List[GateResult]
    uncertain: List[GateResult]
    discarded: List[GateResult]

    @property
    def detections_for_risk_engine(self) -> List[Detection]:
        """
        Convenience accessor: just the underlying Detection objects
        that passed the gate, unwrapped from their GateResult.

        Remember these still need distance_metres added by the depth
        step before RiskEngine.assess() can score them — see the
        module docstring's "distance_metres gap" note.
        """
        return [result.detection for result in self.passed]

    @property
    def has_anything_to_report(self) -> bool:
        """True if there's at least one PASS or UNCERTAIN detection this frame."""
        return bool(self.passed or self.uncertain)


class Detector:
    """
    YOLOv8-nano wrapper. Load once, reuse across frames.

    Thread/process note: a single Detector instance is not guaranteed
    safe for concurrent .detect() calls from multiple threads —
    Ultralytics models share internal state during inference. Use one
    Detector per worker/process, not one shared across threads calling
    detect() concurrently.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        confidence_gate: Optional[ConfidenceGate] = None,
    ) -> None:
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.device = device or settings.device
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.YOLO_DETECTION_CONFIDENCE
        )

        # Confidence Gate (item 17 / 20). Defaults its PASS threshold to
        # settings.YOLO_CONFIDENCE_GATE so there is one source of truth
        # for "0.6" instead of two (this file's old is_confident property
        # and confidence_gate.py's own CONFIDENCE_THRESHOLD constant).
        # Pass an existing ConfidenceGate instance in if you want shared
        # ablation-study counters across multiple Detector instances.
        self.confidence_gate = confidence_gate or ConfidenceGate(
            confidence_threshold=settings.YOLO_CONFIDENCE_GATE
        )

        logger.info(
            "Loading YOLOv8 model '%s' on device '%s' (conf >= %.2f, gate >= %.2f)",
            self.model_path,
            self.device,
            self.confidence_threshold,
            self.confidence_gate.confidence_threshold,
        )
        self._model = YOLO(self.model_path)
        self._model.to(self.device)
        self._class_names: dict = self._model.names

        self._warm_up()

    def _warm_up(self) -> None:
        """
        Runs one dummy inference so the first real frame isn't slowed
        down by lazy MPS/CUDA kernel compilation. Cheap insurance
        toward the sub-200ms voice response latency target.
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self._model.predict(
                source=dummy,
                device=self.device,
                conf=self.confidence_threshold,
                verbose=False,
            )
            logger.info("Detector warm-up complete.")
        except Exception:
            # Warm-up is an optimization, not a correctness requirement —
            # don't let a warm-up failure block startup.
            logger.warning("Detector warm-up failed; continuing anyway.", exc_info=True)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame (as produced by OpenCV).

        Filtering happens at the inference-time threshold
        (settings.YOLO_DETECTION_CONFIDENCE). This is intentionally
        looser than the Confidence Gate (settings.YOLO_CONFIDENCE_GATE)
        — it returns everything YOLO thinks might be an object so the
        gate has something to evaluate. Use detect_gated() if you want
        the gated pipeline split instead of the raw list.

        Returns detections sorted by confidence, descending.
        """
        if frame is None or frame.size == 0:
            logger.warning("detect() called with an empty frame; returning no detections.")
            return []

        results = self._model.predict(
            source=frame,
            device=self.device,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, f"class_{class_id}"),
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_gated(
        self,
        frame: np.ndarray,
        frame_width: Optional[int] = None,
    ) -> GatedDetections:
        """
        Run detection, then immediately route the results through the
        Confidence Gate (item 17) before anything reaches the Risk
        Engine. This is the entry point the main pipeline loop should
        use — see the module docstring's "Updated pipeline" diagram.

        Args:
            frame       : BGR frame, same as detect().
            frame_width : Pixel width used for left/right direction
                          hints on UNCERTAIN speech. Defaults to
                          frame.shape[1] when not supplied, so callers
                          don't need to pass it for the common case —
                          only override it if you're evaluating a
                          detection against a different frame than the
                          one it was just detected in.

        Returns:
            GatedDetections — .passed / .uncertain / .discarded.
        """
        detections = self.detect(frame)

        if frame_width is None and frame is not None and frame.size > 0:
            frame_width = frame.shape[1]

        passed, uncertain, discarded = self.confidence_gate.filter_frame(
            detections, frame_width=frame_width
        )

        if uncertain:
            logger.debug(
                "[Detector] %d uncertain detection(s) gated out before risk engine "
                "(vague speech only, no class name)",
                len(uncertain),
            )
        if discarded:
            logger.debug(
                "[Detector] %d detection(s) discarded (too noisy to voice)",
                len(discarded),
            )

        return GatedDetections(passed=passed, uncertain=uncertain, discarded=discarded)

    def annotate(
        self,
        frame: np.ndarray,
        detections: Optional[List[Detection]] = None,
    ) -> np.ndarray:
        """
        Returns a copy of `frame` with bounding boxes + labels drawn —
        useful for the live debug view and demo video (architecture doc
        Section 9.3: "Show YOLO bounding boxes"). Re-runs detection if
        `detections` isn't supplied. Confident detections draw green;
        sub-gate detections draw orange so you can see the hedge boundary
        live while testing.
        """
        import cv2  # local import: only needed for this optional debug path

        if detections is None:
            detections = self.detect(frame)

        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = (0, 200, 0) if d.is_confident else (0, 165, 255)  # BGR: green / orange
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return annotated


if __name__ == "__main__":
    # Quick manual check: `python -m app.perception.detector`
    # Confirms the model loads, the configured device works, and the
    # confidence gate is wired in.
    logging.basicConfig(level=logging.INFO)
    detector = Detector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = detector.detect(dummy_frame)
    print(
        f"Detector loaded OK on '{detector.device}'. "
        f"{len(results)} raw detection(s) on a blank test frame."
    )

    gated = detector.detect_gated(dummy_frame)
    print(
        f"detect_gated(): passed={len(gated.passed)}  "
        f"uncertain={len(gated.uncertain)}  discarded={len(gated.discarded)}"
    )
    print(detector.confidence_gate.stats_summary())