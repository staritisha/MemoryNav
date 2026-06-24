"""
MemoryNav — EasyOCR Wrapper
backend/app/perception/ocr.py

Module 1 (Perception Layer): offline text reading. Thin wrapper around
EasyOCR (JaidedAI) — reads medicine labels, door signs, and switches,
fully on-device. No frame, image, or recognized text is ever sent to
a cloud service (architecture doc, Privacy Architecture section).

Where this fits in the pipeline
--------------------------------
Unlike Detector, this is NOT part of the per-frame YOLO loop — running
OCR on every frame would be far too slow for a 30fps obstacle stream.
It's invoked on demand from the voice layer (Phase 5 — Whisper voice
input + question answering), when the user asks something like
"what does this say?" or "read this":

    User: "what does this say?"
        ↓
    Whisper transcribes the question
        ↓
    Voice layer points the camera / grabs the current frame
        ↓
    OCRReader.read_text(frame)        ← YOU ARE HERE
        ↓
    Voice speaks the recognized text, or "I can't make out any text
    right now" if nothing was read confidently.

Device note — Apple Silicon
-----------------------------
EasyOCR's GPU acceleration is CUDA-only; it has no Apple MPS backend
(unlike the YOLO detector, which does run on MPS — see detector.py).
On an M2 Mac, OCR therefore always runs on CPU regardless of
settings.device. This wrapper detects that automatically and logs a
one-time warning instead of silently being slow, since on-demand OCR
latency is part of the system's response-time budget.

Usage
-----
    from app.perception.ocr import OCRReader

    ocr = OCRReader()                       # one instance, reuse it

    text = ocr.read_text(frame)             # str, or None if nothing read
    if text:
        voice.speak(text)
    else:
        voice.speak("I can't make out any text right now")

    # Reading text within a specific region (e.g. a detected sign's
    # bounding box from the YOLO detector) instead of the whole frame:
    text = ocr.read_text(frame, region=(x1, y1, x2, y2))

    # Line-level detail (bbox + per-line confidence) for debugging or
    # the OCR character-accuracy ablation metric (doc Table — OCR row):
    lines = ocr.read_lines(frame)

Dependencies: easyocr, numpy, opencv-python (for the BGR→RGB convert
and optional region crop).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# ── Defaults ─────────────────────────────────────────────────────────────────

# EasyOCR confidence is not directly comparable to YOLO confidence (different
# model, different calibration) so it gets its own threshold rather than
# reusing settings.YOLO_CONFIDENCE_GATE. Reads below this are dropped as
# noise — getattr() so this file doesn't hard-fail if config.py hasn't grown
# an OCR section yet; it just falls back to a sane default.
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40
_DEFAULT_LANGUAGES: Tuple[str, ...] = ("en",)


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OCRLine:
    """
    One recognized line of text from EasyOCR.

    Attributes
    ----------
    text       : Recognized text, whitespace-trimmed.
    confidence : EasyOCR's recognition confidence, 0.0–1.0.
    bbox       : Four (x, y) corner points (the text can be rotated,
                 so this is a quadrilateral, not an axis-aligned
                 (x1, y1, x2, y2) box like Detection.bbox).
    """

    text:       str
    confidence: float
    bbox:       Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]

    @property
    def center(self) -> Tuple[int, int]:
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        return (sum(xs) // 4, sum(ys) // 4)


def _reading_order_key(line: OCRLine) -> Tuple[int, int]:
    """Sort lines top-to-bottom, then left-to-right — roughly how a person reads a label."""
    cx, cy = line.center
    return (cy, cx)


# ── Main wrapper class ────────────────────────────────────────────────────────

class OCRReader:
    """
    EasyOCR wrapper. Load once, reuse across calls — model load is the
    expensive part (downloads weights on first run, then loads them
    into memory), exactly like Detector's YOLO model.

    Parameters
    ----------
    languages : Sequence[str]
        EasyOCR language codes, e.g. ("en",) or ("en", "es").
        Defaults to settings.OCR_LANGUAGES if present, else ("en",).
    gpu : Optional[bool]
        Force GPU on/off. Defaults to settings.device == "cuda" — see
        the Device note in the module docstring for why "mps" does not
        enable GPU here.
    confidence_threshold : float
        Recognized lines below this confidence are dropped before
        read_text()/read_lines() return anything. Default 0.40.
    """

    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        gpu: Optional[bool] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        self.languages = list(
            languages or getattr(settings, "OCR_LANGUAGES", _DEFAULT_LANGUAGES)
        )
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(settings, "OCR_CONFIDENCE_THRESHOLD", _DEFAULT_CONFIDENCE_THRESHOLD)
        )

        self._use_gpu = self._resolve_gpu_flag(gpu)

        logger.info(
            "Loading EasyOCR reader (languages=%s, gpu=%s, conf >= %.2f)",
            self.languages, self._use_gpu, self.confidence_threshold,
        )

        # Imported here, not at module scope: easyocr pulls in torch and
        # downloads detection/recognition weights on first import in some
        # environments, which is too heavy a side effect for `import
        # app.perception.ocr` alone — callers that never use OCR shouldn't
        # pay for it.
        import easyocr

        self._reader = easyocr.Reader(self.languages, gpu=self._use_gpu)

        # Ablation study counters (doc: "OCR | Character Accuracy %" metric
        # is computed externally against ground-truth labels, but these
        # coarse counts are useful for spotting a reader that's silently
        # returning nothing on every call).
        self._total_reads:      int = 0
        self._successful_reads: int = 0

        self._warm_up()

    @staticmethod
    def _resolve_gpu_flag(gpu: Optional[bool]) -> bool:
        if gpu is not None:
            return gpu

        device = str(getattr(settings, "device", "cpu")).lower()
        if device == "cuda":
            return True
        if device == "mps":
            logger.warning(
                "EasyOCR has no Apple MPS backend; OCR will run on CPU and "
                "will be slower than YOLO detection on this device. This is "
                "expected on Apple Silicon, not a bug."
            )
        return False

    def _warm_up(self) -> None:
        """
        Runs one dummy read so the first real on-demand OCR call isn't
        slowed down by lazy kernel/model initialization. Best-effort,
        same as Detector._warm_up() — a failed warm-up shouldn't block
        startup.
        """
        dummy = np.zeros((100, 300, 3), dtype=np.uint8)
        try:
            self._reader.readtext(dummy)
            logger.info("OCRReader warm-up complete.")
        except Exception:
            logger.warning("OCRReader warm-up failed; continuing anyway.", exc_info=True)

    # ── Primary API ───────────────────────────────────────────────────────────

    def read_text(
        self,
        frame: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[str]:
        """
        Read all confident text in a frame (or a region of it) and
        return it as a single string, top-to-bottom / left-to-right.

        Args:
            frame  : BGR frame (OpenCV convention, same as Detector.detect()).
            region : Optional (x1, y1, x2, y2) pixel crop — e.g. a YOLO
                     detection's bbox for a sign or label — so OCR only
                     looks at that area instead of the whole frame.

        Returns:
            A whitespace-normalized string of all recognized text above
            confidence_threshold, or None if nothing confident was read
            (including on an empty/invalid frame). Never raises — OCR
            failing should degrade to "nothing read", not crash the
            assistant a visually impaired user is relying on.
        """
        lines = self.read_lines(frame, region=region)
        if not lines:
            return None

        ordered = sorted(lines, key=_reading_order_key)
        text = " ".join(line.text for line in ordered).strip()
        return text or None

    def read_lines(
        self,
        frame: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[OCRLine]:
        """
        Same as read_text() but returns per-line detail (text, confidence,
        bbox) instead of a single joined string. Useful for debugging,
        the character-accuracy ablation metric, or UI overlays that want
        to highlight individual recognized lines.

        Lines below confidence_threshold are filtered out before this
        returns — there is no "uncertain OCR" tier analogous to
        ConfidenceGate; a label is either read or it isn't.
        """
        self._total_reads += 1

        if frame is None or frame.size == 0:
            logger.warning("read_lines() called with an empty frame; returning no lines.")
            return []

        crop = self._crop(frame, region)
        if crop is None:
            return []

        rgb = self._to_rgb(crop)

        try:
            raw_results = self._reader.readtext(rgb)
        except Exception:
            logger.error("EasyOCR readtext() failed; returning no lines.", exc_info=True)
            return []

        lines: List[OCRLine] = []
        for bbox, text, confidence in raw_results:
            text = text.strip()
            if not text or confidence < self.confidence_threshold:
                continue
            corners = tuple((int(x), int(y)) for x, y in bbox)
            lines.append(OCRLine(text=text, confidence=float(confidence), bbox=corners))

        if lines:
            self._successful_reads += 1
            logger.info(
                "[OCRReader] read %d confident line(s): %s",
                len(lines), " | ".join(l.text for l in lines),
            )
        else:
            logger.debug("[OCRReader] no confident text found in frame.")

        return lines

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _crop(
        frame: np.ndarray, region: Optional[Tuple[int, int, int, int]]
    ) -> Optional[np.ndarray]:
        """Crop to `region` if given, clamping to frame bounds. None on a degenerate crop."""
        if region is None:
            return frame

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))

        if x2 <= x1 or y2 <= y1:
            logger.warning("read_lines() region %s is empty after clamping; skipping.", region)
            return None

        return frame[y1:y2, x1:x2]

    @staticmethod
    def _to_rgb(frame: np.ndarray) -> np.ndarray:
        """
        EasyOCR's models are trained on RGB images; the rest of this
        pipeline passes frames around as BGR (OpenCV convention). Convert
        here rather than pushing that detail onto every caller.
        """
        import cv2  # local import: only needed for this conversion

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Ablation study metrics ────────────────────────────────────────────────

    @property
    def counts(self) -> dict:
        return {
            "total_reads":      self._total_reads,
            "successful_reads": self._successful_reads,
            "empty_reads":      self._total_reads - self._successful_reads,
        }

    def reset_counts(self) -> None:
        self._total_reads = 0
        self._successful_reads = 0

    def stats_summary(self) -> str:
        """One-line summary for logs — e.g. how often OCR comes back empty."""
        n = self._total_reads
        if n == 0:
            return "OCRReader: no reads yet"
        hit_rate = self._successful_reads / n
        return (
            f"OCRReader — reads={n}  successful={self._successful_reads}  "
            f"hit_rate={hit_rate:.1%}  languages={self.languages}"
        )


if __name__ == "__main__":
    # Quick manual check: `python -m app.perception.ocr`
    # Confirms the model loads and read_text() degrades gracefully when
    # there's no text to find (a blank frame has none, by construction —
    # this only proves "no crash, returns None", not recognition accuracy;
    # for that, point it at a real photo of a label).
    logging.basicConfig(level=logging.INFO)

    ocr = OCRReader()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = ocr.read_text(blank_frame)
    print(f"read_text() on a blank frame -> {result!r} (expected: None)")
    assert result is None

    print(ocr.stats_summary())
    print("OCRReader loaded and ran OK.")