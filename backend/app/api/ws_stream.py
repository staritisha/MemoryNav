"""
MemoryNav — WebSocket Pipeline Endpoint
backend/app/api/ws_stream.py

FastAPI WebSocket at /ws. Receives raw JPEG frame bytes from the
frontend (or a test script), runs the full pipeline synchronously in a
thread pool (so inference doesn't block the event loop), and returns a
JSON result per frame.

Pipeline per frame:
    1. Decode bytes  → BGR numpy frame
    2. Frame quality → skip + send DEGRADED response if bad
    3. YOLO detect   → List[perception.Detection]
    4. Depth map     → DepthEstimator.estimate(frame)
    5. Per detection → depth_at_bbox → risk.Detection
    6. Motion trend  → short_term.SessionStore → motion_factor
    7. User context  → PreferencesStore.mobility_flags → context_weight
    8. assess_all    → sorted List[RiskAssessment]
    9. Alert gate    → suppression window + confidence gate
   10. TTS           → speak(text, interrupt=HIGH)
   11. Session store → record obstacle + warning
   12. Return JSON   → {detections, dominant_level, spoke, quality, ...}

Mount this router in your main FastAPI app:

    from app.api.ws_stream import router
    app.include_router(router)

Dependencies: fastapi, uvicorn, opencv-python, numpy,
              ultralytics, transformers, pyttsx3.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

from app.config import settings
from app.memory_modules.preferences import PreferencesStore
from app.memory_modules.short_term import SessionStore
from app.perception.depth import DepthEstimator
from app.perception.detector import Detector
from app.perception.frame_quality import check_frame_quality
from app.risk.engine import MotionState, assess_all, motion_factor_for
from app.risk.models import Detection as RiskDetection
from app.voice.tts import TTSEngine

logger = logging.getLogger(__name__)

# One thread per blocking inference call — YOLO and Depth-Anything both
# release the GIL, so they can run concurrently without contention.
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")

# --------------------------------------------------------------------------- #
# Pipeline state — heavy objects initialized once at startup
# --------------------------------------------------------------------------- #

@dataclass
class PipelineState:
    detector: Detector
    depth_estimator: DepthEstimator
    tts: TTSEngine
    session: SessionStore
    prefs_store: PreferencesStore
    # context_weight cache: reloaded lazily, not on every frame
    _context_weight: float = field(default=1.0, repr=False)
    _context_weight_ts: float = field(default=0.0, repr=False)

    _CONTEXT_CACHE_TTL_S: float = field(default=5.0, init=False, repr=False)

    def user_context_weight(self) -> float:
        """
        Derives a risk multiplier from stored mobility flags. Reloaded
        from SQLite at most once per 5s (cheap cache — avoids a DB read
        on every frame while still picking up changes promptly).
        Available flags and their weights are a deliberate first pass;
        tune or extend as user-testing reveals what matters most.
        """
        now = time.monotonic()
        if now - self._context_weight_ts < self._CONTEXT_CACHE_TTL_S:
            return self._context_weight

        prefs = self.prefs_store.load()
        flags = set(prefs.mobility_flags)
        weight = 1.0
        if "limited_mobility" in flags:
            weight *= 1.4
        if "bad_knee" in flags:
            weight *= 1.2
        if "uses_cane" in flags:
            weight *= 1.1
        self._context_weight = weight
        self._context_weight_ts = now
        return weight


# Module-level singleton — shared across all WebSocket connections.
_state: Optional[PipelineState] = None


def get_state() -> PipelineState:
    if _state is None:
        raise RuntimeError("Pipeline not initialized. Use the lifespan app or call init_pipeline().")
    return _state


def init_pipeline() -> PipelineState:
    """Initializes all heavy resources. Called once at app startup."""
    global _state
    logger.info("Initializing MemoryNav pipeline on device '%s'...", settings.device)
    _state = PipelineState(
        detector=Detector(),
        depth_estimator=DepthEstimator(),
        tts=TTSEngine(),
        session=SessionStore(),
        prefs_store=PreferencesStore(),
    )
    logger.info("Pipeline ready.")
    return _state


def shutdown_pipeline() -> None:
    global _state
    if _state is not None:
        _state.tts.shutdown()
        _state = None


# --------------------------------------------------------------------------- #
# Frame → JSON pipeline (runs in thread pool, safe to block)
# --------------------------------------------------------------------------- #

_MOTION_STATE_MAP: Dict[str, MotionState] = {
    "approaching": MotionState.APPROACHING,
    "receding":    MotionState.RECEDING,
    "stationary":  MotionState.STATIONARY,
    "unknown":     MotionState.UNKNOWN,
}

_WARNING_TEMPLATES = {
    "HIGH":   "Stop — {name} very close, {dist:.1f} meters",
    "MEDIUM": "{name} ahead, {dist:.1f} meters",
}


def _warning_text(class_name: str, distance_metres: float, level: str) -> str:
    tpl = _WARNING_TEMPLATES.get(level, "")
    return tpl.format(name=class_name, dist=distance_metres) if tpl else ""


def _run_pipeline_sync(
    frame_bytes: bytes,
    state: PipelineState,
    frame_id: Optional[str],
) -> Dict[str, Any]:
    """
    Blocking pipeline. Runs in a ThreadPoolExecutor so it doesn't block
    the FastAPI event loop. Returns a plain dict (JSON-serializable).
    """
    ts = time.time()

    # ── 1. Decode ───────────────────────────────────────────────────────
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {
            "frame_id": frame_id, "ts": ts,
            "status": "decode_error",
            "quality": {"ok": False, "issue": "decode_error"},
            "detections": [], "dominant_level": None, "spoke": None,
        }

    # ── 2. Frame quality ────────────────────────────────────────────────
    quality = check_frame_quality(frame)
    if not quality.ok:
        return {
            "frame_id": frame_id, "ts": ts,
            "status": "degraded",
            "quality": {
                "ok": False,
                "issue": quality.issue.value,
                "brightness": quality.brightness,
                "blur_variance": quality.blur_variance,
            },
            "detections": [], "dominant_level": None, "spoke": None,
        }

    # ── 3. YOLO detections ──────────────────────────────────────────────
    perception_detections = state.detector.detect(frame)

    if not perception_detections:
        return {
            "frame_id": frame_id, "ts": ts,
            "status": "ok",
            "quality": {"ok": True, "issue": "none",
                        "brightness": quality.brightness,
                        "blur_variance": quality.blur_variance},
            "detections": [], "dominant_level": "NONE", "spoke": None,
        }

    # ── 4. Depth map (one pass for all detections) ───────────────────────
    depth_map = state.depth_estimator.estimate(frame)

    # ── 5–6. Build risk Detections + motion factor per class ─────────────
    risk_detections: List[RiskDetection] = []
    motion_factors: Dict[str, float] = {}

    for pd in perception_detections:
        dist = state.depth_estimator.depth_at_bbox(depth_map, pd.bbox)
        if np.isnan(dist) or dist <= 0:
            dist = float("nan")

        rd = RiskDetection.from_perception(pd, distance_metres=dist)
        risk_detections.append(rd)

        if pd.class_name not in motion_factors:
            trend = state.session.motion_trend(pd.class_name)
            motion_factors[pd.class_name] = motion_factor_for(
                _MOTION_STATE_MAP.get(trend, MotionState.UNKNOWN)
            )

    # ── 7. User context weight ───────────────────────────────────────────
    context_weight = state.user_context_weight()

    # ── 8. Risk assessment (sorted highest first) ────────────────────────
    assessments = assess_all(
        risk_detections,
        motion_factor=1.0,          # per-detection below
        user_context_weight=context_weight,
    )
    # Re-assess using per-class motion factors
    from app.risk.engine import assess_risk  # local import avoids circular at module level
    assessments = sorted(
        [
            assess_risk(
                rd,
                motion_factor=motion_factors.get(rd.class_name, 1.0),
                user_context_weight=context_weight,
            )
            for rd in risk_detections
        ],
        key=lambda a: a.score,
        reverse=True,
    )

    dominant = assessments[0] if assessments else None
    dominant_level = dominant.level.value if dominant else "NONE"

    # ── 9–10. Alert gate + TTS ───────────────────────────────────────────
    spoke: Optional[str] = None

    if dominant and dominant.level.value in _WARNING_TEMPLATES:
        text = _warning_text(
            dominant.detection.class_name,
            dominant.detection.distance_metres,
            dominant.level.value,
        )
        # Confidence gate: hedge if detector isn't sure
        if not dominant.detection.confidence >= settings.YOLO_CONFIDENCE_GATE:
            text = f"something may be ahead, I am not certain"

        # Suppression: skip if same text was already spoken recently
        suppress_window = settings.ALERT_SUPPRESSION_WINDOW_SECONDS
        if text and not state.session.was_recently_warned(text, within_seconds=suppress_window):
            interrupt = dominant.level.value == "HIGH"
            state.tts.speak(text, interrupt=interrupt)
            state.session.record_warning(text, risk_level=dominant.level.value)
            spoke = text

    # ── 11. Record to short-term session store ───────────────────────────
    for rd in risk_detections:
        state.session.record_obstacle_detection(rd)

    # ── 12. Serialize response ───────────────────────────────────────────
    return {
        "frame_id": frame_id,
        "ts": ts,
        "status": "ok",
        "quality": {
            "ok": True,
            "issue": "none",
            "brightness": quality.brightness,
            "blur_variance": quality.blur_variance,
        },
        "detections": [
            {
                "class_name": a.detection.class_name,
                "confidence": round(a.detection.confidence, 3),
                "bbox": list(a.detection.bbox),
                "distance_metres": round(a.detection.distance_metres, 3)
                    if not np.isnan(a.detection.distance_metres) else None,
                "risk_score": round(a.score, 3),
                "risk_level": a.level.value,
                "action": a.action,
                "motion_trend": state.session.motion_trend(a.detection.class_name),
                "is_confident": a.detection.confidence >= settings.YOLO_CONFIDENCE_GATE,
            }
            for a in assessments
        ],
        "dominant_level": dominant_level,
        "spoke": spoke,
    }


# --------------------------------------------------------------------------- #
# FastAPI router
# --------------------------------------------------------------------------- #

router = APIRouter()


@router.websocket("/ws")
async def ws_stream(websocket: WebSocket) -> None:
    """
    WebSocket /ws

    Protocol:
      Client → Server: raw JPEG bytes, optionally prefixed with a
                       frame_id header (see below).
      Server → Client: UTF-8 JSON per frame.

    Optional frame_id header:
      Send a text frame immediately before each binary frame containing
      a client-generated ID string (e.g. "frame-0042"). The response
      will echo it back in "frame_id" for latency measurement. Skip the
      text frame entirely to get null frame_ids.

    Close codes:
      1000 — normal close initiated by the client.
      1011 — server-side pipeline error on a frame (the connection stays
             open; only that frame's result is an error response).
    """
    await websocket.accept()
    state = get_state()
    loop = asyncio.get_running_loop()
    pending_frame_id: Optional[str] = None

    logger.info("WebSocket client connected: %s", websocket.client)
    try:
        while True:
            message = await websocket.receive()

            # Text message = optional frame_id tag for the next binary frame
            if message["type"] == "websocket.receive" and message.get("text"):
                pending_frame_id = message["text"]
                continue

            frame_bytes: Optional[bytes] = message.get("bytes")
            if not frame_bytes:
                continue

            frame_id = pending_frame_id
            pending_frame_id = None

            try:
                result = await loop.run_in_executor(
                    _EXECUTOR,
                    _run_pipeline_sync,
                    frame_bytes,
                    state,
                    frame_id,
                )
            except Exception:
                logger.exception("Pipeline error on frame %s", frame_id)
                result = {
                    "frame_id": frame_id,
                    "ts": time.time(),
                    "status": "pipeline_error",
                    "detections": [],
                    "dominant_level": None,
                    "spoke": None,
                }

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", websocket.client)
    except Exception:
        logger.exception("Unexpected WebSocket error.")


# --------------------------------------------------------------------------- #
# Standalone app (for `uvicorn app.api.ws_stream:app` during development)
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_pipeline()
    yield
    shutdown_pipeline()


app = FastAPI(title="MemoryNav WS Stream", lifespan=_lifespan)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("app.api.ws_stream:app", host=settings.HOST, port=settings.PORT, reload=False)