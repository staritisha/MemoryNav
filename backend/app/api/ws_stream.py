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
    4. Depth map     → DepthEstimator.estimate(frame) [frame-skip cached]
    5. Per detection → depth_at_bbox → risk.Detection
    6. Motion trend  → short_term.SessionStore → motion_factor
    7. Risk engine   → RiskEngine.assess_all() — ContextWeightResolver
                       pulls user_context_weight from ChromaDB + SQLite
    8. Memory write  → persist high-confidence observations to ChromaDB
    9. Ghost alerts  → warn about remembered but currently unseen objects
   10. Alert gate    → TemporalAlertManager (WalkVLM suppression)
   11. TTS           → speak(text, interrupt=HIGH)
   12. Session store → record obstacle + warning
   13. Return JSON

Mount this router in your main FastAPI app:

    from app.api.ws_stream import router
    app.include_router(router)
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

from app.alerts.temporal_manager import TemporalAlertManager
from app.config import settings
from app.memory_modules.long_term import LongTermMemory
from app.memory_modules.preferences import PreferencesStore
from app.memory_modules.short_term import SessionStore
from app.perception.depth import DepthEstimator
from app.perception.detector import Detector
from app.perception.frame_quality import check_frame_quality
from app.perception.tracker import ObjectTracker, TrackedDetection
from app.risk.engine import MotionState, RiskEngine, motion_factor_for
from app.risk.models import Detection as RiskDetection
from app.voice.tts import TTSEngine

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")

# Write a new spatial observation to ChromaDB at most every N frames.
_MEMORY_WRITE_EVERY_N_FRAMES = 30
# Minimum YOLO confidence before we persist an observation to long-term memory.
_MEMORY_WRITE_MIN_CONFIDENCE = 0.55
# Seconds before re-alerting about a ghost (remembered but unseen) object.
_GHOST_ALERT_COOLDOWN_S = 30.0

@dataclass
class PipelineState:
    detector: Detector
    depth_estimator: DepthEstimator
    tts: TTSEngine
    session: SessionStore
    prefs_store: PreferencesStore
    long_term: LongTermMemory
    risk_engine: RiskEngine
    alert_manager: TemporalAlertManager
    tracker: ObjectTracker = None
    frame_counter: int = field(default=0, repr=False)
    _ghost_alerted_at: Dict[str, float] = field(default_factory=dict, repr=False)
    

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
    long_term = LongTermMemory()
    prefs_store = PreferencesStore()
    _state = PipelineState(
        detector=Detector(),
        depth_estimator=DepthEstimator(),
        tts=TTSEngine(),
        session=SessionStore(),
        prefs_store=prefs_store,
        long_term=long_term,
        # Share the live instances so no second DB connection is opened.
        risk_engine=RiskEngine(prefs_store=prefs_store, long_term=long_term),
        alert_manager=TemporalAlertManager(
            suppression_window_s=settings.ALERT_SUPPRESSION_WINDOW_SECONDS
        ),
    )
    _state.tracker = ObjectTracker(frame_rate=30)
    logger.info("Pipeline ready.")
    return _state


def shutdown_pipeline() -> None:
    global _state
    if _state is not None:
        _state.tts.shutdown()
        _state = None


# --------------------------------------------------------------------------- #
# Memory helpers
# --------------------------------------------------------------------------- #

def _write_observation_to_memory(
    state: PipelineState, class_name: str, distance_m: float
) -> None:
    """Persist a high-confidence detection as a spatial memory entry."""
    try:
        if distance_m < 0.7:
            proximity = "very close"
        elif distance_m < 2.0:
            proximity = "nearby"
        else:
            proximity = "in the area"
        text = f"{class_name} observed {proximity} ({distance_m:.1f}m)"
        state.long_term.add_context(
            text,
            metadata={"class_name": class_name, "distance_m": round(distance_m, 2)},
        )
        logger.debug("[Memory] Wrote: %r", text)
    except Exception as exc:
        logger.warning("[Memory] Failed to write observation: %s", exc)


def _check_ghost_alerts(
    state: PipelineState,
    detected_classes: set,
) -> Optional[str]:
    """
    Check ChromaDB for remembered objects NOT currently visible.
    Returns a speech string if a ghost alert should fire, else None.
    """
    now = time.monotonic()
    try:
        query = ("obstacle near " + ", ".join(list(detected_classes)[:3])
                 if detected_classes else "obstacle in this area")
        results = state.long_term.retrieve(query, n_results=3)
        for result in results:
            if result.similarity < 0.50:
                continue
            remembered_class = result.metadata.get("class_name", "")
            if not remembered_class or remembered_class in detected_classes:
                continue
            last = state._ghost_alerted_at.get(remembered_class, 0.0)
            if (now - last) < _GHOST_ALERT_COOLDOWN_S:
                continue
            state._ghost_alerted_at[remembered_class] = now
            return f"Previously observed {remembered_class} here — proceed carefully"
    except Exception as exc:
        logger.warning("[Memory] Ghost alert check failed: %s", exc)
    return None


# --------------------------------------------------------------------------- #
# Frame → JSON pipeline (runs in thread pool, safe to block)
# --------------------------------------------------------------------------- #

_MOTION_STATE_MAP: Dict[str, MotionState] = {
    "approaching": MotionState.APPROACHING,
    "receding":    MotionState.RECEDING,
    "stationary":  MotionState.STATIONARY,
    "unknown":     MotionState.UNKNOWN,
}


def _run_pipeline_sync(
    frame_bytes: bytes,
    state: PipelineState,
    frame_id: Optional[str],
) -> Dict[str, Any]:
    ts = time.time()
    state.frame_counter += 1

    # 1. Decode
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {
            "frame_id": frame_id, "ts": ts, "status": "decode_error",
            "quality": {"ok": False, "issue": "decode_error"},
            "detections": [], "dominant_level": None, "spoke": None,
        }

    # 2. Frame quality
    quality = check_frame_quality(frame)
    if not quality.ok:
        return {
            "frame_id": frame_id, "ts": ts, "status": "degraded",
            "quality": {
                "ok": False, "issue": quality.issue.value,
                "brightness": quality.brightness,
                "blur_variance": quality.blur_variance,
            },
            "detections": [], "dominant_level": None, "spoke": None,
        }

    # 3. YOLO
    perception_detections = state.detector.detect(frame)

    # 3b. Track — assigns track_id and is_new to each detection
    tracked = state.tracker.update(perception_detections, frame.shape)
    detected_classes = {d.class_name for d in tracked}

    # 4. Depth (frame-skip cached inside DepthEstimator)
    depth_map = state.depth_estimator.estimate(frame)

    # 5–6. Build risk detections + per-class motion factors
    risk_detections: List[RiskDetection] = []
    motion_factors: Dict[str, float] = {}
    for pd in tracked:
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

    # 7. Risk — user_context_weight=None → ContextWeightResolver queries
    #    ChromaDB for spatial boost + SQLite for mobility flags.
    #    This is what makes memory change pipeline decisions.
    assessments = sorted(
        [
            state.risk_engine.assess(
                rd,
                motion_factor=motion_factors.get(rd.class_name, 1.0),
                user_context_weight=None,
            )
            for rd in risk_detections
        ],
        key=lambda a: a.score,
        reverse=True,
    ) if risk_detections else []

    dominant = assessments[0] if assessments else None
    dominant_level = dominant.level.value if dominant else "NONE"

    # 8. Write observations to long-term memory (every N frames)
    if state.frame_counter % _MEMORY_WRITE_EVERY_N_FRAMES == 0:
        for pd, rd in zip(tracked, risk_detections):
            if (pd.confidence >= _MEMORY_WRITE_MIN_CONFIDENCE
                    and not np.isnan(rd.distance_metres)):
                _write_observation_to_memory(state, pd.class_name, rd.distance_metres)

    # 9. Ghost alerts — remembered objects not currently visible
    ghost_text: Optional[str] = None
    if state.long_term.count() > 0:
        ghost_text = _check_ghost_alerts(state, detected_classes)

    # 10. Alert gate — TemporalAlertManager
    spoke: Optional[str] = None
    spoke_ghost = False

    if dominant and dominant.level.value in ("HIGH", "MEDIUM"):
        decision = state.alert_manager.evaluate(dominant)
        if decision.should_speak:
            text = decision.speech_text
            if not (dominant.detection.confidence >= settings.YOLO_CONFIDENCE_GATE):
                text = "something may be ahead — I am not certain"
            interrupt = dominant.level.value == "HIGH"
            state.tts.speak(text, interrupt=interrupt)
            state.session.record_warning(text, risk_level=dominant.level.value)
            spoke = text

    # Ghost alert only fires when no primary alert spoken this frame
    if spoke is None and ghost_text is not None:
        if not state.session.was_recently_warned(
            ghost_text, within_seconds=_GHOST_ALERT_COOLDOWN_S
        ):
            state.tts.speak(ghost_text, interrupt=False)
            state.session.record_warning(ghost_text, risk_level="MEMORY")
            spoke = ghost_text
            spoke_ghost = True

    # 11. Record to short-term session store
    for rd in risk_detections:
        state.session.record_obstacle_detection(rd)

    # 12. Serialize
    memory_context: Optional[str] = None
    if assessments and assessments[0].context_result:
        memory_context = assessments[0].context_result.spatial_memory

    return {
        "frame_id": frame_id,
        "ts": ts,
        "status": "ok",
        "quality": {
            "ok": True, "issue": "none",
            "brightness": quality.brightness,
            "blur_variance": quality.blur_variance,
        },
        "detections": [
            {
                "class_name":      a.detection.class_name,
                "confidence":      round(a.detection.confidence, 3),
                "bbox":            list(a.detection.bbox),
                "distance_metres": (
                    round(a.detection.distance_metres, 3)
                    if not np.isnan(a.detection.distance_metres) else None
                ),
                "risk_score":      round(a.score, 3),
                "risk_level":      a.level.value,
                "action":          a.action,
                "motion_trend":    state.session.motion_trend(a.detection.class_name),
                "is_confident":    a.detection.confidence >= settings.YOLO_CONFIDENCE_GATE,
                "context_weight":  (
                    round(a.context_result.final_weight, 3)
                    if a.context_result else None
                ),
                "spatial_memory":  (
                    a.context_result.spatial_memory if a.context_result else None
                ),
            }
            for a in assessments
        ],
        "dominant_level":    dominant_level,
        "spoke":             spoke,
        "spoke_ghost":       spoke_ghost,
        "memory_context":    memory_context,
        "suppression_stats": state.alert_manager.counts,
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