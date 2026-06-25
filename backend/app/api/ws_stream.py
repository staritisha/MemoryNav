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
from app.memory_modules.spatial_map import SpatialMap
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
    spatial_map: SpatialMap = None
    frame_counter: int = field(default=0, repr=False)
    _ghost_alerted_at: Dict[str, float] = field(default_factory=dict, repr=False)
    

# Module-level singleton — shared across all WebSocket connections.
_state: Optional[PipelineState] = None


def get_state() -> PipelineState:
    if _state is None:
        raise RuntimeError("Pipeline not initialized. Use the lifespan app or call init_pipeline().")
    return _state


def init_pipeline(
    detector: Optional[Detector] = None,
    depth_estimator: Optional[DepthEstimator] = None,
    tts: Optional[TTSEngine] = None,
    long_term: Optional[LongTermMemory] = None,
    prefs_store: Optional[PreferencesStore] = None,
    alert_manager: Optional[TemporalAlertManager] = None,
    session: Optional[SessionStore] = None,
) -> PipelineState:
    """
    Initialize the pipeline singleton.

    When called from main.py lifespan, pass in the already-loaded instances
    from app.state so models are loaded exactly ONCE. When called standalone
    (unit tests, dev server), instances are created here.
    """
    global _state
    logger.info("Initializing MemoryNav pipeline on device '%s'...", settings.device)

    _long_term   = long_term   or LongTermMemory()
    _prefs_store = prefs_store or PreferencesStore()

    _state = PipelineState(
        detector      = detector       or Detector(),
        depth_estimator = depth_estimator or DepthEstimator(),
        tts           = tts            or TTSEngine(),
        session       = session        or SessionStore(),
        prefs_store   = _prefs_store,
        long_term     = _long_term,
        risk_engine   = RiskEngine(prefs_store=_prefs_store, long_term=_long_term),
        alert_manager = alert_manager  or TemporalAlertManager(
            suppression_window_s=settings.ALERT_SUPPRESSION_WINDOW_SECONDS
        ),
    )
    _state.tracker = ObjectTracker(frame_rate=30)
    _state.spatial_map = SpatialMap()
    _state.spatial_map.set_room("Living Room")
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


def _decode_frame(frame_bytes: bytes) -> Optional[np.ndarray]:
    """Stage 1: Decode raw JPEG bytes to a BGR numpy frame."""
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _run_perception(
    frame: np.ndarray,
    state: PipelineState,
) -> tuple[list, np.ndarray, set]:
    """
    Stage 2: YOLO detect → ByteTrack → Depth.
    Returns (tracked_detections, depth_map, detected_class_names).
    """
    raw_detections = state.detector.detect(frame)
    tracked = state.tracker.update(raw_detections, frame)
    detected_classes = {d.class_name for d in tracked}
    depth_map = state.depth_estimator.estimate(frame)
    return tracked, depth_map, detected_classes


def _build_risk_detections(
    tracked: list,
    depth_map: np.ndarray,
    frame: np.ndarray,
    state: PipelineState,
) -> tuple[list, dict]:
    """
    Stage 3: Attach distance + motion to each tracked detection, update spatial map.
    Returns (risk_detections, motion_factors_by_class).
    """
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

        state.spatial_map.update(
            class_name=pd.class_name,
            bbox=pd.bbox,
            frame_width=frame.shape[1],
            distance_m=dist,
            confidence=pd.confidence,
        )

    return risk_detections, motion_factors


def _run_risk_engine(
    risk_detections: list,
    motion_factors: dict,
    state: PipelineState,
) -> list:
    """
    Stage 4: Score every detection with the Risk Engine.
    ContextWeightResolver queries ChromaDB + SQLite on demand.
    Returns assessments sorted by score descending.
    """
    if not risk_detections:
        return []
    return sorted(
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
    )


def _run_memory_and_alerts(
    assessments: list,
    risk_detections: list,
    tracked: list,
    detected_classes: set,
    state: PipelineState,
) -> tuple[Optional[str], bool]:
    """
    Stage 5: Persist observations, check ghost alerts, run temporal suppression, speak.
    Returns (spoke_text_or_None, spoke_ghost_bool).
    """
    # Write high-confidence observations to long-term memory every N frames
    if state.frame_counter % _MEMORY_WRITE_EVERY_N_FRAMES == 0:
        for pd, rd in zip(tracked, risk_detections):
            if pd.confidence >= _MEMORY_WRITE_MIN_CONFIDENCE and not np.isnan(rd.distance_metres):
                _write_observation_to_memory(state, pd.class_name, rd.distance_metres)

    # Ghost alerts for remembered-but-unseen objects
    ghost_text: Optional[str] = None
    if state.long_term.count() > 0:
        ghost_text = _check_ghost_alerts(state, detected_classes)

    dominant = assessments[0] if assessments else None
    spoke: Optional[str] = None
    spoke_ghost = False

    if dominant and dominant.level.value in ("HIGH", "MEDIUM"):
        decision = state.alert_manager.evaluate(dominant)
        if decision.should_speak:
            text = decision.speech_text
            if dominant.detection.confidence < settings.YOLO_CONFIDENCE_GATE:
                text = "something may be ahead, I am not certain"
            state.tts.speak(text, interrupt=(dominant.level.value == "HIGH"))
            state.session.record_warning(text, risk_level=dominant.level.value)
            spoke = text

    if spoke is None and ghost_text is not None:
        if not state.session.was_recently_warned(ghost_text, within_seconds=_GHOST_ALERT_COOLDOWN_S):
            state.tts.speak(ghost_text, interrupt=False)
            state.session.record_warning(ghost_text, risk_level="MEMORY")
            spoke = ghost_text
            spoke_ghost = True

    for rd in risk_detections:
        state.session.record_obstacle_detection(rd)

    return spoke, spoke_ghost


def _serialize_frame(
    assessments: list,
    dominant,
    spoke: Optional[str],
    spoke_ghost: bool,
    frame_shape,
    ts: float,
    frame_id: Optional[str],
    state: PipelineState,
) -> Dict[str, Any]:
    """Stage 6: Build the JSON payload matching frontend types.ts DetectionFrame."""
    dominant_level = dominant.level.value if dominant else "NONE"

    memory_context: Optional[str] = None
    if assessments and assessments[0].context_result:
        memory_context = assessments[0].context_result.spatial_memory

    dominant_score = round(dominant.score, 3) if dominant else None
    dominant_reason: Optional[str] = None
    if dominant:
        dist_val = dominant.detection.distance_metres
        dist_str = f"{dist_val:.1f}m" if not np.isnan(dist_val) else "unknown distance"
        dominant_reason = (
            f"{dominant.detection.class_name} {dist_str}, "
            f"{dominant.level.value} risk (score {dominant.score:.2f})"
        )
        if memory_context:
            dominant_reason += f" · memory: {memory_context}"

    latency_ms = round((time.time() - ts) * 1000, 1)

    def _det_dict(a) -> dict:
        h, w = frame_shape[:2] if frame_shape else (480, 640)
        x1, y1, x2, y2 = a.detection.bbox
        dist = a.detection.distance_metres
        return {
            "id":             f"{a.detection.class_name}_{x1}_{y1}_{state.frame_counter}",
            "label":          a.detection.class_name,
            "confidence":     round(a.detection.confidence, 3),
            "box": {
                "x":      round(x1 / w, 4),
                "y":      round(y1 / h, 4),
                "width":  round((x2 - x1) / w, 4),
                "height": round((y2 - y1) / h, 4),
            },
            "distanceMeters": round(dist, 3) if not np.isnan(dist) else None,
            "riskScore":      round(a.score, 3),
            "riskLevel":      a.level.value,
            "motionTrend":    state.session.motion_trend(a.detection.class_name),
            "isConfident":    a.detection.confidence >= settings.YOLO_CONFIDENCE_GATE,
            "contextWeight":  round(a.context_result.final_weight, 3) if a.context_result else None,
            "spatialMemory":  a.context_result.spatial_memory if a.context_result else None,
        }

    return {
        "timestamp":        ts,
        "detections":       [_det_dict(a) for a in assessments],
        "riskLevel":        dominant_level if dominant_level != "NONE" else "LOW",
        "riskScore":        dominant_score,
        "riskReason":       dominant_reason,
        "suppressed":       (spoke is None and dominant is not None
                             and dominant.level.value in ("HIGH", "MEDIUM")),
        "suppressionReason": None,
        "latencyMs":        latency_ms,
        "status":           "ok",
        "spokeGhost":       spoke_ghost,
        "spoke":            spoke,
        "memoryContext":    memory_context,
        "suppressionStats": state.alert_manager.counts,
        "spatialMap":       state.spatial_map.snapshot(),
        "frame_id":         frame_id,
    }


def _run_pipeline_sync(
    frame_bytes: bytes,
    state: PipelineState,
    frame_id: Optional[str],
) -> Dict[str, Any]:
    """
    Orchestrates the 6 pipeline stages for one frame.
    Runs in a ThreadPoolExecutor worker — safe to block.
    """
    ts = time.time()
    state.frame_counter += 1

    # Stage 1: Decode
    frame = _decode_frame(frame_bytes)
    if frame is None:
        return {
            "frame_id": frame_id, "ts": ts, "status": "decode_error",
            "quality": {"ok": False, "issue": "decode_error"},
            "detections": [], "dominant_level": None, "spoke": None,
        }

    # Stage 1b: Frame quality gate
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

    # Stage 2: Perception (YOLO + tracking + depth)
    tracked, depth_map, detected_classes = _run_perception(frame, state)

    # Stage 3: Build risk-ready detections
    risk_detections, motion_factors = _build_risk_detections(tracked, depth_map, frame, state)

    # Stage 4: Risk scoring
    assessments = _run_risk_engine(risk_detections, motion_factors, state)
    dominant = assessments[0] if assessments else None

    # Stage 5: Memory, ghost alerts, temporal suppression, TTS
    spoke, spoke_ghost = _run_memory_and_alerts(
        assessments, risk_detections, tracked, detected_classes, state
    )

    # Stage 6: Serialize to JSON
    return _serialize_frame(
        assessments, dominant, spoke, spoke_ghost,
        frame.shape, ts, frame_id, state,
    )


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

            # Client closed the connection cleanly
            if message["type"] == "websocket.disconnect":
                break

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