"""
MemoryNav — FastAPI Application Entry Point
backend/app/main.py

Phase 6 (Backend API): wires every module built in Phases 1–5 into a
single running server. Responsibilities:

    1. Lifespan — load heavy models ONCE on startup, store in app.state,
       share across every request without reloading.
    2. CORS    — allow Next.js frontend (Phase 7) to connect.
    3. Routers — mount ws_stream, memory_router, preferences_router,
                 voice_router at their correct prefixes.
    4. Health  — GET /health so docker-compose and the frontend know
                 when the server is actually ready.
    5. Logging — structured, level driven by settings.LOG_LEVEL.

Run (dev):
    cd memorynav
    source .venv/bin/activate
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

Run (prod / Docker):
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1
    (workers=1 — models are loaded into app.state once; multiple workers
    would each load their own copy and exceed RAM on most laptops)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings

# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.ws_stream import router as ws_router
from app.api.memory_router import router as memory_router
from app.api.preferences_router import router as preferences_router

# ── Heavy modules loaded once in lifespan ─────────────────────────────────────
from app.perception.detector       import Detector
from app.perception.depth          import DepthEstimator
from app.perception.ocr            import OCRReader
from app.memory_modules.long_term  import LongTermMemory
from app.memory_modules.preferences import PreferencesStore
from app.memory_modules.short_term  import SessionStore as ShortTermMemory
from app.alerts.temporal_manager    import TemporalAlertManager
from app.voice.tts                  import TTSEngine
from app.voice.stt                  import WhisperSTT

logger = logging.getLogger(__name__)


# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Runs once on startup (before any request is served) and once on
    shutdown (after the server stops accepting requests).

    All heavy model loads go here — never in a request handler — so the
    first real frame isn't slowed by lazy initialisation.

    Everything stored in app.state is available inside every router and
    WebSocket handler as:
        request.app.state.<name>      (HTTP routes)
        websocket.app.state.<name>    (WebSocket routes)
    """

    # ── Startup ────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("MemoryNav %s — starting up", settings.APP_VERSION)
    logger.info("Device: %s", settings.device)

    # Config sanity check
    logger.info("Settings loaded — device: %s", settings.device)

    # 1. Perception — loaded in Phase 1-2 order so each builds on the last
    logger.info("[1/7] Loading YOLOv8 detector …")
    app.state.detector = Detector()

    logger.info("[2/7] Loading Depth-Anything estimator …")
    app.state.depth_estimator = DepthEstimator()

    logger.info("[3/7] Loading EasyOCR reader …")
    app.state.ocr_reader = OCRReader()

    # 2. Memory — Phase 3
    logger.info("[4/7] Opening long-term memory (ChromaDB) …")
    app.state.long_term_memory = LongTermMemory()

    logger.info("[5/7] Opening preferences store (SQLite) …")
    app.state.preferences_store = PreferencesStore()

    # Short-term memory is cheap (plain dict) — still initialise here so
    # every request gets the same session object rather than a new one.
    app.state.short_term_memory = ShortTermMemory()

    # 3. Alerts — Phase 4
    logger.info("[6/7] Initialising temporal alert manager …")
    app.state.alert_manager = TemporalAlertManager()

    # 4. Voice — Phase 5
    logger.info("[7/7] Loading TTS + Whisper STT …")
    app.state.tts_engine  = TTSEngine()
    app.state.stt_engine  = WhisperSTT()

    elapsed = time.monotonic() - t0
    logger.info("Startup complete in %.1f s — ready on %s:%s",
                elapsed, settings.HOST, settings.PORT)
    logger.info("=" * 60)

    yield   # server is live — handle requests

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("MemoryNav shutting down — releasing resources …")

    # TTS has a background thread that needs a clean exit.
    if hasattr(app.state, "tts_engine"):
        try:
            app.state.tts_engine.shutdown()
        except Exception:
            logger.warning("TTS shutdown raised; ignoring.", exc_info=True)

    # ChromaDB flushes its WAL automatically on GC, but being explicit
    # avoids the "collection modified after close" warning in logs.
    if hasattr(app.state, "long_term_memory"):
        try:
            app.state.long_term_memory._client.reset()
        except Exception:
            pass

    logger.info("Shutdown complete.")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MemoryNav API",
    description=(
        "Memory-Augmented Spatial Intelligence System for Indoor Navigation. "
        "Real-time obstacle detection, depth estimation, spatial memory retrieval, "
        "and offline voice guidance."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    # Disable default /docs redirect in prod if you want — keep on for dev.
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS ──────────────────────────────────────────────────────────────────────
# Next.js runs on :3000 in dev. In prod (Docker) they're on the same origin,
# but keeping this middleware present doesn't hurt.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,   # ["http://localhost:3000", ...]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────
#
# Prefix map (mirrors frontend/src/lib/api.ts call sites):
#
#   WebSocket  ws://localhost:8000/ws          ← live frame stream
#   REST       /memory/*                       ← home context CRUD
#   REST       /preferences/*                  ← user settings
#   REST       /voice                          ← STT + answer
#
# voice_router already has prefix="/voice" defined internally (same
# pattern as your other routers) so we don't double-prefix here.

app.include_router(ws_router)
app.include_router(memory_router)
app.include_router(preferences_router)
app.include_router(voice_router)


# ── Health endpoint ───────────────────────────────────────────────────────────
@app.get(
    "/health",
    tags=["system"],
    summary="Server + model readiness check",
    response_description="JSON with status, version, device, and loaded-module flags",
)
async def health() -> JSONResponse:
    """
    Called by:
      - docker-compose healthcheck
      - Next.js frontend on mount (to show a "connected" badge)
      - You, when debugging why the WebSocket isn't responding

    Returns 200 only when every model is loaded and ready.
    Returns 503 if startup is still in progress or a model failed to load.
    """
    modules = {
        "detector":          hasattr(app.state, "detector"),
        "depth_estimator":   hasattr(app.state, "depth_estimator"),
        "ocr_reader":        hasattr(app.state, "ocr_reader"),
        "long_term_memory":  hasattr(app.state, "long_term_memory"),
        "preferences_store": hasattr(app.state, "preferences_store"),
        "short_term_memory": hasattr(app.state, "short_term_memory"),
        "alert_manager":     hasattr(app.state, "alert_manager"),
        "tts_engine":        hasattr(app.state, "tts_engine"),
        "stt_engine":        hasattr(app.state, "stt_engine"),
    }

    all_ready = all(modules.values())
    status_code = 200 if all_ready else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status":  "ready" if all_ready else "starting",
            "version": settings.APP_VERSION,
            "device":  settings.device,
            "modules": modules,
        },
    )


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse({
        "name":    "MemoryNav API",
        "version": settings.APP_VERSION,
        "docs":    "/docs",
        "health":  "/health",
    })


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )