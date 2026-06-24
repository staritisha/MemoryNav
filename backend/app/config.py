"""
MemoryNav — Backend Configuration
backend/app/config.py

Single source of truth for: model paths, confidence thresholds, device
selection (mps/cuda/cpu), and per-module settings for all six layers
described in the architecture doc (Perception, Risk Engine, Memory,
Alert Manager, Voice Interface, LLM Layer).

Usage elsewhere in the app:

    from app.config import settings

    model = YOLO(settings.YOLO_MODEL_PATH).to(settings.device)

All values can be overridden via environment variables or a `.env`
file placed in `backend/` (see `.env.example`). Nothing here requires
network access — every default is local/offline per the Privacy-by-Design
section of the architecture doc.

Dependencies: pydantic-settings, torch
    pip install pydantic-settings torch
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent              # backend/app
BACKEND_DIR = APP_DIR.parent                            # backend/
PROJECT_ROOT = BACKEND_DIR.parent                        # memorynav/

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
CHROMA_DIR = DATA_DIR / "chroma_store"
SQLITE_PATH = DATA_DIR / "memorynav.db"

for _dir in (DATA_DIR, MODELS_DIR, CHROMA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def detect_device() -> str:
    """
    Auto-detect the best available inference device.
    Priority: Apple Silicon MPS -> CUDA -> CPU.

    The architecture doc targets an M2 MacBook (MPS backend, ~30 FPS
    YOLOv8-nano). This falls back gracefully so the same code runs on
    CUDA boxes or plain CPU during development/CI.
    """
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass  # older torch builds without MPS support
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    """
    Application-wide settings, grouped by the module that owns them
    (matches Section 3.2 of the architecture doc).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App ----
    APP_NAME: str = "MemoryNav"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # ---- Device ----
    # "auto" resolves via detect_device() at import time. Override with
    # DEVICE=cpu in .env to force CPU (useful for debugging MPS issues).
    DEVICE: Literal["auto", "mps", "cuda", "cpu"] = "auto"

    @property
    def device(self) -> str:
        return detect_device() if self.DEVICE == "auto" else self.DEVICE

    # ------------------------------------------------------------------- #
    # Module 1 — Perception Layer
    # ------------------------------------------------------------------- #
    YOLO_MODEL_PATH: str = str(MODELS_DIR / "yolov8n.pt")
    DEPTH_MODEL_NAME: str = "LiheYoung/depth-anything-small-hf"
    EASYOCR_LANGUAGES: list[str] = ["en"]
    # Note: EasyOCR's GPU flag only accelerates via CUDA, not MPS — on an
    # M2 Mac this silently runs on CPU even when True. Harmless either way.
    EASYOCR_GPU: bool = True

    # Frame Quality Check (prevents bad inference reaching the pipeline)
    FRAME_MIN_BRIGHTNESS: float = 20.0             # mean pixel value, 0-255
    FRAME_MIN_BLUR_VARIANCE: float = 3.0           # Laplacian variance floor

    # Minimum confidence for YOLO to report a detection at all (inference-time)
    YOLO_DETECTION_CONFIDENCE: float = 0.25

    # ------------------------------------------------------------------- #
    # Module 5 — Voice Interface: Confidence Gating
    # Below this, the voice layer hedges instead of asserting an obstacle:
    # "something may be ahead, I am not certain." Never hallucinate.
    # ------------------------------------------------------------------- #
    YOLO_CONFIDENCE_GATE: float = 0.60

    # ------------------------------------------------------------------- #
    # Module 2 — Risk Engine
    # Risk Score = (1 / distance_metres) x motion_factor x user_context_weight
    # ------------------------------------------------------------------- #
    RISK_HIGH_THRESHOLD: float = 0.55       # score > 0.55 -> HIGH, interrupt now
    RISK_MEDIUM_THRESHOLD: float = 0.35     # 0.35-0.55    -> MEDIUM, queue
    RISK_HIGH_DISTANCE_M: float = 0.7       # obstacle under 0.7m
    RISK_MEDIUM_DISTANCE_M: float = 2.0     # obstacle 0.7m - 2m, else LOW/log-only

    # ------------------------------------------------------------------- #
    # Module 3 — Memory System
    # ------------------------------------------------------------------- #
    SHORT_TERM_MEMORY_WINDOW_SECONDS: int = 30
    CHROMA_PERSIST_DIR: str = str(CHROMA_DIR)
    CHROMA_COLLECTION_NAME: str = "memorynav_spatial"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    SQLITE_DB_PATH: str = str(SQLITE_PATH)          # user preference memory

    # ------------------------------------------------------------------- #
    # Module 4 — Temporal Alert Manager (WalkVLM-inspired)
    # ------------------------------------------------------------------- #
    ALERT_SUPPRESSION_WINDOW_SECONDS: float = 4.0

    # ------------------------------------------------------------------- #
    # Module 5 — Voice Interface
    # ------------------------------------------------------------------- #
    WHISPER_MODEL_SIZE: Literal["tiny", "base", "small", "medium", "large"] = "base"
    TTS_ENGINE: Literal["pyttsx3", "elevenlabs"] = "pyttsx3"
    TTS_RATE_WPM: int = 175
    ELEVENLABS_API_KEY: Optional[str] = None
    ELEVENLABS_VOICE_ID: Optional[str] = None

    # ------------------------------------------------------------------- #
    # Module 6 — LLM Enhancement Layer (optional, on-demand only)
    # ------------------------------------------------------------------- #
    LLM_ENABLED: bool = False
    OPENAI_API_KEY: Optional[str] = None
    LLM_VISION_MODEL: str = "gpt-4o"

    # ------------------------------------------------------------------- #
    # Privacy-by-Design (Section 3.3)
    # Master switch — must be explicitly True before any cloud call
    # (ElevenLabs TTS / GPT-4o Vision) is permitted to fire. Everything
    # else (YOLO, Depth-Anything, EasyOCR, Whisper, ChromaDB) is local
    # regardless of this flag.
    # ------------------------------------------------------------------- #
    ALLOW_CLOUD_SERVICES: bool = False

    @property
    def effective_tts_engine(self) -> Literal["pyttsx3", "elevenlabs"]:
        """Falls back to offline TTS if cloud services are disabled."""
        if self.TTS_ENGINE == "elevenlabs" and not self.ALLOW_CLOUD_SERVICES:
            return "pyttsx3"
        return self.TTS_ENGINE

    @property
    def llm_layer_active(self) -> bool:
        """LLM layer only fires when explicitly enabled AND cloud is allowed."""
        return self.LLM_ENABLED and self.ALLOW_CLOUD_SERVICES

    # ------------------------------------------------------------------- #
    # Camera / capture
    # ------------------------------------------------------------------- #
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 30

    # ------------------------------------------------------------------- #
    # FastAPI / server (Module: Backend + API, Phase 6)
    # ------------------------------------------------------------------- #
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    @field_validator("ELEVENLABS_API_KEY", "OPENAI_API_KEY", mode="before")
    @classmethod
    def _blank_string_to_none(cls, v):
        """Treat empty-string env vars (e.g. unset in .env) as None."""
        return v or None


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor. Prefer importing `settings` below; use this
    directly only if you need a fresh instance (e.g. in tests with
    monkeypatched env vars + get_settings.cache_clear()).
    """
    return Settings()


settings = get_settings()


if __name__ == "__main__":
    # Quick sanity check: `python -m app.config`
    print(f"{settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Resolved device:        {settings.device}")
    print(f"YOLO model path:        {settings.YOLO_MODEL_PATH}")
    print(f"Confidence gate:        {settings.YOLO_CONFIDENCE_GATE}")
    print(f"Cloud services allowed: {settings.ALLOW_CLOUD_SERVICES}")
    print(f"Effective TTS engine:   {settings.effective_tts_engine}")