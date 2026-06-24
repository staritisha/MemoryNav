"""
MemoryNav — User Preferences REST Router
backend/app/api/preferences_router.py

Backend API (Phase 6): exposes UserPreferences (speech speed, language,
mobility flags, alert suppression window) over a simple REST interface
so the frontend (Phase 7) can drive a settings screen without touching
SQLite directly.

Routes
------
GET  /preferences                  read current saved preferences
PUT  /preferences                  update all fields in one request
PUT  /preferences/speech_rate      update speech rate only + apply live to TTS engine
PUT  /preferences/language         update language only
PUT  /preferences/alert_frequency  update alert suppression window only
PUT  /preferences/mobility         add or remove individual mobility flags

A PUT that changes speech_rate_wpm has an immediate side-effect:
it calls tts.setProperty('rate', ...) on the live TTSEngine so the
change takes effect on the next spoken alert — without restarting the
app. All other changes are SQLite-only (no live side-effect needed).

Mount in main.py:

    from app.api.preferences_router import router as preferences_router
    app.include_router(preferences_router)

Dependencies: fastapi, pydantic.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.memory_modules.preferences import DEFAULT_USER_ID, PreferencesStore, UserPreferences

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Singleton store + optional live TTS handle
# Both injected at app startup via init_preferences(); neither is required
# for the router to serve requests — TTS side-effects just silently skip
# if no engine was registered.
# --------------------------------------------------------------------------- #

_store: Optional[PreferencesStore] = None
_tts_engine = None  # type: ignore[assignment]  # TTSEngine, avoid circular import


def init_preferences(
    store: Optional[PreferencesStore] = None,
    tts_engine=None,
) -> PreferencesStore:
    """
    Call from your FastAPI lifespan (or main.py startup).
    `tts_engine` is the TTSEngine instance from ws_stream.PipelineState —
    pass it in so speech-rate PUTs take immediate effect.
    """
    global _store, _tts_engine
    _store = store or PreferencesStore()
    _tts_engine = tts_engine
    logger.info("PreferencesStore ready (db: %s).", _store.db_path)
    return _store


def get_store() -> PreferencesStore:
    """FastAPI dependency. Raises 503 if init_preferences() hasn't been called."""
    if _store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Preferences store not initialized. App may still be starting up.",
        )
    return _store


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #

class PreferencesOut(BaseModel):
    """Full preferences — returned by GET and all successful PUTs."""

    user_id: str
    speech_rate_wpm: int
    language: str
    mobility_flags: List[str]
    alert_suppression_seconds: float
    updated_at: Optional[str]

    @classmethod
    def from_domain(cls, p: UserPreferences) -> "PreferencesOut":
        return cls(
            user_id=p.user_id,
            speech_rate_wpm=p.speech_rate_wpm,
            language=p.language,
            mobility_flags=p.mobility_flags,
            alert_suppression_seconds=p.alert_suppression_seconds,
            updated_at=p.updated_at,
        )


class PreferencesPutRequest(BaseModel):
    """Full update body for PUT /preferences."""

    speech_rate_wpm: int = Field(
        ..., ge=80, le=400,
        description="TTS speech rate in words per minute. Typical range 80–400.",
    )
    language: str = Field(
        ..., min_length=2, max_length=10,
        description="BCP-47 language tag, e.g. 'en', 'es', 'hi'.",
    )
    mobility_flags: List[str] = Field(
        default_factory=list,
        description="Mobility context tags, e.g. ['bad_knee', 'uses_cane'].",
    )
    alert_suppression_seconds: float = Field(
        ..., ge=1.0, le=60.0,
        description=(
            "Minimum seconds between repeated spoken alerts for the same obstacle. "
            "Lower = more frequent alerts. Range 1–60s."
        ),
    )


class SpeechRatePutRequest(BaseModel):
    speech_rate_wpm: int = Field(..., ge=80, le=400)


class LanguagePutRequest(BaseModel):
    language: str = Field(..., min_length=2, max_length=10)


class AlertFrequencyPutRequest(BaseModel):
    alert_suppression_seconds: float = Field(
        ..., ge=1.0, le=60.0,
        description="Seconds between repeated spoken alerts for the same obstacle.",
    )


class MobilityFlagRequest(BaseModel):
    flag: str = Field(..., min_length=1, description="Flag to add or remove, e.g. 'bad_knee'.")
    action: str = Field(..., pattern="^(add|remove)$", description="'add' or 'remove'.")


# --------------------------------------------------------------------------- #
# Live side-effect helpers
# --------------------------------------------------------------------------- #

def _apply_speech_rate_to_tts(rate_wpm: int) -> None:
    """
    Applies a new speech rate to the live TTSEngine worker thread.
    pyttsx3's setProperty() is not thread-safe to call from outside the
    worker, but the TTSEngine's worker reads self.rate_wpm before each
    utterance — updating the attribute is enough for the change to land
    on the next speak() call without requiring a lock or queue message.
    """
    if _tts_engine is None:
        return
    try:
        _tts_engine.rate_wpm = rate_wpm
        if _tts_engine._engine is not None:
            _tts_engine._engine.setProperty("rate", rate_wpm)
        logger.info("TTS rate updated live to %d wpm.", rate_wpm)
    except Exception:
        logger.warning("Could not apply speech rate to live TTS engine.", exc_info=True)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/preferences", tags=["preferences"])


@router.get("", response_model=PreferencesOut, summary="Read current preferences")
async def get_preferences(store: PreferencesStore = Depends(get_store)) -> PreferencesOut:
    """
    Returns the current saved preferences for the default user.
    Returns config defaults if no preferences have been saved yet —
    never a 404 on first run.
    """
    prefs = store.load()
    return PreferencesOut.from_domain(prefs)


@router.put("", response_model=PreferencesOut, summary="Update all preferences at once")
async def update_all_preferences(
    body: PreferencesPutRequest,
    store: PreferencesStore = Depends(get_store),
) -> PreferencesOut:
    """
    Full replace of all preference fields in one request.
    Applies speech rate change live to the running TTS engine immediately.
    """
    prefs = store.load()
    prefs.speech_rate_wpm = body.speech_rate_wpm
    prefs.language = body.language
    prefs.mobility_flags = body.mobility_flags
    prefs.alert_suppression_seconds = body.alert_suppression_seconds
    saved = store.save(prefs)
    _apply_speech_rate_to_tts(saved.speech_rate_wpm)
    logger.info("PUT /preferences — full update saved.")
    return PreferencesOut.from_domain(saved)


@router.put("/speech_rate", response_model=PreferencesOut, summary="Update speech rate only")
async def update_speech_rate(
    body: SpeechRatePutRequest,
    store: PreferencesStore = Depends(get_store),
) -> PreferencesOut:
    """
    Updates speech rate and applies it immediately to the running TTS
    engine — no restart needed. The change is audible on the very next
    spoken alert.
    """
    prefs = store.load()
    prefs.speech_rate_wpm = body.speech_rate_wpm
    saved = store.save(prefs)
    _apply_speech_rate_to_tts(saved.speech_rate_wpm)
    return PreferencesOut.from_domain(saved)


@router.put("/language", response_model=PreferencesOut, summary="Update language only")
async def update_language(
    body: LanguagePutRequest,
    store: PreferencesStore = Depends(get_store),
) -> PreferencesOut:
    """
    Updates the language tag. Note: pyttsx3's offline TTS engine uses
    the OS voice configured at startup — changing language here affects
    any future cloud TTS calls (ElevenLabs, if enabled) and stores the
    preference for retrieval, but doesn't auto-switch the offline OS
    voice mid-session.
    """
    prefs = store.load()
    prefs.language = body.language
    saved = store.save(prefs)
    return PreferencesOut.from_domain(saved)


@router.put("/alert_frequency", response_model=PreferencesOut, summary="Update alert suppression window")
async def update_alert_frequency(
    body: AlertFrequencyPutRequest,
    store: PreferencesStore = Depends(get_store),
) -> PreferencesOut:
    """
    Controls how often the same obstacle triggers a spoken alert.
    `alert_suppression_seconds=4` (default) means a "chair ahead" warning
    won't repeat for 4 seconds even if the chair stays in frame.
    Increase for fewer interruptions; decrease for more frequent reminders.
    The WebSocket pipeline reads this from the saved preferences on every
    frame, so the change takes effect immediately with no restart.
    """
    prefs = store.load()
    prefs.alert_suppression_seconds = body.alert_suppression_seconds
    saved = store.save(prefs)
    logger.info("Alert suppression window updated to %.1fs.", saved.alert_suppression_seconds)
    return PreferencesOut.from_domain(saved)


@router.put("/mobility", response_model=PreferencesOut, summary="Add or remove a mobility flag")
async def update_mobility_flag(
    body: MobilityFlagRequest,
    store: PreferencesStore = Depends(get_store),
) -> PreferencesOut:
    """
    Adds or removes a single mobility context tag. Valid flags include
    'bad_knee', 'uses_cane', 'limited_mobility', or any custom string —
    these are free-form tags, not an enum. The Risk Engine uses them to
    scale user_context_weight; any unknown flag is stored but ignored
    by the current weight calculation.

    Example body:
        {"flag": "bad_knee", "action": "add"}
    """
    if body.action == "add":
        saved = store.add_mobility_flag(body.flag)
    else:
        saved = store.remove_mobility_flag(body.flag)
    return PreferencesOut.from_domain(saved)