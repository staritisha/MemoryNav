"""
MemoryNav — Voice Router
backend/app/api/voice_router.py

Phase 6 (Backend API): the single HTTP entry point for all voice
interaction. The frontend (Next.js, Phase 7) and the mobile client
both POST here when the user speaks.

Architecture doc — Module 5 (Voice Interface):
    "User asks: 'what is in front of me?'"
    "GPT-4o Vision is NOT the foundation. It is an optional enhancement
     called only when the user explicitly asks a complex question
     ('describe everything around me', 'read this label').
     The system works fully without it."

This router enforces that contract:
    - Every request goes through WhisperSTT first (offline).
    - The intent is classified by simple keyword matching — no agents,
      no LangChain (doc: "Do NOT add any of the following: LangChain
      agents or complex chains").
    - OBSTACLE_QUERY and MEMORY_QUERY are answered entirely offline
      using short-term + long-term memory.
    - OCR_QUERY routes to EasyOCR (offline).
    - SCENE_DESCRIPTION optionally calls GPT-4o Vision, and only when
      settings.LLM_ENABLED is True.  System degrades gracefully to an
      offline summary when the flag is off or the key is absent.

Pipeline (one request)
-----------------------
    POST /voice  (audio bytes + optional frame)
        ↓
    WhisperSTT.transcribe(audio_bytes)   → question text / None (silence)
        ↓
    _classify_intent(question)           → IntentType enum
        ↓
    ┌── OBSTACLE_QUERY  → ShortTermMemory.get_recent_summary()
    │                     + LongTermMemory.retrieve(question)
    │                     → compose answer
    │
    ├── OCR_QUERY       → frame required
    │                     → OCRReader.read_text(frame)
    │                     → compose answer
    │
    ├── SCENE_DESCRIPTION → LongTermMemory.retrieve(question)
    │                       + GPT-4o Vision (if LLM_ENABLED)
    │                       → compose answer
    │
    └── MEMORY_QUERY    → LongTermMemory.retrieve(question)
                          → compose answer
        ↓
    VoiceResponse (transcription, answer, intent, latency_ms, …)

Request format
--------------
    POST /voice
    Content-Type: multipart/form-data

    Fields:
        audio       : bytes   REQUIRED  WAV or raw PCM int16 at 16 kHz
        frame       : bytes   OPTIONAL  JPEG/PNG frame for OCR / scene queries
        language    : str     OPTIONAL  Whisper language code (default "en")
        session_id  : str     OPTIONAL  Caller session for short-term memory
                                        (single-user device: can omit)

Response (200 OK)
-----------------
    {
        "transcription" : "what is in front of me?",
        "answer"        : "Chair ahead, getting closer. Home note: loose rug near sofa.",
        "intent"        : "obstacle_query",
        "latency_ms"    : 143.7,
        "context"       : ["loose rug near sofa"],   // long-term memory hits
        "ocr_text"      : null                       // raw OCR, null when N/A
    }

Error responses
---------------
    400  Silence or empty audio — nothing was heard; user should try again.
    400  OCR query received but no frame was uploaded.
    422  FastAPI validation error (missing required field, wrong type).
    500  Internal error — logs full traceback, returns safe message to client.

Memory module interfaces assumed
---------------------------------
These modules are Phase 3 (items 12–14) and must exist before this
router is used in production. The interfaces assumed here match their
roadmap descriptions:

    preferences   (item 12):  get(key) → Any
    long_term     (item 13):  retrieve(query) → list[str]
    short_term    (item 14):  get_recent_summary() → str | None
                              record_voice_query(question, answer)

Adjust the TODO-marked call sites below if the actual signatures differ.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.perception.ocr import OCRReader
from app.voice.stt import WhisperSTT

# Import the shared pipeline state accessor so voice queries read from
# the same LongTermMemory and SessionStore instances that ws_stream.py
# writes to.  Calling module-level functions on the memory modules
# directly would create separate instances that never receive pipeline
# data — stale/empty results every time.
from app.api.ws_stream import get_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])


# ── Lazy singletons ───────────────────────────────────────────────────────────
# Loaded once on first request, not at module import time, so that running
# unit tests or importing the router does not trigger model downloads.

_stt: Optional[WhisperSTT] = None
_ocr: Optional[OCRReader]  = None


def _get_stt() -> WhisperSTT:
    global _stt
    if _stt is None:
        _stt = WhisperSTT()
    return _stt


def _get_ocr() -> OCRReader:
    global _ocr
    if _ocr is None:
        _ocr = OCRReader()
    return _ocr


# ── Memory module imports ─────────────────────────────────────────────────────
# Memory instances live on PipelineState (created once in ws_stream.init_pipeline).
# We access them via get_state() so voice queries always read from the same
# LongTermMemory and SessionStore that the WebSocket pipeline writes to.
# Falling back gracefully when the pipeline hasn't started yet (e.g. unit tests).

def _pipeline_available() -> bool:
    try:
        get_state()
        return True
    except RuntimeError:
        return False


# ── GPT-4o Vision (optional) ──────────────────────────────────────────────────
# Only imported when LLM_ENABLED is True so the offline-only path never
# touches the openai package at all.

def _llm_enabled() -> bool:
    return bool(getattr(settings, "LLM_ENABLED", False)) and bool(
        getattr(settings, "OPENAI_API_KEY", None)
    )


# ── Intent classification ─────────────────────────────────────────────────────

class IntentType(str, Enum):
    OBSTACLE_QUERY    = "obstacle_query"      # "what is in front of me?"
    OCR_QUERY         = "ocr_query"           # "what does this say?" / "read this"
    SCENE_DESCRIPTION = "scene_description"   # "describe everything around me"
    MEMORY_QUERY      = "memory_query"        # "where is the step?" (home memory)
    UNKNOWN           = "unknown"             # fallback → obstacle query path


# Keyword sets — kept as simple tuples, not regex.
# Doc: "Do NOT add LangChain agents or complex chains."
_OCR_KEYWORDS: tuple[str, ...] = (
    "what does this say",
    "what does it say",
    "read this",
    "read this label",
    "what is written",
    "what does the sign say",
    "what does the label say",
    "read the sign",
    "read the label",
    "read the text",
    "what does the text",
    "read ",       # catches "read the menu", "read this sign", etc.
)

_SCENE_KEYWORDS: tuple[str, ...] = (
    "describe everything",
    "describe what",
    "describe the room",
    "describe my surroundings",
    "describe around",
    "what is around me",
    "what's around me",
    "tell me everything",
)

_MEMORY_KEYWORDS: tuple[str, ...] = (
    "where is the",
    "remind me",
    "did i tell you",
    "what did i say about",
    "what's near the",
    "what is near the",
    "is there a step",
    "is there a rug",
)

# Obstacle query matches if none of the above matched and question asks about
# proximity.  Also the final fallback for any unrecognised query.
_OBSTACLE_KEYWORDS: tuple[str, ...] = (
    "what is in front",
    "what's in front",
    "what is ahead",
    "what's ahead",
    "what is there",
    "what's there",
    "what do you see",
    "is there anything",
    "is there something",
    "any obstacles",
    "what is close",
    "what's close",
)


def _classify_intent(question: str) -> IntentType:
    """
    Classify a transcribed question into one of four intent buckets.

    Simple keyword matching — deliberately not an LLM call.  Fast,
    deterministic, and works fully offline, which matters when the user
    is mid-stride and needs a sub-200ms response.

    Match priority: OCR > SCENE > MEMORY > OBSTACLE > UNKNOWN.
    UNKNOWN is treated as OBSTACLE_QUERY in the answer builder.
    """
    q = question.lower()

    for kw in _OCR_KEYWORDS:
        if kw in q:
            return IntentType.OCR_QUERY

    for kw in _SCENE_KEYWORDS:
        if kw in q:
            return IntentType.SCENE_DESCRIPTION

    for kw in _MEMORY_KEYWORDS:
        if kw in q:
            return IntentType.MEMORY_QUERY

    for kw in _OBSTACLE_KEYWORDS:
        if kw in q:
            return IntentType.OBSTACLE_QUERY

    return IntentType.UNKNOWN


# ── Pydantic response model ───────────────────────────────────────────────────

class VoiceResponse(BaseModel):
    """
    JSON body returned for every successful POST /voice.

    All fields are present on every response — null rather than absent —
    so the frontend doesn't need to guard every field access.
    """
    transcription: str = Field(
        description="Exact text Whisper heard from the audio."
    )
    answer: str = Field(
        description="The spoken answer to return to the user via TTS."
    )
    intent: str = Field(
        description="Classified intent: obstacle_query / ocr_query / "
                    "scene_description / memory_query / unknown."
    )
    latency_ms: float = Field(
        description="Wall-clock time from request receipt to response ready, "
                    "in milliseconds. Compare against the 200ms target."
    )
    context: List[str] = Field(
        default_factory=list,
        description="Long-term memory snippets retrieved to build this answer. "
                    "Empty list when long-term memory is unavailable or "
                    "returned no relevant context."
    )
    ocr_text: Optional[str] = Field(
        default=None,
        description="Raw OCR output when intent is ocr_query, null otherwise. "
                    "The 'answer' field is the formatted version of this."
    )


# ── Memory helpers ────────────────────────────────────────────────────────────

def _retrieve_long_term(query: str) -> list[str]:
    """
    Pull relevant home-layout context from the shared LongTermMemory instance.
    Returns an empty list when the pipeline is not yet initialized or the
    query returns no results.
    """
    if not _pipeline_available():
        return []
    try:
        results = get_state().long_term.retrieve(query)
        return [r.text for r in results]
    except Exception:
        logger.error("[voice_router] long_term.retrieve() failed.", exc_info=True)
        return []


def _get_short_term_summary() -> Optional[str]:
    """
    Get a text summary of the last 30s obstacles from the shared SessionStore.
    Returns None when the pipeline is not initialized or the window is empty.
    """
    if not _pipeline_available():
        return None
    try:
        return get_state().session.get_recent_summary()
    except Exception:
        logger.error("[voice_router] session.get_recent_summary() failed.", exc_info=True)
        return None


def _record_to_short_term(question: str, answer: str) -> None:
    """
    Log this voice interaction into the shared SessionStore so the temporal
    suppression layer and future queries within this session can see it.
    """
    if not _pipeline_available():
        return
    try:
        get_state().session.record_voice_query(question, answer)
    except Exception:
        logger.warning("[voice_router] session.record_voice_query() failed.", exc_info=True)


# ── Answer builders ───────────────────────────────────────────────────────────

def _answer_obstacle_query(question: str) -> tuple[str, list[str]]:
    """
    Build an answer for "what is in front of me?" and similar queries.

    Sources (in priority order):
        1. Short-term memory  — what YOLO has seen in the last 30s.
        2. Long-term memory   — home layout context relevant to the question.

    If both are empty the answer falls back to an honest "I don't have
    current obstacle data" rather than hallucinating.
    """
    context = _retrieve_long_term(question)
    recent  = _get_short_term_summary()

    parts: list[str] = []
    if recent:
        parts.append(recent)
    if context:
        parts.append("Home note: " + "; ".join(context))

    answer = (
        " ".join(parts)
        if parts
        else "I don't have current obstacle data. Make sure the camera is active."
    )
    return answer, context


def _answer_memory_query(question: str) -> tuple[str, list[str]]:
    """
    Answer a specific home-memory question ("where is the step?") by
    querying long-term ChromaDB directly with the full question string.
    """
    context = _retrieve_long_term(question)

    if context:
        answer = "I have this note about your home: " + "; ".join(context)
    else:
        answer = (
            "I don't have a note about that yet. "
            "You can add home context via the memory settings."
        )
    return answer, context


def _answer_ocr_query(frame_bytes: Optional[bytes]) -> tuple[str, Optional[str]]:
    """
    Run EasyOCR on the uploaded frame and format the result as speech.
    Returns (answer_string, raw_ocr_text_or_none).
    """
    if frame_bytes is None:
        return (
            "I need to see the text. Please point the camera at it and ask again.",
            None,
        )

    frame = _bytes_to_bgr(frame_bytes)
    if frame is None:
        return ("I could not process the image.", None)

    ocr_text = _get_ocr().read_text(frame)

    if ocr_text:
        answer = f"I can read: {ocr_text}"
    else:
        answer = "I can't make out any text right now. Try holding the camera closer."

    return answer, ocr_text


def _answer_scene_description(
    question: str, frame_bytes: Optional[bytes]
) -> tuple[str, list[str]]:
    """
    Describe the scene in detail.

    Offline path (always available):
        Long-term memory context + short-term recent obstacles.

    Enhanced path (LLM_ENABLED=True + OPENAI_API_KEY set):
        GPT-4o Vision call with the frame.
        Falls back to offline path on any error.

    Architecture doc: "GPT-4o Vision is NOT the foundation. It is an
    optional enhancement called only when the user explicitly asks a
    complex question."
    """
    context = _retrieve_long_term(question)
    recent  = _get_short_term_summary()

    if _llm_enabled() and frame_bytes is not None:
        gpt_answer = _call_gpt4o_vision(question, frame_bytes, context)
        if gpt_answer:
            return gpt_answer, context

    # Offline fallback
    parts: list[str] = []
    if recent:
        parts.append(f"Recent obstacles: {recent}.")
    if context:
        parts.append("Home notes: " + "; ".join(context) + ".")
    if not parts:
        parts.append(
            "I can see the camera feed but I don't have a detailed scene description "
            "available offline. Enable the LLM layer in settings for richer descriptions."
        )
    return " ".join(parts), context


def _call_gpt4o_vision(
    question: str, frame_bytes: bytes, context: list[str]
) -> Optional[str]:
    """
    Optional GPT-4o Vision call.  Only reached when LLM_ENABLED=True.
    Returns None on any error so the offline fallback always runs.

    Home context from long-term memory is injected into the system
    prompt so GPT-4o can reference room-specific notes the user has
    added ("there is a step at the kitchen entrance").
    """
    try:
        import base64
        from openai import OpenAI  # only imported on the LLM-enabled path

        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        b64_frame = base64.b64encode(frame_bytes).decode("utf-8")

        system_parts = [
            "You are an assistive navigation AI for a visually impaired user. "
            "Describe the scene concisely and clearly in 2-3 sentences. "
            "Focus on obstacles, distances, and safe paths. "
            "Never mention confidence scores or technical details."
        ]
        if context:
            system_parts.append(
                "Home context (from user's memory): " + "; ".join(context)
            )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": " ".join(system_parts)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_frame}",
                                "detail": "low",   # faster + cheaper for navigation
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                },
            ],
            max_tokens=150,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    except Exception:
        logger.error("[voice_router] GPT-4o Vision call failed.", exc_info=True)
        return None


# ── Image decode helper ───────────────────────────────────────────────────────

def _bytes_to_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode JPEG/PNG bytes to a BGR numpy array (OpenCV convention)
    so it can be passed to OCRReader and Detector unchanged.
    """
    try:
        import cv2
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame if frame is not None else None
    except Exception:
        logger.error("[voice_router] Frame decode failed.", exc_info=True)
        return None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=VoiceResponse,
    summary="Transcribe audio and return a navigation answer",
    responses={
        400: {"description": "Silence detected or audio empty / OCR frame missing"},
        500: {"description": "Internal processing error"},
    },
)
async def post_voice(
    audio: UploadFile = File(
        ...,
        description="Audio clip: WAV bytes or raw PCM int16 at 16 kHz mono. "
                    "Under 10s for reliable sub-200ms response.",
    ),
    frame: Optional[UploadFile] = File(
        None,
        description="Current camera frame (JPEG/PNG). Required for ocr_query "
                    "and scene_description intents; optional otherwise.",
    ),
    language: Optional[str] = Form(
        None,
        description="Whisper language code override: 'en', 'hi', 'ta', 'mr' … "
                    "Defaults to the user's preferred language from settings.",
    ),
    session_id: Optional[str] = Form(
        None,
        description="Caller session identifier for short-term memory isolation. "
                    "Single-user device: leave null — a default session is used.",
    ),
) -> VoiceResponse:
    """
    Primary voice interaction endpoint.

    Accepts an audio clip, transcribes it with Whisper (fully offline),
    classifies the question intent, and returns an answer drawn from
    short-term session state, long-term spatial memory, EasyOCR, or
    optionally GPT-4o Vision.

    All offline paths (Whisper + memory + OCR) target sub-200ms total
    latency. The GPT-4o Vision path adds ~800–1500ms network latency
    and is only reached when `settings.LLM_ENABLED` is True and the
    user explicitly asks for a full scene description.
    """
    t_start = time.monotonic()

    # ── 1. Read uploaded bytes ─────────────────────────────────────────────
    try:
        audio_bytes = await audio.read()
    except Exception:
        logger.error("[POST /voice] Failed to read audio upload.", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read audio data.",
        )

    frame_bytes: Optional[bytes] = None
    if frame is not None:
        try:
            frame_bytes = await frame.read()
        except Exception:
            logger.warning("[POST /voice] Failed to read frame upload — continuing without it.")

    # ── 2. Transcribe ──────────────────────────────────────────────────────
    try:
        question = _get_stt().transcribe(audio_bytes, language=language)
    except Exception:
        logger.error("[POST /voice] WhisperSTT.transcribe() raised.", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Speech transcription failed.",
        )

    if question is None:
        # Silence gate fired — nothing was heard.
        # Return 400 so the frontend can prompt the user to try again,
        # rather than returning a confusing empty answer.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No speech detected. Please try again in a quieter environment.",
        )

    logger.info('[POST /voice] transcribed: "%s"  session=%s', question, session_id or "default")

    # ── 3. Classify intent ─────────────────────────────────────────────────
    intent = _classify_intent(question)
    logger.info("[POST /voice] intent=%s", intent.value)

    # ── 4. Build answer ────────────────────────────────────────────────────
    context:  list[str]   = []
    ocr_text: Optional[str] = None

    try:
        if intent == IntentType.OCR_QUERY:
            answer, ocr_text = _answer_ocr_query(frame_bytes)

        elif intent == IntentType.SCENE_DESCRIPTION:
            answer, context = _answer_scene_description(question, frame_bytes)

        elif intent == IntentType.MEMORY_QUERY:
            answer, context = _answer_memory_query(question)

        else:
            # OBSTACLE_QUERY and UNKNOWN both fall into the obstacle path.
            # UNKNOWN means we couldn't classify — defaulting to "what's
            # near me?" is the safest assumption for an assistive device.
            intent = IntentType.OBSTACLE_QUERY if intent == IntentType.UNKNOWN else intent
            answer, context = _answer_obstacle_query(question)

    except Exception:
        logger.error("[POST /voice] Answer builder raised.", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build an answer. Please try again.",
        )

    # ── 5. Record to short-term memory ────────────────────────────────────
    # Non-blocking — a failure here must not fail the response.
    _record_to_short_term(question, answer)

    # ── 6. Assemble response ───────────────────────────────────────────────
    latency_ms = (time.monotonic() - t_start) * 1000
    logger.info(
        "[POST /voice] done  intent=%s  latency=%.0f ms  llm=%s",
        intent.value, latency_ms, _llm_enabled(),
    )

    return VoiceResponse(
        transcription=question,
        answer=answer,
        intent=intent.value,
        latency_ms=round(latency_ms, 1),
        context=context,
        ocr_text=ocr_text,
    )