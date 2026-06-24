"""
MemoryNav — GPT-4o Vision Scene Describer
backend/app/llm/scene_describer.py

Module 6 (LLM Enhancement Layer — Optional): rich scene description
via GPT-4o Vision, called only when the user explicitly asks for it.

Architecture doc, Module 6:
    "GPT-4o Vision is NOT the foundation. It is an optional enhancement
     called only when the user explicitly asks a complex question
     ('describe everything around me', 'read this label').
     The system works fully without it."

Research backing (VIALM survey, 2024):
    "Large language models frequently fail to provide grounded navigation
     instructions even when they understand the scene. Rule-based
     perception with memory is more reliable for safety-critical guidance.
     LLMs are best used for rich description on demand, not for
     navigation decisions."

This module exists to fulfil the "on demand" half of that sentence —
it is deliberately isolated from the core navigation pipeline (detector,
risk engine, alert manager) so that a GPT-4o failure, API outage, or
the user's choice to run offline never touches those paths.

Privacy guarantee (architecture doc, Privacy Architecture):
    "Cloud services (ElevenLabs, GPT-4o) are opt-in only and clearly
     disclosed to the user. No user data is logged, sold, or used for
     model training."

This module enforces that by:
    1. Hard-gating behind settings.LLM_ENABLED (default False).
    2. Logging a CLOUD_CALL audit line every time a frame is sent.
    3. Never caching frames in memory or on disk.
    4. Raising SceneDescriberDisabledError (not silently skipping)
       when called without LLM_ENABLED — so nothing accidentally
       sends a frame to OpenAI if the flag is unset.

Where this fits in the pipeline
--------------------------------
The core pipeline runs without this module entirely:

    YOLO → Depth → Risk → Alert Manager → Voice
                               ↑
                    Memory (short + long term)

SceneDescriber is called only from voice_router.py, and only when the
intent classifier returns SCENE_DESCRIPTION (e.g. "describe everything
around me"). The caller must first check is_available() or catch
SceneDescriberDisabledError if it wants to degrade gracefully.

Memory grounding
-----------------
GPT-4o alone sees only the current frame — it doesn't know about the
step near the kitchen door the user told the system about last week.
This wrapper accepts `context` (long-term memory retrieval results)
and `recent_obstacles` (short-term session state) and injects them
into the system prompt before the vision call. This is what the
architecture doc means by "grounding" — the LLM's output references
the actual home, not a generic description of the image.

    YOLO detects a sofa at 0.8 m
    Long-term memory retrieves: "loose rug near sofa"
    GPT-4o system prompt includes: "Home note: loose rug near sofa"
    → Output: "There's a sofa about a metre ahead. Be careful of the
               rug you mentioned near it — I can see it to your left."

Usage
-----
    from app.llm.scene_describer import SceneDescriber

    describer = SceneDescriber()

    if describer.is_available():
        text = describer.describe(
            frame,                          # BGR numpy array from OpenCV
            question="describe what's around me",
            context=["loose rug near sofa", "step at kitchen entrance"],
            recent_obstacles="chair at 0.5m on your right",
        )
        if text:
            voice.speak(text)
    else:
        voice.speak("Scene description requires the LLM layer. "
                    "Enable it in settings.")

    # Simple call (question + context optional):
    text = describer.describe(frame)

Dependencies: openai (>=1.0), opencv-python, numpy.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class SceneDescriberDisabledError(RuntimeError):
    """
    Raised when describe() is called but the LLM layer is not enabled.

    Callers must either check is_available() before calling, or catch
    this exception and fall back to the offline pipeline.  It is never
    silently swallowed here — the caller must make a conscious choice.
    """


class SceneDescriberError(RuntimeError):
    """Raised for unrecoverable errors during a describe() call (API failures,
    frame encoding errors, etc.) when the caller asked for strict mode."""


# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_MODEL:      str   = "gpt-4o"
_DEFAULT_MAX_TOKENS: int   = 150    # concise for TTS — 150 tokens ≈ 2-3 sentences
_DEFAULT_DETAIL:     str   = "low"  # "low" = 85 tokens, fast; adequate for navigation

# "low" detail pros: ~3x cheaper, ~2x faster, sufficient for obstacle-level
# spatial understanding. "high" is only needed for fine text reading, which
# OCRReader (EasyOCR, offline) already handles — no need for "high" here.

_JPEG_QUALITY: int = 85   # encode frame before b64; 85 = good quality / size balance


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DescribeResult:
    """
    Output of SceneDescriber.describe().

    Attributes
    ----------
    text         : GPT-4o's spoken description — ready to pass to TTS.
                   None when the call failed and strict_mode=False.
    latency_ms   : Wall-clock time of the API call in milliseconds.
                   Useful for ablation study latency measurement.
    used_context : True when long-term memory context was injected into
                   the system prompt (i.e. context list was non-empty).
    model        : Which model produced this result (e.g. "gpt-4o").
    """
    text:          Optional[str]
    latency_ms:    float
    used_context:  bool
    model:         str

    @property
    def succeeded(self) -> bool:
        return self.text is not None


# ── System prompt builder ─────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = (
    "You are a navigation assistant for a visually impaired or elderly user. "
    "The user is wearing a chest-mounted camera in their home. "
    "Describe the scene in 2-3 short, clear sentences. "
    "Focus on: obstacles and their approximate distance, safe paths forward, "
    "and anything that could cause a fall (steps, rugs, cables, wet surfaces). "
    "Use plain language — no technical terms, no confidence scores, no hedging. "
    "Do not say 'I can see' or 'the image shows' — speak directly to the user."
)

_CONTEXT_PREFIX  = "Home memory notes (facts the user has told you about this space): "
_OBSTACLE_PREFIX = "Recent obstacles detected in the last 30 seconds: "


def _build_system_prompt(
    context:          Optional[List[str]],
    recent_obstacles: Optional[str],
) -> str:
    """
    Compose the system prompt by appending available memory context.

    Both context sources are optional — if neither is provided the base
    prompt is used as-is and GPT-4o works from the image alone.

    The order matters for GPT-4o:
        1. Role + output format instructions (always first)
        2. Home memory notes from long-term ChromaDB (high reliability)
        3. Recent YOLO obstacle state (high recency, lower reliability)
    This ordering gives the model stable home facts before the noisier
    real-time sensor data, matching how a human would want to reason.
    """
    parts = [_BASE_SYSTEM_PROMPT]

    if context:
        clean = [c.strip() for c in context if c.strip()]
        if clean:
            parts.append(_CONTEXT_PREFIX + "; ".join(clean) + ".")

    if recent_obstacles and recent_obstacles.strip():
        parts.append(_OBSTACLE_PREFIX + recent_obstacles.strip() + ".")

    return " ".join(parts)


# ── Frame encoder ─────────────────────────────────────────────────────────────

def _encode_frame(frame: np.ndarray) -> Optional[str]:
    """
    Encode a BGR numpy array as a JPEG data URL string for the OpenAI
    Vision API.

    Returns None if encoding fails (e.g. empty / corrupted frame).
    Logs a WARNING rather than raising so the caller can fall back
    instead of crashing.
    """
    if frame is None or frame.size == 0:
        logger.warning("[SceneDescriber] encode_frame: empty frame received.")
        return None

    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not ok or buf is None:
            logger.warning("[SceneDescriber] cv2.imencode returned failure.")
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        logger.error("[SceneDescriber] Frame encoding failed.", exc_info=True)
        return None


# ── Main class ────────────────────────────────────────────────────────────────

class SceneDescriber:
    """
    GPT-4o Vision wrapper — describe a scene frame on explicit user request.

    This class is intentionally thin: one method (describe), one purpose
    (rich on-demand description), no navigation decisions.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key. Defaults to settings.OPENAI_API_KEY.
        Passing it explicitly is useful in tests (avoids reading settings).
    model : str
        OpenAI model to use. Default "gpt-4o". The doc says "pick one
        (GPT-4o)" — do not add alternative providers here.
    max_tokens : int
        Maximum tokens in the description. 150 ≈ 2–3 sentences, which
        is the limit of what a user can comfortably absorb while walking.
    image_detail : str
        "low" (default) or "high". "low" is sufficient for obstacle-level
        spatial understanding and is ~3× cheaper and ~2× faster.
        Use "high" only if users consistently report missing detail.
    strict_mode : bool
        If True, failures raise SceneDescriberError instead of returning
        None. Useful in tests. Default False (production degrades gracefully).
    """

    def __init__(
        self,
        api_key:      Optional[str] = None,
        model:        str           = _DEFAULT_MODEL,
        max_tokens:   int           = _DEFAULT_MAX_TOKENS,
        image_detail: str           = _DEFAULT_DETAIL,
        strict_mode:  bool          = False,
    ) -> None:
        self._api_key     = api_key or getattr(settings, "OPENAI_API_KEY", None)
        self.model        = model
        self.max_tokens   = max_tokens
        self.image_detail = image_detail
        self.strict_mode  = strict_mode

        # Ablation / audit counters
        self._total_calls:    int   = 0
        self._successful:     int   = 0
        self._failed:         int   = 0
        self._disabled_calls: int   = 0   # calls blocked by LLM_ENABLED=False
        self._total_latency:  float = 0.0

    # ── Guard ─────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        True when the LLM layer is enabled in settings AND an API key is
        present. Use this to decide whether to even offer a scene description
        to the user — don't call describe() and catch the exception in hot
        paths.
        """
        return (
            bool(getattr(settings, "LLM_ENABLED", False))
            and bool(self._api_key)
        )

    # ── Primary API ───────────────────────────────────────────────────────────

    def describe(
        self,
        frame:            np.ndarray,
        question:         Optional[str]       = None,
        context:          Optional[List[str]] = None,
        recent_obstacles: Optional[str]       = None,
    ) -> Optional[str]:
        """
        Describe the scene in `frame` via GPT-4o Vision.

        Args:
            frame            : BGR numpy array (OpenCV convention).
                               The only data sent to the cloud — never
                               cached or stored by this module.
            question         : The user's verbatim question (e.g.
                               "describe everything around me"). Included
                               in the user message so GPT-4o can tailor
                               its answer. If None a default prompt is used.
            context          : Long-term memory retrieval results for this
                               scene (e.g. ["loose rug near sofa",
                               "step at kitchen entrance"]). Injected into
                               the system prompt to ground GPT-4o's output
                               in the actual home, not a generic description.
            recent_obstacles : Short-term memory summary string from the
                               last 30s of YOLO detections (e.g. "chair at
                               0.5m on your right"). Also injected into the
                               system prompt.

        Returns:
            A 2–3 sentence spoken description string, or None when:
              - LLM_ENABLED is False / API key missing  (also raises
                SceneDescriberDisabledError — callers must handle this)
              - Frame encoding fails
              - The OpenAI API call fails
              (None is returned in the last two cases only when
               strict_mode=False; otherwise SceneDescriberError is raised.)

        Privacy note: every call emits an INFO-level CLOUD_CALL audit log
        line so it is always visible in production logs that a frame left
        the device. This satisfies the architecture doc's "clearly disclosed"
        requirement at the technical level (the UI disclosure happens in
        the settings/onboarding flow, not here).
        """
        self._total_calls += 1

        # ── Gate: LLM_ENABLED + API key ───────────────────────────────────
        if not self.is_available():
            self._disabled_calls += 1
            msg = (
                "SceneDescriber.describe() called but LLM_ENABLED is False "
                "or OPENAI_API_KEY is not set. Call is_available() first, or "
                "enable the LLM layer in settings."
            )
            logger.warning("[SceneDescriber] %s", msg)
            raise SceneDescriberDisabledError(msg)

        # ── Encode frame ──────────────────────────────────────────────────
        data_url = _encode_frame(frame)
        if data_url is None:
            self._failed += 1
            err = "Frame encoding failed — cannot send to GPT-4o."
            logger.error("[SceneDescriber] %s", err)
            if self.strict_mode:
                raise SceneDescriberError(err)
            return None

        # ── Build prompts ─────────────────────────────────────────────────
        system_prompt = _build_system_prompt(context, recent_obstacles)
        user_text     = question or "Please describe what you can see."

        # ── CLOUD_CALL audit log ──────────────────────────────────────────
        # This line must appear in production logs every time a frame is
        # sent to OpenAI. The architecture doc's privacy guarantee requires
        # that cloud calls are "clearly disclosed" — at the code level that
        # means an unambiguous audit trail.
        logger.info(
            "[SceneDescriber] CLOUD_CALL  model=%s  detail=%s  "
            "context_items=%d  question=%r",
            self.model,
            self.image_detail,
            len(context) if context else 0,
            user_text[:80],
        )

        # ── GPT-4o Vision call ────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            result = self._call_api(data_url, system_prompt, user_text)
        except Exception as exc:
            self._failed += 1
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._total_latency += elapsed_ms
            logger.error(
                "[SceneDescriber] API call failed after %.0f ms: %s",
                elapsed_ms, exc, exc_info=True,
            )
            if self.strict_mode:
                raise SceneDescriberError(f"GPT-4o call failed: {exc}") from exc
            return None

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._total_latency += elapsed_ms
        self._successful += 1

        logger.info(
            "[SceneDescriber] SUCCESS  latency=%.0f ms  tokens_approx=%d  "
            "text=%r",
            elapsed_ms,
            len(result.split()),   # rough word count for the log
            result[:120],
        )
        return result

    def describe_result(
        self,
        frame:            np.ndarray,
        question:         Optional[str]       = None,
        context:          Optional[List[str]] = None,
        recent_obstacles: Optional[str]       = None,
    ) -> DescribeResult:
        """
        Same as describe() but returns a DescribeResult dataclass instead
        of a plain string — useful when the caller wants latency data or
        needs to know whether memory context was used (ablation study).

        Raises SceneDescriberDisabledError if the LLM layer is not enabled
        (same as describe()).  Never raises on API failure — returns a
        DescribeResult with text=None and succeeded=False instead.
        """
        t0           = time.monotonic()
        used_context = bool(context)

        try:
            text = self.describe(
                frame,
                question=question,
                context=context,
                recent_obstacles=recent_obstacles,
            )
        except SceneDescriberDisabledError:
            raise
        except Exception:
            text = None

        latency_ms = (time.monotonic() - t0) * 1000
        return DescribeResult(
            text=text,
            latency_ms=round(latency_ms, 1),
            used_context=used_context,
            model=self.model,
        )

    # ── Private: OpenAI call ──────────────────────────────────────────────────

    def _call_api(
        self,
        data_url:      str,
        system_prompt: str,
        user_text:     str,
    ) -> str:
        """
        Make the GPT-4o Vision API call and return the response text.

        Raises openai.OpenAIError (or subclasses) on API failures —
        the caller (describe()) handles these.

        API parameters:
            temperature=0    Deterministic — safety-critical assistive use;
                             stochastic output variation in navigation
                             descriptions is undesirable.
            max_tokens=150   Keeps descriptions concise enough for TTS.
            detail="low"     85-token image budget — fast and cheap for
                             obstacle-level spatial understanding.
        """
        # openai is imported inside the method, not at module scope, for the
        # same reason as easyocr and whisper — callers that never use the LLM
        # layer (i.e. LLM_ENABLED=False) should not pay the import cost.
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type":      "image_url",
                            "image_url": {
                                "url":    data_url,
                                "detail": self.image_detail,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_text,
                        },
                    ],
                },
            ],
            max_tokens=self.max_tokens,
            temperature=0,
        )

        text = response.choices[0].message.content or ""
        return text.strip()

    # ── Ablation / audit metrics ──────────────────────────────────────────────

    @property
    def counts(self) -> dict:
        """
        Running counters since init or last reset_counts().

        Keys:
            total_calls    : every call to describe() including blocked ones
            successful     : returned a non-empty string
            failed         : API error or frame encoding failure
            disabled_calls : blocked by LLM_ENABLED=False / missing key
        """
        return {
            "total_calls":    self._total_calls,
            "successful":     self._successful,
            "failed":         self._failed,
            "disabled_calls": self._disabled_calls,
        }

    @property
    def mean_latency_ms(self) -> float:
        """
        Mean API call latency in milliseconds across calls that actually
        reached OpenAI (successful + failed, excluding disabled_calls).
        Zero if no API calls have been made.
        """
        api_calls = self._successful + self._failed
        if api_calls == 0:
            return 0.0
        return self._total_latency / api_calls

    def reset_counts(self) -> None:
        """Reset counters and latency accumulator (does not affect settings)."""
        self._total_calls    = 0
        self._successful     = 0
        self._failed         = 0
        self._disabled_calls = 0
        self._total_latency  = 0.0

    def stats_summary(self) -> str:
        """One-line summary for ablation study logs."""
        c = self.counts
        n = c["total_calls"]
        if n == 0:
            return "SceneDescriber: no calls yet"
        return (
            f"SceneDescriber — calls={n}  successful={c['successful']}  "
            f"failed={c['failed']}  disabled={c['disabled_calls']}  "
            f"mean_latency={self.mean_latency_ms:.0f}ms  "
            f"model={self.model}  detail={self.image_detail}  "
            f"llm_enabled={self.is_available()}"
        )


# ── Convenience module-level function ─────────────────────────────────────────

_default_describer: Optional[SceneDescriber] = None


def describe(
    frame:            np.ndarray,
    question:         Optional[str]       = None,
    context:          Optional[List[str]] = None,
    recent_obstacles: Optional[str]       = None,
) -> Optional[str]:
    """
    Module-level shortcut using a lazily-created default SceneDescriber.

    Raises SceneDescriberDisabledError if LLM_ENABLED is False — the
    caller must handle this rather than silently getting None, to avoid
    accidentally omitting the privacy disclosure step.

        from app.llm.scene_describer import describe, SceneDescriberDisabledError

        try:
            text = describe(frame, question="describe what's around me",
                            context=memory_context)
        except SceneDescriberDisabledError:
            text = offline_fallback_answer()
    """
    global _default_describer
    if _default_describer is None:
        _default_describer = SceneDescriber()
    return _default_describer.describe(
        frame,
        question=question,
        context=context,
        recent_obstacles=recent_obstacles,
    )


if __name__ == "__main__":
    # Smoke check: `python -m app.llm.scene_describer`
    # Verifies the disabled-gate and frame-encoder without needing an API key.
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    print("\n── MemoryNav SceneDescriber — smoke test ──\n")

    # ── Test 1: disabled gate raises correctly ──────────────────────────────
    print("TEST 1 — LLM_ENABLED=False → SceneDescriberDisabledError")

    # Temporarily force-disable to test the gate
    original_flag = getattr(settings, "LLM_ENABLED", False)
    settings.LLM_ENABLED = False

    describer = SceneDescriber()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        describer.describe(blank_frame)
        assert False, "Should have raised SceneDescriberDisabledError"
    except SceneDescriberDisabledError as e:
        print(f"  Correctly raised SceneDescriberDisabledError: {e}")
    print("  ✅ PASS\n")

    # ── Test 2: is_available() returns False when disabled ─────────────────
    print("TEST 2 — is_available() returns False without key/flag")
    assert not describer.is_available()
    print(f"  is_available()={describer.is_available()}  ✅ PASS\n")

    # ── Test 3: frame encoder handles blank frame without crashing ──────────
    print("TEST 3 — _encode_frame on a valid blank frame returns a data URL")
    data_url = _encode_frame(blank_frame)
    assert data_url is not None
    assert data_url.startswith("data:image/jpeg;base64,")
    print(f"  data_url[:40]={data_url[:40]}…  ✅ PASS\n")

    # ── Test 4: _encode_frame on empty array returns None ──────────────────
    print("TEST 4 — _encode_frame on empty array returns None")
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert _encode_frame(empty) is None
    print("  returned None  ✅ PASS\n")

    # ── Test 5: system prompt includes context and obstacle strings ─────────
    print("TEST 5 — _build_system_prompt injects memory context")
    prompt = _build_system_prompt(
        context=["loose rug near sofa", "step at kitchen entrance"],
        recent_obstacles="chair at 0.5m on your right",
    )
    assert "loose rug near sofa" in prompt
    assert "step at kitchen entrance" in prompt
    assert "chair at 0.5m" in prompt
    assert "Home memory notes" in prompt
    assert "Recent obstacles" in prompt
    print(f"  context + obstacles injected correctly  ✅ PASS\n")

    # ── Test 6: disabled call counter increments ────────────────────────────
    print("TEST 6 — disabled_calls counter tracks blocked requests")
    assert describer.counts["disabled_calls"] == 1   # from TEST 1
    assert describer.counts["total_calls"]    == 1
    print(f"  {describer.stats_summary()}")
    print("  ✅ PASS\n")

    # Restore original flag
    settings.LLM_ENABLED = original_flag

    print("── All 6 smoke tests passed ──\n")
    print("To test a live GPT-4o call, set LLM_ENABLED=True and OPENAI_API_KEY "
          "in your .env, then run again with a real camera frame.")