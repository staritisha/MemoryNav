"""
MemoryNav — Offline Text-to-Speech (pyttsx3)
backend/app/voice/tts.py

Module 5 (Voice Interface): offline, privacy-by-design speech output.
pyttsx3 wraps the OS's native TTS engine (NSSpeechSynthesizer on macOS,
SAPI5 on Windows, espeak on Linux) — no network call, no API key,
works even with zero internet connectivity.

This file is specifically the pyttsx3 backend. Picking between this
and a cloud engine (ElevenLabs) based on settings.effective_tts_engine
is a higher-level concern that belongs to a voice-output dispatcher,
not here.

--- Sub-200ms target: what it actually means ---
pyttsx3's runAndWait() blocks until the ENTIRE utterance finishes
playing — for a real sentence ("a chair is two meters ahead") that's
1-2+ seconds, physically. No offline TTS engine makes speaking a full
sentence happen in 200ms. What the 200ms budget actually applies to is
DISPATCH latency: the time from calling speak() to audio starting —
i.e. how long the caller (Risk Engine -> Alert Manager) is blocked
before speech *begins*. This module hits that by:
  1. Initializing the engine once at startup (engine init/voice
     enumeration is the slow part — don't pay for it per call).
  2. Running playback on a single dedicated background thread, so
     speak() just enqueues and returns immediately instead of blocking
     on runAndWait().
  3. Logging a warning if dispatch ever exceeds the budget, so
     regressions are visible instead of silent.

Usage:

    from app.voice.tts import TTSEngine

    tts = TTSEngine()
    tts.speak("Chair ahead, two meters")                    # queued
    tts.speak("Stop, obstacle very close", interrupt=True)  # cuts in immediately

Dependencies: pyttsx3.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import pyttsx3

from app.config import settings

logger = logging.getLogger(__name__)

# Dispatch-latency budget (see module docstring). Not in config.py — this
# is an internal implementation target for this module, not a tunable
# the rest of the app needs to know about.
_DISPATCH_LATENCY_TARGET_MS = 200


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    enqueued_at: float  # time.perf_counter() timestamp, for dispatch-latency logging


class TTSEngine:
    """
    Offline TTS via pyttsx3, owned by a single dedicated worker thread.

    pyttsx3 engines are NOT thread-safe and only tolerate one
    say()/runAndWait() cycle in flight at a time — calling runAndWait()
    concurrently from multiple threads is a well-known source of
    "run loop already started" errors and hangs, especially on macOS's
    NSSpeechSynthesizer driver. Routing every utterance through one
    queue + one worker thread sidesteps that entirely: speak() is safe
    to call from any thread (the main capture loop, the Alert Manager,
    a FastAPI request handler) without touching the engine directly.
    """

    def __init__(self, rate_wpm: Optional[int] = None) -> None:
        self.rate_wpm = rate_wpm or settings.TTS_RATE_WPM
        self.available = False
        self._engine: Optional[Any] = None
        self._queue: "queue.Queue[Optional[SpeechRequest]]" = queue.Queue()
        self._speaking = threading.Event()
        self._shutdown = threading.Event()

        self._init_engine()

        self._worker = threading.Thread(target=self._run_worker, daemon=True, name="tts-worker")
        self._worker.start()

    def _init_engine(self) -> None:
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate_wpm)
            self.available = True
            logger.info("pyttsx3 engine initialized (rate=%d wpm).", self.rate_wpm)
        except Exception:
            # No native TTS driver available (e.g. headless CI without
            # espeak) — degrade gracefully instead of crashing startup.
            self._engine = None
            self.available = False
            logger.warning(
                "pyttsx3 failed to initialize; TTSEngine will log instead of "
                "speaking. Install/verify your OS's native TTS driver.",
                exc_info=True,
            )

    def _run_worker(self) -> None:
        """
        Owns the pyttsx3 engine exclusively. Pulls one request at a time
        off the queue and speaks it to completion before pulling the
        next — the only thread ever allowed to touch self._engine.
        """
        while not self._shutdown.is_set():
            request = self._queue.get()
            if request is None:  # shutdown sentinel
                break

            dispatch_ms = (time.perf_counter() - request.enqueued_at) * 1000
            if dispatch_ms > _DISPATCH_LATENCY_TARGET_MS:
                logger.warning(
                    "TTS dispatch latency %.0fms exceeded %dms target for: %r",
                    dispatch_ms,
                    _DISPATCH_LATENCY_TARGET_MS,
                    request.text,
                )

            if not self.available or self._engine is None:
                logger.info("[TTS unavailable] would speak: %r", request.text)
                continue

            self._speaking.set()
            try:
                self._engine.say(request.text)
                self._engine.runAndWait()
            except Exception:
                logger.error("pyttsx3 failed mid-utterance: %r", request.text, exc_info=True)
                # A failed engine can get stuck — reinitialize so the next
                # request isn't doomed too.
                self._init_engine()
            finally:
                self._speaking.clear()

    def speak(self, text: str, interrupt: bool = False) -> None:
        """
        Queues `text` for speech. Returns immediately — dispatch only,
        see module docstring for what the 200ms target actually covers.

        interrupt=True: stops whatever's currently playing and drops
        anything else queued, so this utterance starts next. Use for
        HIGH-risk alerts that must cut in over a MEDIUM-risk one already
        playing.
        interrupt=False (default): appends behind whatever's already
        playing. Use for MEDIUM-risk alerts that should queue, not
        interrupt.
        """
        if not text or not text.strip():
            return

        if interrupt:
            self._clear_queue()
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    logger.warning("pyttsx3 stop() failed during interrupt.", exc_info=True)

        self._queue.put(SpeechRequest(text=text, enqueued_at=time.perf_counter()))

    def _clear_queue(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def stop(self) -> None:
        """
        Immediately silences any current/queued speech without queuing
        anything new — a hard mute, not an interrupt-with-new-alert.
        """
        self._clear_queue()
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                logger.warning("pyttsx3 stop() failed.", exc_info=True)

    def shutdown(self) -> None:
        """Cleanly stops the worker thread. Call on app shutdown."""
        self.stop()
        self._shutdown.set()
        self._queue.put(None)  # wake the worker so it can exit
        self._worker.join(timeout=2.0)


if __name__ == "__main__":
    # Quick manual check: `python -m app.voice.tts`
    # Note: this actually speaks out loud if a native TTS driver is available.
    logging.basicConfig(level=logging.INFO)
    tts = TTSEngine()

    start = time.perf_counter()
    tts.speak("Memory nav voice test, chair two meters ahead.")
    print(f"speak() returned in {(time.perf_counter() - start) * 1000:.1f}ms (dispatch only)")

    time.sleep(4)  # let it actually finish talking before the process exits
    tts.shutdown()