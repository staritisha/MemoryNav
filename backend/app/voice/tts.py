"""
MemoryNav — Text-to-Speech Engine
backend/app/voice/tts.py

Supports two offline backends, selected by settings.effective_tts_engine:

  "kokoro"  (default) — Kokoro-82M neural TTS. Natural-sounding, fully
                        offline, ~330MB RAM. Runs on CPU/MPS via PyTorch.
                        Produces WAV audio played via sounddevice.

  "pyttsx3" (fallback) — OS-native TTS (NSSpeechSynthesizer on macOS).
                          Robotic but zero dependencies and instant startup.
                          Used automatically if kokoro fails to load.

Both backends expose the same public interface:

    tts = TTSEngine()
    tts.speak("Chair two meters ahead")              # queued, non-blocking
    tts.speak("Stop — obstacle very close", interrupt=True)  # cuts queue
    tts.shutdown()                                   # clean thread exit

The dispatch latency target is 200ms — speak() must return within that
window even though the utterance itself takes 1-3s to finish playing.
This is achieved by doing all synthesis and playback on a dedicated
daemon thread; speak() just enqueues and returns immediately.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

_DISPATCH_LATENCY_TARGET_MS = 200


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    enqueued_at: float = field(default_factory=time.perf_counter)


# ── Kokoro backend ────────────────────────────────────────────────────────────

class _KokoroBackend:
    """
    Kokoro-82M neural TTS backend.

    Voice: 'af_heart' — warm, natural American English female voice.
    Audio is synthesised to a float32 numpy array and played via
    sounddevice. If sounddevice is unavailable, falls back to writing
    a WAV file and playing with the OS afplay/aplay command.

    First call downloads ~330MB of model weights to the HuggingFace
    cache (~/.cache/huggingface). Subsequent startups load from cache
    in ~2-4s.
    """

    VOICE = "af_heart"   # warm, clear, natural — best for navigation alerts
    SAMPLE_RATE = 24000  # Kokoro native output sample rate

    def __init__(self) -> None:
        self.available = False
        self._pipeline = None
        self._sd = None          # sounddevice module, loaded lazily
        self._load()

    def _load(self) -> None:
        try:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code="a")  # "a" = American English
            # Warm-up: synthesise a short silent token so first real call
            # doesn't absorb model-init latency.
            _ = list(self._pipeline("hello", voice=self.VOICE, speed=1.0))
            self.available = True
            logger.info("Kokoro TTS ready (voice=%s, sr=%d).", self.VOICE, self.SAMPLE_RATE)
        except Exception:
            logger.warning(
                "Kokoro failed to initialise — will fall back to pyttsx3.",
                exc_info=True,
            )

    def speak_blocking(self, text: str) -> None:
        """Synthesise and play `text` synchronously. Called from worker thread."""
        if not self.available or self._pipeline is None:
            return
        try:
            import numpy as np
            audio_chunks = []
            for _, _, audio in self._pipeline(text, voice=self.VOICE, speed=1.0):
                if audio is not None:
                    audio_chunks.append(audio)

            if not audio_chunks:
                return

            audio_np = np.concatenate(audio_chunks).astype(np.float32)
            self._play(audio_np)
        except Exception:
            logger.error("Kokoro synthesis failed for: %r", text, exc_info=True)

    def _play(self, audio: "np.ndarray") -> None:
        """Play float32 audio array. Tries sounddevice first, then afplay."""
        try:
            import sounddevice as sd
            sd.play(audio, samplerate=self.SAMPLE_RATE)
            sd.wait()
            return
        except Exception:
            pass  # sounddevice unavailable or failed — try file-based fallback

        try:
            import io
            import wave
            import subprocess
            import numpy as np

            pcm = (audio * 32767).astype(np.int16).tobytes()
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm)
            wav_bytes = buf.getvalue()

            # Write to a temp file and play with system command
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            # macOS: afplay; Linux: aplay
            player = "afplay" if os.uname().sysname == "Darwin" else "aplay"
            subprocess.run([player, tmp_path], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.unlink(tmp_path)
        except Exception:
            logger.error("Kokoro audio playback failed.", exc_info=True)


# ── pyttsx3 backend ───────────────────────────────────────────────────────────

class _Pyttsx3Backend:
    """pyttsx3 OS-native TTS fallback. Robotic but zero extra dependencies."""

    def __init__(self, rate_wpm: int) -> None:
        self.available = False
        self._engine = None
        self._rate_wpm = rate_wpm
        self._load()

    def _load(self) -> None:
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate_wpm)
            self.available = True
            logger.info("pyttsx3 engine initialized (rate=%d wpm).", self._rate_wpm)
        except Exception:
            logger.warning(
                "pyttsx3 failed to initialize; TTS will log-only.",
                exc_info=True,
            )

    def speak_blocking(self, text: str) -> None:
        if not self.available or self._engine is None:
            logger.info("[TTS log-only] %r", text)
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            logger.error("pyttsx3 mid-utterance error: %r", text, exc_info=True)
            self._load()  # reinitialise so next call isn't also broken

    def stop(self) -> None:
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass


# ── TTSEngine — public interface ──────────────────────────────────────────────

class TTSEngine:
    """
    Unified TTS engine. Picks Kokoro or pyttsx3 based on config, runs
    all synthesis and playback on a single background daemon thread so
    speak() always returns within the 200ms dispatch budget.

    Thread-safe: speak() and stop()/shutdown() can be called from any thread.
    """

    def __init__(self, rate_wpm: Optional[int] = None) -> None:
        self._rate_wpm = rate_wpm or settings.TTS_RATE_WPM
        self._queue: "queue.Queue[Optional[SpeechRequest]]" = queue.Queue()
        self._speaking = threading.Event()
        self._shutdown_flag = threading.Event()

        self._backend = self._load_backend()

        self._worker = threading.Thread(
            target=self._run_worker,
            daemon=True,
            name="tts-worker",
        )
        self._worker.start()

    def _load_backend(self) -> "_KokoroBackend | _Pyttsx3Backend":
        engine_choice = settings.effective_tts_engine

        if engine_choice == "kokoro":
            backend = _KokoroBackend()
            if backend.available:
                return backend
            # Kokoro failed — fall through to pyttsx3
            logger.warning("Kokoro unavailable, falling back to pyttsx3.")

        # pyttsx3 path (explicit choice or kokoro fallback)
        return _Pyttsx3Backend(rate_wpm=self._rate_wpm)

    @property
    def available(self) -> bool:
        return self._backend.available

    def _run_worker(self) -> None:
        while not self._shutdown_flag.is_set():
            request = self._queue.get()
            if request is None:
                break

            dispatch_ms = (time.perf_counter() - request.enqueued_at) * 1000
            if dispatch_ms > _DISPATCH_LATENCY_TARGET_MS:
                logger.warning(
                    "TTS dispatch %.0fms exceeded %dms target: %r",
                    dispatch_ms, _DISPATCH_LATENCY_TARGET_MS, request.text,
                )

            self._speaking.set()
            try:
                self._backend.speak_blocking(request.text)
            finally:
                self._speaking.clear()

    def speak(self, text: str, interrupt: bool = False) -> None:
        """
        Queue `text` for speech. Returns immediately (dispatch only).

        interrupt=True  — clears current queue and stops any ongoing
                          utterance, so this message starts next.
                          Use for HIGH-risk alerts.
        interrupt=False — appends behind whatever is playing.
                          Use for MEDIUM-risk alerts.
        """
        if not text or not text.strip():
            return

        if interrupt:
            self._clear_queue()
            if isinstance(self._backend, _Pyttsx3Backend):
                self._backend.stop()

        self._queue.put(SpeechRequest(text=text))

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
        """Hard mute — silence current and queued speech."""
        self._clear_queue()
        if isinstance(self._backend, _Pyttsx3Backend):
            self._backend.stop()

    def shutdown(self) -> None:
        """Clean shutdown. Call once on app exit."""
        self.stop()
        self._shutdown_flag.set()
        self._queue.put(None)
        self._worker.join(timeout=3.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tts = TTSEngine()
    backend_name = type(tts._backend).__name__
    print(f"Backend: {backend_name}  available={tts.available}")

    start = time.perf_counter()
    tts.speak("Memory nav voice test. Chair two meters ahead.")
    print(f"speak() returned in {(time.perf_counter() - start) * 1000:.1f}ms")

    time.sleep(5)
    tts.shutdown()
    print("Done.")
