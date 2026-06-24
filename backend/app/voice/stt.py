"""
MemoryNav — Whisper Local Speech-to-Text
backend/app/voice/stt.py

Module 5 (Voice Interface): offline speech transcription. Thin wrapper
around OpenAI Whisper (the local openai-whisper package, not the API)
— audio never leaves the device. This is the "voice input" side of the
voice loop; the "voice output" side lives in tts.py.

Privacy guarantee (architecture doc, Privacy Architecture section):
    "Whisper speech recognition runs on-device (M2 Neural Engine) —
    no audio sent to cloud."
This file enforces that — there are no network calls here at all.

Where this fits in the pipeline
--------------------------------
    Microphone (sounddevice / PyAudio)
        ↓  raw audio bytes (PCM-int16 or WAV)
    WhisperSTT.transcribe(audio_bytes)    ← YOU ARE HERE
        ↓  transcribed question string (or None if silence / noise)
    Voice layer intent parser
        ↓
    "what is in front of me?" → YOLO + Risk + Memory answer
    "what does this say?"     → OCRReader.read_text(frame)
    "describe everything"     → optional GPT-4o Vision call

Audio format accepted
----------------------
transcribe() accepts audio bytes in two formats — callers don't need
to pick one; the method auto-detects by WAV header:

    1. WAV bytes  (b"RIFF..." header)
       What sounddevice.rec(..., out=buffer) + scipy.io.wavfile.write
       produce; also what most off-the-shelf microphone capture code
       returns. No extra conversion needed from the caller.

    2. Raw PCM int16 bytes, 16 kHz, mono, little-endian
       What PyAudio stream.read() returns by default (paInt16 format).
       Frame rate must be 16 000 Hz — Whisper's native sample rate.

Both are converted to float32 numpy arrays internally before being
passed to Whisper.

Silence gate
-------------
Whisper is well-known for hallucinating filler text on silent or very
quiet audio ("Thank you.", ".", "you", etc.). This wrapper computes
the RMS amplitude of the audio before transcription and returns None
if it falls below a configurable silence threshold, preventing those
hallucinations from ever reaching the voice layer. The threshold is
in the same float32 [-1, 1] amplitude scale Whisper uses internally.

Device — Apple Silicon
-----------------------
Whisper supports torch MPS (unlike EasyOCR). On an M2 the wrapper
auto-selects "mps" when settings.device is "mps" or "cpu" and the
torch.backends.mps.is_available() check passes. This uses the M2
Neural Engine and produces meaningfully faster transcription than CPU
on the "base" and "small" models. Falls back to CPU gracefully.

Latency target
--------------
Architecture doc: "sub-200ms voice response latency." Whisper "tiny"
achieves this comfortably on M2 for short utterances (< 5s). "base"
is close but may spike over 200ms on first call without a warm-up
run — hence _warm_up() in __init__. model_size defaults to "base"
for the better accuracy/latency balance; swap to "tiny" if your
measured latency exceeds the 200ms budget after Phase 5 integration.

Usage
-----
    from app.voice.stt import WhisperSTT

    stt = WhisperSTT()                          # load once, reuse

    # Raw audio captured from microphone:
    audio_bytes = mic.record(seconds=4)         # bytes (WAV or PCM)
    text = stt.transcribe(audio_bytes)

    if text:
        handle_user_question(text)              # e.g. "what is ahead?"
    else:
        pass                                    # silence — no-op

    # Override language per-call (e.g. pulled from user preferences):
    text = stt.transcribe(audio_bytes, language="hi")   # Hindi

Dependencies: openai-whisper, torch, numpy, soundfile (for WAV decode).
soundfile is lighter than scipy for the WAV → float32 conversion and
has no scipy-on-arm compile issues on Apple Silicon.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

WHISPER_SAMPLE_RATE: int = 16_000   # Whisper always expects 16 kHz

# Default model — balance of accuracy vs latency on M2.
# "tiny"  → fastest, ~39 M params, accuracy drops on accented speech.
# "base"  → default here, ~74 M params, still sub-200ms for <5s clips on M2.
# "small" → better accuracy for multilingual (Hindi/Tamil/Marathi), slower.
_DEFAULT_MODEL_SIZE: str = "base"

# Silence threshold — float32 RMS in [-1, 1] amplitude space.
# Audio below this is discarded without running Whisper.
# 0.01 = ~-40 dBFS, comfortably above mic thermal noise and well below
# normal speech (~0.10–0.30 RMS). Lower this if users speak very softly.
_DEFAULT_SILENCE_THRESHOLD: float = 0.01

# WAV file magic bytes — used to auto-detect format in transcribe()
_WAV_HEADER: bytes = b"RIFF"


# ── WhisperSTT ────────────────────────────────────────────────────────────────

class WhisperSTT:
    """
    Local Whisper STT wrapper. Load once per session, reuse across calls.

    Parameters
    ----------
    model_size : str
        Whisper model identifier: "tiny", "base", "small", "medium", "large".
        Default "base" (doc: M2-optimised, sub-200ms target).
    device : Optional[str]
        "cpu", "cuda", or "mps". Auto-detected from settings.device when
        None — see the Device note in the module docstring.
    language : Optional[str]
        Default transcription language as a Whisper language code ("en",
        "hi", "ta", "mr" …). None lets Whisper auto-detect per call, which
        adds ~20ms overhead but supports the multilingual use case without
        locking a language at init time. Pulled from user preferences at the
        voice-layer level and passed in here; defaults to "en" from settings.
    silence_threshold : float
        RMS amplitude below which audio is considered silence and None is
        returned without running Whisper. Default 0.01 (~-40 dBFS).
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        language: Optional[str] = None,
        silence_threshold: float = _DEFAULT_SILENCE_THRESHOLD,
    ) -> None:
        self.model_size = model_size or getattr(
            settings, "WHISPER_MODEL_SIZE", _DEFAULT_MODEL_SIZE
        )
        self.language = language or getattr(settings, "WHISPER_LANGUAGE", "en")
        self.silence_threshold = silence_threshold

        self._device = self._resolve_device(device)

        logger.info(
            "Loading Whisper '%s' on device '%s' (language='%s', silence_threshold=%.3f)",
            self.model_size, self._device, self.language, self.silence_threshold,
        )

        # Import lazily: `import whisper` triggers torch load + model weight
        # download on first run — callers that never use STT shouldn't pay that
        # cost at import time (same pattern as OCRReader).
        import whisper as _whisper

        self._model = _whisper.load_model(self.model_size, device=self._device)

        # Ablation / latency metrics
        self._total_calls:      int   = 0
        self._successful_calls: int   = 0   # returned non-empty text
        self._silence_drops:    int   = 0   # returned None (silence gate)
        self._total_latency_s:  float = 0.0

        self._warm_up()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """
        Pick the best available torch device for Whisper.

        Priority: explicit arg > settings.device (if valid) > MPS (if
        available on Apple Silicon) > CPU.

        Unlike EasyOCR (CUDA-only GPU), Whisper's torch backend supports
        MPS, so M2 users get hardware acceleration here.
        """
        import torch

        if device is not None:
            return device

        configured = str(getattr(settings, "device", "cpu")).lower()

        if configured == "cuda" and torch.cuda.is_available():
            return "cuda"

        if configured == "mps" or torch.backends.mps.is_available():
            try:
                # A small tensor round-trip to confirm MPS actually works —
                # torch.backends.mps.is_available() can return True on machines
                # where the MPS runtime then errors on first use.
                _ = torch.zeros(1, device="mps")
                return "mps"
            except Exception:
                logger.warning(
                    "MPS reported as available but failed the probe; falling back to CPU."
                )

        return "cpu"

    def _warm_up(self) -> None:
        """
        Run one silent dummy transcription so the first real call doesn't
        absorb model-initialization latency and blow the 200ms budget.
        The silence gate is bypassed here (silence_threshold=0) so Whisper
        actually runs through its inference path, not just our gating code.
        """
        dummy = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)  # 1 s of silence
        try:
            t0 = time.monotonic()
            self._run_whisper(dummy, language=self.language)
            elapsed = time.monotonic() - t0
            logger.info(
                "WhisperSTT warm-up complete (%.0f ms, device=%s).",
                elapsed * 1000, self._device,
            )
        except Exception:
            logger.warning("WhisperSTT warm-up failed; continuing anyway.", exc_info=True)

    # ── Primary API ────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> Optional[str]:
        """
        Transcribe an audio clip and return the spoken text.

        Args:
            audio_bytes : Raw audio — either WAV bytes (auto-detected by
                          RIFF header) or raw PCM int16 bytes at 16 kHz
                          mono. See module docstring for details.
            language    : Whisper language code override for this call
                          ("en", "hi", "ta", "mr" …). Falls back to
                          self.language (from user preferences) when None.

        Returns:
            Stripped transcription string, or None if:
              - audio_bytes is empty / invalid
              - RMS amplitude is below self.silence_threshold (silence gate)
              - Whisper returns blank text after stripping
              - Any unexpected error (logs at ERROR level, never raises)

        Never raises — a transcription failure must degrade to silence,
        not crash the assistant the user is relying on.
        """
        self._total_calls += 1
        lang = language or self.language

        # ── Decode bytes → float32 numpy array ────────────────────────────
        audio = self._decode(audio_bytes)
        if audio is None:
            return None

        # ── Silence gate — reject before Whisper ever runs ────────────────
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.silence_threshold:
            self._silence_drops += 1
            logger.debug(
                "[WhisperSTT] Silence gate — RMS=%.4f below threshold=%.4f; skipping.",
                rms, self.silence_threshold,
            )
            return None

        # ── Transcription ──────────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            result = self._run_whisper(audio, language=lang)
        except Exception:
            logger.error("[WhisperSTT] Whisper inference failed.", exc_info=True)
            return None
        finally:
            elapsed = time.monotonic() - t0
            self._total_latency_s += elapsed
            logger.debug("[WhisperSTT] transcription took %.0f ms", elapsed * 1000)

        text = result.strip() if result else ""
        if not text:
            logger.debug("[WhisperSTT] Whisper returned empty text.")
            return None

        self._successful_calls += 1
        logger.info('[WhisperSTT] → "%s"', text)
        return text

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _decode(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Convert audio_bytes to a float32 numpy array at 16 kHz, mono.

        Handles two formats (see module docstring):
          1. WAV bytes (detected by "RIFF" magic header) — decoded via soundfile.
          2. Raw PCM int16 bytes at 16 kHz, mono — cast directly.

        Returns None if decoding fails or the array is empty.
        """
        if not audio_bytes:
            logger.warning("[WhisperSTT] transcribe() called with empty bytes.")
            return None

        try:
            if audio_bytes[:4] == _WAV_HEADER:
                return self._decode_wav(audio_bytes)
            return self._decode_raw_pcm(audio_bytes)
        except Exception:
            logger.error("[WhisperSTT] Audio decode failed.", exc_info=True)
            return None

    @staticmethod
    def _decode_wav(audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode WAV bytes using soundfile.

        soundfile handles sample rate conversion for us — if the WAV was
        captured at a sample rate other than 16 kHz (e.g. 44.1 kHz from
        some microphones) resampling is needed. soundfile reads the stored
        rate but does NOT resample; we resample here using librosa-style
        linear interpolation via numpy to avoid a hard librosa dependency.

        If the audio is already at 16 kHz (the common case with sounddevice
        using sr=16000), the resample step is a no-op.
        """
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)

        # Downmix stereo → mono by averaging channels
        if data.ndim == 2:
            data = data.mean(axis=1)

        # Resample to 16 kHz if necessary
        if sr != WHISPER_SAMPLE_RATE:
            logger.debug(
                "[WhisperSTT] WAV sample rate %d Hz → resampling to %d Hz.",
                sr, WHISPER_SAMPLE_RATE,
            )
            data = _resample(data, orig_sr=sr, target_sr=WHISPER_SAMPLE_RATE)

        return data if data.size > 0 else None

    @staticmethod
    def _decode_raw_pcm(audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode raw PCM int16 bytes (PyAudio default format, 16 kHz, mono).

        int16 range [-32768, 32767] → float32 [-1.0, 1.0].
        """
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
        return audio if audio.size > 0 else None

    def _run_whisper(self, audio: np.ndarray, language: str) -> str:
        """
        Call Whisper's transcribe() with MemoryNav-appropriate options and
        return the raw text string (may be empty; stripping happens upstream).

        fp16 is enabled only on CUDA (where it gives a genuine speed boost).
        On MPS and CPU it causes warnings or errors in some torch versions,
        so it's explicitly disabled for those devices.
        """
        result = self._model.transcribe(
            audio,
            language=language,
            fp16=self._device == "cuda",
            # temperature=0 removes sampling randomness — deterministic output
            # is important for safety-critical assistive use; we don't want
            # stochastic variation in transcribed navigation questions.
            temperature=0,
            # Suppress Whisper's verbose per-segment logging — MemoryNav has
            # its own logging at the transcribe() call site above.
            verbose=False,
        )
        return result.get("text", "")

    # ── Ablation / latency metrics ────────────────────────────────────────────

    @property
    def counts(self) -> dict:
        """
        Running counters since init or last reset_counts().
        Use for the ablation study's voice-input metrics.
        """
        return {
            "total_calls":      self._total_calls,
            "successful_calls": self._successful_calls,
            "silence_drops":    self._silence_drops,
            "failed_calls":     (
                self._total_calls
                - self._successful_calls
                - self._silence_drops
            ),
        }

    @property
    def mean_latency_ms(self) -> float:
        """
        Mean transcription latency in milliseconds across all calls that
        actually ran Whisper (silence-gated calls are excluded since Whisper
        never ran). Zero if no Whisper calls have been made yet.

        Compare against the 200ms sub-latency target from the architecture doc.
        """
        whisper_calls = self._successful_calls + self.counts["failed_calls"]
        if whisper_calls == 0:
            return 0.0
        return (self._total_latency_s / whisper_calls) * 1000

    def reset_counts(self) -> None:
        """Reset counters and latency accumulator (does not unload the model)."""
        self._total_calls      = 0
        self._successful_calls = 0
        self._silence_drops    = 0
        self._total_latency_s  = 0.0

    def stats_summary(self) -> str:
        """One-line summary for ablation study logs."""
        c = self.counts
        n = c["total_calls"]
        if n == 0:
            return "WhisperSTT: no transcription calls yet"
        return (
            f"WhisperSTT — calls={n}  successful={c['successful_calls']}  "
            f"silence_drops={c['silence_drops']}  failed={c['failed_calls']}  "
            f"mean_latency={self.mean_latency_ms:.0f}ms  "
            f"model={self.model_size}  device={self._device}"
        )


# ── Resampling helper ─────────────────────────────────────────────────────────

def _resample(
    audio: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """
    Linear interpolation resample — avoids a hard librosa or resampy
    dependency for the common case of WAV files at a non-16 kHz rate.

    Not high-quality (linear interp, not sinc), but perfectly adequate
    for speech — users won't notice, and Whisper is robust to mild
    resampling artifacts. If you add librosa to requirements.txt, swap
    this for librosa.resample(..., res_type="kaiser_fast") for higher
    fidelity at negligible latency cost.
    """
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, num_samples),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


# ── Convenience module-level function ─────────────────────────────────────────

_default_stt: Optional[WhisperSTT] = None


def transcribe(audio_bytes: bytes, language: Optional[str] = None) -> Optional[str]:
    """
    Module-level shortcut using a lazily-created default WhisperSTT
    instance (model="base", language from settings).

    Useful for quick integration without managing a WhisperSTT instance:

        from app.voice.stt import transcribe
        text = transcribe(audio_bytes)

    The first call pays the model-load cost; subsequent calls reuse
    the same loaded model.
    """
    global _default_stt
    if _default_stt is None:
        _default_stt = WhisperSTT()
    return _default_stt.transcribe(audio_bytes, language=language)


if __name__ == "__main__":
    # Quick smoke check: `python -m app.voice.stt`
    # Confirms Whisper loads, the device resolves, and the silence gate
    # correctly returns None for a silent clip without hallucinating text.
    logging.basicConfig(level=logging.INFO)

    stt = WhisperSTT()

    # -- Test 1: silence gate returns None (not a hallucination string) ----
    silent_pcm = np.zeros(WHISPER_SAMPLE_RATE * 2, dtype=np.float32)  # 2s silence
    silent_bytes = (silent_pcm * 32767).astype(np.int16).tobytes()

    result = stt.transcribe(silent_bytes)
    assert result is None, f"Expected None for silence, got: {result!r}"
    print(f"Silence gate: transcribe(silent audio) → {result!r}  ✅")

    # -- Test 2: stats summary reflects the silence drop -------------------
    print(stt.stats_summary())

    print("\nWhisperSTT loaded and smoke test passed.")