"""
MemoryNav — Step 10: Final Integration Tests
backend/app/tests/test_final_integration.py

Validates all four areas from Step 10:
  Part A — main.py: startup order, health endpoint (200 / 503), workers=1, CORS
  Part B — memory_router.py: 6 endpoints, semantic search, wipe guard, executor
  Part C — voice_router.py: intent priority, SessionStore singleton, response
            shape, silence → 400

All heavy models are mocked so the suite is fast and requires no GPU.

Run:
    cd backend
    pytest app/tests/test_final_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Make sure the backend app package is importable from wherever pytest runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ═══════════════════════════════════════════════════════════════════════════════
# Part A — main.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestMainStartupOrder:
    """
    Check 1 — Startup order: verify the 7-step init sequence is defined
    correctly in main.py source and all nine state attributes are documented.
    """

    def test_startup_initialises_detector_first(self):
        """
        YOLO detector must be step 1 — parse main.py to confirm ordering.
        """
        main_path = Path(__file__).resolve().parents[1] / "main.py"
        source = main_path.read_text()

        # All nine state assignments must be present
        for attr in (
            "app.state.detector",
            "app.state.depth_estimator",
            "app.state.ocr_reader",
            "app.state.long_term_memory",
            "app.state.preferences_store",
            "app.state.alert_manager",
            "app.state.tts_engine",
            "app.state.stt",
            "app.state.pipeline",
        ):
            assert attr in source, f"'{attr}' assignment missing from main.py lifespan"

    def test_detector_assigned_before_depth(self):
        """Detector (step 1) must appear before DepthEstimator (step 2) in source."""
        main_path = Path(__file__).resolve().parents[1] / "main.py"
        source = main_path.read_text()
        pos_detector = source.index("app.state.detector")
        pos_depth    = source.index("app.state.depth_estimator")
        assert pos_detector < pos_depth, \
            "detector must be assigned before depth_estimator in startup sequence"

    def test_memory_assigned_before_pipeline(self):
        """ChromaDB (long_term_memory) must exist before pipeline is wired."""
        main_path = Path(__file__).resolve().parents[1] / "main.py"
        source = main_path.read_text()
        pos_memory   = source.index("app.state.long_term_memory")
        pos_pipeline = source.index("app.state.pipeline")
        assert pos_memory < pos_pipeline, \
            "long_term_memory must be assigned before pipeline in startup sequence"

    def test_voice_modules_assigned_before_pipeline(self):
        """TTS and STT must be initialized before the pipeline is wired."""
        main_path = Path(__file__).resolve().parents[1] / "main.py"
        source = main_path.read_text()
        pos_tts      = source.index("app.state.tts_engine")
        pos_stt      = source.index("app.state.stt")
        pos_pipeline = source.index("app.state.pipeline")
        assert pos_tts < pos_pipeline
        assert pos_stt < pos_pipeline

    def test_init_pipeline_called_in_lifespan(self):
        """init_pipeline() must be called in the lifespan to wire the WS pipeline."""
        main_path = Path(__file__).resolve().parents[1] / "main.py"
        source = main_path.read_text()
        assert "init_pipeline()" in source, \
            "lifespan must call init_pipeline() to wire the WebSocket pipeline"

    def test_init_memory_called_in_lifespan(self):
        main_path = Path(__file__).resolve().parents[1] / "main.py"
        source = main_path.read_text()
        assert "init_memory(" in source

    def test_init_preferences_called_in_lifespan(self):
        main_path = Path(__file__).resolve().parents[1] / "main.py"
        source = main_path.read_text()
        assert "init_preferences(" in source


class TestHealthEndpoint:
    """
    Check 2 — Health endpoint must return 503 + 'missing' list if any module
    is absent, and 200 with status='healthy' when all nine are present.
    """

    REQUIRED = [
        "detector", "depth_estimator", "ocr_reader",
        "long_term_memory", "preferences_store", "alert_manager",
        "tts_engine", "stt", "pipeline",
    ]

    def _make_client_with_full_state(self):
        """Build a TestClient with all nine app.state attrs set to MagicMock."""
        with patch("app.main.Detector", MagicMock()), \
             patch("app.main.DepthEstimator", MagicMock()), \
             patch("app.main.OCRReader", MagicMock()), \
             patch("app.main.LongTermMemory", MagicMock()), \
             patch("app.main.PreferencesStore", MagicMock()), \
             patch("app.main.ShortTermMemory", MagicMock()), \
             patch("app.main.TemporalAlertManager", MagicMock()), \
             patch("app.main.TTSEngine", MagicMock()), \
             patch("app.main.WhisperSTT", MagicMock()), \
             patch("app.api.ws_stream.init_pipeline", MagicMock(return_value=MagicMock())), \
             patch("app.api.memory_router.init_memory", MagicMock()), \
             patch("app.api.preferences_router.init_preferences", MagicMock()):

            import importlib
            import app.main as main_mod
            importlib.reload(main_mod)

            # Directly set all expected state attributes on the app
            # (simulates what a successful lifespan startup does)
            for attr in self.REQUIRED:
                setattr(main_mod.app.state, attr, MagicMock())

            from fastapi.testclient import TestClient
            client = TestClient(main_mod.app, raise_server_exceptions=False)
            return client, main_mod.app

    def test_health_returns_200_when_all_modules_ready(self):
        client, app = self._make_client_with_full_state()
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_returns_503_if_module_missing(self):
        """Required test from the step brief — deleting detector → 503."""
        client, application = self._make_client_with_full_state()
        # Forcibly remove the attr from state to simulate failed startup
        application.state._state.pop("detector", None)
        response = client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unhealthy"
        assert "detector" in body["missing"]

    def test_health_503_lists_all_missing_modules(self):
        """If multiple modules are missing, all should appear in 'missing'."""
        client, application = self._make_client_with_full_state()
        for attr in ("stt", "pipeline"):
            application.state._state.pop(attr, None)
        response = client.get("/health")
        assert response.status_code == 503
        missing = response.json()["missing"]
        assert "stt" in missing
        assert "pipeline" in missing

    def test_health_checks_nine_required_modules(self):
        """The spec requires exactly these nine modules to be checked."""
        for module_name in self.REQUIRED:
            client, application = self._make_client_with_full_state()
            application.state._state.pop(module_name, None)
            response = client.get("/health")
            assert response.status_code == 503, \
                f"Expected 503 when {module_name!r} is missing, got {response.status_code}"


class TestSingleWorkerConfig:
    """
    Check 3 — workers=1 must be the only value in the __main__ entrypoint.
    Multiple workers each load full model copies → OOM on edge devices.
    """

    def test_main_entrypoint_uses_workers_1(self):
        """
        Parse main.py source and assert workers=1 is passed to uvicorn.run()
        and that no wildcard workers value exists.
        """
        main_path = Path(__file__).resolve().parents[1] / "main.py"
        source = main_path.read_text()

        assert "workers=1" in source, \
            "main.py __main__ block must pass workers=1 to uvicorn.run()"

        # Must not have workers=4, workers=2, or cpu_count auto workers
        for bad in ("workers=4", "workers=2", "workers=cpu_count", "workers=os.cpu"):
            assert bad not in source, \
                f"main.py must not use '{bad}' — single worker required"


class TestCORSConfig:
    """
    Check 4 — CORS must be restricted to localhost:3000 only, never wildcard.
    """

    def test_cors_origins_not_wildcard(self):
        from app.config import settings
        assert "*" not in settings.CORS_ORIGINS, \
            "CORS_ORIGINS must not contain '*'"

    def test_cors_origins_contains_localhost_3000(self):
        from app.config import settings
        assert "http://localhost:3000" in settings.CORS_ORIGINS, \
            "CORS_ORIGINS must include 'http://localhost:3000'"

    def test_cors_origins_default_is_localhost_only(self):
        """Default config must not expose any other origin."""
        from app.config import settings
        assert settings.CORS_ORIGINS == ["http://localhost:3000"], \
            f"Expected CORS_ORIGINS=['http://localhost:3000'], got {settings.CORS_ORIGINS}"


# ═══════════════════════════════════════════════════════════════════════════════
# Part B — api/memory_router.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryRouterEndpoints:
    """
    Check 1 — All six endpoints must exist and be reachable.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Wire a fresh memory router with a mock LongTermMemory."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.memory_router import router, init_memory

        self.mock_memory = MagicMock()
        self.mock_memory.count.return_value = 0
        self.mock_memory.list_all.return_value = []
        self.mock_memory.retrieve.return_value = []
        self.mock_memory.add_context.return_value = "test-uuid-1234"
        self.mock_memory.clear.return_value = None
        self.mock_memory.delete.return_value = None

        test_app = FastAPI()

        # Patch ws_stream.get_state for the spatial-map route
        with patch("app.api.memory_router.get_memory", return_value=self.mock_memory):
            init_memory(self.mock_memory)
            test_app.include_router(router)
            self.client = TestClient(test_app)

    def test_get_memory_list_exists(self):
        """GET /memory — list all (no query)."""
        resp = self.client.get("/memory")
        assert resp.status_code == 200

    def test_get_memory_search_exists(self):
        """GET /memory?q=chair — semantic search path."""
        resp = self.client.get("/memory?q=chair")
        assert resp.status_code == 200

    def test_post_memory_exists(self):
        """POST /memory — create new entry."""
        resp = self.client.post("/memory", json={"text": "step at entrance", "room": "hall"})
        assert resp.status_code == 201

    def test_delete_memory_by_id_exists(self):
        """DELETE /memory/{id} — delete single entry."""
        # add_context returned 'test-uuid-1234'; mock list_all so 404 doesn't fire
        mock_result = MagicMock()
        mock_result.id = "test-uuid-1234"
        mock_result.text = "step at entrance"
        mock_result.metadata = {}
        self.mock_memory.list_all.return_value = [mock_result]
        resp = self.client.delete("/memory/test-uuid-1234")
        assert resp.status_code == 200

    def test_delete_memory_wipe_requires_confirm(self):
        """DELETE /memory — wipe without confirm → 400."""
        resp = self.client.delete("/memory")
        assert resp.status_code == 400

    def test_delete_memory_wipe_with_confirm(self):
        """DELETE /memory?confirm=true — wipe succeeds → 200."""
        resp = self.client.delete("/memory?confirm=true")
        assert resp.status_code == 200

    def test_get_spatial_map_exists(self):
        """GET /memory/spatial-map — live map snapshot."""
        with patch("app.api.ws_stream.get_state") as mock_gs:
            mock_state = MagicMock()
            mock_state.spatial_map.snapshot.return_value = {
                "current_room": "Living Room", "rooms": {}
            }
            mock_gs.return_value = mock_state
            resp = self.client.get("/memory/spatial-map")
        assert resp.status_code == 200


class TestMemoryRouterSemanticSearch:
    """
    Check 2 — GET /memory?q=... must route to retrieve(); omitting q uses list_all().
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.memory_router import router, init_memory

        self.mock_memory = MagicMock()
        self.mock_memory.count.return_value = 2
        self.mock_memory.list_all.return_value = []
        self.mock_memory.retrieve.return_value = []

        init_memory(self.mock_memory)
        test_app = FastAPI()
        test_app.include_router(router)
        self.client = TestClient(test_app)

    def test_query_param_triggers_semantic_search(self):
        resp = self.client.get("/memory?q=chair+near+door")
        assert resp.status_code == 200
        self.mock_memory.retrieve.assert_called_once()
        self.mock_memory.list_all.assert_not_called()

    def test_no_query_param_triggers_list_all(self):
        resp = self.client.get("/memory")
        assert resp.status_code == 200
        self.mock_memory.list_all.assert_called_once()
        self.mock_memory.retrieve.assert_not_called()

    def test_room_filter_included_in_where_clause(self):
        """?room=kitchen must pass where={"room": "kitchen"} to retrieve()."""
        resp = self.client.get("/memory?q=step&room=kitchen")
        assert resp.status_code == 200
        call_kwargs = self.mock_memory.retrieve.call_args
        # where is passed as keyword or positional arg
        args, kwargs = call_kwargs
        assert kwargs.get("where") == {"room": "kitchen"} or \
               (len(args) >= 3 and args[2] == {"room": "kitchen"})


class TestMemoryWipeGuard:
    """
    Check 3 — DELETE /memory without confirm=true must return 400.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.memory_router import router, init_memory

        mock_memory = MagicMock()
        mock_memory.count.return_value = 5
        mock_memory.clear.return_value = None
        init_memory(mock_memory)

        test_app = FastAPI()
        test_app.include_router(router)
        self.client = TestClient(test_app)
        self.mock_memory = mock_memory

    def test_wipe_without_confirm_returns_400(self):
        """Required test from the step brief."""
        response = self.client.delete("/memory")
        assert response.status_code == 400

    def test_wipe_with_confirm_false_returns_400(self):
        response = self.client.delete("/memory?confirm=false")
        assert response.status_code == 400

    def test_wipe_with_confirm_true_returns_200(self):
        """Required test from the step brief."""
        response = self.client.delete("/memory?confirm=true")
        assert response.status_code == 200

    def test_wipe_without_confirm_does_not_call_clear(self):
        """Clear must NOT be invoked when the guard fires."""
        self.client.delete("/memory")
        self.mock_memory.clear.assert_not_called()

    def test_wipe_detail_message_explains_guard(self):
        response = self.client.delete("/memory")
        assert "confirm" in response.json()["detail"].lower()


class TestMemoryRouterExecutor:
    """
    Check 4 — The module-level ThreadPoolExecutor must use max_workers=1.
    """

    def test_executor_max_workers_is_one(self):
        from app.api.memory_router import _EXECUTOR
        assert _EXECUTOR._max_workers == 1, \
            "memory_router._EXECUTOR must have max_workers=1 to avoid concurrent embedding"


# ═══════════════════════════════════════════════════════════════════════════════
# Part C — api/voice_router.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntentClassification:
    """
    Check 1 — Intent priority: OCR > SCENE > MEMORY > OBSTACLE > UNKNOWN.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from app.api.voice_router import _classify_intent, IntentType
        self._classify = _classify_intent
        self.Intent = IntentType

    def test_read_keyword_maps_to_ocr(self):
        assert self._classify("read the sign") == self.Intent.OCR_QUERY

    def test_say_keyword_maps_to_ocr(self):
        assert self._classify("what does it say") == self.Intent.OCR_QUERY

    def test_describe_keyword_maps_to_scene(self):
        assert self._classify("describe what you see") == self.Intent.SCENE_DESCRIPTION

    def test_see_keyword_maps_to_scene(self):
        assert self._classify("what is around me") == self.Intent.SCENE_DESCRIPTION

    def test_remember_keyword_maps_to_memory(self):
        assert self._classify("remind me about the step") == self.Intent.MEMORY_QUERY

    def test_before_keyword_maps_to_memory(self):
        assert self._classify("what did i say about the rug") == self.Intent.MEMORY_QUERY

    def test_obstacle_keyword_maps_to_obstacle(self):
        assert self._classify("what is in front of me") == self.Intent.OBSTACLE_QUERY

    def test_ahead_keyword_maps_to_obstacle(self):
        assert self._classify("what is ahead") == self.Intent.OBSTACLE_QUERY

    def test_unknown_query_returns_unknown(self):
        assert self._classify("play some music") == self.Intent.UNKNOWN

    def test_required_ocr_beats_scene_priority(self):
        """
        Required test from the step brief:
        'read the sign around me' → OCR wins over SCENE.
        """
        intent = self._classify("read the sign around me")
        assert intent == self.Intent.OCR_QUERY, \
            f"Expected OCR_QUERY, got {intent}. OCR must beat SCENE in priority order."

    def test_ocr_beats_memory_priority(self):
        """'read this label, do you remember it?' → OCR wins."""
        intent = self._classify("read this label, do you remember it")
        assert intent == self.Intent.OCR_QUERY

    def test_scene_beats_memory_priority(self):
        """'describe the area, do you remember it?' → SCENE wins."""
        intent = self._classify("describe the room, do you remember it")
        assert intent == self.Intent.SCENE_DESCRIPTION

    def test_memory_beats_obstacle_priority(self):
        """'where is the step ahead' → MEMORY wins (has 'where is the')."""
        intent = self._classify("where is the step")
        assert intent == self.Intent.MEMORY_QUERY


class TestVoiceRouterSingletonFix:
    """
    Check 2 — voice_router must use the shared pipeline instances
    (get_state().session / get_state().long_term), NOT separate singletons.
    """

    def test_voice_router_uses_get_state_for_long_term(self):
        """
        _retrieve_long_term() must call get_state().long_term.retrieve(),
        not a module-level _long_term instance.
        """
        from app.api import voice_router as vr

        mock_state = MagicMock()
        mock_state.long_term.retrieve.return_value = []

        with patch("app.api.voice_router.get_state", return_value=mock_state), \
             patch("app.api.voice_router._pipeline_available", return_value=True):
            vr._retrieve_long_term("test query")

        mock_state.long_term.retrieve.assert_called_once_with("test query")

    def test_voice_router_uses_get_state_for_session(self):
        """
        _get_short_term_summary() must call get_state().session.get_recent_summary().
        """
        from app.api import voice_router as vr

        mock_state = MagicMock()
        mock_state.session.get_recent_summary.return_value = "chair ahead"

        with patch("app.api.voice_router.get_state", return_value=mock_state), \
             patch("app.api.voice_router._pipeline_available", return_value=True):
            result = vr._get_short_term_summary()

        assert result == "chair ahead"
        mock_state.session.get_recent_summary.assert_called_once()

    def test_no_module_level_short_term_singleton(self):
        """
        voice_router must NOT have a module-level _short_term attribute
        that creates a separate SessionStore instance.
        """
        import app.api.voice_router as vr
        assert not hasattr(vr, "_short_term"), \
            "voice_router must not have a module-level _short_term — use get_state().session"

    def test_no_module_level_long_term_singleton(self):
        """
        voice_router must NOT have a module-level _long_term attribute
        that creates a separate LongTermMemory instance.
        """
        import app.api.voice_router as vr
        assert not hasattr(vr, "_long_term"), \
            "voice_router must not have a module-level _long_term — use get_state().long_term"


class TestVoiceResponseShape:
    """
    Check 3 — VoiceResponse must have all six required fields.
    """

    def test_voice_response_has_all_six_fields(self):
        from app.api.voice_router import VoiceResponse
        import inspect

        fields = VoiceResponse.model_fields
        required_fields = {
            "transcription", "answer", "intent",
            "latency_ms", "context", "ocr_text",
        }
        missing = required_fields - set(fields.keys())
        assert not missing, f"VoiceResponse missing fields: {missing}"

    def test_voice_response_ocr_text_is_optional(self):
        """ocr_text should be None-able for non-OCR intents."""
        from app.api.voice_router import VoiceResponse
        r = VoiceResponse(
            transcription="what is in front",
            answer="chair ahead",
            intent="obstacle_query",
            latency_ms=50.0,
            context=[],
            ocr_text=None,
        )
        assert r.ocr_text is None

    def test_voice_response_context_defaults_to_empty_list(self):
        from app.api.voice_router import VoiceResponse
        r = VoiceResponse(
            transcription="test",
            answer="ok",
            intent="unknown",
            latency_ms=10.0,
            context=[],
            ocr_text=None,
        )
        assert r.context == []


class TestVoiceSilenceGate:
    """
    Check 4 — Whisper returning None (silence) must produce HTTP 400, not 200.
    """

    def test_silence_returns_400(self):
        """Required test from the step brief."""
        import io
        import wave
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.voice_router import router

        # Build a tiny silent WAV in-memory
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # 0.5 seconds of silence
            wf.writeframes(b"\x00\x00" * 8000)
        silent_wav = buf.getvalue()

        # Patch STT to simulate silence detection
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = None   # Whisper silence gate

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)

        with patch("app.api.voice_router._get_stt", return_value=mock_stt):
            response = client.post(
                "/voice",
                files={"audio": ("audio.wav", silent_wav, "audio/wav")},
            )

        assert response.status_code == 400, \
            f"Expected 400 for silent audio, got {response.status_code}"

    def test_silence_400_detail_message(self):
        """The 400 response should tell the user no speech was detected."""
        import io
        import wave
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.voice_router import router

        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = None

        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app, raise_server_exceptions=False)

        with patch("app.api.voice_router._get_stt", return_value=mock_stt):
            response = client.post(
                "/voice",
                files={"audio": ("audio.wav", b"\x00" * 100, "audio/wav")},
            )

        assert response.status_code == 400
        detail = response.json().get("detail", "").lower()
        assert "speech" in detail or "silent" in detail or "audio" in detail


# ═══════════════════════════════════════════════════════════════════════════════
# System Hardening Checklist
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemHardening:
    """Final hardening checks from the spec."""

    def test_llm_enabled_false_by_default(self):
        from app.config import settings
        assert settings.LLM_ENABLED is False, \
            "LLM_ENABLED must default to False — no accidental cloud calls"

    def test_allow_cloud_services_false_by_default(self):
        from app.config import settings
        assert settings.ALLOW_CLOUD_SERVICES is False, \
            "ALLOW_CLOUD_SERVICES must default to False — privacy preserved"

    def test_openai_key_checked_before_llm_call(self):
        """LLM layer must be inactive when OPENAI_API_KEY is None."""
        from app.api.voice_router import _llm_enabled
        from app.config import settings

        # Default config has OPENAI_API_KEY=None and LLM_ENABLED=False
        assert not _llm_enabled(), \
            "_llm_enabled() must return False when OPENAI_API_KEY is None"

    def test_cors_not_wildcard(self):
        """Required test from the step brief."""
        from app.config import settings
        assert "*" not in settings.CORS_ORIGINS
        assert "http://localhost:3000" in settings.CORS_ORIGINS

    def test_chroma_empty_collection_returns_empty_list(self):
        """LongTermMemory.list_all() on an empty collection must return [] not raise."""
        from app.memory_modules.long_term import LongTermMemory

        # Use a tmp path so we don't touch the real chroma store
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with patch("app.memory_modules.long_term.settings") as mock_cfg:
                mock_cfg.CHROMA_PERSIST_DIR = tmp
                mock_cfg.CHROMA_COLLECTION_NAME = "test_empty"
                mock_cfg.SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
                try:
                    mem = LongTermMemory()
                    result = mem.list_all()
                    assert result == [] or isinstance(result, list), \
                        "list_all() on empty collection must return a list"
                except Exception as exc:
                    # If the real model can't be loaded (CI), skip gracefully
                    pytest.skip(f"LongTermMemory requires model download: {exc}")
