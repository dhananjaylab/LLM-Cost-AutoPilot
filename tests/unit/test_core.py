"""
Unit tests for libs/core — config, registry, schemas.
No external services required.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest
from llm_autopilot_api import main as api_main
from llm_autopilot_core.config import Settings, get_settings
from llm_autopilot_core.registry import (
    BASELINE_MODEL_KEY,
    MODEL_REGISTRY,
    compute_cost,
    get_cheapest_model_for_tier,
    get_model,
    get_models_by_quality_tier,
)
from llm_autopilot_core.schemas import (
    CompletionRequest,
    ComplexityTier,
    Message,
    Provider,
    QualityTier,
)
from pydantic import ValidationError

# ── Config tests ──────────────────────────────────────────────────────────────


class TestSettings:
    def test_default_environment(self) -> None:
        s = Settings(
            database_url="postgresql+asyncpg://u:p@localhost/db",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert s.environment == "development"
        assert not s.is_production

    def test_invalid_environment_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(
                environment="invalid",
                database_url="postgresql+asyncpg://u:p@localhost/db",
                _env_file=None,  # type: ignore[call-arg]
            )

    def test_available_providers_no_keys(self) -> None:
        s = Settings(
            database_url="postgresql+asyncpg://u:p@localhost/db",
            _env_file=None,  # type: ignore[call-arg]
        )
        # Ollama is always available regardless of API keys
        assert "ollama" in s.available_providers

    def test_get_settings_is_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


# ── Registry tests ────────────────────────────────────────────────────────────


def test_should_sample_for_verification_uses_secure_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SecureRandomStub:
        def random(self) -> float:
            return 0.0

    cache_module = types.ModuleType("llm_autopilot_core.cache")
    semantic_cache_module = types.ModuleType("llm_autopilot_core.cache.semantic_cache")
    semantic_cache_module.get_semantic_cache = lambda: None
    monkeypatch.setitem(sys.modules, "llm_autopilot_core.cache", cache_module)
    monkeypatch.setitem(
        sys.modules,
        "llm_autopilot_core.cache.semantic_cache",
        semantic_cache_module,
    )

    classifier_module = types.ModuleType("llm_autopilot_core.classifier")
    classifier_module.get_classifier = lambda: None
    monkeypatch.setitem(sys.modules, "llm_autopilot_core.classifier", classifier_module)

    completions_module = importlib.import_module("llm_autopilot_core.completions")
    monkeypatch.setattr(completions_module, "_rng", _SecureRandomStub(), raising=True)

    settings = Settings(
        database_url="postgresql+asyncpg://u:p@localhost/db",
        _env_file=None,  # type: ignore[call-arg]
    )

    assert completions_module._should_sample_for_verification(0.5, settings) is True


def test_configure_tracing_uses_instrumentor_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    instrumented: list[str] = []

    class DummyInstrumentor:
        def __init__(self) -> None:
            self.instantiated = True

        def instrument(self) -> None:
            instrumented.append(type(self).__name__)

    fake_trace = types.ModuleType("opentelemetry.trace")
    fake_trace.set_tracer_provider = lambda provider: None

    fake_sdk_resources = types.ModuleType("opentelemetry.sdk.resources")

    class DummyResource:
        @staticmethod
        def create(_: object) -> object:
            return {}

    fake_sdk_resources.Resource = DummyResource

    fake_sdk_trace = types.ModuleType("opentelemetry.sdk.trace")

    class DummyTracerProvider:
        def __init__(self, resource: object | None = None) -> None:
            self.resource = resource

        def add_span_processor(self, processor: object) -> None:
            self.processor = processor

    fake_sdk_trace.TracerProvider = DummyTracerProvider

    fake_sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")

    class DummyBatchSpanProcessor:
        def __init__(self, exporter: object) -> None:
            self.exporter = exporter

    fake_sdk_trace_export.BatchSpanProcessor = DummyBatchSpanProcessor

    fake_otlp_exporter = types.ModuleType("opentelemetry.exporter.otlp.proto.http.trace_exporter")

    class DummyOTLPSpanExporter:
        def __init__(self, endpoint: str | None = None) -> None:
            self.endpoint = endpoint

    fake_otlp_exporter.OTLPSpanExporter = DummyOTLPSpanExporter

    fake_fastapi_module = types.ModuleType("opentelemetry.instrumentation.fastapi")
    fake_fastapi_module.FastAPIInstrumentor = DummyInstrumentor

    fake_sqlalchemy_module = types.ModuleType("opentelemetry.instrumentation.sqlalchemy")
    fake_sqlalchemy_module.SQLAlchemyInstrumentor = DummyInstrumentor

    monkeypatch.setitem(sys.modules, "opentelemetry", types.ModuleType("opentelemetry"))
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", fake_trace)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk", types.ModuleType("opentelemetry.sdk"))
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.resources", fake_sdk_resources)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", fake_sdk_trace)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace.export", fake_sdk_trace_export)
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter",
        types.ModuleType("opentelemetry.exporter"),
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp",
        types.ModuleType("opentelemetry.exporter.otlp"),
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp.proto",
        types.ModuleType("opentelemetry.exporter.otlp.proto"),
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp.proto.http",
        types.ModuleType("opentelemetry.exporter.otlp.proto.http"),
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        fake_otlp_exporter,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation",
        types.ModuleType("opentelemetry.instrumentation"),
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.fastapi",
        fake_fastapi_module,
    )
    monkeypatch.setitem(
        sys.modules, "opentelemetry.instrumentation.sqlalchemy", fake_sqlalchemy_module
    )

    api_main._configure_tracing()

    assert instrumented == ["DummyInstrumentor", "DummyInstrumentor"]


class TestModelRegistry:
    def test_registry_not_empty(self) -> None:
        # Ollama entries are deliberately commented out in registry.py for now
        # (local-model support paused, not removed) — 9 cloud-provider models.
        assert len(MODEL_REGISTRY) >= 9

    def test_all_providers_represented(self) -> None:
        providers = {m.provider for m in MODEL_REGISTRY.values()}
        assert Provider.OPENAI in providers
        assert Provider.ANTHROPIC in providers
        assert Provider.GOOGLE in providers
        assert Provider.GROQ in providers
        # Provider.OLLAMA intentionally NOT asserted here — its registry
        # entries are commented out for now. Re-add this assertion when
        # Ollama support is turned back on.

    def test_baseline_model_exists(self) -> None:
        assert BASELINE_MODEL_KEY in MODEL_REGISTRY

    def test_get_model_valid_key(self) -> None:
        model = get_model("openai/gpt-4o")
        assert model is not None
        assert model.model_id == "gpt-4o"
        assert model.provider == Provider.OPENAI

    def test_get_model_invalid_key(self) -> None:
        assert get_model("nonexistent/model") is None

    def test_ollama_models_are_free_when_enabled(self) -> None:
        """
        Ollama entries are currently commented out in registry.py, so this
        is a forward-looking check rather than a live assertion: if/when
        Ollama is re-enabled, every entry for it must still be free.
        """
        ollama_models = [m for m in MODEL_REGISTRY.values() if m.provider == Provider.OLLAMA]
        for m in ollama_models:
            assert m.cost_per_input_token == 0.0
            assert m.cost_per_output_token == 0.0

    def test_get_models_by_quality_tier(self) -> None:
        high_tier = get_models_by_quality_tier(QualityTier.HIGH)
        assert len(high_tier) >= 3  # GPT-4o, Sonnet, Gemini Pro

    def test_get_cheapest_model_for_low_tier(self) -> None:
        cheapest = get_cheapest_model_for_tier(QualityTier.LOW)
        assert cheapest is not None
        # With Ollama ($0) currently disabled, Groq's low-latency 8B variant
        # is the cheapest LOW-tier model in the registry.
        assert cheapest.cost_per_input_token == 5e-08

    def test_compute_cost(self) -> None:
        model = get_model("openai/gpt-4o")
        assert model is not None
        cost = compute_cost(model, input_tokens=1000, output_tokens=500)
        expected = (0.000_005 * 1000) + (0.000_015 * 500)
        assert abs(cost - expected) < 1e-10

    def test_groq_cheaper_than_openai_for_same_tier(self) -> None:
        groq_model = get_model("meta-llama/llama-prompt-guard-2-22m")
        openai_model = get_model("openai/gpt-4o-mini")
        assert groq_model is not None
        assert openai_model is not None
        assert groq_model.cost_per_input_token < openai_model.cost_per_input_token


# ── Schema tests ──────────────────────────────────────────────────────────────


class TestSchemas:
    def test_completion_request_valid(self) -> None:
        req = CompletionRequest(
            messages=[Message(role="user", content="Hello, world!")],
        )
        assert len(req.messages) == 1
        assert req.max_tokens == 1024
        assert req.temperature == 0.7

    def test_completion_request_invalid_role(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="invalid_role", content="test")

    def test_completion_request_empty_messages(self) -> None:
        with pytest.raises(ValidationError):
            CompletionRequest(messages=[])

    def test_complexity_tier_values(self) -> None:
        assert ComplexityTier.SIMPLE.value == "simple"
        assert ComplexityTier.MODERATE.value == "moderate"
        assert ComplexityTier.COMPLEX.value == "complex"
