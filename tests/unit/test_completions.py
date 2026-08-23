from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from llm_autopilot_core.classifier.model import ClassificationResult
from llm_autopilot_core.completions import handle_completion_request
from llm_autopilot_core.routing import (
    CostBaselineConfig,
    RoutingConfig,
    TierRoute,
    VerificationRoutingConfig,
)
from llm_autopilot_core.schemas import (
    CompletionRequest,
    ComplexityTier,
    Message,
    Provider,
    ProviderResponse,
)


def _routing_config() -> RoutingConfig:
    return RoutingConfig(
        version="1",
        tiers={
            ComplexityTier.SIMPLE: TierRoute(
                description="", models=["groq/llama-3.1-8b-instant"], max_latency_ms=3000
            ),
            ComplexityTier.MODERATE: TierRoute(
                description="", models=["openai/gpt-4o-mini"], max_latency_ms=5000
            ),
            ComplexityTier.COMPLEX: TierRoute(
                description="", models=["anthropic/claude-sonnet-4-6"], max_latency_ms=15000
            ),
        },
        verification=VerificationRoutingConfig(
            judge_model="anthropic/claude-haiku-4-5", judge_max_tokens=512
        ),
        cost_baseline=CostBaselineConfig(model="openai/gpt-4o"),
    )


@asynccontextmanager
async def _fake_managed_session():
    yield MagicMock()


@pytest.fixture(autouse=True)
def _patch_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_autopilot_core.completions.get_routing_config", _routing_config)
    monkeypatch.setattr("llm_autopilot_core.completions.managed_session", _fake_managed_session)
    monkeypatch.setattr("llm_autopilot_core.completions.enqueue_verify_response", MagicMock())


def _dispatch_response(content: str = "4") -> ProviderResponse:
    return ProviderResponse(
        content=content,
        input_tokens=5,
        output_tokens=1,
        latency_ms=50.0,
        cost_usd=0.000_01,
        model_id="llama-3.1-8b-instant",
        provider=Provider.GROQ,
    )


class TestCacheHit:
    async def test_returns_cached_response_without_calling_classifier_or_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = MagicMock()
        cache.acheck = AsyncMock(
            return_value=[
                {
                    "response": "Paris",
                    "metadata": {
                        "model_id": "gpt-4o-mini",
                        "provider": "openai",
                        "complexity_tier": "simple",
                        "classifier_confidence": 0.95,
                        "input_tokens": 10,
                        "output_tokens": 5,
                    },
                }
            ]
        )
        monkeypatch.setattr("llm_autopilot_core.completions.get_semantic_cache", lambda: cache)

        classifier = MagicMock()
        monkeypatch.setattr("llm_autopilot_core.completions.get_classifier", lambda: classifier)
        send_mock = AsyncMock()
        monkeypatch.setattr("llm_autopilot_core.completions.send_request", send_mock)

        request = CompletionRequest(
            messages=[Message(role="user", content="What is the capital of France?")]
        )
        response = await handle_completion_request(request)

        assert response.cache_hit is True
        assert response.content == "Paris"
        assert response.cost_usd == 0.0
        classifier.predict.assert_not_called()
        send_mock.assert_not_awaited()

    async def test_malformed_hit_falls_through_to_cache_miss_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = MagicMock()
        # Missing model_id/provider in metadata — _is_well_formed_hit()
        # must reject this rather than let a KeyError blow up the request.
        cache.acheck = AsyncMock(return_value=[{"response": "Paris", "metadata": {}}])
        cache.astore = AsyncMock(return_value="cache:entry:xyz")
        monkeypatch.setattr("llm_autopilot_core.completions.get_semantic_cache", lambda: cache)

        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            tier=ComplexityTier.SIMPLE, confidence=0.9, probabilities={}
        )
        monkeypatch.setattr("llm_autopilot_core.completions.get_classifier", lambda: classifier)
        monkeypatch.setattr(
            "llm_autopilot_core.completions.send_request",
            AsyncMock(return_value=_dispatch_response("Paris")),
        )

        request = CompletionRequest(
            messages=[Message(role="user", content="What is the capital of France?")]
        )
        response = await handle_completion_request(request)

        assert response.cache_hit is False
        classifier.predict.assert_called_once()


class TestCacheMiss:
    async def test_full_pipeline_dispatches_and_stores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = MagicMock()
        cache.acheck = AsyncMock(return_value=[])
        cache.astore = AsyncMock(return_value="cache:entry:new")
        monkeypatch.setattr("llm_autopilot_core.completions.get_semantic_cache", lambda: cache)

        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            tier=ComplexityTier.SIMPLE, confidence=0.5, probabilities={}
        )
        monkeypatch.setattr("llm_autopilot_core.completions.get_classifier", lambda: classifier)

        fake_send = AsyncMock(return_value=_dispatch_response("4"))
        monkeypatch.setattr("llm_autopilot_core.completions.send_request", fake_send)

        request = CompletionRequest(messages=[Message(role="user", content="What is 2+2?")])
        response = await handle_completion_request(request)

        assert response.cache_hit is False
        assert response.content == "4"
        fake_send.assert_awaited_once()
        cache.astore.assert_awaited_once()

    async def test_low_confidence_always_samples_for_verification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = MagicMock()
        cache.acheck = AsyncMock(return_value=[])
        cache.astore = AsyncMock(return_value="cache:entry:new")
        monkeypatch.setattr("llm_autopilot_core.completions.get_semantic_cache", lambda: cache)

        classifier = MagicMock()
        # Below classifier_confidence_threshold (0.85 default) →
        # verification_sample_rate_low_confidence (default 1.0) → always enqueue.
        classifier.predict.return_value = ClassificationResult(
            tier=ComplexityTier.SIMPLE, confidence=0.1, probabilities={}
        )
        monkeypatch.setattr("llm_autopilot_core.completions.get_classifier", lambda: classifier)
        monkeypatch.setattr(
            "llm_autopilot_core.completions.send_request",
            AsyncMock(return_value=_dispatch_response("4")),
        )
        enqueue_mock = MagicMock()
        monkeypatch.setattr("llm_autopilot_core.completions.enqueue_verify_response", enqueue_mock)

        request = CompletionRequest(messages=[Message(role="user", content="What is 2+2?")])
        await handle_completion_request(request)

        enqueue_mock.assert_called_once()

    async def test_force_tier_overrides_classifier_prediction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = MagicMock()
        cache.acheck = AsyncMock(return_value=[])
        cache.astore = AsyncMock(return_value="cache:entry:new")
        monkeypatch.setattr("llm_autopilot_core.completions.get_semantic_cache", lambda: cache)

        classifier = MagicMock()
        classifier.predict.return_value = ClassificationResult(
            tier=ComplexityTier.SIMPLE, confidence=0.9, probabilities={}
        )
        monkeypatch.setattr("llm_autopilot_core.completions.get_classifier", lambda: classifier)
        monkeypatch.setattr(
            "llm_autopilot_core.completions.send_request",
            AsyncMock(
                return_value=ProviderResponse(
                    content="a thoughtful essay",
                    input_tokens=50,
                    output_tokens=200,
                    latency_ms=1200.0,
                    cost_usd=0.01,
                    model_id="claude-sonnet-4-6",
                    provider=Provider.ANTHROPIC,
                )
            ),
        )

        request = CompletionRequest(
            messages=[Message(role="user", content="What is 2+2?")],
            force_tier=ComplexityTier.COMPLEX,
        )
        response = await handle_completion_request(request)

        assert response.complexity_tier == ComplexityTier.COMPLEX
        assert response.provider == Provider.ANTHROPIC

    async def test_cache_key_from_astore_is_threaded_to_verification_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Phase 5 — the semantic-cache key astore() returns must reach
        enqueue_verify_response() so a later escalation can write the
        corrected answer back to this exact entry."""
        cache = MagicMock()
        cache.acheck = AsyncMock(return_value=[])
        cache.astore = AsyncMock(return_value="cache:entry:threaded-key")
        monkeypatch.setattr("llm_autopilot_core.completions.get_semantic_cache", lambda: cache)

        classifier = MagicMock()
        # Force sampling so enqueue_verify_response is definitely called.
        classifier.predict.return_value = ClassificationResult(
            tier=ComplexityTier.SIMPLE, confidence=0.1, probabilities={}
        )
        monkeypatch.setattr("llm_autopilot_core.completions.get_classifier", lambda: classifier)
        monkeypatch.setattr(
            "llm_autopilot_core.completions.send_request",
            AsyncMock(return_value=_dispatch_response("4")),
        )
        enqueue_mock = MagicMock()
        monkeypatch.setattr("llm_autopilot_core.completions.enqueue_verify_response", enqueue_mock)

        request = CompletionRequest(messages=[Message(role="user", content="What is 2+2?")])
        await handle_completion_request(request)

        enqueue_mock.assert_called_once()
        assert enqueue_mock.call_args.kwargs["cache_key"] == "cache:entry:threaded-key"
