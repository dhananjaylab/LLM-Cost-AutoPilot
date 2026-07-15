"""
Unit tests for llm_autopilot_core.providers.

These mock each SDK's response object rather than hitting real APIs —
consistent with tests/unit/ not requiring external services (see
tests/unit/test_core.py, test_models.py). Live, end-to-end provider
calls belong in the Phase 1 Task 3 baseline script, which needs real
credentials and is meant to be run manually against .env.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from llm_autopilot_core.providers import ProviderError, send_request
from llm_autopilot_core.providers.base import BaseProviderAdapter
from llm_autopilot_core.providers.dispatcher import _ADAPTERS, _BREAKERS
from llm_autopilot_core.registry import MODEL_REGISTRY, compute_cost
from llm_autopilot_core.schemas import Message, ModelConfig, Provider, QualityTier

# ── Fixtures: fake SDK response shapes (mirrors what the real SDKs return) ────


def _fake_openai_completion(
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="chatcmpl-fake",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _fake_anthropic_message(content: str, input_tokens: int, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg-fake",
        content=[SimpleNamespace(type="text", text=content)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _fake_ollama_response(content: str, prompt_eval: int, eval_count: int) -> dict:
    return {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
        "done_reason": "stop",
    }


# ── send_request() end-to-end (dispatcher + one real adapter, SDK mocked) ─────


class TestSendRequestOpenAI:
    async def test_send_request_returns_costed_response(self) -> None:
        model_config = MODEL_REGISTRY["openai/gpt-4o-mini"]
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(return_value=_fake_openai_completion("Hello there!", 10, 5))
                )
            )
        )

        with patch.object(_ADAPTERS[Provider.OPENAI], "_get_client", return_value=fake_client):
            response = await send_request("Hi", model_config)

        assert response.content == "Hello there!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.total_tokens == 15
        assert response.model_id == "gpt-4o-mini"
        assert response.provider == Provider.OPENAI
        # latency_ms must be filled in by the dispatcher, not left at 0
        assert response.latency_ms > 0
        # cost must match registry pricing, not be left at the adapter default of 0.0
        expected_cost = compute_cost(model_config, 10, 5)
        assert response.cost_usd == pytest.approx(expected_cost)
        assert response.cost_usd > 0

    async def test_prompt_string_is_wrapped_as_single_user_message(self) -> None:
        model_config = MODEL_REGISTRY["openai/gpt-4o-mini"]
        captured: dict = {}

        async def _capture_and_respond(**kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return _fake_openai_completion("ok", 1, 1)

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=_capture_and_respond))
            )
        )
        with patch.object(_ADAPTERS[Provider.OPENAI], "_get_client", return_value=fake_client):
            await send_request("plain string prompt", model_config)

        assert captured["messages"] == [{"role": "user", "content": "plain string prompt"}]


class TestSendRequestAnthropic:
    async def test_system_message_is_split_out(self) -> None:
        model_config = MODEL_REGISTRY["anthropic/claude-haiku-4-5"]
        captured: dict = {}

        async def _capture_and_respond(**kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return _fake_anthropic_message("Hi back", 20, 8)

        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(side_effect=_capture_and_respond))
        )
        with patch.object(_ADAPTERS[Provider.ANTHROPIC], "_get_client", return_value=fake_client):
            response = await send_request(
                [
                    Message(role="system", content="You are terse."),
                    Message(role="user", content="Hello"),
                ],
                model_config,
            )

        assert captured["system"] == "You are terse."
        assert captured["messages"] == [{"role": "user", "content": "Hello"}]
        assert response.content == "Hi back"
        assert response.input_tokens == 20
        assert response.output_tokens == 8


class TestSendRequestOllama:
    async def test_ollama_needs_no_api_key(self) -> None:
        """
        Ollama is the only provider that should work with zero configured
        secrets. NOTE: as of this test, MODEL_REGISTRY has no ollama/* entry
        (it was removed after Task 2 was implemented) — this ModelConfig is
        built locally so the adapter itself stays covered regardless. If
        Ollama support is intentionally dropped, this test (and the adapter/
        dispatcher wiring for Provider.OLLAMA) should be removed too.
        """
        model_config = ModelConfig(
            provider=Provider.OLLAMA,
            model_id="llama3.1",
            display_name="Llama 3.1 8B (Local)",
            cost_per_input_token=0.0,
            cost_per_output_token=0.0,
            avg_latency_ms=2_500,
            quality_tier=QualityTier.LOW,
            context_window=128_000,
            max_output_tokens=4_096,
        )
        fake_resp = SimpleNamespace(
            json=lambda: _fake_ollama_response("local reply", 12, 6),
            raise_for_status=lambda: None,
        )
        with patch.object(_ADAPTERS[Provider.OLLAMA], "_get_client") as get_client:
            get_client.return_value = SimpleNamespace(post=AsyncMock(return_value=fake_resp))
            response = await send_request("Hi", model_config)

        assert response.content == "local reply"
        assert response.cost_usd == 0.0  # Ollama is priced at $0 in the registry


# ── Error handling ─────────────────────────────────────────────────────────────


class TestErrorHandling:
    async def test_missing_api_key_raises_provider_error(self) -> None:
        model_config = MODEL_REGISTRY["openai/gpt-4o-mini"]
        adapter = _ADAPTERS[Provider.OPENAI]
        adapter._client = None  # force lazy re-creation to hit the missing-key branch

        with patch("llm_autopilot_core.providers.openai_adapter.get_settings") as mock_settings:
            mock_settings.return_value = SimpleNamespace(openai_api_key=None)
            with pytest.raises(ProviderError) as exc_info:
                await send_request("Hi", model_config)

            assert exc_info.value.provider == "openai"
            assert exc_info.value.retryable is False

    async def test_provider_error_message_includes_provider_and_model(self) -> None:
        err = ProviderError("anthropic", "claude-3-5-sonnet-20241022", "boom", retryable=True)
        assert "anthropic" in str(err)
        assert "claude-3-5-sonnet-20241022" in str(err)
        assert err.retryable is True


# ── Adapter registry coverage ───────────────────────────────────────────────────


class TestAdapterRegistry:
    def test_every_provider_enum_member_has_an_adapter(self) -> None:
        for provider in Provider:
            assert provider in _ADAPTERS
            assert isinstance(_ADAPTERS[provider], BaseProviderAdapter)

    def test_every_provider_enum_member_has_a_circuit_breaker(self) -> None:
        for provider in Provider:
            assert provider in _BREAKERS

    def test_every_registry_model_has_a_matching_adapter(self) -> None:
        """Every model in MODEL_REGISTRY must be routable — no orphaned entries."""
        for model in MODEL_REGISTRY.values():
            assert model.provider in _ADAPTERS
