"""
send_request() — the single entrypoint every caller (router, Celery verifier,
baseline test script) goes through to call any model in MODEL_REGISTRY.

Phase 2 addition: is_provider_available() — a thin read-only accessor over
the same _BREAKERS dict used by send_request(), so llm_autopilot_core.routing
can skip providers whose circuit is OPEN without importing dispatcher
internals or duplicating breaker state. No behavior of send_request() itself
changes.
"""

from __future__ import annotations

import time

import structlog

from llm_autopilot_core.metrics import (
    circuit_breaker_state,
    cost_usd_total,
    request_latency_ms,
    requests_errors_total,
)
from llm_autopilot_core.providers.anthropic_adapter import AnthropicAdapter
from llm_autopilot_core.providers.base import BaseProviderAdapter, ProviderError
from llm_autopilot_core.providers.circuit_breaker import (
    AsyncCircuitBreaker,
    BreakerState,
    CircuitOpenError,
)
from llm_autopilot_core.providers.google_adapter import GoogleAdapter
from llm_autopilot_core.providers.groq_adapter import GroqAdapter
from llm_autopilot_core.providers.ollama_adapter import OllamaAdapter
from llm_autopilot_core.providers.openai_adapter import OpenAIAdapter
from llm_autopilot_core.registry import compute_cost
from llm_autopilot_core.schemas import Message, ModelConfig, Provider, ProviderResponse

logger = structlog.get_logger(__name__)

_ADAPTERS: dict[Provider, BaseProviderAdapter] = {
    Provider.OPENAI: OpenAIAdapter(),
    Provider.ANTHROPIC: AnthropicAdapter(),
    Provider.GOOGLE: GoogleAdapter(),
    Provider.GROQ: GroqAdapter(),
    Provider.OLLAMA: OllamaAdapter(),
}

_BREAKERS: dict[Provider, AsyncCircuitBreaker] = {
    provider: AsyncCircuitBreaker(
        name=provider.value,
        fail_max=3 if provider == Provider.OLLAMA else 5,
        reset_timeout=15.0 if provider == Provider.OLLAMA else 30.0,
    )
    for provider in Provider
}

_BREAKER_STATE_VALUE: dict[BreakerState, int] = {
    BreakerState.CLOSED: 0,
    BreakerState.OPEN: 1,
    BreakerState.HALF_OPEN: 2,
}


def is_provider_available(provider: Provider) -> bool:
    """
    Read-only check used by the routing layer: True unless the provider's
    circuit breaker is currently OPEN. HALF_OPEN counts as available (that's
    the probe state — the breaker itself decides whether the probe succeeds).
    """
    return _BREAKERS[provider].current_state != BreakerState.OPEN


async def send_request(
    prompt: str | list[Message],
    model_config: ModelConfig,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> ProviderResponse:
    messages = [Message(role="user", content=prompt)] if isinstance(prompt, str) else prompt
    adapter = _ADAPTERS[model_config.provider]
    breaker = _BREAKERS[model_config.provider]

    log = logger.bind(provider=model_config.provider.value, model_id=model_config.model_id)
    start = time.perf_counter()

    try:
        raw = await breaker.call(
            adapter.send, messages, model_config, max_tokens=max_tokens, temperature=temperature
        )
    except CircuitOpenError as exc:
        circuit_breaker_state.labels(provider=model_config.provider.value).set(
            _BREAKER_STATE_VALUE[breaker.current_state]
        )
        requests_errors_total.labels(
            provider=model_config.provider.value, error_type="circuit_open"
        ).inc()
        log.error("circuit_open", error=str(exc))
        raise ProviderError(
            model_config.provider.value,
            model_config.model_id,
            "circuit breaker open",
            retryable=True,
        ) from exc
    except ProviderError as exc:
        error_type = "transient" if exc.retryable else "permanent"
        requests_errors_total.labels(
            provider=model_config.provider.value, error_type=error_type
        ).inc()
        circuit_breaker_state.labels(provider=model_config.provider.value).set(
            _BREAKER_STATE_VALUE[breaker.current_state]
        )
        log.error("provider_call_failed", error=str(exc), retryable=exc.retryable)
        raise

    latency_ms = (time.perf_counter() - start) * 1_000
    cost_usd = compute_cost(model_config, raw.input_tokens, raw.output_tokens)
    response = raw.model_copy(update={"latency_ms": latency_ms, "cost_usd": cost_usd})

    cost_usd_total.labels(provider=model_config.provider.value, model_id=model_config.model_id).inc(
        cost_usd
    )
    request_latency_ms.labels(
        complexity_tier="unclassified", provider=model_config.provider.value
    ).observe(latency_ms)
    circuit_breaker_state.labels(provider=model_config.provider.value).set(
        _BREAKER_STATE_VALUE[breaker.current_state]
    )

    log.info(
        "provider_call_succeeded",
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        cost_usd=round(cost_usd, 6),
        latency_ms=round(latency_ms, 1),
    )
    return response
