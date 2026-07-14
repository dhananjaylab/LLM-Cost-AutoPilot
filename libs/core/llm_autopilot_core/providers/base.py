"""
Provider adapter interface.

Every LLM provider (OpenAI, Anthropic, Google, Groq, Ollama) implements
BaseProviderAdapter.send() the same way: take messages + a ModelConfig,
return an uncosted ProviderResponse (latency_ms and cost_usd are filled
in by the dispatcher, not the adapter — the adapter shouldn't need to
know about pricing or timing wrapper logic).

Adapters are intentionally "dumb": they translate our Message list into
the provider's wire format, make the call, and translate the raw usage
fields back into ProviderResponse. All retry/circuit-breaker/cost/metric
concerns live in dispatcher.send_request(), not here, so adding a sixth
provider never touches that shared logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_autopilot_core.schemas import Message, ModelConfig, ProviderResponse


class ProviderError(Exception):
    """
    Raised by an adapter when a provider call fails.

    `retryable` distinguishes transient failures (rate limits, timeouts,
    5xx, connection errors) — worth retrying / worth tripping the circuit
    breaker — from permanent ones (bad API key, malformed request) that
    tenacity should NOT retry.
    """

    def __init__(self, provider: str, model_id: str, message: str, *, retryable: bool = False):
        self.provider = provider
        self.model_id = model_id
        self.retryable = retryable
        super().__init__(f"[{provider}/{model_id}] {message}")


class BaseProviderAdapter(ABC):
    """One stateless instance per provider; SDK clients are created lazily and cached."""

    @abstractmethod
    async def send(
        self,
        messages: list[Message],
        model_config: ModelConfig,
        *,
        max_tokens: int,
        temperature: float,
    ) -> ProviderResponse:
        """
        Call the provider and return content + token usage.

        Implementations MUST leave `latency_ms` and the (not-yet-added)
        cost fields at their defaults — the dispatcher overwrites them.
        Implementations MUST raise ProviderError (not the raw SDK
        exception) so callers only ever need to catch one type.
        """
        raise NotImplementedError
