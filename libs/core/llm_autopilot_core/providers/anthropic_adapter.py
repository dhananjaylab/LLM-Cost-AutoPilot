"""Anthropic adapter — Claude Sonnet, Claude Haiku, etc. via the official AsyncAnthropic client."""

from __future__ import annotations

from typing import cast

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_autopilot_core.config import get_settings
from llm_autopilot_core.providers.base import BaseProviderAdapter, ProviderError
from llm_autopilot_core.schemas import Message, ModelConfig, Provider, ProviderResponse

# Verified against anthropic>=0.x's exception hierarchy.
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
)


class AnthropicAdapter(BaseProviderAdapter):
    def __init__(self) -> None:
        self._client: AsyncAnthropic | None = None

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            settings = get_settings()
            if not settings.anthropic_api_key:
                raise ProviderError(
                    Provider.ANTHROPIC.value, "unknown", "ANTHROPIC_API_KEY is not configured"
                )
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        return self._client

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def send(
        self,
        messages: list[Message],
        model_config: ModelConfig,
        *,
        max_tokens: int,
        temperature: float,
    ) -> ProviderResponse:
        client = self._get_client()
        # Anthropic takes `system` as its own top-level param, not a message
        # with role="system" — split it out before building the messages list.
        system_parts = [m.content for m in messages if m.role == "system"]
        conversation = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]

        try:
            response = await client.messages.create(
                model=model_config.model_id,
                messages=cast(list[MessageParam], conversation),
                system="\n".join(system_parts) if system_parts else anthropic.omit,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except _RETRYABLE as exc:
            raise ProviderError(
                Provider.ANTHROPIC.value, model_config.model_id, str(exc), retryable=True
            ) from exc
        except anthropic.AnthropicError as exc:
            raise ProviderError(
                Provider.ANTHROPIC.value, model_config.model_id, str(exc), retryable=False
            ) from exc

        # content is a list of blocks (text, tool_use, ...); we only care about text here.
        text = "".join(block.text for block in response.content if block.type == "text")

        return ProviderResponse(
            content=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=0.0,
            model_id=model_config.model_id,
            provider=Provider.ANTHROPIC,
            raw_response={"stop_reason": response.stop_reason, "id": response.id},
        )
