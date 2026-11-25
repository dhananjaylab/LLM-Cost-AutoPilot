"""Anthropic adapter — explicit prompt caching via cache_control breakpoints."""

from __future__ import annotations

from typing import Any, cast

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_autopilot_core.config import get_settings
from llm_autopilot_core.providers.base import BaseProviderAdapter, ProviderError
from llm_autopilot_core.providers.caching import MAX_BREAKPOINTS, estimate_tokens
from llm_autopilot_core.schemas import Message, ModelConfig, Provider, ProviderResponse

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
)
_EPHEMERAL = {"type": "ephemeral"}


def _build_request_parts(
    messages: list[Message], model_config: ModelConfig
) -> tuple[Any, list[dict[str, Any]]]:
    """
    Split system/conversation and attach cache_control to flagged messages whose
    cumulative prefix (system first, then turns) reaches the model's minimum
    cacheable size — below it Anthropic silently ignores the marker.
    """
    system_msgs = [m for m in messages if m.role == "system"]
    convo_msgs = [m for m in messages if m.role != "system"]

    used = 0
    prefix_tokens = 0

    def eligible(m: Message) -> bool:
        nonlocal used
        return (
            m.cache
            and used < MAX_BREAKPOINTS
            and (prefix_tokens >= model_config.prompt_cache_min_tokens)
        )

    system_blocks: list[dict[str, Any]] = []
    for m in system_msgs:
        prefix_tokens += estimate_tokens(m.content)
        block: dict[str, Any] = {"type": "text", "text": m.content}
        if eligible(m):
            block["cache_control"] = _EPHEMERAL
            used += 1
        system_blocks.append(block)

    conversation: list[dict[str, Any]] = []
    for m in convo_msgs:
        prefix_tokens += estimate_tokens(m.content)
        if eligible(m):
            used += 1
            conversation.append(
                {
                    "role": m.role,
                    "content": [{"type": "text", "text": m.content, "cache_control": _EPHEMERAL}],
                }
            )
        else:
            conversation.append({"role": m.role, "content": m.content})

    if not system_blocks:
        system: Any = anthropic.omit
    elif any("cache_control" in b for b in system_blocks):
        system = system_blocks  # block form is required to carry cache_control
    else:
        system = "\n".join(b["text"] for b in system_blocks)  # unchanged legacy shape
    return system, conversation


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
        system, conversation = _build_request_parts(messages, model_config)
        try:
            response = await client.messages.create(
                model=model_config.model_id,
                messages=cast(list[MessageParam], conversation),
                system=system,
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

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        # Anthropic's input_tokens EXCLUDES cache reads/writes — normalise to a
        # total so compute_cost() sees one consistent shape across providers.
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        return ProviderResponse(
            content=text,
            input_tokens=usage.input_tokens + cache_read + cache_write,
            output_tokens=usage.output_tokens,
            cached_input_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_ms=0.0,
            model_id=model_config.model_id,
            provider=Provider.ANTHROPIC,
            raw_response={"stop_reason": response.stop_reason, "id": response.id},
        )
