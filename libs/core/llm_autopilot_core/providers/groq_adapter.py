"""Groq adapter — Llama/Mixtral on Groq's custom silicon via the official AsyncGroq client."""

from __future__ import annotations

import groq
from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_autopilot_core.config import get_settings
from llm_autopilot_core.providers.base import BaseProviderAdapter, ProviderError
from llm_autopilot_core.schemas import Message, ModelConfig, Provider, ProviderResponse

_RETRYABLE = (
    groq.RateLimitError,
    groq.APITimeoutError,
    groq.APIConnectionError,
    groq.InternalServerError,
)


class GroqAdapter(BaseProviderAdapter):
    def __init__(self) -> None:
        self._client: AsyncGroq | None = None

    def _get_client(self) -> AsyncGroq:
        if self._client is None:
            settings = get_settings()
            if not settings.groq_api_key:
                raise ProviderError(
                    Provider.GROQ.value, "unknown", "GROQ_API_KEY is not configured"
                )
            self._client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())
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
        messages_param: list[ChatCompletionMessageParam] = [
            {
                "role": m.role,
                "content": m.content,
            }  # type: ignore[typeddict-item]
            for m in messages
        ]
        try:
            completion = await client.chat.completions.create(
                model=model_config.model_id,
                messages=messages_param,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except _RETRYABLE as exc:
            raise ProviderError(
                Provider.GROQ.value, model_config.model_id, str(exc), retryable=True
            ) from exc
        except groq.GroqError as exc:
            raise ProviderError(
                Provider.GROQ.value, model_config.model_id, str(exc), retryable=False
            ) from exc

        choice = completion.choices[0]
        usage = completion.usage
        return ProviderResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=0.0,
            model_id=model_config.model_id,
            provider=Provider.GROQ,
            raw_response={"finish_reason": choice.finish_reason, "id": completion.id},
        )
