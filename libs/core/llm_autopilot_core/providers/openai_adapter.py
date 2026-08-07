"""OpenAI adapter — GPT-4o, GPT-4o-mini, etc. via the official AsyncOpenAI client."""

from __future__ import annotations

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_autopilot_core.config import get_settings
from llm_autopilot_core.providers.base import BaseProviderAdapter, ProviderError
from llm_autopilot_core.schemas import Message, ModelConfig, Provider, ProviderResponse

# Transient errors worth retrying — verified against openai>=1.x's exception hierarchy.
_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIAdapter(BaseProviderAdapter):
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            settings = get_settings()
            if not settings.openai_api_key:
                raise ProviderError(
                    Provider.OPENAI.value, "unknown", "OPENAI_API_KEY is not configured"
                )
            self._client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
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
            {  # type: ignore[misc]
                "role": m.role,
                "content": m.content,
            }
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
                Provider.OPENAI.value, model_config.model_id, str(exc), retryable=True
            ) from exc
        except openai.OpenAIError as exc:
            raise ProviderError(
                Provider.OPENAI.value, model_config.model_id, str(exc), retryable=False
            ) from exc

        choice = completion.choices[0]
        usage = completion.usage
        return ProviderResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=0.0,  # filled in by dispatcher
            model_id=model_config.model_id,
            provider=Provider.OPENAI,
            raw_response={"finish_reason": choice.finish_reason, "id": completion.id},
        )
