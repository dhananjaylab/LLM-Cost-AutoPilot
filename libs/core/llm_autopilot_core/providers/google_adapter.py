"""Google adapter — Gemini 1.5 Pro/Flash via the unified `google-genai` async client."""

from __future__ import annotations

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.providers.base import BaseProviderAdapter, ProviderError
from llm_autopilot_core.schemas import Message, ModelConfig, Provider, ProviderResponse
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# google-genai raises ClientError (4xx) vs ServerError (5xx); only the
# latter (and its base APIError for connection-level failures) is worth retrying.
_RETRYABLE = (genai_errors.ServerError,)


class GoogleAdapter(BaseProviderAdapter):
    def __init__(self) -> None:
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            settings = get_settings()
            if not settings.google_api_key:
                raise ProviderError(
                    Provider.GOOGLE.value, "unknown", "GOOGLE_API_KEY is not configured"
                )
            self._client = genai.Client(api_key=settings.google_api_key.get_secret_value())
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
        # Gemini takes system instructions out-of-band (config.system_instruction),
        # and wants plain conversational turns for everything else.
        system_parts = [m.content for m in messages if m.role == "system"]
        contents = [
            genai_types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[genai_types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction="\n".join(system_parts) if system_parts else None,
        )

        try:
            response = await client.aio.models.generate_content(
                model=model_config.model_id,
                contents=contents,
                config=config,
            )
        except _RETRYABLE as exc:
            raise ProviderError(
                Provider.GOOGLE.value, model_config.model_id, str(exc), retryable=True
            ) from exc
        except genai_errors.APIError as exc:
            raise ProviderError(
                Provider.GOOGLE.value, model_config.model_id, str(exc), retryable=False
            ) from exc

        usage = response.usage_metadata
        return ProviderResponse(
            content=response.text or "",
            input_tokens=usage.prompt_token_count or 0 if usage else 0,
            output_tokens=usage.candidates_token_count or 0 if usage else 0,
            latency_ms=0.0,
            model_id=model_config.model_id,
            provider=Provider.GOOGLE,
            raw_response={"response_id": response.response_id},
        )
