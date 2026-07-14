"""
Ollama adapter — local models via Ollama's plain REST API (no official
async SDK, so we talk to POST {base_url}/api/chat directly with httpx).

No API key: this is the one provider that's "free" both in cost
(registry.py prices it at $0.00) and in setup — anyone with `ollama serve`
running locally can exercise the whole pipeline without any provider keys.
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_autopilot_core.config import get_settings
from llm_autopilot_core.providers.base import BaseProviderAdapter, ProviderError
from llm_autopilot_core.schemas import Message, ModelConfig, Provider, ProviderResponse

_RETRYABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)


class OllamaAdapter(BaseProviderAdapter):
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            settings = get_settings()
            self._client = httpx.AsyncClient(
                base_url=settings.ollama_base_url,
                timeout=settings.ollama_timeout,
            )
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
        try:
            resp = await client.post(
                "/api/chat",
                json={
                    "model": model_config.model_id,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
        except _RETRYABLE as exc:
            raise ProviderError(
                Provider.OLLAMA.value, model_config.model_id, str(exc), retryable=True
            ) from exc
        except httpx.HTTPStatusError as exc:
            # 404 = model not pulled locally; treat as non-retryable operator error.
            raise ProviderError(
                Provider.OLLAMA.value, model_config.model_id, str(exc), retryable=False
            ) from exc

        data = resp.json()
        return ProviderResponse(
            content=data.get("message", {}).get("content", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=0.0,
            model_id=model_config.model_id,
            provider=Provider.OLLAMA,
            raw_response={"done_reason": data.get("done_reason")},
        )
