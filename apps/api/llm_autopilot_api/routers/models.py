"""
GET /v1/models — list every model in MODEL_REGISTRY with pricing, quality
tier, and live circuit-breaker availability. Read-only, no auth.
"""

from __future__ import annotations

from fastapi import APIRouter
from llm_autopilot_core.providers.dispatcher import is_provider_available
from llm_autopilot_core.registry import MODEL_REGISTRY
from llm_autopilot_core.schemas import ModelInfo, ModelListResponse

router = APIRouter(tags=["models"])


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List every model in the registry, with pricing and live availability",
)
async def list_models() -> ModelListResponse:
    models = [
        ModelInfo(
            registry_key=key,
            provider=model.provider,
            model_id=model.model_id,
            display_name=model.display_name,
            quality_tier=model.quality_tier,
            cost_per_input_token=model.cost_per_input_token,
            cost_per_output_token=model.cost_per_output_token,
            cost_per_1k_tokens=model.cost_per_1k_tokens,
            avg_latency_ms=model.avg_latency_ms,
            context_window=model.context_window,
            max_output_tokens=model.max_output_tokens,
            enabled=model.enabled,
            circuit_breaker_available=is_provider_available(model.provider),
        )
        for key, model in MODEL_REGISTRY.items()
    ]
    return ModelListResponse(models=models)
