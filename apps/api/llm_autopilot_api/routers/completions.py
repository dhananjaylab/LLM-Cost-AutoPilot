"""
POST /v1/completions — the main routing endpoint.

Thin FastAPI wrapper around llm_autopilot_core.completions.
handle_completion_request(); see that module for the actual
cache → classify → route → dispatch → persist → verify pipeline.
The user doesn't choose the model — the router does.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, status
from llm_autopilot_core.completions import handle_completion_request
from llm_autopilot_core.providers.base import ProviderError
from llm_autopilot_core.routing import RoutingConfigError
from llm_autopilot_core.schemas import CompletionRequest, CompletionResponse

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["completions"])


@router.post(
    "/completions",
    response_model=CompletionResponse,
    summary="Route a completion request to the cheapest capable model",
    responses={
        502: {"description": "The selected provider failed"},
        503: {"description": "Routing configuration could not resolve a model"},
    },
)
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    try:
        return await handle_completion_request(request)
    except RoutingConfigError as exc:
        logger.error("completion_routing_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ProviderError as exc:
        logger.error("completion_provider_error", error=str(exc), retryable=exc.retryable)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
