"""
Admin/config endpoints — Phase 5.

GET  /v1/admin/routing-config           — currently active routing config
GET  /v1/admin/routing-config/versions  — audit history of past versions
PUT  /v1/admin/routing-config           — replace the active routing config

Only PUT is behind the admin API key (apps/api/.../auth.py) — the GETs
are read-only operational visibility, same trust level as GET /v1/models
and GET /v1/stats.
"""

from __future__ import annotations

from typing import cast

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from llm_autopilot_core.routing import (
    RoutingConfig,
    RoutingConfigError,
    RoutingConfigUpdateRequest,
    RoutingConfigVersionSummary,
    get_routing_config,
    list_routing_config_versions,
    persist_routing_config,
)

from llm_autopilot_api.auth import require_admin_api_key

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/routing-config",
    response_model=RoutingConfig,
    summary="The routing config this replica is currently using",
)
async def get_current_routing_config() -> RoutingConfig:
    return get_routing_config()


@router.get(
    "/routing-config/versions",
    response_model=list[RoutingConfigVersionSummary],
    summary="Audit history of past routing config versions, most recent first",
)
async def get_routing_config_history(
    limit: int = Query(default=20, ge=1, le=100),
) -> list[RoutingConfigVersionSummary]:
    return cast(list[RoutingConfigVersionSummary], await list_routing_config_versions(limit=limit))


@router.put(
    "/routing-config",
    response_model=RoutingConfig,
    dependencies=[Depends(require_admin_api_key)],
    responses={
        400: {"description": "Payload references a model key not in MODEL_REGISTRY"},
        401: {"description": "Missing or invalid X-Admin-API-Key header"},
        503: {"description": "ADMIN_API_KEY is not configured on the server"},
    },
    summary="Replace the active routing config (full document, not a patch)",
)
async def update_routing_config(payload: RoutingConfigUpdateRequest) -> RoutingConfig:
    config = RoutingConfig(
        version=payload.version,
        tiers=payload.tiers,
        verification=payload.verification,
        cost_baseline=payload.cost_baseline,
    )
    try:
        version = await persist_routing_config(
            config, notes=payload.notes, updated_by=payload.updated_by
        )
    except RoutingConfigError as exc:
        logger.error("routing_config_update_rejected", error=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info(
        "routing_config_updated", version=version.version_number, updated_by=payload.updated_by
    )
    return get_routing_config()
