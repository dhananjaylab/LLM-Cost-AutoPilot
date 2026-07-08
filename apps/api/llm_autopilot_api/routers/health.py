"""
Health probes.

GET /v1/healthz   — liveness: process is alive (no external checks)
GET /v1/readyz    — readiness: DB + Redis are reachable (controls traffic ingress)
"""

from __future__ import annotations

from datetime import UTC, datetime

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, HTTPException, status
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.database import engine
from llm_autopilot_core.registry import MODEL_REGISTRY
from pydantic import BaseModel
from sqlalchemy import text

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["health"])
settings = get_settings()


class LivenessResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    timestamp: datetime
    checks: dict[str, str]
    models_loaded: int
    available_providers: list[str]


@router.get(
    "/healthz",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Returns 200 as long as the process is running. No external checks.",
)
async def liveness() -> LivenessResponse:
    return LivenessResponse(
        status="ok",
        timestamp=datetime.now(tz=UTC),
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get(
    "/readyz",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Checks PostgreSQL and Redis connectivity. Returns 503 if either dependency is unreachable."
    ),
    responses={503: {"description": "One or more dependencies are not ready"}},
)
async def readiness() -> ReadinessResponse:
    checks: dict[str, str] = {}
    all_ok = True

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        logger.warning("readiness_postgres_fail", error=str(exc))
        checks["postgres"] = f"error: {exc}"
        all_ok = False

    # ── Redis ─────────────────────────────────────────────────────────────────
    try:
        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.close()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.warning("readiness_redis_fail", error=str(exc))
        checks["redis"] = f"error: {exc}"
        all_ok = False

    response = ReadinessResponse(
        status="ready" if all_ok else "degraded",
        timestamp=datetime.now(tz=UTC),
        checks=checks,
        models_loaded=len(MODEL_REGISTRY),
        available_providers=settings.available_providers,
    )

    if not all_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )

    return response
