"""
GET /v1/stats — cost/quality summary rolled up from the cost_aggregates
table (see apps/worker/.../tasks/retraining.aggregate_daily_costs, which
writes one row per UTC day at 01:00 UTC).

Deliberately reads only fully-aggregated days rather than computing a
live "today" bucket on the fly (confirmed with the project owner) —
keeps this endpoint a fast, predictable read against one small table
instead of an ad hoc aggregate query over the full
requests/responses/verifications tables on every call. Today's data
becomes visible here the following morning once the beat task runs;
Grafana's cost_overview dashboard is the place for live/real-time
numbers, since it reads Prometheus counters directly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, cast

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from llm_autopilot_core.models import CostAggregate
from llm_autopilot_core.schemas import ComplexityTier, CostStats, Provider
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_autopilot_api.dependencies import get_session

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["stats"])

_DEFAULT_WINDOW_DAYS = 7


def _merge_counts(dicts: list[dict[str, Any]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for d in dicts:
        for key, value in d.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def _weighted_avg(rows: list[CostAggregate], field: str) -> float:
    """Average of `field` across `rows`, weighted by each row's
    total_requests — a day with 10x the traffic should count 10x as much
    toward the period average, not be treated equally with a quiet day."""
    total_requests = sum(r.total_requests for r in rows)
    if total_requests == 0:
        return 0.0
    weighted_sum = sum(cast(float, getattr(r, field)) * r.total_requests for r in rows)
    return cast(float, weighted_sum / total_requests)


@router.get(
    "/stats",
    response_model=CostStats,
    responses={
        400: {"description": "start_date is after end_date"},
        404: {"description": "No aggregated cost data for the requested range yet"},
    },
)
async def get_stats(
    start_date: Annotated[
        date | None,
        Query(
            default=None,
            description="Inclusive range start (UTC date). Defaults to 7 days before end_date.",
        ),
    ],
    end_date: Annotated[
        date | None,
        Query(
            default=None,
            description="Inclusive range end (UTC date). Defaults to yesterday; clamped to "
            "yesterday even if a later date is supplied, since today isn't aggregated yet.",
        ),
    ],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CostStats:
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    resolved_end = min(end_date, yesterday) if end_date is not None else yesterday
    resolved_start = (
        start_date
        if start_date is not None
        else resolved_end - timedelta(days=_DEFAULT_WINDOW_DAYS - 1)
    )

    if resolved_start > resolved_end:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date must be on or before end_date",
        )

    stmt = (
        select(CostAggregate)
        .where(CostAggregate.date >= resolved_start, CostAggregate.date <= resolved_end)
        .order_by(CostAggregate.date)
    )
    rows = list((await session.execute(stmt)).scalars().all())

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No aggregated cost data between {resolved_start} and {resolved_end}. "
                "The daily rollup runs at 01:00 UTC, so very recent days won't have data yet."
            ),
        )

    total_requests = sum(r.total_requests for r in rows)
    total_cost_usd = sum(r.total_cost_usd for r in rows)
    hypothetical_cost_usd = sum(r.hypothetical_cost_usd for r in rows)
    cost_savings_usd = sum(r.cost_savings_usd for r in rows)
    cost_savings_pct = (
        (cost_savings_usd / hypothetical_cost_usd * 100) if hypothetical_cost_usd else 0.0
    )

    requests_by_tier_raw = _merge_counts([r.requests_by_tier for r in rows])
    requests_by_provider_raw = _merge_counts([r.requests_by_provider for r in rows])

    return CostStats(
        period_start=datetime.combine(resolved_start, datetime.min.time(), tzinfo=UTC),
        period_end=datetime.combine(resolved_end, datetime.min.time(), tzinfo=UTC),
        total_requests=total_requests,
        total_cost_usd=total_cost_usd,
        hypothetical_cost_usd=hypothetical_cost_usd,
        cost_savings_usd=cost_savings_usd,
        cost_savings_pct=cost_savings_pct,
        cache_hit_rate=_weighted_avg(rows, "cache_hit_rate"),
        escalation_rate=_weighted_avg(rows, "escalation_rate"),
        avg_quality_score=_weighted_avg(rows, "avg_quality_score"),
        requests_by_tier={ComplexityTier(k): v for k, v in requests_by_tier_raw.items()},
        requests_by_provider={Provider(k): v for k, v in requests_by_provider_raw.items()},
    )
