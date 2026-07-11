"""
ORM model — daily cost/quality rollup consumed by GET /v1/stats and the
Grafana cost-overview dashboard.

One row per calendar day (UTC). Written by the Celery `aggregate_daily_costs`
beat task; the task should upsert on `date` so re-runs for the same day
stay idempotent.

Unlike the other tables, this one uses `date` as a natural primary key
instead of the UUIDPrimaryKeyMixin — there's exactly one row per day by
definition, so a surrogate UUID would just add an extra unique index for
no benefit.
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from typing import Any

from sqlalchemy import Date, DateTime, Float, Integer, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from llm_autopilot_core.database import Base


class CostAggregate(Base):
    __tablename__ = "cost_aggregates"

    date: Mapped[date_type] = mapped_column(Date, primary_key=True)

    total_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    hypothetical_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cost_savings_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cache_hit_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    escalation_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    requests_by_tier: Mapped[dict[str, int]] = mapped_column(JSONB, nullable=False, default=dict)
    requests_by_provider: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
