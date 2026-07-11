"""
ORM model — one row per model response actually produced.

On a cache hit, we still write a Response row (cost_usd=0.0, latency_ms
reflecting the cache lookup, not a provider call) so `/v1/stats` and the
Grafana cost-overview dashboard have a single source of truth for "what
model/tier served this" regardless of whether the tokens came from a
fresh provider call or the semantic cache.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from llm_autopilot_core.database import Base
from llm_autopilot_core.models.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin
from llm_autopilot_core.models.types import enum_column
from llm_autopilot_core.schemas import ComplexityTier, Provider


class Response(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "responses"

    request_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Full content is nullable so a future privacy-conscious deployment can
    # write NULL here (e.g. via a settings flag) while everything else in
    # this row — cost, tokens, tier, model — stays intact for analytics.
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    provider: Mapped[Provider] = mapped_column(enum_column(Provider), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    complexity_tier: Mapped[ComplexityTier] = mapped_column(
        enum_column(ComplexityTier), nullable=False
    )
    classifier_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("ix_responses_request_id", "request_id"),
        # Composite indexes back the "routing distribution by tier" /
        # "requests by provider" Grafana panels and GET /v1/stats queries,
        # which always filter by a created_at range first.
        Index("ix_responses_created_at_provider", "created_at", "provider"),
        Index("ix_responses_created_at_tier", "created_at", "complexity_tier"),
    )
