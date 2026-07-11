"""
ORM model — logged for every request that reaches the router.

Per RoutingDecision's docstring in schemas.py, this is only written on
cache misses (a cache hit never invokes the classifier/router at all). A
single request can produce more than one row here if the async
verification loop's escalation check triggers a re-route to a
higher-tier model — `reason` and `circuit_breaker_overrides` distinguish
the original decision from an escalation rerun.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from llm_autopilot_core.database import Base
from llm_autopilot_core.models.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin
from llm_autopilot_core.models.types import enum_column
from llm_autopilot_core.schemas import ComplexityTier, Provider


class RoutingDecision(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "routing_decisions"

    request_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    complexity_tier: Mapped[ComplexityTier] = mapped_column(
        enum_column(ComplexityTier), nullable=False
    )
    classifier_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    selected_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_provider: Mapped[Provider] = mapped_column(enum_column(Provider), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    alternatives_considered: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    circuit_breaker_overrides: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )

    __table_args__ = (Index("ix_routing_decisions_request_id", "request_id"),)
