"""
ORM model — result of the async Celery verification task
(`llm_autopilot_worker.tasks.verification.verify_response`).

A request may have zero verification rows (not sampled — see
VERIFICATION_SAMPLE_RATE_* in config.py), one (verified, passed/failed),
or additional rows if escalation reruns the request against a
higher-tier model and that rerun gets verified too.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Float, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from llm_autopilot_core.database import Base
from llm_autopilot_core.models.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin
from llm_autopilot_core.models.types import enum_column
from llm_autopilot_core.schemas import EscalationReason, VerificationStatus


class Verification(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """`created_at` here corresponds to VerificationResult.checked_at in schemas.py."""

    __tablename__ = "verifications"

    request_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    original_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    judge_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[VerificationStatus] = mapped_column(
        enum_column(VerificationStatus), nullable=False
    )
    quality_gap: Mapped[float | None] = mapped_column(Float, nullable=True)
    escalation_reason: Mapped[EscalationReason | None] = mapped_column(
        enum_column(EscalationReason), nullable=True
    )
    escalated_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    cost_delta_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_verifications_request_id", "request_id"),
        Index("ix_verifications_status", "status"),
    )
