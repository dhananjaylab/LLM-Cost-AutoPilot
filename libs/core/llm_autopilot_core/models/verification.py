"""
ORM model — result of the async Celery verification task
(`llm_autopilot_worker.tasks.verification.verify_response`).

A request may have zero verification rows (not sampled), one
(verified, passed/failed), or additional rows if escalation reruns the
request against a higher-tier model and that rerun gets verified too.

Phase 4 additions — all nullable, purely additive:
  - escalated_content: the corrected output when escalation succeeds
    (fresh rerun, or reused from the pairwise comparison call for
    creative/reasoning categories).
  - feature_vector: the 11-float vector classifier/features.py would
    extract from the prompt. Stored instead of the raw prompt — the
    `requests` table intentionally never persists prompt text — so this
    is the only way retrain_classifier can turn a routing failure into a
    training example without reversing that privacy decision.
  - corrected_tier: the tier this request should have been routed to.
    Only populated when escalation succeeded; this is the label
    retrain_classifier() trains against.
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
from llm_autopilot_core.schemas import ComplexityTier, EscalationReason, VerificationStatus


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

    # ── Phase 4 additions ────────────────────────────────────────────────────
    escalated_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_vector: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)
    corrected_tier: Mapped[ComplexityTier | None] = mapped_column(
        enum_column(ComplexityTier), nullable=True
    )

    __table_args__ = (
        Index("ix_verifications_request_id", "request_id"),
        Index("ix_verifications_status", "status"),
        # Backs retrain_classifier()'s "new feedback examples since last
        # retrain" query — always filters on corrected_tier IS NOT NULL
        # first, then a created_at range.
        Index("ix_verifications_corrected_tier_created_at", "corrected_tier", "created_at"),
    )
