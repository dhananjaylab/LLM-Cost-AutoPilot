"""
ORM model — one row per inbound POST /v1/completions request.

This is the top-level audit record that Response, RoutingDecision, and
Verification all hang off of via `request_id`. We deliberately store a
hash of the prompt rather than raw text — per Phase 4's audit-trail
requirement ("prompt hash") — so the table stays useful for dedup and
cost analytics without holding onto potentially sensitive user content.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Float, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from llm_autopilot_core.database import Base
from llm_autopilot_core.models.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin


class Request(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Inbound completion request, logged before cache lookup / routing occurs."""

    __tablename__ = "requests"

    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    # Mirrors CompletionRequest.force_tier; stored as plain text (not the
    # enum_column helper) since it's caller-supplied and optional — a
    # constraint here would reject valid requests if the enum ever grows.
    force_tier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Named `caller_metadata`, not `metadata` — that name is reserved by
    # SQLAlchemy's declarative Base (Base.metadata is the schema registry).
    caller_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_requests_prompt_hash", "prompt_hash"),
        Index("ix_requests_created_at", "created_at"),
    )
