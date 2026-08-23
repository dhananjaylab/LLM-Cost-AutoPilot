"""
ORM model — versioned snapshots of the routing configuration (tier ->
model chains, verification judge model, cost baseline).

Written by PUT /v1/admin/routing-config (apps/api/.../routers/admin.py,
via llm_autopilot_core.routing.persist_routing_config) and read back by
llm_autopilot_core.routing.refresh_routing_config_from_db() — the
DB-backed counterpart to routing.load_routing_config(), which now only
parses configs/routing.yaml to *bootstrap* the very first row (see that
function's docstring for the exact precedence).

Mirrors classifier_versions' shape on purpose: exactly one row has
promoted=True at a time, and every PUT keeps prior versions around for
an audit trail / manual rollback rather than overwriting them in place —
same reasoning classifier retraining already established in this
codebase for its own versioned artifact.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from llm_autopilot_core.database import Base


class RoutingConfigVersion(Base):
    __tablename__ = "routing_config_versions"

    version_number: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Free-text audit fields — there's no identity/auth system beyond the
    # single shared admin API key, so `updated_by` is caller-supplied
    # rather than derived from a verified principal.
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_routing_config_versions_promoted", "promoted"),)
