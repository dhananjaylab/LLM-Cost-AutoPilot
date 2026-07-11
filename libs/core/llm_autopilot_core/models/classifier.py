"""
ORM model — versioned snapshots of the trained complexity classifier.

Written by the weekly `retrain_classifier` Celery beat task after each
shadow-test run (see apps/worker/.../tasks/retraining.py). Exactly one
version should have `promoted=True` at a time — that's the artifact the
classifier-loading code should use for live routing decisions.

This table has no matching Pydantic schema in schemas.py yet, since
Phase 2 (the classifier itself) hasn't been built. The shape here
anticipates what retrain_classifier's TODO comments describe; expect it
to gain fields (e.g. per-tier precision/recall) once that phase lands.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from llm_autopilot_core.database import Base


class ClassifierVersion(Base):
    __tablename__ = "classifier_versions"

    version_number: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
    confusion_matrix: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    training_examples_count: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_classifier_versions_promoted", "promoted"),)
