"""
Shared mixins for ORM models.

Every model in this package uses a UUID primary key (matching the `id` /
`request_id` fields already used throughout schemas.py) and a `created_at`
timestamp, so both live here once instead of being repeated per-table.

Note: `Verification.created_at` is what schemas.VerificationResult calls
`checked_at`. We keep the DB column name uniform across tables (so ops
queries like "rows created since X" work the same everywhere) and remap
the name at the service layer when converting ORM rows to Pydantic models.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column


class UUIDPrimaryKeyMixin:
    """Adds a UUID `id` primary key, generated client-side via uuid4."""

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


class CreatedAtMixin:
    """Adds a timezone-aware `created_at`, defaulted both in Python and at the DB."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
