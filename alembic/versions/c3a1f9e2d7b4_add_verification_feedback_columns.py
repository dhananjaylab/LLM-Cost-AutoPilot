"""add verification feedback loop columns (escalated_content, feature_vector, corrected_tier)

Revision ID: c3a1f9e2d7b4
Revises: bf95f2172f94
Create Date: 2026-07-29 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3a1f9e2d7b4"
down_revision: str | None = "bf95f2172f94"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "verifications",
        sa.Column("escalated_content", sa.Text(), nullable=True),
    )
    op.add_column(
        "verifications",
        sa.Column(
            "feature_vector",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "verifications",
        sa.Column("corrected_tier", sa.String(length=32), nullable=True),
    )
    op.create_index(
        "ix_verifications_corrected_tier_created_at",
        "verifications",
        ["corrected_tier", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_verifications_corrected_tier_created_at", table_name="verifications")
    op.drop_column("verifications", "corrected_tier")
    op.drop_column("verifications", "feature_vector")
    op.drop_column("verifications", "escalated_content")
