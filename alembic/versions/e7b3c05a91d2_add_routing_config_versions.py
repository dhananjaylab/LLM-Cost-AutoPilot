"""add routing_config_versions table

Revision ID: e7b3c05a91d2
Revises: c3a1f9e2d7b4
Create Date: 2026-08-10 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7b3c05a91d2"
down_revision: str | None = "c3a1f9e2d7b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "routing_config_versions",
        sa.Column("version_number", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("config_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("promoted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("updated_by", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_routing_config_versions_promoted", "routing_config_versions", ["promoted"])


def downgrade() -> None:
    op.drop_index("ix_routing_config_versions_promoted", table_name="routing_config_versions")
    op.drop_table("routing_config_versions")
