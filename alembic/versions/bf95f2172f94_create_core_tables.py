"""create core request/routing/verification/cost tables

Revision ID: bf95f2172f94
Revises:
Create Date: 2026-07-11 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bf95f2172f94"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── requests ──────────────────────────────────────────────────────────────
    op.create_table(
        "requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("message_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("max_tokens", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("force_tier", sa.String(length=16), nullable=True),
        sa.Column(
            "caller_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_requests_prompt_hash", "requests", ["prompt_hash"])
    op.create_index("ix_requests_created_at", "requests", ["created_at"])

    # ── responses ─────────────────────────────────────────────────────────────
    op.create_table(
        "responses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("requests.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("complexity_tier", sa.String(length=32), nullable=False),
        sa.Column("classifier_confidence", sa.Float(), nullable=False),
    )
    op.create_index("ix_responses_request_id", "responses", ["request_id"])
    op.create_index("ix_responses_created_at_provider", "responses", ["created_at", "provider"])
    op.create_index("ix_responses_created_at_tier", "responses", ["created_at", "complexity_tier"])

    # ── routing_decisions ─────────────────────────────────────────────────────
    op.create_table(
        "routing_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("requests.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("complexity_tier", sa.String(length=32), nullable=False),
        sa.Column("classifier_confidence", sa.Float(), nullable=False),
        sa.Column("selected_model_id", sa.String(length=128), nullable=False),
        sa.Column("selected_provider", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "alternatives_considered",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "circuit_breaker_overrides",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
    )
    op.create_index("ix_routing_decisions_request_id", "routing_decisions", ["request_id"])

    # ── verifications ─────────────────────────────────────────────────────────
    op.create_table(
        "verifications",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("requests.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("original_model_id", sa.String(length=128), nullable=False),
        sa.Column("judge_model_id", sa.String(length=128), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("quality_gap", sa.Float(), nullable=True),
        sa.Column("escalation_reason", sa.String(length=32), nullable=True),
        sa.Column("escalated_model_id", sa.String(length=128), nullable=True),
        sa.Column("cost_delta_usd", sa.Float(), nullable=True),
    )
    op.create_index("ix_verifications_request_id", "verifications", ["request_id"])
    op.create_index("ix_verifications_status", "verifications", ["status"])

    # ── cost_aggregates ───────────────────────────────────────────────────────
    op.create_table(
        "cost_aggregates",
        sa.Column("date", sa.Date(), primary_key=True),
        sa.Column("total_requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("hypothetical_cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cost_savings_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cache_hit_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("escalation_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("avg_quality_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "requests_by_tier",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "requests_by_provider",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ── classifier_versions ───────────────────────────────────────────────────
    op.create_table(
        "classifier_versions",
        sa.Column("version_number", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("accuracy", sa.Float(), nullable=False),
        sa.Column("precision_macro", sa.Float(), nullable=True),
        sa.Column("recall_macro", sa.Float(), nullable=True),
        sa.Column("confusion_matrix", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("training_examples_count", sa.Integer(), nullable=False),
        sa.Column("artifact_path", sa.String(length=512), nullable=False),
        sa.Column("promoted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_classifier_versions_promoted", "classifier_versions", ["promoted"])


def downgrade() -> None:
    op.drop_table("classifier_versions")
    op.drop_table("cost_aggregates")
    op.drop_index("ix_verifications_status", table_name="verifications")
    op.drop_index("ix_verifications_request_id", table_name="verifications")
    op.drop_table("verifications")
    op.drop_index("ix_routing_decisions_request_id", table_name="routing_decisions")
    op.drop_table("routing_decisions")
    op.drop_index("ix_responses_created_at_tier", table_name="responses")
    op.drop_index("ix_responses_created_at_provider", table_name="responses")
    op.drop_index("ix_responses_request_id", table_name="responses")
    op.drop_table("responses")
    op.drop_index("ix_requests_created_at", table_name="requests")
    op.drop_index("ix_requests_prompt_hash", table_name="requests")
    op.drop_table("requests")
