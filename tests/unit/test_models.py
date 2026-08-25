"""
Unit tests for libs/core/llm_autopilot_core/models.

These only inspect SQLAlchemy metadata — no Postgres connection required,
consistent with the rest of tests/unit/. Live-DB behavior (constraints,
FK enforcement, cascade deletes) belongs in tests/integration/.
"""

from __future__ import annotations

from llm_autopilot_core.database import Base
from llm_autopilot_core.models import (
    ClassifierVersion,
    CostAggregate,
    Request,
    Response,
    RoutingDecision,
    Verification,
)

EXPECTED_TABLES = {
    "requests",
    "responses",
    "routing_decisions",
    "verifications",
    "cost_aggregates",
    "classifier_versions",
}


class TestModelRegistration:
    def test_all_tables_registered_on_metadata(self) -> None:
        assert set(Base.metadata.tables.keys()) >= EXPECTED_TABLES

    def test_importing_models_package_is_sufficient(self) -> None:
        # Re-importing here (already imported above) should not raise and
        # should not duplicate table registration.
        from llm_autopilot_core import models as models_pkg

        assert set(models_pkg.__all__) == {
            "ClassifierVersion",
            "CostAggregate",
            "Request",
            "Response",
            "RoutingConfigVersion",
            "RoutingDecision",
            "Verification",
        }


class TestForeignKeys:
    def test_response_fks_to_requests(self) -> None:
        table = Response.__table__
        fk_targets = {fk.target_fullname for fk in table.foreign_keys}
        assert "requests.id" in fk_targets

    def test_routing_decision_fks_to_requests(self) -> None:
        table = RoutingDecision.__table__
        fk_targets = {fk.target_fullname for fk in table.foreign_keys}
        assert "requests.id" in fk_targets

    def test_verification_fks_to_requests(self) -> None:
        table = Verification.__table__
        fk_targets = {fk.target_fullname for fk in table.foreign_keys}
        assert "requests.id" in fk_targets

    def test_fk_cascade_deletes_are_configured(self) -> None:
        for model in (Response, RoutingDecision, Verification):
            for fk in model.__table__.foreign_keys:
                assert fk.ondelete == "CASCADE"


class TestPrimaryKeys:
    def test_uuid_pk_tables_use_uuid_id(self) -> None:
        for model in (Request, Response, RoutingDecision, Verification):
            pk_cols = [c.name for c in model.__table__.primary_key.columns]
            assert pk_cols == ["id"]

    def test_cost_aggregate_uses_date_as_natural_key(self) -> None:
        pk_cols = [c.name for c in CostAggregate.__table__.primary_key.columns]
        assert pk_cols == ["date"]

    def test_classifier_version_uses_autoincrement_version(self) -> None:
        pk_cols = [c.name for c in ClassifierVersion.__table__.primary_key.columns]
        assert pk_cols == ["version_number"]


class TestNoReservedNameCollisions:
    def test_request_does_not_shadow_declarative_metadata(self) -> None:
        # `metadata` is reserved by SQLAlchemy's DeclarativeBase — the JSONB
        # column must be named something else (caller_metadata).
        assert "metadata" not in Request.__table__.columns
        assert "caller_metadata" in Request.__table__.columns
