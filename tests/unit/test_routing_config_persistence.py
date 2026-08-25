"""
Unit tests for the Phase 5 additions to llm_autopilot_core.routing:
validate_routing_config_strict(), refresh_routing_config_from_db(),
persist_routing_config(), and the get_routing_config()/
reset_routing_config_cache() process-local cache.

All DB access goes through managed_session(), which every test here
replaces with a fake async context manager yielding a MagicMock session
— the same pattern already used throughout this test suite (see
test_completions.py, test_verification_task.py) — so none of this needs
a live Postgres. tests/integration/test_admin_routing_config.py covers
the real round trip.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from llm_autopilot_core.routing import (
    CostBaselineConfig,
    RoutingConfig,
    RoutingConfigError,
    TierRoute,
    VerificationRoutingConfig,
    get_routing_config,
    load_routing_config,
    persist_routing_config,
    refresh_routing_config_from_db,
    reset_routing_config_cache,
    validate_routing_config_strict,
)

_REAL_ROUTING_YAML = "configs/routing.yaml"


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_routing_config_cache()
    yield
    reset_routing_config_cache()


def _valid_config() -> RoutingConfig:
    return RoutingConfig(
        version="1",
        tiers={
            "simple": TierRoute(  # type: ignore[arg-type]
                description="", models=["meta-llama/llama-prompt-guard-2-22m"], max_latency_ms=3000
            ),
        },
        verification=VerificationRoutingConfig(
            judge_model="anthropic/claude-haiku-4-5", judge_max_tokens=512
        ),
        cost_baseline=CostBaselineConfig(model="openai/gpt-4o"),
    )


class TestGetRoutingConfigFallback:
    def test_falls_back_to_yaml_when_cache_empty(self) -> None:
        config = get_routing_config()
        expected = load_routing_config(_REAL_ROUTING_YAML)
        assert config.version == expected.version
        assert set(config.tiers) == set(expected.tiers)

    def test_returns_cached_value_once_populated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = get_routing_config()
        second = get_routing_config()
        assert first is second  # same object, no re-parse


class TestValidateRoutingConfigStrict:
    def test_accepts_valid_config(self) -> None:
        validate_routing_config_strict(_valid_config())  # must not raise

    def test_rejects_unknown_tier_model(self) -> None:
        config = _valid_config()
        config.tiers["simple"].models = ["openai/does-not-exist"]
        with pytest.raises(RoutingConfigError, match="unknown model keys"):
            validate_routing_config_strict(config)

    def test_rejects_unknown_judge_model(self) -> None:
        config = _valid_config()
        config.verification.judge_model = "openai/does-not-exist"
        with pytest.raises(RoutingConfigError, match="judge_model"):
            validate_routing_config_strict(config)

    def test_rejects_unknown_cost_baseline_model(self) -> None:
        config = _valid_config()
        config.cost_baseline.model = "openai/does-not-exist"
        with pytest.raises(RoutingConfigError, match="cost_baseline"):
            validate_routing_config_strict(config)


def _fake_session_with_scalar_result(row: object | None) -> MagicMock:
    session = MagicMock()
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = row
    session.execute = AsyncMock(return_value=execute_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


class TestRefreshRoutingConfigFromDb:
    async def test_bootstraps_from_yaml_when_no_promoted_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = _fake_session_with_scalar_result(None)

        @asynccontextmanager
        async def _fake_managed_session():
            yield session

        monkeypatch.setattr("llm_autopilot_core.routing.managed_session", _fake_managed_session)

        config = await refresh_routing_config_from_db()

        expected = load_routing_config(_REAL_ROUTING_YAML)
        assert config.version == expected.version
        session.add.assert_called_once()
        added_row = session.add.call_args.args[0]
        assert added_row.promoted is True
        assert "bootstrap" in (added_row.notes or "")
        # get_routing_config() must reflect the bootstrapped config immediately
        assert get_routing_config() is config

    async def test_loads_promoted_row_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stored_config = _valid_config()
        fake_row = MagicMock()
        fake_row.config_json = stored_config.model_dump(mode="json")
        session = _fake_session_with_scalar_result(fake_row)

        @asynccontextmanager
        async def _fake_managed_session():
            yield session

        monkeypatch.setattr("llm_autopilot_core.routing.managed_session", _fake_managed_session)

        config = await refresh_routing_config_from_db()

        assert config.cost_baseline.model == stored_config.cost_baseline.model
        assert get_routing_config() is config
        session.add.assert_not_called()


class TestPersistRoutingConfig:
    async def test_rejects_invalid_config_without_touching_db(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session_factory_called = False

        @asynccontextmanager
        async def _fake_managed_session():
            nonlocal session_factory_called
            session_factory_called = True
            yield MagicMock()

        monkeypatch.setattr("llm_autopilot_core.routing.managed_session", _fake_managed_session)

        config = _valid_config()
        config.tiers["simple"].models = ["openai/does-not-exist"]

        with pytest.raises(RoutingConfigError):
            await persist_routing_config(config)

        assert session_factory_called is False

    async def test_unpromotes_previous_and_updates_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        session = MagicMock()
        session.execute = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        def _set_version_number(obj: object) -> None:
            # Mimics what a real flush() against Postgres would populate —
            # the autoincrement PK and the server-side created_at default —
            # since session.flush() itself is a no-op mock here.
            obj.version_number = 7  # type: ignore[attr-defined]
            obj.created_at = datetime.now(UTC)  # type: ignore[attr-defined]

        session.add.side_effect = _set_version_number

        @asynccontextmanager
        async def _fake_managed_session():
            yield session

        monkeypatch.setattr("llm_autopilot_core.routing.managed_session", _fake_managed_session)

        config = _valid_config()
        summary = await persist_routing_config(config, notes="test update", updated_by="partha")

        assert summary.version_number == 7
        assert summary.updated_by == "partha"
        # First execute() call unpromotes the previous version.
        assert session.execute.await_count == 1
        assert get_routing_config().cost_baseline.model == config.cost_baseline.model
