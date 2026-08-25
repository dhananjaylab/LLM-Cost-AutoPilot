"""
Unit tests for the admin router (GET/PUT /v1/admin/routing-config).

GET /v1/admin/routing-config needs no DB (reads the process-local cache,
which falls back to configs/routing.yaml). PUT delegates persistence to
llm_autopilot_core.routing.persist_routing_config(), which this file
monkeypatches at the router's import site — the same boundary-mocking
pattern used throughout this suite — so these stay fast, DB-free unit
tests. tests/integration/test_admin_routing_config.py covers the real
DB round trip plus the actual RoutingConfigError → 400 validation path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.routing import (
    RoutingConfigError,
    RoutingConfigVersionSummary,
    reset_routing_config_cache,
)


@pytest.fixture
async def client():
    # Imported lazily so ADMIN_API_KEY env changes made per-test (via
    # monkeypatch, before the client is constructed) are already in
    # place before the app's dependency graph is exercised.
    from llm_autopilot_api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_routing_config_cache()
    yield
    reset_routing_config_cache()


_VALID_PAYLOAD = {
    "version": "1",
    "tiers": {
        "simple": {
            "description": "test",
            "models": ["meta-llama/llama-prompt-guard-2-22m"],
            "max_latency_ms": 3000,
        }
    },
    "verification": {"judge_model": "anthropic/claude-haiku-4-5", "judge_max_tokens": 512},
    "cost_baseline": {"model": "openai/gpt-4o"},
    "notes": "test update",
    "updated_by": "pytest",
}


class TestGetRoutingConfig:
    async def test_returns_current_config_without_auth(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/admin/routing-config")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["tiers"].keys()) == {"simple", "moderate", "complex"}


class TestGetRoutingConfigVersions:
    async def test_returns_mocked_history(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        summaries = [
            RoutingConfigVersionSummary(
                version_number=2,
                promoted=True,
                promoted_at=datetime.now(UTC),
                notes="latest",
                updated_by="partha",
                created_at=datetime.now(UTC),
            )
        ]
        monkeypatch.setattr(
            "llm_autopilot_api.routers.admin.list_routing_config_versions",
            AsyncMock(return_value=summaries),
        )
        resp = await client.get("/v1/admin/routing-config/versions")
        assert resp.status_code == 200
        assert resp.json()[0]["version_number"] == 2


class TestPutRoutingConfigAuth:
    async def test_returns_503_when_admin_key_not_configured(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ADMIN_API_KEY", "")
        get_settings.cache_clear()
        resp = await client.put("/v1/admin/routing-config", json=_VALID_PAYLOAD)
        assert resp.status_code == 503

    async def test_returns_401_without_header(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ADMIN_API_KEY", "test-secret-key")
        get_settings.cache_clear()
        resp = await client.put("/v1/admin/routing-config", json=_VALID_PAYLOAD)
        assert resp.status_code == 401

    async def test_returns_401_with_wrong_key(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ADMIN_API_KEY", "test-secret-key")
        get_settings.cache_clear()
        resp = await client.put(
            "/v1/admin/routing-config",
            json=_VALID_PAYLOAD,
            headers={"X-Admin-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401


class TestPutRoutingConfigSuccess:
    async def test_valid_payload_persists_and_returns_effective_config(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        monkeypatch.setenv("ADMIN_API_KEY", "test-secret-key")
        get_settings.cache_clear()

        persist_mock = AsyncMock(
            return_value=RoutingConfigVersionSummary(
                version_number=3,
                promoted=True,
                promoted_at=datetime.now(UTC),
                notes="test update",
                updated_by="pytest",
                created_at=datetime.now(UTC),
            )
        )
        monkeypatch.setattr("llm_autopilot_api.routers.admin.persist_routing_config", persist_mock)

        resp = await client.put(
            "/v1/admin/routing-config",
            json=_VALID_PAYLOAD,
            headers={"X-Admin-API-Key": "test-secret-key"},
        )

        assert resp.status_code == 200
        persist_mock.assert_awaited_once()
        assert persist_mock.await_args.kwargs["notes"] == "test update"
        assert persist_mock.await_args.kwargs["updated_by"] == "pytest"
        assert resp.json()["cost_baseline"]["model"] == "openai/gpt-4o"

    async def test_rejected_payload_returns_400(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ADMIN_API_KEY", "test-secret-key")
        get_settings.cache_clear()

        monkeypatch.setattr(
            "llm_autopilot_api.routers.admin.persist_routing_config",
            AsyncMock(side_effect=RoutingConfigError("unknown model keys referenced")),
        )

        resp = await client.put(
            "/v1/admin/routing-config",
            json=_VALID_PAYLOAD,
            headers={"X-Admin-API-Key": "test-secret-key"},
        )

        assert resp.status_code == 400
        assert "unknown model keys" in resp.json()["detail"]
