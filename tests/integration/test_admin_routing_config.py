"""
Integration tests for Phase 5 routing-config persistence — require live
Postgres. Run with `make test-integration` (or `make up-core` first).

Unlike tests/unit/test_admin_router.py (which mocks persist_routing_config
at the router boundary) and tests/unit/test_routing_config_persistence.py
(which mocks managed_session), this exercises the real
routing_config_versions table end to end: PUT via the actual FastAPI app,
reading the row back from Postgres, and confirming only one version stays
promoted at a time.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.database import managed_session
from llm_autopilot_core.models import RoutingConfigVersion
from llm_autopilot_core.routing import reset_routing_config_cache
from sqlalchemy import select

_TEST_ADMIN_KEY = "integration-test-admin-key"

_VALID_PAYLOAD = {
    "version": "integration-test",
    "tiers": {
        "simple": {
            "description": "integration test tier",
            "models": ["meta-llama/llama-prompt-guard-2-22m"],
            "max_latency_ms": 3000,
        },
        "moderate": {
            "description": "integration test tier",
            "models": ["openai/gpt-4o-mini"],
            "max_latency_ms": 5000,
        },
        "complex": {
            "description": "integration test tier",
            "models": ["anthropic/claude-sonnet-4-6"],
            "max_latency_ms": 15000,
        },
    },
    "verification": {"judge_model": "anthropic/claude-haiku-4-5", "judge_max_tokens": 512},
    "cost_baseline": {"model": "openai/gpt-4o"},
    "notes": "integration test run",
    "updated_by": "pytest-integration",
}


@pytest.fixture(autouse=True)
def _admin_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ADMIN_API_KEY", _TEST_ADMIN_KEY)
    get_settings.cache_clear()
    reset_routing_config_cache()
    yield
    get_settings.cache_clear()
    reset_routing_config_cache()


@pytest.fixture
async def client():
    from llm_autopilot_api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


class TestPutRoutingConfigRealDb:
    async def test_put_persists_a_new_promoted_version_and_unpromotes_the_old_one(
        self, client: AsyncClient
    ) -> None:
        resp = await client.put(
            "/v1/admin/routing-config",
            json=_VALID_PAYLOAD,
            headers={"X-Admin-API-Key": _TEST_ADMIN_KEY},
        )
        assert resp.status_code == 200
        assert resp.json()["version"] == "integration-test"

        async with managed_session() as session:
            promoted_rows = (
                (
                    await session.execute(
                        select(RoutingConfigVersion).where(RoutingConfigVersion.promoted.is_(True))
                    )
                )
                .scalars()
                .all()
            )
            assert len(promoted_rows) == 1
            assert promoted_rows[0].notes == "integration test run"
            assert promoted_rows[0].updated_by == "pytest-integration"

    async def test_get_after_put_reflects_the_new_config(self, client: AsyncClient) -> None:
        await client.put(
            "/v1/admin/routing-config",
            json=_VALID_PAYLOAD,
            headers={"X-Admin-API-Key": _TEST_ADMIN_KEY},
        )
        resp = await client.get("/v1/admin/routing-config")
        assert resp.status_code == 200
        assert resp.json()["cost_baseline"]["model"] == "openai/gpt-4o"

    async def test_invalid_model_key_is_rejected_before_writing(self, client: AsyncClient) -> None:
        bad_payload = dict(_VALID_PAYLOAD)
        bad_payload["cost_baseline"] = {"model": "openai/does-not-exist"}

        resp = await client.put(
            "/v1/admin/routing-config",
            json=bad_payload,
            headers={"X-Admin-API-Key": _TEST_ADMIN_KEY},
        )
        assert resp.status_code == 400

    async def test_versions_endpoint_lists_history(self, client: AsyncClient) -> None:
        await client.put(
            "/v1/admin/routing-config",
            json=_VALID_PAYLOAD,
            headers={"X-Admin-API-Key": _TEST_ADMIN_KEY},
        )
        resp = await client.get("/v1/admin/routing-config/versions")
        assert resp.status_code == 200
        versions = resp.json()
        assert len(versions) >= 1
        assert versions[0]["promoted"] is True  # most recent first
