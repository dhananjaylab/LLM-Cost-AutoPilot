"""
Unit tests for GET /v1/models — pure registry read, no DB/Redis needed,
so this runs against the real FastAPI app via ASGI transport the same
way tests/integration/test_health.py does, but doesn't require live
services since models.py's router never touches the database.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from llm_autopilot_api.main import app
from llm_autopilot_core.registry import MODEL_REGISTRY


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


class TestListModels:
    async def test_returns_every_registry_entry(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        keys = {m["registry_key"] for m in data["models"]}
        assert keys == set(MODEL_REGISTRY.keys())

    async def test_entry_shape_includes_pricing_and_live_availability(
        self, client: AsyncClient
    ) -> None:
        resp = await client.get("/v1/models")
        entry = next(m for m in resp.json()["models"] if m["registry_key"] == "openai/gpt-4o")
        assert entry["provider"] == "openai"
        assert entry["model_id"] == "gpt-4o"
        assert entry["cost_per_input_token"] == pytest.approx(0.000_005)
        assert entry["cost_per_1k_tokens"] > 0
        assert "circuit_breaker_available" in entry
        assert isinstance(entry["circuit_breaker_available"], bool)

    async def test_no_auth_required(self, client: AsyncClient) -> None:
        # Deliberately no X-Admin-API-Key header — GET /v1/models is
        # public read-only, unlike PUT /v1/admin/routing-config.
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
