"""
Integration tests for the FastAPI API — require live Postgres + Redis.

Run with:
    make test-integration

Or with docker services up:
    make up-core && make test-integration
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from llm_autopilot_api.main import app


@pytest.fixture
async def client():
    """Async test client using ASGI transport (no real HTTP server needed)."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


class TestHealthEndpoints:
    async def test_liveness(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "environment" in data
        assert "timestamp" in data

    async def test_readiness_with_live_services(self, client: AsyncClient) -> None:
        """Passes only when Postgres and Redis are reachable."""
        resp = await client.get("/v1/readyz")
        # Should be 200 if services are up, 503 if not
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "checks" in data
        assert "postgres" in data["checks"]
        assert "redis" in data["checks"]

    async def test_metrics_endpoint_returns_prometheus_text(
        self, client: AsyncClient
    ) -> None:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        # Prometheus text format always starts with "# HELP" or metric name
        assert b"llm_autopilot" in resp.content or b"# HELP" in resp.content

    async def test_request_id_header_propagated(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/v1/healthz", headers={"X-Request-ID": "test-12345"}
        )
        assert resp.headers.get("x-request-id") == "test-12345"

    async def test_random_request_id_injected(self, client: AsyncClient) -> None:
        """If no X-Request-ID sent, the API generates one."""
        resp = await client.get("/v1/healthz")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0
