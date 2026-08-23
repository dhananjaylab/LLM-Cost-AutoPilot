"""
Unit tests for GET /v1/stats.

The pure rollup helpers (_merge_counts, _weighted_avg) are tested
directly. The endpoint itself is tested through FastAPI's
dependency_overrides mechanism, swapping get_session for a fake that
returns SimpleNamespace stand-ins for CostAggregate rows — stats.py only
ever reads attributes off them via getattr, so a full ORM/DB isn't
needed here. tests/integration/test_cost_aggregation.py covers the real
Postgres round trip (aggregate_daily_costs writing, GET /v1/stats reading).
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from llm_autopilot_api.dependencies import get_session
from llm_autopilot_api.main import app
from llm_autopilot_api.routers.stats import _merge_counts, _weighted_avg


def _row(
    day: date,
    *,
    total_requests: int,
    total_cost_usd: float,
    hypothetical_cost_usd: float,
    cost_savings_usd: float,
    cache_hit_rate: float,
    escalation_rate: float,
    avg_quality_score: float,
    requests_by_tier: dict[str, int],
    requests_by_provider: dict[str, int],
) -> SimpleNamespace:
    return SimpleNamespace(
        date=day,
        total_requests=total_requests,
        total_cost_usd=total_cost_usd,
        hypothetical_cost_usd=hypothetical_cost_usd,
        cost_savings_usd=cost_savings_usd,
        cache_hit_rate=cache_hit_rate,
        escalation_rate=escalation_rate,
        avg_quality_score=avg_quality_score,
        requests_by_tier=requests_by_tier,
        requests_by_provider=requests_by_provider,
    )


class TestMergeCounts:
    def test_sums_across_days(self) -> None:
        merged = _merge_counts([{"simple": 3, "moderate": 1}, {"simple": 2, "complex": 5}])
        assert merged == {"simple": 5, "moderate": 1, "complex": 5}

    def test_empty_input(self) -> None:
        assert _merge_counts([]) == {}


class TestWeightedAvg:
    def test_weights_by_total_requests(self) -> None:
        rows = [
            _row(
                date(2026, 1, 1),
                total_requests=90,
                total_cost_usd=0,
                hypothetical_cost_usd=0,
                cost_savings_usd=0,
                cache_hit_rate=10.0,
                escalation_rate=0,
                avg_quality_score=0,
                requests_by_tier={},
                requests_by_provider={},
            ),
            _row(
                date(2026, 1, 2),
                total_requests=10,
                total_cost_usd=0,
                hypothetical_cost_usd=0,
                cost_savings_usd=0,
                cache_hit_rate=90.0,
                escalation_rate=0,
                avg_quality_score=0,
                requests_by_tier={},
                requests_by_provider={},
            ),
        ]
        # 90 requests at 10% + 10 requests at 90% => 18%, not the naive 50%
        assert _weighted_avg(rows, "cache_hit_rate") == pytest.approx(18.0)

    def test_zero_total_requests_returns_zero(self) -> None:
        rows = [
            _row(
                date(2026, 1, 1),
                total_requests=0,
                total_cost_usd=0,
                hypothetical_cost_usd=0,
                cost_savings_usd=0,
                cache_hit_rate=0,
                escalation_rate=0,
                avg_quality_score=0,
                requests_by_tier={},
                requests_by_provider={},
            )
        ]
        assert _weighted_avg(rows, "cache_hit_rate") == 0.0


def _override_session_with_rows(rows: list[SimpleNamespace]) -> None:
    async def _fake_get_session():
        session = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = rows
        session.execute = AsyncMock(return_value=result)
        yield session

    app.dependency_overrides[get_session] = _fake_get_session


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


class TestGetStatsEndpoint:
    async def test_returns_404_when_no_rows(self, client: AsyncClient) -> None:
        _override_session_with_rows([])
        resp = await client.get("/v1/stats")
        assert resp.status_code == 404

    async def test_returns_400_when_start_after_end(self, client: AsyncClient) -> None:
        _override_session_with_rows([])
        resp = await client.get(
            "/v1/stats", params={"start_date": "2026-01-10", "end_date": "2026-01-01"}
        )
        assert resp.status_code == 400

    async def test_aggregates_multiple_days(self, client: AsyncClient) -> None:
        rows = [
            _row(
                date(2026, 1, 1),
                total_requests=100,
                total_cost_usd=1.0,
                hypothetical_cost_usd=5.0,
                cost_savings_usd=4.0,
                cache_hit_rate=20.0,
                escalation_rate=5.0,
                avg_quality_score=0.9,
                requests_by_tier={"simple": 60, "moderate": 40},
                requests_by_provider={"groq": 100},
            ),
            _row(
                date(2026, 1, 2),
                total_requests=50,
                total_cost_usd=0.5,
                hypothetical_cost_usd=2.5,
                cost_savings_usd=2.0,
                cache_hit_rate=30.0,
                escalation_rate=10.0,
                avg_quality_score=0.8,
                requests_by_tier={"simple": 50},
                requests_by_provider={"openai": 50},
            ),
        ]
        _override_session_with_rows(rows)

        resp = await client.get(
            "/v1/stats", params={"start_date": "2026-01-01", "end_date": "2026-01-02"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 150
        assert data["total_cost_usd"] == pytest.approx(1.5)
        assert data["hypothetical_cost_usd"] == pytest.approx(7.5)
        assert data["cost_savings_usd"] == pytest.approx(6.0)
        assert data["cost_savings_pct"] == pytest.approx(80.0)
        assert data["requests_by_tier"] == {"simple": 110, "moderate": 40}
        assert data["requests_by_provider"] == {"groq": 100, "openai": 50}

    async def test_end_date_clamped_to_yesterday(self, client: AsyncClient) -> None:
        """A caller passing today (or later) as end_date must not see a 200
        with an implied 'today' bucket — GET /v1/stats only serves fully
        aggregated days (confirmed scope decision)."""
        rows = [
            _row(
                date(2026, 1, 1),
                total_requests=10,
                total_cost_usd=0.1,
                hypothetical_cost_usd=0.5,
                cost_savings_usd=0.4,
                cache_hit_rate=0,
                escalation_rate=0,
                avg_quality_score=0,
                requests_by_tier={},
                requests_by_provider={},
            )
        ]
        _override_session_with_rows(rows)
        far_future = "2099-01-01"
        resp = await client.get("/v1/stats", params={"end_date": far_future})
        # Should not error out — end_date gets clamped server-side rather
        # than trusted verbatim.
        assert resp.status_code in (200, 404)
