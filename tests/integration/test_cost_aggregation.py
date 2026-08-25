"""
Integration tests for Phase 5 cost aggregation — require live Postgres.

Run with:
    make test-integration

Or with docker services up:
    make up-core && make test-integration

Covers the part test_retraining_task.py's unit tests deliberately don't:
real SQL aggregation across requests/responses/verifications, the
upsert-by-date idempotency of cost_aggregates, and GET /v1/stats actually
reading back what the Celery task wrote.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from llm_autopilot_api.main import app
from llm_autopilot_core.database import managed_session
from llm_autopilot_core.models import CostAggregate, Request, Response, Verification
from llm_autopilot_core.schemas import ComplexityTier, Provider, VerificationStatus
from llm_autopilot_worker.tasks.retraining import _aggregate_daily_costs_async
from sqlalchemy import delete


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


async def _seed_day(target_date: date) -> None:
    """Write two requests/responses (one cache hit, one miss + escalated
    verification) directly into the DB for target_date, bypassing the
    full completion pipeline — this test only cares about aggregation."""
    at = datetime.combine(target_date, datetime.min.time(), tzinfo=UTC) + timedelta(hours=10)

    async with managed_session() as session:
        await session.execute(delete(CostAggregate).where(CostAggregate.date == target_date))
        await session.execute(
            delete(Request).where(
                Request.prompt_hash.in_(["hash-hit", "hash-miss"]),
                Request.created_at
                >= datetime.combine(target_date, datetime.min.time(), tzinfo=UTC),
                Request.created_at
                < datetime.combine(
                    target_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC
                ),
            )
        )

        hit_request_id = uuid.uuid4()
        session.add(
            Request(
                id=hit_request_id,
                prompt_hash="hash-hit",
                message_count=1,
                max_tokens=100,
                temperature=0.7,
                cache_hit=True,
                created_at=at,
            )
        )
        session.add(
            Response(
                request_id=hit_request_id,
                content="cached answer",
                model_id="llama-3.1-8b-instant",
                provider=Provider.GROQ,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.0,
                latency_ms=5.0,
                complexity_tier=ComplexityTier.SIMPLE,
                classifier_confidence=0.9,
                created_at=at,
            )
        )

        miss_request_id = uuid.uuid4()
        session.add(
            Request(
                id=miss_request_id,
                prompt_hash="hash-miss",
                message_count=1,
                max_tokens=100,
                temperature=0.7,
                cache_hit=False,
                created_at=at,
            )
        )
        session.add(
            Response(
                request_id=miss_request_id,
                content="a mediocre poem",
                model_id="llama-3.1-8b-instant",
                provider=Provider.GROQ,
                input_tokens=20,
                output_tokens=40,
                cost_usd=0.01,
                latency_ms=200.0,
                complexity_tier=ComplexityTier.SIMPLE,
                classifier_confidence=0.4,
                created_at=at,
            )
        )
        session.add(
            Verification(
                request_id=miss_request_id,
                original_model_id="llama-3.1-8b-instant",
                judge_model_id="claude-haiku-4-5",
                quality_score=0.2,
                status=VerificationStatus.ESCALATED,
                escalated_model_id="claude-sonnet-4-6",
                escalated_content="a much better poem",
                corrected_tier=ComplexityTier.COMPLEX,
                created_at=at,
            )
        )


class TestAggregateDailyCosts:
    async def test_writes_expected_rollup_for_seeded_day(self) -> None:
        target_date = date(2026, 1, 15)
        await _seed_day(target_date)

        result = await _aggregate_daily_costs_async(target_date)

        assert result["total_requests"] == 2
        assert result["cache_hit_rate"] == pytest.approx(50.0)
        assert result["escalation_rate"] == pytest.approx(50.0)
        assert result["avg_quality_score"] == pytest.approx(0.2)
        assert result["requests_by_tier"] == {"simple": 2}
        assert result["requests_by_provider"] == {"groq": 2}

        async with managed_session() as session:
            row = await session.get(CostAggregate, target_date)
            assert row is not None
            assert row.total_requests == 2

    async def test_rerunning_the_same_day_is_idempotent(self) -> None:
        target_date = date(2026, 1, 16)
        await _seed_day(target_date)

        await _aggregate_daily_costs_async(target_date)
        second = await _aggregate_daily_costs_async(target_date)

        assert second["total_requests"] == 2  # not double-counted on rerun

        async with managed_session() as session:
            row = await session.get(CostAggregate, target_date)
            assert row is not None
            assert row.total_requests == 2


class TestStatsEndpointReadsAggregatedData:
    async def test_get_stats_returns_seeded_aggregate(self, client: AsyncClient) -> None:
        target_date = date(2026, 1, 17)
        await _seed_day(target_date)
        await _aggregate_daily_costs_async(target_date)

        resp = await client.get(
            "/v1/stats",
            params={"start_date": target_date.isoformat(), "end_date": target_date.isoformat()},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 2
        assert data["requests_by_tier"] == {"simple": 2}

    async def test_get_stats_404s_for_a_range_with_no_data(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/v1/stats", params={"start_date": "1999-01-01", "end_date": "1999-01-02"}
        )
        assert resp.status_code == 404
