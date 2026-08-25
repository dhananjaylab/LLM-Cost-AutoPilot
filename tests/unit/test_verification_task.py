from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from llm_autopilot_core.routing import (
    CostBaselineConfig,
    RoutingConfig,
    TierRoute,
    VerificationRoutingConfig,
)
from llm_autopilot_core.schemas import (
    ComplexityTier,
    Provider,
    ProviderResponse,
    VerificationStatus,
)
from llm_autopilot_core.verification.scoring import ScoringResult
from llm_autopilot_worker.tasks.verification import _verify_response_async


def _routing_config() -> RoutingConfig:
    return RoutingConfig(
        version="1",
        tiers={
            ComplexityTier.SIMPLE: TierRoute(
                description="", models=["meta-llama/llama-prompt-guard-2-22m"], max_latency_ms=3000
            ),
            ComplexityTier.MODERATE: TierRoute(
                description="", models=["openai/gpt-4o-mini"], max_latency_ms=5000
            ),
            ComplexityTier.COMPLEX: TierRoute(
                description="", models=["anthropic/claude-sonnet-4-6"], max_latency_ms=15000
            ),
        },
        verification=VerificationRoutingConfig(
            judge_model="anthropic/claude-haiku-4-5", judge_max_tokens=512
        ),
        cost_baseline=CostBaselineConfig(model="openai/gpt-4o"),
    )


async def _refresh_routing_config() -> RoutingConfig:
    """Async stand-in for refresh_routing_config_from_db() — Phase 5 moved
    the verification task off the sync get_routing_config() accessor onto
    this DB-backed refresh, so the monkeypatch target changed with it."""
    return _routing_config()


@asynccontextmanager
async def _fake_managed_session():
    yield MagicMock()


@pytest.fixture(autouse=True)
def _patch_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_autopilot_worker.tasks.verification.refresh_routing_config_from_db",
        _refresh_routing_config,
    )
    monkeypatch.setattr(
        "llm_autopilot_worker.tasks.verification.managed_session", _fake_managed_session
    )


class TestSelfJudgeSkip:
    async def test_skips_when_original_model_is_the_judge(self) -> None:
        result = await _verify_response_async(
            request_id=str(uuid.uuid4()),
            prompt="hi",
            original_response="hello",
            model_id="claude-haiku-4-5",
            provider="anthropic",
            complexity_tier="moderate",
            classifier_confidence=0.5,
        )
        assert result.status == VerificationStatus.SKIPPED
        assert result.quality_score == 1.0


class TestPassed:
    async def test_score_above_threshold_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.score_response",
            AsyncMock(return_value=ScoringResult(quality_score=0.95, judge_output="ok")),
        )
        result = await _verify_response_async(
            request_id=str(uuid.uuid4()),
            prompt="Summarize this",
            original_response="a summary",
            model_id="gpt-4o-mini",
            provider="openai",
            complexity_tier="moderate",
            classifier_confidence=0.9,
        )
        assert result.status == VerificationStatus.PASSED
        assert result.feature_vector is not None
        assert result.corrected_tier is None


class TestEscalationViaPairwiseReuse:
    async def test_reuses_comparison_content_without_extra_rerun(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.score_response",
            AsyncMock(
                return_value=ScoringResult(
                    quality_score=0.0,
                    judge_output="pass1=B pass2=A",
                    escalation_candidate_content="better poem",
                    escalation_candidate_model_id="claude-sonnet-4-6",
                    escalation_candidate_provider=Provider.ANTHROPIC,
                    escalation_candidate_cost_usd=0.002,
                )
            ),
        )
        send_mock = AsyncMock()
        monkeypatch.setattr("llm_autopilot_worker.tasks.verification.send_request", send_mock)

        result = await _verify_response_async(
            request_id=str(uuid.uuid4()),
            prompt="Write a poem",
            original_response="a mediocre poem",
            model_id="llama-3.1-8b-instant",
            provider="groq",
            complexity_tier="simple",
            classifier_confidence=0.5,
        )
        assert result.status == VerificationStatus.ESCALATED
        assert result.escalated_content == "better poem"
        assert result.corrected_tier == ComplexityTier.COMPLEX
        assert result.cost_delta_usd == 0.002
        send_mock.assert_not_called()  # reused pairwise content — no redundant rerun


class TestEscalationViaFreshRerun:
    async def test_reruns_against_complex_tier_when_no_pairwise_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.score_response",
            AsyncMock(return_value=ScoringResult(quality_score=0.3, judge_output="wrong label")),
        )
        send_mock = AsyncMock(
            return_value=ProviderResponse(
                content="corrected label",
                input_tokens=5,
                output_tokens=2,
                latency_ms=100.0,
                cost_usd=0.003,
                model_id="claude-sonnet-4-6",
                provider=Provider.ANTHROPIC,
            )
        )
        monkeypatch.setattr("llm_autopilot_worker.tasks.verification.send_request", send_mock)

        result = await _verify_response_async(
            request_id=str(uuid.uuid4()),
            prompt="Classify this",
            original_response="wrong label",
            model_id="gemini-3.5-flash",
            provider="google",
            complexity_tier="moderate",
            classifier_confidence=0.6,
        )
        assert result.status == VerificationStatus.ESCALATED
        assert result.escalated_content == "corrected label"
        send_mock.assert_awaited_once()


class TestNoEscalationAtTopTier:
    async def test_complex_tier_failure_does_not_escalate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.score_response",
            AsyncMock(return_value=ScoringResult(quality_score=0.0, judge_output="loss")),
        )
        result = await _verify_response_async(
            request_id=str(uuid.uuid4()),
            prompt="Design a system",
            original_response="a bad design",
            model_id="claude-sonnet-4-6",
            provider="anthropic",
            complexity_tier="complex",
            classifier_confidence=0.7,
        )
        assert result.status == VerificationStatus.FAILED
        assert result.corrected_tier is None


class TestEscalationTimeoutOrError:
    async def test_rerun_failure_marks_failed_not_escalated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.score_response",
            AsyncMock(return_value=ScoringResult(quality_score=0.2, judge_output="bad")),
        )

        async def _failing_send(*args: object, **kwargs: object) -> ProviderResponse:
            raise TimeoutError

        monkeypatch.setattr("llm_autopilot_worker.tasks.verification.send_request", _failing_send)

        result = await _verify_response_async(
            request_id=str(uuid.uuid4()),
            prompt="Summarize",
            original_response="a bad summary",
            model_id="gpt-4o-mini",
            provider="openai",
            complexity_tier="moderate",
            classifier_confidence=0.5,
        )
        assert result.status == VerificationStatus.FAILED
        assert result.escalation_reason is not None
        assert result.escalated_content is None


class TestCacheWriteback:
    """Phase 5 — successful escalation overwrites the semantic cache entry
    the original (wrong) response populated, in place."""

    async def test_writes_back_via_aupdate_when_cache_key_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.score_response",
            AsyncMock(
                return_value=ScoringResult(
                    quality_score=0.0,
                    judge_output="pass1=B pass2=A",
                    escalation_candidate_content="better poem",
                    escalation_candidate_model_id="claude-sonnet-4-6",
                    escalation_candidate_provider=Provider.ANTHROPIC,
                    escalation_candidate_cost_usd=0.002,
                )
            ),
        )
        fake_cache = MagicMock()
        fake_cache.aupdate = AsyncMock()
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.get_semantic_cache",
            lambda: fake_cache,
        )

        result = await _verify_response_async(
            request_id=str(uuid.uuid4()),
            prompt="Write a poem",
            original_response="a mediocre poem",
            model_id="llama-3.1-8b-instant",
            provider="groq",
            complexity_tier="simple",
            classifier_confidence=0.5,
            cache_key="cache:entry:abc123",
        )

        assert result.status == VerificationStatus.ESCALATED
        fake_cache.aupdate.assert_awaited_once()
        call = fake_cache.aupdate.await_args
        assert call.args[0] == "cache:entry:abc123"
        assert call.kwargs["response"] == "better poem"
        assert call.kwargs["metadata"]["model_id"] == "claude-sonnet-4-6"
        assert call.kwargs["metadata"]["provider"] == "anthropic"
        assert call.kwargs["metadata"]["complexity_tier"] == "complex"
        assert call.kwargs["metadata"]["corrected_by_escalation"] is True

    async def test_no_cache_interaction_when_cache_key_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.score_response",
            AsyncMock(
                return_value=ScoringResult(
                    quality_score=0.0,
                    judge_output="pass1=B pass2=A",
                    escalation_candidate_content="better poem",
                    escalation_candidate_model_id="claude-sonnet-4-6",
                    escalation_candidate_provider=Provider.ANTHROPIC,
                    escalation_candidate_cost_usd=0.002,
                )
            ),
        )
        get_cache_mock = MagicMock()
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.get_semantic_cache", get_cache_mock
        )

        result = await _verify_response_async(
            request_id=str(uuid.uuid4()),
            prompt="Write a poem",
            original_response="a mediocre poem",
            model_id="llama-3.1-8b-instant",
            provider="groq",
            complexity_tier="simple",
            classifier_confidence=0.5,
            cache_key=None,
        )

        assert result.status == VerificationStatus.ESCALATED
        get_cache_mock.assert_not_called()

    async def test_cache_writeback_failure_does_not_fail_the_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.score_response",
            AsyncMock(
                return_value=ScoringResult(
                    quality_score=0.0,
                    judge_output="pass1=B pass2=A",
                    escalation_candidate_content="better poem",
                    escalation_candidate_model_id="claude-sonnet-4-6",
                    escalation_candidate_provider=Provider.ANTHROPIC,
                    escalation_candidate_cost_usd=0.002,
                )
            ),
        )
        fake_cache = MagicMock()
        fake_cache.aupdate = AsyncMock(side_effect=RuntimeError("redis unavailable"))
        monkeypatch.setattr(
            "llm_autopilot_worker.tasks.verification.get_semantic_cache",
            lambda: fake_cache,
        )

        result = await _verify_response_async(
            request_id=str(uuid.uuid4()),
            prompt="Write a poem",
            original_response="a mediocre poem",
            model_id="llama-3.1-8b-instant",
            provider="groq",
            complexity_tier="simple",
            classifier_confidence=0.5,
            cache_key="cache:entry:abc123",
        )

        # Escalation itself still succeeded and was persisted — a cache
        # failure is best-effort and must not surface as a task failure.
        assert result.status == VerificationStatus.ESCALATED
        assert result.escalated_content == "better poem"
