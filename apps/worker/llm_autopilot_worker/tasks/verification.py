"""
Async quality verification task.

Triggered by the API after every request that passes the sampling filter.
Sampling decision (made in the API layer before enqueueing):

  confidence < threshold → sample at verification_sample_rate_low_confidence  (default 100%)
  confidence ≥ threshold → sample at verification_sample_rate_high_confidence  (default 5%)
  random baseline        → sample at verification_random_baseline_rate         (default 2%)

This task:
  1. Buckets the prompt into a TaskCategory (extraction / classification /
     summarization / creative / reasoning) and scores the original
     response against a judge model using the matching strategy from
     verification/scoring.py — including true pairwise comparison for
     creative and reasoning.
  2. If the score clears the category's threshold: PASSED.
  3. If not, and the original response wasn't already at the top tier:
     escalates to COMPLEX and reruns (or reuses the pairwise strategy's
     comparison response directly, when available) — ESCALATED on
     success, FAILED if the rerun times out or errors.
  4. Persists a Verification row, always including the prompt's feature
     vector (never the raw prompt) and — on successful escalation — the
     corrected_tier, which is what retrain_classifier() trains against.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import UUID

import structlog
from celery import Task
from llm_autopilot_core.classifier.features import feature_vector
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.database import managed_session
from llm_autopilot_core.metrics import (
    celery_tasks_total,
    escalations_total,
    verifications_total,
)
from llm_autopilot_core.metrics import (
    quality_score as quality_score_metric,
)
from llm_autopilot_core.models import Verification
from llm_autopilot_core.providers.base import ProviderError
from llm_autopilot_core.providers.dispatcher import send_request
from llm_autopilot_core.registry import get_model
from llm_autopilot_core.routing import (
    RoutingConfigError,
    get_routing_config,
    select_model_for_tier,
)
from llm_autopilot_core.schemas import (
    ComplexityTier,
    EscalationReason,
    Provider,
    VerificationResult,
    VerificationStatus,
)
from llm_autopilot_core.verification import (
    classify_task_category,
    get_threshold_for_category,
    is_self_judge,
    score_response,
)

from llm_autopilot_worker.main import celery_app

logger = structlog.get_logger(__name__)


async def _persist_verification(result: VerificationResult) -> None:
    async with managed_session() as session:
        session.add(
            Verification(
                request_id=result.request_id,
                original_model_id=result.original_model_id,
                judge_model_id=result.judge_model_id,
                quality_score=result.quality_score,
                status=result.status,
                quality_gap=result.quality_gap,
                escalation_reason=result.escalation_reason,
                escalated_model_id=result.escalated_model_id,
                escalated_content=result.escalated_content,
                cost_delta_usd=result.cost_delta_usd,
                feature_vector=result.feature_vector,
                corrected_tier=result.corrected_tier,
            )
        )


async def _verify_response_async(
    *,
    request_id: str,
    prompt: str,
    original_response: str,
    model_id: str,
    provider: str,
    complexity_tier: str,
    classifier_confidence: float,
) -> VerificationResult:
    settings = get_settings()
    routing_config = get_routing_config()

    judge_config = get_model(routing_config.verification.judge_model)
    if judge_config is None:
        raise RoutingConfigError(
            f"configured judge model '{routing_config.verification.judge_model}' "
            "not found in MODEL_REGISTRY"
        )

    original_provider = Provider(provider)
    tier = ComplexityTier(complexity_tier)
    request_uuid = UUID(request_id)

    if is_self_judge(original_provider, model_id, judge_config):
        result = VerificationResult(
            request_id=request_uuid,
            original_model_id=model_id,
            judge_model_id=judge_config.model_id,
            quality_score=1.0,
            status=VerificationStatus.SKIPPED,
        )
        await _persist_verification(result)
        verifications_total.labels(status=result.status.value).inc()
        return result

    task_category = classify_task_category(prompt)
    scoring_result = await score_response(
        task_category=task_category,
        prompt=prompt,
        original_response=original_response,
        judge_config=judge_config,
        routing_config=routing_config,
    )
    quality_score_metric.labels(complexity_tier=tier.value, model_id=model_id).observe(
        scoring_result.quality_score
    )

    threshold = get_threshold_for_category(task_category, settings.escalation_quality_threshold)
    prompt_features = feature_vector(prompt)

    if scoring_result.quality_score >= threshold:
        result = VerificationResult(
            request_id=request_uuid,
            original_model_id=model_id,
            judge_model_id=judge_config.model_id,
            quality_score=scoring_result.quality_score,
            status=VerificationStatus.PASSED,
            feature_vector=prompt_features,
        )
        await _persist_verification(result)
        verifications_total.labels(status=result.status.value).inc()
        return result

    quality_gap = threshold - scoring_result.quality_score

    if tier == ComplexityTier.COMPLEX:
        # Already at the top tier — nowhere to escalate to. This is a
        # model-quality problem, not a routing problem, so it isn't fed
        # back to the classifier (no corrected_tier).
        result = VerificationResult(
            request_id=request_uuid,
            original_model_id=model_id,
            judge_model_id=judge_config.model_id,
            quality_score=scoring_result.quality_score,
            status=VerificationStatus.FAILED,
            quality_gap=quality_gap,
            feature_vector=prompt_features,
        )
        await _persist_verification(result)
        verifications_total.labels(status=result.status.value).inc()
        return result

    escalated_content: str | None
    escalated_model_id: str | None
    escalation_cost: float

    if scoring_result.escalation_candidate_content is not None:
        # Pairwise strategy already generated a COMPLEX-tier comparison
        # response — reuse it rather than pay for a second rerun.
        escalated_content = scoring_result.escalation_candidate_content
        escalated_model_id = scoring_result.escalation_candidate_model_id
        escalation_cost = scoring_result.escalation_candidate_cost_usd
    else:
        try:
            escalation_decision = select_model_for_tier(
                ComplexityTier.COMPLEX, classifier_confidence, routing_config
            )
            escalation_model_config = get_model(
                f"{escalation_decision.selected_provider.value}/"
                f"{escalation_decision.selected_model_id}"
            )
            if escalation_model_config is None:
                raise RoutingConfigError("escalation target model not found in registry")
            escalation_response = await asyncio.wait_for(
                send_request(prompt, escalation_model_config),
                timeout=settings.escalation_max_latency_ms / 1000,
            )
            escalated_content = escalation_response.content
            escalated_model_id = escalation_model_config.model_id
            escalation_cost = escalation_response.cost_usd
        except (TimeoutError, ProviderError, RoutingConfigError) as exc:
            logger.warning("escalation_rerun_failed", request_id=request_id, error=str(exc))
            result = VerificationResult(
                request_id=request_uuid,
                original_model_id=model_id,
                judge_model_id=judge_config.model_id,
                quality_score=scoring_result.quality_score,
                status=VerificationStatus.FAILED,
                quality_gap=quality_gap,
                escalation_reason=EscalationReason.QUALITY_GAP,
                feature_vector=prompt_features,
            )
            await _persist_verification(result)
            verifications_total.labels(status=result.status.value).inc()
            return result

    result = VerificationResult(
        request_id=request_uuid,
        original_model_id=model_id,
        judge_model_id=judge_config.model_id,
        quality_score=scoring_result.quality_score,
        status=VerificationStatus.ESCALATED,
        quality_gap=quality_gap,
        escalation_reason=EscalationReason.QUALITY_GAP,
        escalated_model_id=escalated_model_id,
        escalated_content=escalated_content,
        cost_delta_usd=escalation_cost,
        feature_vector=prompt_features,
        corrected_tier=ComplexityTier.COMPLEX,
    )
    await _persist_verification(result)
    verifications_total.labels(status=result.status.value).inc()
    escalations_total.labels(
        original_model=model_id,
        escalated_model=escalated_model_id or "unknown",
        reason=EscalationReason.QUALITY_GAP.value,
    ).inc()
    return result


@celery_app.task(
    name="llm_autopilot_worker.tasks.verification.verify_response",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queue="verification",
)
def verify_response(
    self: Task,
    request_id: str,
    prompt: str,
    original_response: str,
    model_id: str,
    provider: str,
    complexity_tier: str,
    classifier_confidence: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> dict[str, Any]:
    """
    Async quality verification for a single completed request.

    `input_tokens` / `output_tokens` / `cost_usd` describe the *original*
    response and aren't used by the scoring logic itself — they're kept
    on the signature as an audit-trail hook for future cost-delta
    reporting (e.g. comparing original vs. escalation cost) without
    another schema change.
    """
    log = logger.bind(
        request_id=request_id,
        model_id=model_id,
        tier=complexity_tier,
        confidence=classifier_confidence,
    )
    log.info("verification_started")

    try:
        result = asyncio.run(
            _verify_response_async(
                request_id=request_id,
                prompt=prompt,
                original_response=original_response,
                model_id=model_id,
                provider=provider,
                complexity_tier=complexity_tier,
                classifier_confidence=classifier_confidence,
            )
        )
        celery_tasks_total.labels(task_name="verify_response", status="success").inc()
        log.info(
            "verification_complete",
            status=result.status.value,
            quality_score=round(result.quality_score, 3),
        )
        return cast(dict[str, Any], result.model_dump(mode="json"))

    except Exception as exc:
        celery_tasks_total.labels(task_name="verify_response", status="failure").inc()
        log.error("verification_error", error=str(exc), exc_info=True)
        raise self.retry(exc=exc) from exc
