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
  5. (Phase 5) On successful escalation, also overwrites the semantic
     cache entry the original response populated with the corrected
     answer — see _write_back_escalated_response() below — so a cache
     hit on the same/similar prompt doesn't keep serving the answer the
     verifier just found wrong. Best-effort: a cache failure here never
     fails the task, since the escalation itself already succeeded and
     is already durably recorded in the Verification row.

Reads the routing config fresh from Postgres on every run
(refresh_routing_config_from_db()) rather than through the API's
process-local cache — this task runs far less often than a request, so
the extra DB round trip is cheap, and it means a routing-config change
made via PUT /v1/admin/routing-config is visible here immediately rather
than waiting out the API's refresh TTL.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import structlog
from celery import Task
from llm_autopilot_core.cache import get_semantic_cache
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
    refresh_routing_config_from_db,
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


async def _write_back_escalated_response(
    *,
    cache_key: str,
    escalated_content: str,
    escalated_model_id: str | None,
    escalated_provider: Provider | None,
    classifier_confidence: float,
    request_id: str,
) -> None:
    """
    Best-effort: overwrite the semantic-cache entry this request
    originally populated with the escalated (corrected) response, in
    place, so future callers with the same or a similar prompt get the
    fixed answer instead of the one the verifier just found wrong.

    Uses RedisVL's aupdate(), which updates fields on an existing entry
    rather than creating a second, competing one the way a second
    astore() call would — important since acheck() has no defined
    tie-break between two entries that both match closely. Passing a
    fresh `metadata` dict replaces the whole field rather than merging
    into it, so this rebuilds every key completions._response_from_cache_hit()
    reads; input/output token counts are omitted (the escalated response
    came from a different model, and neither escalation path here tracks
    its own token counts) — completions.py already defaults those to 0
    on read, so this only affects cosmetic token accounting on a future
    cache hit, not correctness of the served content.

    Never raises — a cache write-back failure shouldn't turn an
    already-successful, already-persisted escalation into a task
    failure/retry.
    """
    try:
        cache = get_semantic_cache()
        await cache.aupdate(
            cache_key,
            response=escalated_content,
            metadata={
                "model_id": escalated_model_id or "unknown",
                "provider": escalated_provider.value if escalated_provider else "unknown",
                "complexity_tier": ComplexityTier.COMPLEX.value,
                "classifier_confidence": classifier_confidence,
                "corrected_by_escalation": True,
            },
        )
        logger.info("cache_writeback_succeeded", request_id=request_id, cache_key=cache_key)
    except Exception as exc:  # noqa: BLE001 — best-effort, escalation already succeeded
        logger.warning("cache_writeback_failed", request_id=request_id, error=str(exc))


async def _verify_response_async(
    *,
    request_id: str,
    prompt: str,
    original_response: str,
    model_id: str,
    provider: str,
    complexity_tier: str,
    classifier_confidence: float,
    cache_key: str | None = None,
) -> VerificationResult:
    settings = get_settings()
    routing_config = await refresh_routing_config_from_db()

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
    escalated_provider: Provider | None
    escalation_cost: float

    if scoring_result.escalation_candidate_content is not None:
        # Pairwise strategy already generated a COMPLEX-tier comparison
        # response — reuse it rather than pay for a second rerun.
        escalated_content = scoring_result.escalation_candidate_content
        escalated_model_id = scoring_result.escalation_candidate_model_id
        escalated_provider = scoring_result.escalation_candidate_provider
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
            escalated_provider = escalation_model_config.provider
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

    if cache_key is not None and escalated_content is not None:
        await _write_back_escalated_response(
            cache_key=cache_key,
            escalated_content=escalated_content,
            escalated_model_id=escalated_model_id,
            escalated_provider=escalated_provider,
            classifier_confidence=classifier_confidence,
            request_id=request_id,
        )

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
    cache_key: str | None = None,
) -> dict[str, Any]:
    """
    Async quality verification for a single completed request.

    `input_tokens` / `output_tokens` / `cost_usd` describe the *original*
    response and aren't used by the scoring logic itself — they're kept
    on the signature as an audit-trail hook for future cost-delta
    reporting (e.g. comparing original vs. escalation cost) without
    another schema change.

    `cache_key` (Phase 5) is the semantic-cache entry the original
    response populated, if any — enqueue_verify_response() only ever
    passes one when the original completion was a cache miss (a cache
    hit isn't sampled for verification at all), so it's always present
    when there's actually something to write a correction back to.
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
                cache_key=cache_key,
            )
        )
        celery_tasks_total.labels(task_name="verify_response", status="success").inc()
        log.info(
            "verification_complete",
            status=result.status.value,
            quality_score=round(result.quality_score, 3),
        )
        result_json: dict[str, Any] = result.model_dump(mode="json")
        return result_json

    except Exception as exc:
        celery_tasks_total.labels(task_name="verify_response", status="failure").inc()
        log.error("verification_error", error=str(exc), exc_info=True)
        raise self.retry(exc=exc) from exc
