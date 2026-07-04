"""
Async quality verification task.

Triggered by the API after every request that passes the sampling filter.
Sampling decision (made in the API layer before enqueueing):

  confidence < threshold → sample at verification_sample_rate_low_confidence  (default 100%)
  confidence ≥ threshold → sample at verification_sample_rate_high_confidence  (default 5%)
  random baseline        → sample at verification_random_baseline_rate         (default 2%)

This task:
  1. Sends the same prompt to the judge model
  2. Scores the original response vs the judge's response
  3. Logs the result to the verifications table
  4. If quality_gap > threshold → triggers escalation (rerun with higher-tier model)
  5. Writes failure as a new training example for the classifier retraining task

Phase 4 TODO:
  - Implement LLM-as-judge scoring with structured rubric
  - Implement escalation rerun logic
  - Implement training example writing
"""

from __future__ import annotations

from typing import Any

import structlog
from celery import Task
from llm_autopilot_core.metrics import (
    celery_tasks_total,
    verifications_total,
)

from llm_autopilot_worker.main import celery_app

logger = structlog.get_logger(__name__)


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

    Returns a typed dictionary matching the VerificationResult schema.
    """
    log = logger.bind(
        request_id=request_id,
        model_id=model_id,
        tier=complexity_tier,
        confidence=classifier_confidence,
    )
    log.info("verification_started")

    try:
        # ── Phase 4 implementation goes here ─────────────────────────────────
        # 1. Call judge model
        # 2. Compute quality_score
        # 3. Decide whether to escalate
        # 4. Write verification record to DB
        # 5. If failed, enqueue training example

        result: dict[str, Any] = {
            "request_id": request_id,
            "original_model_id": model_id,
            "judge_model_id": "not_implemented",  # Phase 4
            "quality_score": 0.0,  # Phase 4 placeholder
            "status": "skipped",
            "escalation_reason": None,
            "escalated_model_id": None,
            "cost_delta_usd": 0.0,
        }

        verifications_total.labels(status="skipped").inc()
        celery_tasks_total.labels(task_name="verify_response", status="success").inc()
        log.info("verification_complete", status=result["status"])
        return result

    except Exception as exc:
        celery_tasks_total.labels(task_name="verify_response", status="failure").inc()
        log.error("verification_error", error=str(exc), exc_info=True)
        raise self.retry(exc=exc) from exc
