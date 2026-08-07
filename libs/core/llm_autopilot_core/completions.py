"""
Phase 3 — request orchestration for POST /v1/completions.

handle_completion_request() is the single place that wires together:
  semantic cache lookup → classification → routing → provider dispatch →
  persistence → async verification trigger (sampled).

Lives in libs/core (not the API router) so it's testable without FastAPI
and reusable from a CLI/load-test script the same way baseline_test.py
exercises providers.dispatcher directly — "shared logic over drift".
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from typing import Any

import structlog

from llm_autopilot_core.cache.semantic_cache import get_semantic_cache
from llm_autopilot_core.classifier import get_classifier
from llm_autopilot_core.config import Settings, get_settings
from llm_autopilot_core.database import managed_session
from llm_autopilot_core.metrics import (
    cache_hits_total,
    cache_misses_total,
    cost_savings_usd_total,
    hypothetical_cost_usd_total,
    request_latency_ms,
    requests_total,
)
from llm_autopilot_core.metrics import (
    classifier_confidence as classifier_confidence_metric,
)
from llm_autopilot_core.models import Request as RequestModel
from llm_autopilot_core.models import Response as ResponseModel
from llm_autopilot_core.models import RoutingDecision as RoutingDecisionModel
from llm_autopilot_core.providers.dispatcher import send_request
from llm_autopilot_core.registry import compute_cost, get_model
from llm_autopilot_core.routing import (
    RoutingConfig,
    RoutingConfigError,
    get_routing_config,
    select_model_for_tier,
)
from llm_autopilot_core.schemas import (
    CompletionRequest,
    CompletionResponse,
    ComplexityTier,
    Message,
    Provider,
)
from llm_autopilot_core.schemas import RoutingDecision as RoutingDecisionSchema
from llm_autopilot_core.tasks_client import enqueue_verify_response

logger = structlog.get_logger(__name__)
_rng = secrets.SystemRandom()


def _hash_prompt(flattened_prompt: str) -> str:
    return hashlib.sha256(flattened_prompt.encode("utf-8")).hexdigest()


def _flatten_messages(messages: list[Message]) -> str:
    """Single string used for both the semantic cache key and classification."""
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def _should_sample_for_verification(confidence: float, settings: Settings) -> bool:
    if _rng.random() < settings.verification_random_baseline_rate:
        return True
    rate = (
        settings.verification_sample_rate_low_confidence
        if confidence < settings.classifier_confidence_threshold
        else settings.verification_sample_rate_high_confidence
    )
    return _rng.random() < rate


def _is_well_formed_hit(hit: dict[str, Any]) -> bool:
    metadata = hit.get("metadata")
    return (
        bool(hit.get("response"))
        and isinstance(metadata, dict)
        and "model_id" in metadata
        and "provider" in metadata
    )


def _response_from_cache_hit(
    request_id: uuid.UUID, hit: dict[str, Any], *, latency_ms: float
) -> CompletionResponse:
    metadata = hit["metadata"]
    return CompletionResponse(
        id=request_id,
        content=hit["response"],
        model_id=metadata["model_id"],
        provider=Provider(metadata["provider"]),
        input_tokens=int(metadata.get("input_tokens", 0)),
        output_tokens=int(metadata.get("output_tokens", 0)),
        cost_usd=0.0,
        latency_ms=latency_ms,
        complexity_tier=ComplexityTier(
            metadata.get("complexity_tier", ComplexityTier.MODERATE.value)
        ),
        classifier_confidence=float(metadata.get("classifier_confidence", 0.0)),
        cache_hit=True,
    )


def _record_savings(
    routing_config: RoutingConfig, response: CompletionResponse, *, actual_cost_usd: float
) -> None:
    baseline_model = get_model(routing_config.cost_baseline.model)
    if baseline_model is None:
        return
    hypothetical_cost = compute_cost(baseline_model, response.input_tokens, response.output_tokens)
    hypothetical_cost_usd_total.inc(hypothetical_cost)
    cost_savings_usd_total.inc(max(0.0, hypothetical_cost - actual_cost_usd))


async def _persist(
    request: CompletionRequest,
    request_id: uuid.UUID,
    prompt_hash: str,
    response: CompletionResponse,
    *,
    routing_decision: RoutingDecisionSchema | None,
    cache_hit: bool,
) -> None:
    async with managed_session() as session:
        session.add(
            RequestModel(
                id=request_id,
                prompt_hash=prompt_hash,
                message_count=len(request.messages),
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                force_tier=request.force_tier.value if request.force_tier else None,
                caller_metadata=request.metadata,
                cache_hit=cache_hit,
            )
        )
        session.add(
            ResponseModel(
                request_id=request_id,
                content=response.content,
                model_id=response.model_id,
                provider=response.provider,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                complexity_tier=response.complexity_tier,
                classifier_confidence=response.classifier_confidence,
            )
        )
        if routing_decision is not None:
            session.add(
                RoutingDecisionModel(
                    request_id=request_id,
                    complexity_tier=routing_decision.complexity_tier,
                    classifier_confidence=routing_decision.classifier_confidence,
                    selected_model_id=routing_decision.selected_model_id,
                    selected_provider=routing_decision.selected_provider,
                    reason=routing_decision.reason,
                    alternatives_considered=routing_decision.alternatives_considered,
                    circuit_breaker_overrides=routing_decision.circuit_breaker_overrides,
                )
            )


async def handle_completion_request(request: CompletionRequest) -> CompletionResponse:
    settings = get_settings()
    routing_config = get_routing_config()
    request_id = uuid.uuid4()
    flattened = _flatten_messages(request.messages)
    prompt_hash = _hash_prompt(flattened)

    # ── 1. Semantic cache lookup ─────────────────────────────────────────────
    cache = get_semantic_cache()
    cache_start = time.perf_counter()
    hits = await cache.acheck(prompt=flattened, num_results=1)
    cache_lookup_ms = (time.perf_counter() - cache_start) * 1_000

    if hits and _is_well_formed_hit(hits[0]):
        response = _response_from_cache_hit(request_id, hits[0], latency_ms=cache_lookup_ms)
        cache_hits_total.inc()
        requests_total.labels(
            complexity_tier=response.complexity_tier.value,
            provider=response.provider.value,
            model_id=response.model_id,
            cache_hit="true",
        ).inc()
        await _persist(
            request, request_id, prompt_hash, response, routing_decision=None, cache_hit=True
        )
        _record_savings(routing_config, response, actual_cost_usd=0.0)
        return response

    cache_misses_total.inc()

    # ── 2. Classify ───────────────────────────────────────────────────────────
    classifier = get_classifier()
    classification = classifier.predict(flattened)
    tier = request.force_tier or classification.tier
    classifier_confidence_metric.labels(complexity_tier=classification.tier.value).observe(
        classification.confidence
    )

    # ── 3. Route ─────────────────────────────────────────────────────────────
    routing_decision = select_model_for_tier(
        tier, classification.confidence, routing_config, request_id=request_id
    )
    model_config = get_model(
        f"{routing_decision.selected_provider.value}/{routing_decision.selected_model_id}"
    )
    if model_config is None:
        raise RoutingConfigError(
            f"routed model '{routing_decision.selected_model_id}' not found in MODEL_REGISTRY"
        )

    # ── 4. Dispatch ──────────────────────────────────────────────────────────
    provider_response = await send_request(
        request.messages,
        model_config,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    response = CompletionResponse(
        id=request_id,
        content=provider_response.content,
        model_id=provider_response.model_id,
        provider=provider_response.provider,
        input_tokens=provider_response.input_tokens,
        output_tokens=provider_response.output_tokens,
        cost_usd=provider_response.cost_usd,
        latency_ms=provider_response.latency_ms,
        complexity_tier=tier,
        classifier_confidence=classification.confidence,
        cache_hit=False,
    )
    requests_total.labels(
        complexity_tier=tier.value,
        provider=response.provider.value,
        model_id=response.model_id,
        cache_hit="false",
    ).inc()
    request_latency_ms.labels(complexity_tier=tier.value, provider=response.provider.value).observe(
        response.latency_ms
    )

    # ── 5. Persist ───────────────────────────────────────────────────────────
    await _persist(
        request,
        request_id,
        prompt_hash,
        response,
        routing_decision=routing_decision,
        cache_hit=False,
    )
    _record_savings(routing_config, response, actual_cost_usd=response.cost_usd)

    # ── 6. Populate the cache for next time ─────────────────────────────────
    await cache.astore(
        prompt=flattened,
        response=response.content,
        metadata={
            "model_id": response.model_id,
            "provider": response.provider.value,
            "complexity_tier": response.complexity_tier.value,
            "classifier_confidence": response.classifier_confidence,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
    )

    # ── 7. Sampled async verification trigger ───────────────────────────────
    if _should_sample_for_verification(classification.confidence, settings):
        enqueue_verify_response(
            request_id=str(request_id),
            prompt=flattened,
            original_response=response.content,
            model_id=response.model_id,
            provider=response.provider.value,
            complexity_tier=response.complexity_tier.value,
            classifier_confidence=response.classifier_confidence,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )

    return response
