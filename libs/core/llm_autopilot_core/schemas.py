"""
Shared Pydantic schemas used across API, worker, and core.

Design rules:
- All external-facing models use strict validation (no coercion).
- Internal models can be more permissive.
- Enums are str-based so they serialise cleanly to JSON / Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

# ── Enums ─────────────────────────────────────────────────────────────────────


class ComplexityTier(StrEnum):
    SIMPLE = "simple"  # reformatting, extraction, basic Q&A
    MODERATE = "moderate"  # summarisation, classification, structured analysis
    COMPLEX = "complex"  # multi-step reasoning, creative, nuanced judgement


class QualityTier(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"


class EscalationReason(StrEnum):
    QUALITY_GAP = "quality_gap"  # verifier score below threshold
    CIRCUIT_OPEN = "circuit_open"  # primary provider circuit breaker tripped
    PROVIDER_ERROR = "provider_error"  # non-retriable provider error
    MANUAL = "manual"  # operator-initiated


class VerificationStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ESCALATED = "escalated"
    SKIPPED = "skipped"  # sampling decided not to verify


# ── Model Registry ────────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    """Static config for a single LLM. Populated from registry.py."""

    provider: Provider
    model_id: str
    display_name: str
    cost_per_input_token: float = Field(ge=0.0, description="USD per token")
    cost_per_output_token: float = Field(ge=0.0, description="USD per token")
    avg_latency_ms: float = Field(ge=0.0)
    quality_tier: QualityTier
    context_window: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    supports_system_prompt: bool = True
    enabled: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_per_1k_tokens(self) -> float:
        """Blended cost (input+output) per 1k tokens, assuming 1:1 ratio."""
        return (self.cost_per_input_token + self.cost_per_output_token) * 1000


# ── Request / Response contracts ──────────────────────────────────────────────


class Message(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1)


class CompletionRequest(BaseModel):
    """What the caller sends to POST /v1/completions."""

    messages: list[Message] = Field(..., min_length=1)
    max_tokens: int = Field(default=1024, ge=1, le=100_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    # Caller can hint the tier; router may override it
    force_tier: ComplexityTier | None = None
    # Arbitrary key/value for audit logging (e.g. user_id, session_id)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """
    Raw result of a single provider call — the "standardized Response
    object" from Phase 1, Task 2: output text, tokens (in + out), latency,
    cost, and the model ID.

    Adapters (providers/*_adapter.py) construct this with latency_ms=0.0
    and no cost — providers.dispatcher.send_request() fills both in after
    timing the call and pricing it via registry.compute_cost(), since
    adapters shouldn't need to know about pricing.
    """

    content: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    model_id: str
    provider: Provider
    raw_response: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CompletionResponse(BaseModel):
    """What POST /v1/completions returns to the caller."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    model_id: str
    provider: Provider
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    complexity_tier: ComplexityTier
    classifier_confidence: float = Field(ge=0.0, le=1.0)
    cache_hit: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ── Routing ───────────────────────────────────────────────────────────────────


class RoutingDecision(BaseModel):
    """Logged for every request that reaches the router (cache misses only)."""

    request_id: UUID
    complexity_tier: ComplexityTier
    classifier_confidence: float = Field(ge=0.0, le=1.0)
    selected_model_id: str
    selected_provider: Provider
    reason: str
    alternatives_considered: list[str] = Field(default_factory=list)
    circuit_breaker_overrides: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Verification / Escalation ─────────────────────────────────────────────────


class VerificationResult(BaseModel):
    """
    Produced by the async Celery verification task.

    Phase 4 additions (feedback loop into the complexity classifier):
      - escalated_content: the corrected output when escalation succeeds.
        Populated either from a fresh rerun against the escalation target,
        or — for pairwise-judged categories (creative/reasoning) — reused
        directly from the comparison response generated during scoring,
        since that call already happened against a top-tier model.
      - feature_vector: the same 11-float vector classifier/features.py
        would extract from the prompt, stored so the weekly retraining
        job never needs the raw prompt text (which is never persisted —
        see models/requests.py's prompt_hash design).
      - corrected_tier: the tier this request *should* have been routed
        to, i.e. the escalation target's tier. Only set when escalation
        succeeded; this is the label retrain_classifier() trains against.
    """

    request_id: UUID
    original_model_id: str
    judge_model_id: str
    quality_score: float = Field(ge=0.0, le=1.0)
    status: VerificationStatus
    quality_gap: float | None = None  # None if not escalated
    escalation_reason: EscalationReason | None = None
    escalated_model_id: str | None = None
    escalated_content: str | None = None
    cost_delta_usd: float | None = None  # escalation overhead
    feature_vector: list[float] | None = None
    corrected_tier: ComplexityTier | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Dashboard / Stats ─────────────────────────────────────────────────────────


class CostStats(BaseModel):
    """Returned by GET /v1/stats."""

    period_start: datetime
    period_end: datetime
    total_requests: int
    total_cost_usd: float
    hypothetical_cost_usd: float  # if everything went to GPT-4o
    cost_savings_usd: float
    cost_savings_pct: float
    cache_hit_rate: float
    escalation_rate: float
    requests_by_tier: dict[ComplexityTier, int]
    requests_by_provider: dict[Provider, int]
    avg_quality_score: float
