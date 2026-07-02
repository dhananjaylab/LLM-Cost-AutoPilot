"""
Prometheus metric definitions — single source of truth.

Import these objects wherever you need to record observations.
All metrics are registered with the default Prometheus registry,
which FastAPI exposes via /metrics (make_asgi_app).

Naming convention: <namespace>_<subsystem>_<unit>
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

from llm_autopilot_core.config import get_settings

_ns = get_settings().metrics_namespace   # "llm_autopilot"

# ── Request counters ──────────────────────────────────────────────────────────

requests_total = Counter(
    f"{_ns}_requests_total",
    "Total completion requests received",
    ["complexity_tier", "provider", "model_id", "cache_hit"],
)

requests_errors_total = Counter(
    f"{_ns}_requests_errors_total",
    "Total failed requests",
    ["provider", "error_type"],
)

# ── Cost tracking ─────────────────────────────────────────────────────────────

cost_usd_total = Counter(
    f"{_ns}_cost_usd_total",
    "Cumulative USD spent on LLM API calls",
    ["provider", "model_id"],
)

cost_savings_usd_total = Counter(
    f"{_ns}_cost_savings_usd_total",
    "Cumulative USD saved vs routing everything to the baseline model",
)

hypothetical_cost_usd_total = Counter(
    f"{_ns}_hypothetical_cost_usd_total",
    "What cumulative cost would have been if using baseline model for all requests",
)

# ── Latency ───────────────────────────────────────────────────────────────────

request_latency_ms = Histogram(
    f"{_ns}_request_latency_ms",
    "End-to-end request latency in milliseconds",
    ["complexity_tier", "provider"],
    buckets=[50, 100, 250, 500, 1_000, 2_000, 5_000, 10_000],
)

# ── Classifier ────────────────────────────────────────────────────────────────

classifier_confidence = Histogram(
    f"{_ns}_classifier_confidence",
    "Classifier confidence score distribution",
    ["complexity_tier"],
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
)

# ── Cache ─────────────────────────────────────────────────────────────────────

cache_hits_total = Counter(
    f"{_ns}_cache_hits_total",
    "Semantic cache hit count",
)

cache_misses_total = Counter(
    f"{_ns}_cache_misses_total",
    "Semantic cache miss count",
)

# ── Verification & escalation ─────────────────────────────────────────────────

verifications_total = Counter(
    f"{_ns}_verifications_total",
    "Async verification tasks executed",
    ["status"],   # passed | failed | escalated | skipped
)

quality_score = Histogram(
    f"{_ns}_quality_score",
    "LLM-as-judge quality score from verification tasks",
    ["complexity_tier", "model_id"],
    buckets=[0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
)

escalations_total = Counter(
    f"{_ns}_escalations_total",
    "Auto-escalation count",
    ["original_model", "escalated_model", "reason"],
)

# ── Circuit breakers ──────────────────────────────────────────────────────────

circuit_breaker_state = Gauge(
    f"{_ns}_circuit_breaker_state",
    "Circuit breaker state per provider (0=closed, 1=open, 2=half-open)",
    ["provider"],
)

# ── Worker ────────────────────────────────────────────────────────────────────

celery_tasks_total = Counter(
    f"{_ns}_celery_tasks_total",
    "Celery task execution count",
    ["task_name", "status"],
)

# ── Application info ──────────────────────────────────────────────────────────

app_info = Info(
    f"{_ns}_app",
    "Static application metadata",
)

def initialise_info() -> None:
    s = get_settings()
    app_info.info({
        "version": s.app_version,
        "environment": s.environment,
        "embedding_model": s.embedding_model,
    })
