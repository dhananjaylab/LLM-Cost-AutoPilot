"""
Tier → model routing.

configs/routing.yaml bootstraps the very first routing config; from then
on, Postgres (routing_config_versions) is the source of truth and
configs/routing.yaml is only read again if the table is ever empty (e.g.
a fresh database). This module is the code that turns (tier, confidence)
into an actual ModelConfig, respecting live circuit breaker state from
providers.dispatcher so a tripped provider doesn't get selected just
because it's first in the chain.

Phase 5 additions (PUT /v1/admin/routing-config, see
apps/api/.../routers/admin.py):
  - get_routing_config() stays a fast, synchronous, process-local
    accessor — the hot request path in completions.py calls it on every
    single request and can't afford a DB round trip there.
  - refresh_routing_config_from_db() is the only thing that talks to
    Postgres. It's called once at API startup (fail-fast, like the
    existing DB connectivity check), then on a periodic background loop
    per settings.routing_config_cache_ttl_seconds (see main.py's
    lifespan), and once per Celery verification task (the worker doesn't
    keep a long-lived process-local cache — a small DB read per task is
    cheap and always fresh, unlike the API's hot path).
  - persist_routing_config() is what PUT actually calls: validates
    strictly (no silent model-key pruning — an admin write should fail
    loudly on a typo, unlike the lenient YAML-bootstrap parser below),
    writes a new promoted RoutingConfigVersion row, unpromotes the
    previous one, and updates the process-local cache immediately so the
    replica that served the PUT reflects it without waiting for its own
    next refresh tick.

The result is built as schemas.RoutingDecision directly rather than a
bespoke dataclass — that schema already has exactly the fields a router
needs (alternatives_considered, circuit_breaker_overrides), and a future
completions endpoint can persist it to the routing_decisions table
unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update

from llm_autopilot_core.config import get_settings
from llm_autopilot_core.database import managed_session
from llm_autopilot_core.providers.dispatcher import is_provider_available
from llm_autopilot_core.registry import MODEL_REGISTRY, get_model
from llm_autopilot_core.schemas import ComplexityTier, RoutingDecision

logger = structlog.get_logger(__name__)


class RoutingConfigError(Exception):
    """Raised when routing.yaml (or a PUT payload) can't produce at least
    one routable model for a tier, or references a model that doesn't
    exist in MODEL_REGISTRY."""


class TierRoute(BaseModel):
    description: str
    models: list[str]
    max_latency_ms: int = Field(gt=0)


class VerificationRoutingConfig(BaseModel):
    judge_model: str
    judge_max_tokens: int = Field(gt=0)


class CostBaselineConfig(BaseModel):
    model: str


class RoutingConfig(BaseModel):
    version: str
    tiers: dict[ComplexityTier, TierRoute]
    verification: VerificationRoutingConfig
    cost_baseline: CostBaselineConfig


class RoutingConfigUpdateRequest(RoutingConfig):
    """
    PUT /v1/admin/routing-config request body — a full RoutingConfig plus
    free-text audit fields. PUT replaces the whole document (tiers,
    verification, and cost_baseline together) rather than patching
    individual tiers, matching REST's full-replacement semantics for PUT
    and avoiding "which fields did the caller actually mean to change"
    ambiguity.
    """

    notes: str | None = Field(default=None, max_length=500)
    updated_by: str | None = Field(default=None, max_length=128)


class RoutingConfigVersionSummary(BaseModel):
    """Audit-trail row shape returned by GET /v1/admin/routing-config/versions."""

    model_config = ConfigDict(from_attributes=True)

    version_number: int
    promoted: bool
    promoted_at: datetime | None
    notes: str | None
    updated_by: str | None
    created_at: datetime


# ── YAML parsing (bootstrap only, from Phase 5 onward) ─────────────────────────


def load_routing_config(path: str) -> RoutingConfig:
    """
    Parse configs/routing.yaml and drop any model keys that no longer
    exist in MODEL_REGISTRY (logged as an error — this is config drift,
    not a runtime routing decision). A tier left with zero routable
    models raises RoutingConfigError rather than silently producing an
    empty chain that would only fail later, mid-request.

    Used to bootstrap the first-ever RoutingConfigVersion row in
    Postgres (see refresh_routing_config_from_db()) and as the last-resort
    fallback for get_routing_config() before anything has been loaded
    into the process-local cache yet.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise RoutingConfigError(f"routing config not found: {path}")

    with yaml_path.open() as f:
        raw = yaml.safe_load(f) or {}

    tiers_raw = raw.get("routing", {}).get("tiers", {})
    if not tiers_raw:
        raise RoutingConfigError(f"routing config has no tiers defined: {path}")

    tiers: dict[ComplexityTier, TierRoute] = {}
    for tier_name, tier_data in tiers_raw.items():
        tier = ComplexityTier(tier_name)
        configured_models = tier_data.get("models", [])
        valid_models = [key for key in configured_models if key in MODEL_REGISTRY]
        dropped = set(configured_models) - set(valid_models)
        if dropped:
            logger.error("routing_config_unknown_models", tier=tier_name, models=sorted(dropped))
        if not valid_models:
            raise RoutingConfigError(
                f"tier '{tier_name}' has no models that exist in MODEL_REGISTRY"
            )
        tiers[tier] = TierRoute(
            description=tier_data.get("description", ""),
            models=valid_models,
            max_latency_ms=tier_data["max_latency_ms"],
        )

    try:
        verification_cfg = VerificationRoutingConfig(**raw["verification"])
        cost_baseline_cfg = CostBaselineConfig(**raw["cost_baseline"])
    except KeyError as exc:
        raise RoutingConfigError(f"routing config missing required section: {exc}") from exc

    return RoutingConfig(
        version=str(raw.get("version", "1")),
        tiers=tiers,
        verification=verification_cfg,
        cost_baseline=cost_baseline_cfg,
    )


def validate_routing_config_strict(config: RoutingConfig) -> None:
    """
    Reject — rather than silently prune, unlike load_routing_config's
    YAML-bootstrap path — any tier, judge model, or cost baseline that
    references a model key not present in MODEL_REGISTRY. An admin
    explicitly PUTting a routing config expects every model they named to
    take effect; silently dropping a typo would be a much worse failure
    mode here than at YAML-bootstrap time.
    """
    unknown: dict[str, list[str]] = {}
    for tier, tier_route in config.tiers.items():
        bad = [key for key in tier_route.models if key not in MODEL_REGISTRY]
        if bad:
            unknown[tier.value] = bad
    if unknown:
        raise RoutingConfigError(f"unknown model keys referenced in tiers: {unknown}")

    if config.verification.judge_model not in MODEL_REGISTRY:
        raise RoutingConfigError(
            f"verification.judge_model '{config.verification.judge_model}' "
            "not found in MODEL_REGISTRY"
        )
    if config.cost_baseline.model not in MODEL_REGISTRY:
        raise RoutingConfigError(
            f"cost_baseline.model '{config.cost_baseline.model}' not found in MODEL_REGISTRY"
        )


# ── Process-local cache (Phase 5) ───────────────────────────────────────────────

_routing_config_cache: RoutingConfig | None = None


def get_routing_config() -> RoutingConfig:
    """
    Fast, synchronous, process-local accessor — safe to call on every
    request. Returns whatever refresh_routing_config_from_db() last
    loaded (via the periodic background refresh, or immediately after a
    PUT on the replica that served it). Falls back to parsing
    configs/routing.yaml directly if nothing has been loaded into this
    process yet — covers scripts, one-off tools, and any test that
    doesn't exercise the DB-backed refresh path, the same role this
    function played before Phase 5.
    """
    global _routing_config_cache
    if _routing_config_cache is None:
        settings = get_settings()
        _routing_config_cache = load_routing_config(settings.routing_config_path)
    return _routing_config_cache


def reset_routing_config_cache() -> None:
    """Test helper — clears the process-local cache so the next
    get_routing_config() call re-bootstraps from YAML. Call this in test
    teardown/setup the same way tests already clear get_settings()."""
    global _routing_config_cache
    _routing_config_cache = None


async def refresh_routing_config_from_db() -> RoutingConfig:
    """
    Reload the promoted RoutingConfigVersion row from Postgres and update
    the process-local cache get_routing_config() reads from. On the very
    first call ever (no promoted row exists — e.g. a fresh database),
    seeds one from configs/routing.yaml so there's always exactly one
    promoted version once the app has started at least once.

    Raises whatever the underlying DB call raises on failure — callers
    that run this on a timer (see main.py's background refresh loop)
    should catch and log rather than let a transient DB blip kill the
    loop; the one-time startup call is meant to fail fast instead,
    consistent with the existing check_connection() startup guard.
    """
    global _routing_config_cache
    settings = get_settings()

    from llm_autopilot_core.models import RoutingConfigVersion

    async with managed_session() as session:
        stmt = select(RoutingConfigVersion).where(RoutingConfigVersion.promoted.is_(True))
        row = (await session.execute(stmt)).scalar_one_or_none()

        if row is None:
            bootstrap = load_routing_config(settings.routing_config_path)
            session.add(
                RoutingConfigVersion(
                    config_json=bootstrap.model_dump(mode="json"),
                    promoted=True,
                    promoted_at=datetime.now(UTC),
                    notes="bootstrap from configs/routing.yaml",
                )
            )
            _routing_config_cache = bootstrap
            logger.info("routing_config_bootstrapped_from_yaml")
            return bootstrap

        config = RoutingConfig.model_validate(row.config_json)

    _routing_config_cache = config
    return config


async def persist_routing_config(
    config: RoutingConfig, *, notes: str | None = None, updated_by: str | None = None
) -> RoutingConfigVersionSummary:
    """
    Validate, then write a new promoted RoutingConfigVersion row
    (unpromoting whichever version was previously active) and update the
    process-local cache immediately, so the replica that served the PUT
    reflects the change without waiting for the next background refresh
    tick. Raises RoutingConfigError (→ 400 at the API layer) if the
    payload references any model key not in MODEL_REGISTRY; the DB is
    never touched in that case.
    """
    global _routing_config_cache
    validate_routing_config_strict(config)
    from llm_autopilot_core.models import RoutingConfigVersion

    async with managed_session() as session:
        await session.execute(update(RoutingConfigVersion).values(promoted=False))
        version = RoutingConfigVersion(
            config_json=config.model_dump(mode="json"),
            promoted=True,
            promoted_at=datetime.now(UTC),
            notes=notes,
            updated_by=updated_by,
        )
        session.add(version)
        await session.flush()
        summary = RoutingConfigVersionSummary.model_validate(version)

    _routing_config_cache = config
    logger.info("routing_config_persisted", version=summary.version_number, updated_by=updated_by)
    return summary


async def list_routing_config_versions(*, limit: int = 20) -> list[RoutingConfigVersionSummary]:
    """Audit history for GET /v1/admin/routing-config/versions, most recent first."""
    from llm_autopilot_core.models import RoutingConfigVersion

    async with managed_session() as session:
        stmt = (
            select(RoutingConfigVersion)
            .order_by(RoutingConfigVersion.version_number.desc())
            .limit(limit)
        )
        rows = (await session.execute(stmt)).scalars().all()
        return [RoutingConfigVersionSummary.model_validate(row) for row in rows]


# ── Tier selection (unchanged from Phase 2) ─────────────────────────────────────


def select_model_for_tier(
    tier: ComplexityTier,
    confidence: float,
    routing_config: RoutingConfig,
    *,
    request_id: UUID | None = None,
) -> RoutingDecision:
    """
    Walk the tier's model chain in order, skipping any provider whose
    circuit breaker is currently OPEN. Falls back to the last model in
    the chain (flagged in circuit_breaker_overrides) rather than raising
    if every provider in the chain is tripped — a degraded routing
    decision beats no decision.
    """
    tier_route = routing_config.tiers[tier]
    skipped: list[str] = []

    for key in tier_route.models:
        model = get_model(key)
        if model is None or not model.enabled:
            skipped.append(key)
            continue
        if is_provider_available(model.provider):
            return RoutingDecision(
                request_id=request_id or uuid4(),
                complexity_tier=tier,
                classifier_confidence=confidence,
                selected_model_id=model.model_id,
                selected_provider=model.provider,
                reason=f"first available model in '{tier.value}' chain",
                alternatives_considered=tier_route.models,
                circuit_breaker_overrides=skipped,
            )
        skipped.append(key)

    # Every candidate was skipped (missing / disabled / breaker open).
    # Force the last configured model rather than failing the request.
    fallback_key = tier_route.models[-1]
    fallback_model = get_model(fallback_key)
    if fallback_model is None:
        raise RoutingConfigError(
            f"no routable model for tier '{tier.value}' — all candidates unavailable"
        )
    logger.warning(
        "routing_forced_fallback", tier=tier.value, fallback_model=fallback_key, skipped=skipped
    )
    return RoutingDecision(
        request_id=request_id or uuid4(),
        complexity_tier=tier,
        classifier_confidence=confidence,
        selected_model_id=fallback_model.model_id,
        selected_provider=fallback_model.provider,
        reason=f"all providers in '{tier.value}' chain unavailable; forced fallback",
        alternatives_considered=tier_route.models,
        circuit_breaker_overrides=skipped,
    )
