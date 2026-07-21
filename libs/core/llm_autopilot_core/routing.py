"""
Tier → model routing.

configs/routing.yaml is the "routing map" Phase 2 asks for; this module
is the code that reads it and turns (tier, confidence) into an actual
ModelConfig, respecting live circuit breaker state from
providers.dispatcher so a tripped provider doesn't get selected just
because it's first in the chain.

The result is built as schemas.RoutingDecision directly rather than a
bespoke dataclass — that schema already has exactly the fields a router
needs (alternatives_considered, circuit_breaker_overrides), and a future
completions endpoint can persist it to the routing_decisions table
unchanged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from uuid import UUID, uuid4

import structlog
import yaml
from pydantic import BaseModel, Field

from llm_autopilot_core.config import get_settings
from llm_autopilot_core.providers.dispatcher import is_provider_available
from llm_autopilot_core.registry import MODEL_REGISTRY, get_model
from llm_autopilot_core.schemas import ComplexityTier, RoutingDecision

logger = structlog.get_logger(__name__)


class RoutingConfigError(Exception):
    """Raised when routing.yaml can't produce at least one routable model for a tier."""


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


def load_routing_config(path: str) -> RoutingConfig:
    """
    Parse configs/routing.yaml and drop any model keys that no longer
    exist in MODEL_REGISTRY (logged as an error — this is config drift,
    not a runtime routing decision). A tier left with zero routable
    models raises RoutingConfigError rather than silently producing an
    empty chain that would only fail later, mid-request.
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


@lru_cache(maxsize=1)
def get_routing_config() -> RoutingConfig:
    """Cached singleton. Call get_routing_config.cache_clear() in tests."""
    settings = get_settings()
    return load_routing_config(settings.routing_config_path)


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
