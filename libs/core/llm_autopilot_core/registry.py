"""
Model registry — source of truth for which models exist, pricing, and capability tiers.

Phase 6 / Workstream A: pricing re-verified against provider docs (Oct 2026).
Every entry carries `pricing_verified_at`; entries with `None` have NOT been
re-verified and must not be quoted in portfolio material.

Prices are USD per token (per-MTok figure / 1_000_000).
Cache-read rates: Anthropic 0.1x input, OpenAI 0.5x (gpt-4o / mini), Google 0.1x.
Sources: platform.claude.com/docs/en/about-claude/pricing, OpenAI pricing page,
ai.google.dev Gemini pricing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from llm_autopilot_core.schemas import ModelConfig, Provider, QualityTier

_VERIFIED = date(2026, 10, 6)

MODEL_REGISTRY: dict[str, ModelConfig] = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "openai/gpt-4o": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        display_name="GPT-4o",
        cost_per_input_token=0.000_002_5,  # $2.50 / 1M  (was $5.00 — stale)
        cost_per_output_token=0.000_010,  # $10.00 / 1M (was $15.00 — stale)
        cache_input_cost_per_token=0.000_001_25,  # $1.25 / 1M
        supports_prompt_caching=True,  # automatic, >=1024-token prefix
        supports_batch_api=True,
        pricing_verified_at=_VERIFIED,
        avg_latency_ms=1_800,
        quality_tier=QualityTier.HIGH,
        context_window=128_000,
        max_output_tokens=4_096,
    ),
    "openai/gpt-4o-mini": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        cost_per_input_token=0.000_000_15,
        cost_per_output_token=0.000_000_60,
        cache_input_cost_per_token=0.000_000_075,
        supports_prompt_caching=True,
        supports_batch_api=True,
        pricing_verified_at=_VERIFIED,
        avg_latency_ms=700,
        quality_tier=QualityTier.MEDIUM,
        context_window=128_000,
        max_output_tokens=4_096,
    ),
    # ── Anthropic ─────────────────────────────────────────────────────────────
    "anthropic/claude-sonnet-4-6": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-sonnet-4-6",
        display_name="Claude 4.6 Sonnet (legacy)",
        cost_per_input_token=0.000_003,
        cost_per_output_token=0.000_015,
        cache_input_cost_per_token=0.000_000_30,
        supports_prompt_caching=True,  # explicit cache_control, >=1024 tokens
        supports_batch_api=True,
        pricing_verified_at=_VERIFIED,
        avg_latency_ms=1_500,
        quality_tier=QualityTier.HIGH,
        context_window=1_000_000,
        max_output_tokens=8_192,
    ),
    "anthropic/claude-haiku-4-5": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-haiku-4-5",
        display_name="Claude 4.5 Haiku",
        cost_per_input_token=0.000_001,  # $1.00 / 1M (was $0.80 — stale)
        cost_per_output_token=0.000_005,  # $5.00 / 1M (was $4.00 — stale)
        cache_input_cost_per_token=0.000_000_10,
        supports_prompt_caching=True,  # needs >=4096-token prefix on Haiku 4.5
        supports_batch_api=True,
        pricing_verified_at=_VERIFIED,
        avg_latency_ms=600,
        quality_tier=QualityTier.MEDIUM,
        context_window=200_000,
        max_output_tokens=8_192,
    ),
    # ── Google ────────────────────────────────────────────────────────────────
    "google/gemini-3.1-pro-preview": ModelConfig(
        provider=Provider.GOOGLE,
        model_id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro",
        cost_per_input_token=0.000_002,  # <=200k prompt; >200k is $4 / $18 (not modeled)
        cost_per_output_token=0.000_012,
        cache_input_cost_per_token=0.000_000_20,
        supports_prompt_caching=True,
        supports_batch_api=True,
        pricing_verified_at=_VERIFIED,
        avg_latency_ms=1_600,
        quality_tier=QualityTier.HIGH,
        context_window=1_000_000,
        max_output_tokens=8_192,
    ),
    "google/gemini-3.5-flash": ModelConfig(
        provider=Provider.GOOGLE,
        model_id="gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        cost_per_input_token=0.000_001_5,  # $1.50 / 1M (was $0.075 — 20x stale)
        cost_per_output_token=0.000_009,  # $9.00 / 1M (was $0.30 — 30x stale)
        cache_input_cost_per_token=0.000_000_15,
        supports_prompt_caching=True,
        supports_batch_api=True,
        pricing_verified_at=_VERIFIED,
        avg_latency_ms=500,
        quality_tier=QualityTier.MEDIUM,
        context_window=1_000_000,
        max_output_tokens=8_192,
    ),
    # ── Groq — NOT re-verified; pricing_verified_at intentionally None ────────
    "meta-llama/llama-prompt-guard-2-22m": ModelConfig(
        provider=Provider.GROQ,
        model_id="llama-3.1-8b-instant",
        display_name="Llama 3.1 8B Instant (Groq)",
        cost_per_input_token=0.000_000_05,
        cost_per_output_token=0.000_000_08,
        avg_latency_ms=350,
        quality_tier=QualityTier.MEDIUM,
        context_window=128_000,
        max_output_tokens=8_192,
    ),
    "groq/llama-3.1-8b-instant": ModelConfig(
        provider=Provider.GROQ,
        model_id="llama-3.1-8b-instant",
        display_name="Llama 3.1 8B Instant (Groq)",
        cost_per_input_token=0.000_000_05,
        cost_per_output_token=0.000_000_08,
        avg_latency_ms=350,
        quality_tier=QualityTier.MEDIUM,
        context_window=128_000,
        max_output_tokens=8_192,
    ),
    "meta-llama/llama-prompt-guard-2-22m-low": ModelConfig(
        provider=Provider.GROQ,
        model_id="llama-3.1-8b-instant",
        display_name="Llama 3.1 8B Instant Low (Groq)",
        cost_per_input_token=0.000_000_05,
        cost_per_output_token=0.000_000_08,
        avg_latency_ms=150,
        quality_tier=QualityTier.LOW,
        context_window=128_000,
        max_output_tokens=8_192,
    ),
    "groq/llama-3.3-70b-versatile": ModelConfig(
        provider=Provider.GROQ,
        model_id="llama-3.3-70b-versatile",
        display_name="Llama 3.3 70B Versatile (Groq)",
        cost_per_input_token=0.000_000_24,
        cost_per_output_token=0.000_000_24,
        avg_latency_ms=250,
        quality_tier=QualityTier.MEDIUM,
        context_window=32_768,
        max_output_tokens=4_096,
    ),
}

# Ollama entries remain disabled (see git history for the commented block).

BASELINE_MODEL_KEY = "openai/gpt-4o"


def get_model(registry_key: str) -> ModelConfig | None:
    """Lookup by '<provider>/<model_id>' key."""
    model = MODEL_REGISTRY.get(registry_key)
    if model is not None:
        return model

    # Routing can also receive a provider/model-id pair produced from a
    # RoutingDecision. Support that form for legacy registry aliases.
    provider_name, separator, model_id = registry_key.partition("/")
    if not separator:
        return None
    return next(
        (
            candidate
            for candidate in MODEL_REGISTRY.values()
            if candidate.provider.value == provider_name and candidate.model_id == model_id
        ),
        None,
    )


def get_models_by_quality_tier(tier: QualityTier) -> list[ModelConfig]:
    return [m for m in MODEL_REGISTRY.values() if m.quality_tier == tier and m.enabled]


def get_cheapest_model_for_tier(tier: QualityTier) -> ModelConfig | None:
    models = get_models_by_quality_tier(tier)
    if not models:
        return None
    return min(models, key=lambda m: m.cost_per_input_token + m.cost_per_output_token)


def compute_cost(
    model: ModelConfig,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """
    USD cost for one response.

    `input_tokens` is the TOTAL prompt size including cache hits; adapters
    must normalise (Anthropic reports cache reads separately — add them in).
    `cached_input_tokens` of those are billed at the cache-read rate, falling
    back to the full input rate when the model has no cache price.
    """
    cached = max(0, min(cached_input_tokens, input_tokens))
    cache_rate = (
        model.cache_input_cost_per_token
        if model.cache_input_cost_per_token is not None
        else model.cost_per_input_token
    )
    return (
        model.cost_per_input_token * (input_tokens - cached)
        + cache_rate * cached
        + model.cost_per_output_token * output_tokens
    )


def stale_pricing_keys(max_age_days: int = 90, today: date | None = None) -> list[str]:
    """Registry keys whose pricing is unverified or older than max_age_days."""
    today = today or date.today()
    return sorted(
        key
        for key, m in MODEL_REGISTRY.items()
        if m.pricing_verified_at is None or (today - m.pricing_verified_at).days > max_age_days
    )


def load_yaml_overrides(path: str) -> None:
    """Apply runtime price / availability overrides from configs/models.yaml."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        return
    with yaml_path.open() as f:
        data = yaml.safe_load(f) or {}
    for key, patch in (data.get("overrides") or {}).items():
        if key in MODEL_REGISTRY:
            for field, value in patch.items():
                if hasattr(MODEL_REGISTRY[key], field):
                    MODEL_REGISTRY[key] = MODEL_REGISTRY[key].model_copy(update={field: value})
