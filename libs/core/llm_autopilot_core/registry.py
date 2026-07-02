"""
Model registry — the source of truth for which models exist, their pricing,
and their capability tiers.

Pricing as of mid-2024 (USD per token). Update via configs/models.yaml
at runtime without redeploying — the YAML overrides take precedence over
these defaults when load_yaml_overrides() is called during startup.

Design decision: Groq hosts open-weight models (Llama, Mixtral) on
custom silicon at extremely low cost and latency. Use Groq for Tier 1/2
before touching paid OpenAI/Anthropic/Google tokens.
"""
from __future__ import annotations

import yaml
from pathlib import Path

from llm_autopilot_core.schemas import ModelConfig, Provider, QualityTier


# ── Registry ──────────────────────────────────────────────────────────────────
# Key format: "<provider>/<model_id>" for uniqueness across providers.

MODEL_REGISTRY: dict[str, ModelConfig] = {

    # ── OpenAI ────────────────────────────────────────────────────────────────
    "openai/gpt-4o": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        display_name="GPT-4o",
        cost_per_input_token=0.000_005,     # $5.00 / 1M
        cost_per_output_token=0.000_015,    # $15.00 / 1M
        avg_latency_ms=1_800,
        quality_tier=QualityTier.HIGH,
        context_window=128_000,
        max_output_tokens=4_096,
    ),
    "openai/gpt-4o-mini": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        cost_per_input_token=0.000_000_15,  # $0.15 / 1M
        cost_per_output_token=0.000_000_60, # $0.60 / 1M
        avg_latency_ms=700,
        quality_tier=QualityTier.MEDIUM,
        context_window=128_000,
        max_output_tokens=4_096,
    ),

    # ── Anthropic ─────────────────────────────────────────────────────────────
    "anthropic/claude-3-5-sonnet-20241022": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        cost_per_input_token=0.000_003,     # $3.00 / 1M
        cost_per_output_token=0.000_015,    # $15.00 / 1M
        avg_latency_ms=1_500,
        quality_tier=QualityTier.HIGH,
        context_window=200_000,
        max_output_tokens=8_192,
    ),
    "anthropic/claude-3-5-haiku-20241022": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        cost_per_input_token=0.000_000_80,  # $0.80 / 1M
        cost_per_output_token=0.000_004,    # $4.00 / 1M
        avg_latency_ms=600,
        quality_tier=QualityTier.MEDIUM,
        context_window=200_000,
        max_output_tokens=8_192,
    ),

    # ── Google ────────────────────────────────────────────────────────────────
    "google/gemini-1.5-pro": ModelConfig(
        provider=Provider.GOOGLE,
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        cost_per_input_token=0.000_001_25,  # $1.25 / 1M
        cost_per_output_token=0.000_005,    # $5.00 / 1M
        avg_latency_ms=1_600,
        quality_tier=QualityTier.HIGH,
        context_window=1_000_000,
        max_output_tokens=8_192,
    ),
    "google/gemini-1.5-flash": ModelConfig(
        provider=Provider.GOOGLE,
        model_id="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        cost_per_input_token=0.000_000_075, # $0.075 / 1M
        cost_per_output_token=0.000_000_30, # $0.30 / 1M
        avg_latency_ms=500,
        quality_tier=QualityTier.MEDIUM,
        context_window=1_000_000,
        max_output_tokens=8_192,
    ),

    # ── Groq (open-weight models on custom silicon) ───────────────────────────
    "groq/llama-3.1-70b-versatile": ModelConfig(
        provider=Provider.GROQ,
        model_id="llama-3.1-70b-versatile",
        display_name="Llama 3.1 70B (Groq)",
        cost_per_input_token=0.000_000_59,  # $0.59 / 1M
        cost_per_output_token=0.000_000_79, # $0.79 / 1M
        avg_latency_ms=350,
        quality_tier=QualityTier.MEDIUM,
        context_window=128_000,
        max_output_tokens=8_192,
    ),
    "groq/llama-3.1-8b-instant": ModelConfig(
        provider=Provider.GROQ,
        model_id="llama-3.1-8b-instant",
        display_name="Llama 3.1 8B Instant (Groq)",
        cost_per_input_token=0.000_000_05,  # $0.05 / 1M
        cost_per_output_token=0.000_000_08, # $0.08 / 1M
        avg_latency_ms=150,
        quality_tier=QualityTier.LOW,
        context_window=128_000,
        max_output_tokens=8_192,
    ),
    "groq/mixtral-8x7b-32768": ModelConfig(
        provider=Provider.GROQ,
        model_id="mixtral-8x7b-32768",
        display_name="Mixtral 8x7B (Groq)",
        cost_per_input_token=0.000_000_24,  # $0.24 / 1M
        cost_per_output_token=0.000_000_24, # $0.24 / 1M
        avg_latency_ms=250,
        quality_tier=QualityTier.MEDIUM,
        context_window=32_768,
        max_output_tokens=4_096,
    ),

    # ── Ollama (local — zero marginal cost) ───────────────────────────────────
    "ollama/llama3.1": ModelConfig(
        provider=Provider.OLLAMA,
        model_id="llama3.1",
        display_name="Llama 3.1 8B (Local)",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency_ms=2_500,               # slower without GPU
        quality_tier=QualityTier.LOW,
        context_window=128_000,
        max_output_tokens=4_096,
    ),
    "ollama/mistral": ModelConfig(
        provider=Provider.OLLAMA,
        model_id="mistral",
        display_name="Mistral 7B (Local)",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency_ms=2_800,
        quality_tier=QualityTier.LOW,
        context_window=32_768,
        max_output_tokens=4_096,
    ),
}


# ── Baseline model (used to compute cost savings) ─────────────────────────────
# "What would we have paid if every request went to the most expensive model?"
BASELINE_MODEL_KEY = "openai/gpt-4o"


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_model(registry_key: str) -> ModelConfig | None:
    """Lookup by '<provider>/<model_id>' key."""
    return MODEL_REGISTRY.get(registry_key)


def get_models_by_quality_tier(tier: QualityTier) -> list[ModelConfig]:
    return [m for m in MODEL_REGISTRY.values() if m.quality_tier == tier and m.enabled]


def get_cheapest_model_for_tier(tier: QualityTier) -> ModelConfig | None:
    models = get_models_by_quality_tier(tier)
    if not models:
        return None
    return min(models, key=lambda m: m.cost_per_input_token + m.cost_per_output_token)


def compute_cost(model: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost for a specific response."""
    return (
        model.cost_per_input_token * input_tokens
        + model.cost_per_output_token * output_tokens
    )


def load_yaml_overrides(path: str) -> None:
    """
    Apply runtime price / availability overrides from configs/models.yaml.
    Called once at application startup. Mutates MODEL_REGISTRY in place.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        return

    with yaml_path.open() as f:
        data = yaml.safe_load(f) or {}

    overrides: dict = data.get("overrides", {})
    for key, patch in overrides.items():
        if key in MODEL_REGISTRY:
            for field, value in patch.items():
                if hasattr(MODEL_REGISTRY[key], field):
                    # Pydantic v2: model_copy(update=...) returns new instance
                    MODEL_REGISTRY[key] = MODEL_REGISTRY[key].model_copy(
                        update={field: value}
                    )
