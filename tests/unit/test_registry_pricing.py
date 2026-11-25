"""Phase 6 / Workstream A — cache-aware pricing and staleness guard."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from llm_autopilot_core.registry import (
    MODEL_REGISTRY,
    compute_cost,
    get_model,
    stale_pricing_keys,
)


class TestCacheAwareCost:
    def test_no_cached_tokens_matches_legacy_formula(self) -> None:
        m = get_model("openai/gpt-4o")
        assert m is not None
        assert compute_cost(m, 1000, 500) == pytest.approx(
            m.cost_per_input_token * 1000 + m.cost_per_output_token * 500
        )

    def test_cached_tokens_billed_at_cache_rate(self) -> None:
        m = get_model("anthropic/claude-sonnet-4-6")
        assert m is not None
        cost = compute_cost(m, 10_000, 0, cached_input_tokens=9_000)
        expected = 1_000 * 3e-6 + 9_000 * 3e-7
        assert cost == pytest.approx(expected)

    def test_falls_back_to_input_rate_without_cache_price(self) -> None:
        m = get_model("groq/llama-3.3-70b-versatile")
        assert m is not None
        assert m.cache_input_cost_per_token is None
        assert compute_cost(m, 1000, 0, cached_input_tokens=1000) == pytest.approx(
            compute_cost(m, 1000, 0)
        )

    def test_cached_tokens_clamped_to_input_size(self) -> None:
        m = get_model("openai/gpt-4o-mini")
        assert m is not None
        assert compute_cost(m, 100, 0, cached_input_tokens=999) == pytest.approx(
            100 * m.cache_input_cost_per_token  # type: ignore[operator]
        )


class TestRegistryIntegrity:
    def test_cache_price_never_exceeds_input_price(self) -> None:
        for key, m in MODEL_REGISTRY.items():
            if m.cache_input_cost_per_token is not None:
                assert m.cache_input_cost_per_token <= m.cost_per_input_token, key

    def test_caching_flag_requires_cache_price(self) -> None:
        for key, m in MODEL_REGISTRY.items():
            if m.supports_prompt_caching:
                assert m.cache_input_cost_per_token is not None, key

    def test_baseline_is_priced_at_verified_rates(self) -> None:
        m = get_model("openai/gpt-4o")
        assert m is not None
        assert m.cost_per_input_token == pytest.approx(2.5e-6)
        assert m.cost_per_output_token == pytest.approx(1e-5)


class TestStaleness:
    def test_unverified_entries_are_flagged(self) -> None:
        stale = stale_pricing_keys(today=date(2026, 10, 7))
        assert "groq/llama-3.3-70b-versatile" in stale
        assert "openai/gpt-4o" not in stale

    def test_verified_entries_go_stale_after_window(self) -> None:
        later = date(2026, 10, 6) + timedelta(days=91)
        assert "openai/gpt-4o" in stale_pricing_keys(today=later)
