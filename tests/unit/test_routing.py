"""
Unit tests for llm_autopilot_core.routing.

Circuit breaker state lives in a module-level dict shared with
providers.dispatcher (the same single source of truth send_request()
uses), so tests that trip a breaker must reset it afterwards — the
autouse fixture below does that the same way it's done manually in
scripts/classify_demo.py's --trip-breaker flag.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from llm_autopilot_core.providers.circuit_breaker import BreakerState
from llm_autopilot_core.providers.dispatcher import _BREAKERS
from llm_autopilot_core.routing import (
    RoutingConfigError,
    load_routing_config,
    select_model_for_tier,
)
from llm_autopilot_core.schemas import ComplexityTier, Provider

_REAL_ROUTING_YAML = "configs/routing.yaml"


@pytest.fixture(autouse=True)
def _reset_breakers():
    yield
    for breaker in _BREAKERS.values():
        breaker._state = BreakerState.CLOSED  # noqa: SLF001
        breaker._fail_count = 0  # noqa: SLF001
        breaker._opened_at = None  # noqa: SLF001


def _trip(provider: Provider) -> None:
    breaker = _BREAKERS[provider]
    breaker._state = BreakerState.OPEN  # noqa: SLF001
    breaker._opened_at = float("inf")  # noqa: SLF001  # never rolls to half-open mid-test


class TestLoadRoutingConfig:
    def test_parses_real_routing_yaml(self) -> None:
        config = load_routing_config(_REAL_ROUTING_YAML)
        assert set(config.tiers) == set(ComplexityTier)
        assert config.verification.judge_model
        assert config.cost_baseline.model

    def test_missing_file_raises(self) -> None:
        with pytest.raises(RoutingConfigError):
            load_routing_config("/nonexistent/routing.yaml")

    def test_drops_unknown_model_but_keeps_valid_ones(self, tmp_path: Path) -> None:
        yaml_text = """
version: "1"
routing:
  tiers:
    simple:
      description: "test"
      models:
        - meta-llama/llama-prompt-guard-2-22m
        - openai/model-that-does-not-exist
      max_latency_ms: 1000
    moderate:
      description: "test"
      models:
        - openai/gpt-4o-mini
      max_latency_ms: 1000
    complex:
      description: "test"
      models:
        - openai/gpt-4o
      max_latency_ms: 1000
verification:
  judge_model: openai/gpt-4o-mini
  judge_max_tokens: 512
cost_baseline:
  model: openai/gpt-4o
"""
        path = tmp_path / "routing.yaml"
        path.write_text(yaml_text)

        config = load_routing_config(str(path))
        assert config.tiers[ComplexityTier.SIMPLE].models == ["meta-llama/llama-prompt-guard-2-22m"]

    def test_raises_when_tier_has_no_valid_models(self, tmp_path: Path) -> None:
        yaml_text = """
version: "1"
routing:
  tiers:
    simple:
      description: "test"
      models:
        - openai/does-not-exist
      max_latency_ms: 1000
verification:
  judge_model: openai/gpt-4o-mini
  judge_max_tokens: 512
cost_baseline:
  model: openai/gpt-4o
"""
        path = tmp_path / "routing.yaml"
        path.write_text(yaml_text)

        with pytest.raises(RoutingConfigError):
            load_routing_config(str(path))


class TestSelectModelForTier:
    def test_picks_first_model_in_chain_when_all_available(self) -> None:
        config = load_routing_config(_REAL_ROUTING_YAML)
        decision = select_model_for_tier(ComplexityTier.SIMPLE, 0.9, config)

        expected_first = config.tiers[ComplexityTier.SIMPLE].models[0]
        assert f"{decision.selected_provider.value}/{decision.selected_model_id}" == expected_first
        assert decision.circuit_breaker_overrides == []
        assert decision.complexity_tier == ComplexityTier.SIMPLE
        assert decision.classifier_confidence == 0.9

    def test_skips_provider_with_open_breaker(self) -> None:
        config = load_routing_config(_REAL_ROUTING_YAML)
        # moderate chain's first entry is google/gemini-3.5-flash
        _trip(Provider.GOOGLE)

        decision = select_model_for_tier(ComplexityTier.MODERATE, 0.8, config)

        assert decision.selected_provider != Provider.GOOGLE
        assert "google/gemini-3.5-flash" in decision.circuit_breaker_overrides

    def test_forces_fallback_when_every_provider_in_chain_is_open(self) -> None:
        config = load_routing_config(_REAL_ROUTING_YAML)
        moderate_models = config.tiers[ComplexityTier.MODERATE].models
        providers_in_chain = {m.split("/")[0] for m in moderate_models}
        for provider_name in providers_in_chain:
            _trip(Provider(provider_name))

        decision = select_model_for_tier(ComplexityTier.MODERATE, 0.5, config)

        last_key = moderate_models[-1]
        assert f"{decision.selected_provider.value}/{decision.selected_model_id}" == last_key
        assert "forced fallback" in decision.reason
        assert set(decision.circuit_breaker_overrides) == set(moderate_models)

    def test_respects_supplied_request_id(self) -> None:
        config = load_routing_config(_REAL_ROUTING_YAML)
        request_id = uuid4()
        decision = select_model_for_tier(ComplexityTier.COMPLEX, 0.7, config, request_id=request_id)
        assert decision.request_id == request_id

    def test_generates_request_id_when_not_supplied(self) -> None:
        config = load_routing_config(_REAL_ROUTING_YAML)
        decision = select_model_for_tier(ComplexityTier.COMPLEX, 0.7, config)
        assert decision.request_id is not None
