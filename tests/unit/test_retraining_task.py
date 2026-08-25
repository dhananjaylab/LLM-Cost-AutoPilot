from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from llm_autopilot_core.schemas import ModelConfig, Provider, QualityTier
from llm_autopilot_worker.tasks import retraining
from llm_autopilot_worker.tasks.retraining import _compute_daily_aggregate, _DailyRollupInputs


def _write_seed_dataset(path: Path) -> None:
    rows = (
        [{"prompt": f"What is {i}+{i}?", "tier": "simple"} for i in range(15)]
        + [
            {"prompt": f"Summarize article number {i} in one sentence.", "tier": "moderate"}
            for i in range(15)
        ]
        + [
            {"prompt": f"Analyze scenario {i} and recommend a course of action.", "tier": "complex"}
            for i in range(15)
        ]
    )
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestRetrainClassifierAsync:
    async def test_promotes_when_no_prior_version_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed_path = tmp_path / "training_data.jsonl"
        _write_seed_dataset(seed_path)
        monkeypatch.setattr(retraining, "_SEED_DATA_PATH", seed_path)
        monkeypatch.setattr(retraining, "_VERSIONED_ARTIFACT_DIR", tmp_path / "versions")

        monkeypatch.setattr(retraining, "_current_promoted_version", AsyncMock(return_value=None))
        monkeypatch.setattr(
            retraining,
            "_fetch_feedback_examples",
            AsyncMock(return_value=(np.empty((0, 11)), np.empty((0,), dtype=str))),
        )
        record_mock = AsyncMock()
        monkeypatch.setattr(retraining, "_record_version", record_mock)

        settings = MagicMock()
        settings.classifier_model_path = str(tmp_path / "model.joblib")
        monkeypatch.setattr(retraining, "get_settings", lambda: settings)
        monkeypatch.setattr(retraining, "get_classifier", MagicMock(cache_clear=MagicMock()))

        result = await retraining._retrain_classifier_async()

        assert result["status"] == "promoted"
        assert result["previous_accuracy"] == 0.0
        assert (tmp_path / "model.joblib").exists()
        assert (tmp_path / "model.meta.json").exists()
        record_mock.assert_awaited_once()

    async def test_does_not_promote_when_new_accuracy_is_not_better(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed_path = tmp_path / "training_data.jsonl"
        _write_seed_dataset(seed_path)
        monkeypatch.setattr(retraining, "_SEED_DATA_PATH", seed_path)
        monkeypatch.setattr(retraining, "_VERSIONED_ARTIFACT_DIR", tmp_path / "versions")

        current_version = MagicMock(accuracy=1.0, created_at=datetime.now(UTC))
        monkeypatch.setattr(
            retraining, "_current_promoted_version", AsyncMock(return_value=current_version)
        )
        monkeypatch.setattr(
            retraining,
            "_fetch_feedback_examples",
            AsyncMock(return_value=(np.empty((0, 11)), np.empty((0,), dtype=str))),
        )
        record_mock = AsyncMock()
        monkeypatch.setattr(retraining, "_record_version", record_mock)

        settings = MagicMock()
        settings.classifier_model_path = str(tmp_path / "model.joblib")
        monkeypatch.setattr(retraining, "get_settings", lambda: settings)

        result = await retraining._retrain_classifier_async()

        assert result["status"] == "not_promoted"
        assert not (tmp_path / "model.joblib").exists()
        # A versioned artifact is still written for audit history even when
        # not promoted.
        assert any((tmp_path / "versions").iterdir())
        record_mock.assert_awaited_once()

    async def test_seed_dataset_missing_short_circuits_before_any_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(retraining, "_SEED_DATA_PATH", tmp_path / "does-not-exist.jsonl")

        result = retraining.retrain_classifier()

        assert result["status"] == "skipped"


# ── Daily cost aggregation (Phase 5) ─────────────────────────────────────────────


def _baseline_model() -> ModelConfig:
    return ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        display_name="GPT-4o",
        cost_per_input_token=0.00005,  # $0.00005 per input token
        cost_per_output_token=0.0002,  # $0.0002 per output token
        avg_latency_ms=1_800,
        quality_tier=QualityTier.HIGH,
        context_window=128_000,
        max_output_tokens=4_096,
    )


class TestComputeDailyAggregate:
    def test_basic_rollup_with_baseline(self) -> None:
        inputs = _DailyRollupInputs(
            total_requests=10,
            cache_hits=3,
            response_rows=[
                (0.01, 100, 50, "simple", "groq"),
                (0.02, 200, 100, "moderate", "openai"),
            ],
            verification_rows=[("passed", 0.9), ("escalated", 0.3)],
        )
        result = _compute_daily_aggregate(inputs, baseline_model=_baseline_model())

        assert result["cache_hit_rate"] == pytest.approx(30.0)
        assert result["escalation_rate"] == pytest.approx(10.0)
        assert result["avg_quality_score"] == pytest.approx(0.6)
        assert result["requests_by_tier"] == {"simple": 1, "moderate": 1}
        assert result["requests_by_provider"] == {"groq": 1, "openai": 1}
        assert result["total_cost_usd"] == pytest.approx(0.03)
        # baseline (gpt-4o) is pricier than what was actually spent
        assert result["hypothetical_cost_usd"] > result["total_cost_usd"]
        assert result["cost_savings_usd"] == pytest.approx(
            result["hypothetical_cost_usd"] - result["total_cost_usd"]
        )

    def test_zero_requests_does_not_divide_by_zero(self) -> None:
        result = _compute_daily_aggregate(
            _DailyRollupInputs(
                total_requests=0, cache_hits=0, response_rows=[], verification_rows=[]
            ),
            baseline_model=None,
        )
        assert result["cache_hit_rate"] == 0.0
        assert result["escalation_rate"] == 0.0
        assert result["avg_quality_score"] == 0.0
        assert result["hypothetical_cost_usd"] == 0.0
        assert result["cost_savings_usd"] == 0.0

    def test_no_baseline_model_gives_zero_hypothetical_cost(self) -> None:
        inputs = _DailyRollupInputs(
            total_requests=1,
            cache_hits=0,
            response_rows=[(0.01, 100, 50, "simple", "groq")],
            verification_rows=[],
        )
        result = _compute_daily_aggregate(inputs, baseline_model=None)
        assert result["hypothetical_cost_usd"] == 0.0
        assert result["cost_savings_usd"] == 0.0  # can't be negative from a missing baseline

    def test_no_verifications_gives_zero_avg_quality_and_escalation(self) -> None:
        inputs = _DailyRollupInputs(
            total_requests=5,
            cache_hits=1,
            response_rows=[(0.01, 100, 50, "simple", "groq")] * 5,
            verification_rows=[],
        )
        result = _compute_daily_aggregate(inputs, baseline_model=_baseline_model())
        assert result["avg_quality_score"] == 0.0
        assert result["escalation_rate"] == 0.0
        assert result["cache_hit_rate"] == pytest.approx(20.0)
