from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from llm_autopilot_worker.tasks import retraining


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
