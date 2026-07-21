"""
Unit tests for llm_autopilot_core.classifier.model.

Deliberately does NOT depend on the real var/classifier/model.joblib
artifact produced by scripts/train_classifier.py — that keeps these tests
fast, deterministic, and runnable on a fresh checkout before anyone has
trained anything. Each test that needs a "trained" classifier builds a
tiny toy pipeline in-place and saves it to tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from llm_autopilot_core.classifier.features import FEATURE_NAMES
from llm_autopilot_core.classifier.model import ComplexityClassifier, get_classifier
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.schemas import ComplexityTier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_toy_pipeline() -> Pipeline:
    """A tiny pipeline that's trivially separable on one dominant feature."""
    n_features = len(FEATURE_NAMES)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, n_features))
    # Bias feature 0 strongly per class so the toy model is well-separated.
    y = []
    for i in range(30):
        cls = i % 3
        X[i, 0] += cls * 10
        y.append(["complex", "moderate", "simple"][cls])

    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])
    pipeline.fit(X, np.array(y))
    return pipeline


def _save_toy_artifact(tmp_path: Path, *, feature_names: tuple[str, ...] = FEATURE_NAMES) -> Path:
    pipeline = _build_toy_pipeline()
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    meta_path = model_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({"feature_names": list(feature_names)}))
    return model_path


class TestUntrainedFallback:
    def test_missing_artifact_is_not_trained(self) -> None:
        clf = ComplexityClassifier.load("/nonexistent/path/model.joblib")
        assert clf.is_trained is False

    def test_predict_on_untrained_returns_moderate_zero_confidence(self) -> None:
        clf = ComplexityClassifier.load("/nonexistent/path/model.joblib")
        result = clf.predict("Any prompt at all")
        assert result.tier == ComplexityTier.MODERATE
        assert result.confidence == 0.0
        assert result.probabilities == {}


class TestLoadingATrainedArtifact:
    def test_load_and_predict_returns_valid_tier(self, tmp_path: Path) -> None:
        model_path = _save_toy_artifact(tmp_path)
        clf = ComplexityClassifier.load(str(model_path))
        assert clf.is_trained is True

        result = clf.predict("some prompt")
        assert result.tier in set(ComplexityTier)
        assert 0.0 <= result.confidence <= 1.0
        assert set(result.probabilities) <= set(ComplexityTier)
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_missing_metadata_sidecar_still_loads(self, tmp_path: Path) -> None:
        """No .meta.json — should load fine, just skip the schema check."""
        pipeline = _build_toy_pipeline()
        model_path = tmp_path / "model.joblib"
        joblib.dump(pipeline, model_path)

        clf = ComplexityClassifier.load(str(model_path))
        assert clf.is_trained is True

    def test_feature_schema_mismatch_falls_back_to_untrained(self, tmp_path: Path) -> None:
        stale_features = ("only_one_feature",)
        model_path = _save_toy_artifact(tmp_path, feature_names=stale_features)

        clf = ComplexityClassifier.load(str(model_path))
        assert clf.is_trained is False
        result = clf.predict("anything")
        assert result.tier == ComplexityTier.MODERATE


class TestGetClassifierSingleton:
    def test_get_classifier_uses_configured_path(self, tmp_path: Path, monkeypatch) -> None:
        model_path = _save_toy_artifact(tmp_path)
        monkeypatch.setenv("CLASSIFIER_MODEL_PATH", str(model_path))
        get_settings.cache_clear()
        get_classifier.cache_clear()
        try:
            clf = get_classifier()
            assert clf.is_trained is True
        finally:
            get_classifier.cache_clear()
            get_settings.cache_clear()

    def test_get_classifier_is_cached(self, tmp_path: Path, monkeypatch) -> None:
        model_path = _save_toy_artifact(tmp_path)
        monkeypatch.setenv("CLASSIFIER_MODEL_PATH", str(model_path))
        get_settings.cache_clear()
        get_classifier.cache_clear()
        try:
            first = get_classifier()
            second = get_classifier()
            assert first is second
        finally:
            get_classifier.cache_clear()
            get_settings.cache_clear()
