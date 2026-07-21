"""
Complexity classifier inference — loads a trained sklearn pipeline and
turns a prompt into a (tier, confidence) prediction.

Training happens offline in scripts/train_classifier.py; that's the only
place FEATURE_NAMES ordering and the fitted pipeline need to agree. This
module trusts the artifact's metadata sidecar to confirm that agreement
before using it, and falls back to a neutral default (rather than raising)
if the artifact is missing or its feature schema has drifted — a fresh
checkout that hasn't run train_classifier.py yet should degrade, not crash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import structlog
from sklearn.pipeline import Pipeline

from llm_autopilot_core.classifier.features import FEATURE_NAMES, feature_vector
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.schemas import ComplexityTier

logger = structlog.get_logger(__name__)

# Confidence returned when no trained artifact is available yet. Callers
# (e.g. Phase 3 verification sampling) treat confidence <=
# classifier_confidence_threshold as "low confidence", so 0.0 correctly
# forces the conservative/verify-everything path rather than silently
# looking trustworthy.
_UNTRAINED_FALLBACK_TIER = ComplexityTier.MODERATE
_UNTRAINED_FALLBACK_CONFIDENCE = 0.0


@dataclass(frozen=True)
class ClassificationResult:
    tier: ComplexityTier
    confidence: float
    probabilities: dict[ComplexityTier, float]


def _metadata_path_for(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.json")


class ComplexityClassifier:
    """
    Wraps a fitted sklearn Pipeline (StandardScaler + classifier) trained
    on the FEATURE_NAMES feature vector. Stateless beyond the loaded
    pipeline — safe to share as a module-level singleton.
    """

    def __init__(self, pipeline: Pipeline | None, feature_names: tuple[str, ...]) -> None:
        self._pipeline = pipeline
        self._feature_names = feature_names

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    @classmethod
    def load(cls, model_path: str) -> ComplexityClassifier:
        path = Path(model_path)
        if not path.exists():
            logger.warning(
                "classifier_artifact_missing",
                path=model_path,
                fallback_tier=_UNTRAINED_FALLBACK_TIER.value,
            )
            return cls(pipeline=None, feature_names=FEATURE_NAMES)

        pipeline: Pipeline = joblib.load(path)

        feature_names = FEATURE_NAMES
        meta_path = _metadata_path_for(path)
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            trained_features = tuple(metadata.get("feature_names", FEATURE_NAMES))
            if trained_features != FEATURE_NAMES:
                logger.error(
                    "classifier_feature_schema_mismatch",
                    trained=trained_features,
                    current=FEATURE_NAMES,
                )
                return cls(pipeline=None, feature_names=FEATURE_NAMES)
            feature_names = trained_features

        logger.info("classifier_loaded", path=model_path, feature_count=len(feature_names))
        return cls(pipeline=pipeline, feature_names=feature_names)

    def predict(self, prompt: str) -> ClassificationResult:
        if self._pipeline is None:
            return ClassificationResult(
                tier=_UNTRAINED_FALLBACK_TIER,
                confidence=_UNTRAINED_FALLBACK_CONFIDENCE,
                probabilities={},
            )

        vector = [feature_vector(prompt)]
        proba = self._pipeline.predict_proba(vector)[0]
        classes = self._pipeline.classes_

        probabilities = {
            ComplexityTier(str(cls_label)): float(p)
            for cls_label, p in zip(classes, proba, strict=True)
        }
        tier = max(probabilities, key=lambda t: probabilities[t])
        confidence = probabilities[tier]

        return ClassificationResult(tier=tier, confidence=confidence, probabilities=probabilities)


@lru_cache(maxsize=1)
def get_classifier() -> ComplexityClassifier:
    """Cached singleton. Call get_classifier.cache_clear() in tests / after retraining."""
    settings = get_settings()
    return ComplexityClassifier.load(settings.classifier_model_path)
