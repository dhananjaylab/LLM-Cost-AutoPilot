"""
Phase 2 — complexity classifier.

Public API:
    from llm_autopilot_core.classifier import get_classifier, ClassificationResult

    result = get_classifier().predict("Summarize this article in 3 bullet points")
    result.tier           # ComplexityTier.MODERATE
    result.confidence     # 0.0-1.0, max class probability
    result.probabilities  # dict[ComplexityTier, float], all three classes

Training lives in scripts/train_classifier.py, not in this package — the
package only knows how to *load* an already-trained artifact and predict.
"""

from __future__ import annotations

from llm_autopilot_core.classifier.features import FEATURE_NAMES, extract_features
from llm_autopilot_core.classifier.model import (
    ClassificationResult,
    ComplexityClassifier,
    get_classifier,
)

__all__ = [
    "FEATURE_NAMES",
    "ClassificationResult",
    "ComplexityClassifier",
    "extract_features",
    "get_classifier",
]
