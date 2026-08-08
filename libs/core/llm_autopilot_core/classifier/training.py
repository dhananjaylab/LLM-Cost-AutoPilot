"""
Shared training/evaluation logic for the complexity classifier.

Factored out of scripts/train_classifier.py so both the manual CLI
script and the weekly retrain_classifier Celery task
(apps/worker/.../tasks/retraining.py) run one training implementation
instead of two that could silently drift apart.

Deliberately has no database or Celery imports — it's pure sklearn/numpy
logic, usable from a script, a task, or a notebook alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    accuracy: float
    precision_macro: float
    recall_macro: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    confusion_matrix: dict[str, Any]
    labels: list[str]
    training_examples_count: int


def build_pipeline(model_type: str, seed: int) -> Pipeline:
    classifier: LogisticRegression | RandomForestClassifier
    if model_type == "logistic_regression":
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed
        )
    else:
        raise ValueError(f"unknown model type: {model_type}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def train_and_evaluate(
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.str_],
    x_holdout: npt.NDArray[np.float64],
    y_holdout: npt.NDArray[np.str_],
    *,
    model_type: str = "logistic_regression",
    cv_folds: int = 5,
    seed: int = 42,
) -> TrainingResult:
    """
    Fit on (x_train, y_train), evaluate against a caller-supplied holdout
    split rather than doing an internal train_test_split.

    This lets retrain_classifier() shadow-test against a *fixed* slice of
    the seed dataset across every weekly run — necessary for accuracy
    comparisons to mean anything week over week — while
    scripts/train_classifier.py's CLI usage still passes in its own
    fresh split each invocation.
    """
    pipeline = build_pipeline(model_type, seed)
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_holdout)
    accuracy = float(accuracy_score(y_holdout, y_pred))
    labels = sorted(set(y_train.tolist()) | set(y_holdout.tolist()))
    precision, recall, _f1, _support = precision_recall_fscore_support(
        y_holdout, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_holdout, y_pred, labels=labels)

    # Guard against StratifiedKFold requiring more folds than the
    # smallest class has members — matters most on early retrain runs
    # when accumulated feedback examples are still sparse per tier.
    _, class_counts = np.unique(y_train, return_counts=True)
    effective_cv_folds = max(2, min(cv_folds, int(class_counts.min())))
    cv = StratifiedKFold(n_splits=effective_cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(build_pipeline(model_type, seed), x_train, y_train, cv=cv)

    return TrainingResult(
        pipeline=pipeline,
        accuracy=accuracy,
        precision_macro=float(precision),
        recall_macro=float(recall),
        cv_accuracy_mean=float(cv_scores.mean()),
        cv_accuracy_std=float(cv_scores.std()),
        confusion_matrix={"labels": labels, "matrix": cm.tolist()},
        labels=labels,
        training_examples_count=len(x_train),
    )
