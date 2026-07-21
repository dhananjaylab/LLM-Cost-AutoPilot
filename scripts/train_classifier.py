#!/usr/bin/env python3
"""
Phase 2 — train the complexity classifier.

Loads the labeled dataset, extracts features via
llm_autopilot_core.classifier.features, trains a scikit-learn pipeline,
evaluates it against a held-out split plus stratified k-fold
cross-validation, and saves the fitted pipeline + a metadata sidecar.

Anything above 80% held-out accuracy is the bar the phase doc sets for a
V1 routing skeleton — this script prints a clear PASS/WARN against that,
the same way baseline_test.py prints a pass/fail summary table rather
than silently succeeding or failing.

Usage:
    uv run python scripts/train_classifier.py
    uv run python scripts/train_classifier.py --model random_forest
    uv run python scripts/train_classifier.py --record-db
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libs" / "core"))

from llm_autopilot_core.classifier.features import FEATURE_NAMES, feature_vector  # noqa: E402
from llm_autopilot_core.schemas import ComplexityTier  # noqa: E402

_ACCURACY_TARGET = 0.80
_MIN_EXAMPLES_PER_TIER = 20


def _load_dataset(path: Path) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    tiers: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["prompt"])
            tiers.append(row["tier"])
    return prompts, tiers


def _build_pipeline(model_type: str, seed: int) -> Pipeline:
    if model_type == "logistic_regression":
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed
        )
    else:
        raise ValueError(f"unknown model type: {model_type}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def _print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    col_width = max(len(label) for label in labels) + 2
    header = " " * col_width + "".join(f"{label:>{col_width}}" for label in labels)
    print(header)
    for i, row_label in enumerate(labels):
        cells = "".join(f"{cm[i][j]:>{col_width}}" for j in range(len(labels)))
        print(f"{row_label:<{col_width}}" + cells)


async def _record_classifier_version(
    *,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    confusion: dict[str, object],
    training_examples_count: int,
    artifact_path: str,
) -> None:
    """
    Optional: write a ClassifierVersion row and promote it, unpromoting
    any prior version. Kept isolated from the rest of training so a
    missing/unreachable database degrades to a clear warning rather than
    failing a training run that otherwise succeeded.
    """
    from llm_autopilot_core.database import managed_session
    from llm_autopilot_core.models import ClassifierVersion
    from sqlalchemy import update

    async with managed_session() as session:
        await session.execute(update(ClassifierVersion).values(promoted=False))
        version = ClassifierVersion(
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            confusion_matrix=confusion,
            training_examples_count=training_examples_count,
            artifact_path=artifact_path,
            promoted=True,
            promoted_at=datetime.now(UTC),
            notes="Phase 2 initial training run",
        )
        session.add(version)
    print("Recorded and promoted a new ClassifierVersion row.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/classifier/training_data.jsonl")
    parser.add_argument(
        "--model", choices=["logistic_regression", "random_forest"], default="logistic_regression"
    )
    parser.add_argument("--output", default="var/classifier/model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--record-db",
        action="store_true",
        help="Also write + promote a ClassifierVersion row (requires a reachable database)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    prompts, tiers = _load_dataset(data_path)
    print(f"Loaded {len(prompts)} examples from {data_path}")

    counts = Counter(tiers)
    for tier in ComplexityTier:
        n = counts.get(tier.value, 0)
        flag = "" if n >= _MIN_EXAMPLES_PER_TIER else "  <-- WARN: fewer than 20 examples"
        print(f"  {tier.value:<10} {n}{flag}")

    X = np.array([feature_vector(p) for p in prompts])
    y = np.array(tiers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    pipeline = _build_pipeline(args.model, args.seed)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    labels = sorted(set(tiers))
    precision, recall, _f1, _support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Stratified k-fold CV over the full dataset — more robust estimate
    # than a single train/test split when the dataset is only ~200 rows.
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(_build_pipeline(args.model, args.seed), X, y, cv=cv)

    verdict = "PASS" if accuracy >= _ACCURACY_TARGET else "WARN"
    print()
    print(f"Held-out test accuracy: {accuracy:.3f}  ({verdict} — target is {_ACCURACY_TARGET:.0%})")
    print(f"{args.cv_folds}-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"Macro precision: {precision:.3f}   Macro recall: {recall:.3f}")
    print()
    print("Confusion matrix (rows = actual, columns = predicted):")
    _print_confusion_matrix(cm, labels)
    print()
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    metadata = {
        "model_type": args.model,
        "feature_names": list(FEATURE_NAMES),
        "labels": labels,
        "accuracy": accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "training_examples_count": len(prompts),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved pipeline to {output_path}")
    print(f"Saved metadata to {meta_path}")

    if args.record_db:
        try:
            asyncio.run(
                _record_classifier_version(
                    accuracy=accuracy,
                    precision_macro=float(precision),
                    recall_macro=float(recall),
                    confusion=metadata["confusion_matrix"],
                    training_examples_count=len(prompts),
                    artifact_path=str(output_path),
                )
            )
        except Exception as exc:  # noqa: BLE001 — best-effort, training already succeeded
            print(f"WARNING: could not record ClassifierVersion to the database: {exc}")

    return 0 if accuracy >= _ACCURACY_TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
