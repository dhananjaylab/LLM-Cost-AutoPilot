"""
Scheduled Celery tasks.

retrain_classifier     — weekly, Monday 00:00 UTC
    Phase 4 feedback loop: combine the frozen seed dataset with
    (feature_vector, corrected_tier) examples accumulated from verify_response
    escalations since the last retrain, fit a fresh pipeline, shadow-test it
    against a *fixed* holdout slice of the seed data, and promote only if it
    beats the currently-promoted ClassifierVersion's accuracy. Every run is
    recorded (promoted or not) for audit history.

aggregate_daily_costs  — daily, 01:00 UTC
    Phase 5 placeholder — unchanged.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import structlog
from llm_autopilot_core.classifier.features import FEATURE_NAMES, feature_vector
from llm_autopilot_core.classifier.model import get_classifier
from llm_autopilot_core.classifier.training import TrainingResult, train_and_evaluate
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.database import managed_session
from llm_autopilot_core.metrics import celery_tasks_total
from llm_autopilot_core.models import ClassifierVersion, Verification
from sklearn.model_selection import train_test_split
from sqlalchemy import select, update

from llm_autopilot_worker.main import celery_app

logger = structlog.get_logger(__name__)

_SEED_DATA_PATH = Path("data/classifier/training_data.jsonl")
_SEED_HOLDOUT_SIZE = 0.2
# Fixed forever, deliberately — this is what makes accuracy comparable
# retrain over retrain. Do not wire this to config; changing it silently
# invalidates every past ClassifierVersion.accuracy comparison.
_SEED_SPLIT_SEED = 42
_VERSIONED_ARTIFACT_DIR = Path("var/classifier/versions")


def _load_seed_dataset(path: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.str_]]:
    prompts: list[str] = []
    tiers: list[str] = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["prompt"])
            tiers.append(row["tier"])
    x = np.array([feature_vector(p) for p in prompts], dtype=np.float64)
    y = np.array(tiers, dtype=np.str_)
    return x, y


async def _fetch_feedback_examples(
    since: datetime | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.str_]]:
    """
    New (feature_vector, corrected_tier) pairs written by verify_response
    on successful escalations. `since` bounds this to rows created after
    the last retrain so re-running weekly doesn't reprocess the same
    examples every time.
    """
    async with managed_session() as session:
        stmt = select(Verification.feature_vector, Verification.corrected_tier).where(
            Verification.corrected_tier.is_not(None),
            Verification.feature_vector.is_not(None),
        )
        if since is not None:
            stmt = stmt.where(Verification.created_at > since)
        rows = (await session.execute(stmt)).all()

    if not rows:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float64),
            np.empty((0,), dtype=np.str_),
        )

    x_new = np.array([row[0] for row in rows], dtype=np.float64)
    y_new = np.array([row[1].value for row in rows], dtype=np.str_)
    return x_new, y_new


async def _current_promoted_version() -> ClassifierVersion | None:
    async with managed_session() as session:
        stmt = select(ClassifierVersion).where(ClassifierVersion.promoted.is_(True))
        return (await session.execute(stmt)).scalar_one_or_none()


async def _record_version(
    *, result: TrainingResult, artifact_path: str, promoted: bool, notes: str
) -> None:
    async with managed_session() as session:
        if promoted:
            await session.execute(update(ClassifierVersion).values(promoted=False))
        session.add(
            ClassifierVersion(
                accuracy=result.accuracy,
                precision_macro=result.precision_macro,
                recall_macro=result.recall_macro,
                confusion_matrix=result.confusion_matrix,
                training_examples_count=result.training_examples_count,
                artifact_path=artifact_path,
                promoted=promoted,
                promoted_at=datetime.now(UTC) if promoted else None,
                notes=notes,
            )
        )


async def _retrain_classifier_async() -> dict[str, Any]:
    settings = get_settings()

    x_seed, y_seed = _load_seed_dataset(_SEED_DATA_PATH)
    x_seed_train, x_seed_holdout, y_seed_train, y_seed_holdout = train_test_split(
        x_seed,
        y_seed,
        test_size=_SEED_HOLDOUT_SIZE,
        stratify=y_seed,
        random_state=_SEED_SPLIT_SEED,
    )

    current_version = await _current_promoted_version()
    since = current_version.created_at if current_version is not None else None
    x_new, y_new = await _fetch_feedback_examples(since)

    x_train = np.vstack([x_seed_train, x_new]) if len(x_new) else x_seed_train
    y_train = np.concatenate([y_seed_train, y_new]) if len(y_new) else y_seed_train

    training_result = train_and_evaluate(
        x_train,
        y_train,
        x_seed_holdout,
        y_seed_holdout,
        model_type="logistic_regression",
        cv_folds=5,
        seed=_SEED_SPLIT_SEED,
    )

    current_accuracy = current_version.accuracy if current_version is not None else 0.0
    promote = training_result.accuracy > current_accuracy

    # Every attempt gets a versioned artifact for audit history; only a
    # promotion overwrites the canonical path ComplexityClassifier reads.
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    versioned_path = _VERSIONED_ARTIFACT_DIR / f"model_{timestamp}.joblib"
    versioned_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(training_result.pipeline, versioned_path)

    if promote:
        canonical_path = Path(settings.classifier_model_path)
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(training_result.pipeline, canonical_path)
        meta_path = canonical_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "model_type": "logistic_regression",
                    "feature_names": list(FEATURE_NAMES),
                    "labels": training_result.labels,
                    "accuracy": training_result.accuracy,
                    "training_examples_count": training_result.training_examples_count,
                    "trained_at": datetime.now(UTC).isoformat(),
                    "seed": _SEED_SPLIT_SEED,
                },
                indent=2,
            )
        )
        # Only invalidates this worker process's cached pipeline. A live
        # API process won't pick up the new model until its next
        # deploy/restart — see the design note on hot-reload scope.
        get_classifier.cache_clear()

    notes = (
        f"new_feedback_examples={len(x_new)}; "
        f"{'promoted' if promote else 'not promoted (no accuracy improvement)'}"
    )
    await _record_version(
        result=training_result,
        artifact_path=str(versioned_path),
        promoted=promote,
        notes=notes,
    )

    return {
        "status": "promoted" if promote else "not_promoted",
        "accuracy": training_result.accuracy,
        "previous_accuracy": current_accuracy,
        "new_feedback_examples": len(x_new),
        "training_examples_count": training_result.training_examples_count,
    }


@celery_app.task(
    name="llm_autopilot_worker.tasks.retraining.retrain_classifier",
    queue="retraining",
    max_retries=1,
)
def retrain_classifier() -> dict[str, Any]:
    """
    Weekly classifier retraining job. See module docstring for the full
    feedback-loop design.
    """
    log = logger.bind(task="retrain_classifier")
    log.info("retraining_started")

    if not _SEED_DATA_PATH.exists():
        result: dict[str, Any] = {
            "status": "skipped",
            "message": f"seed dataset not found at {_SEED_DATA_PATH}",
        }
        celery_tasks_total.labels(task_name="retrain_classifier", status="success").inc()
        log.warning("retraining_skipped", **result)
        return result

    try:
        result = asyncio.run(_retrain_classifier_async())
        celery_tasks_total.labels(task_name="retrain_classifier", status="success").inc()
        log.info("retraining_complete", **result)
        return result

    except Exception as exc:
        celery_tasks_total.labels(task_name="retrain_classifier", status="failure").inc()
        log.error("retraining_error", error=str(exc), exc_info=True)
        raise


@celery_app.task(
    name="llm_autopilot_worker.tasks.retraining.aggregate_daily_costs",
    queue="retraining",
    max_retries=2,
    default_retry_delay=300,
)
def aggregate_daily_costs() -> dict[str, str]:
    """
    Daily cost aggregation.

    Summarises yesterday's request data into cost_aggregates table.
    Grafana dashboards read from this table for trend panels.

    Phase 5 implementation placeholder — out of scope for Phase 4.
    """
    log = logger.bind(task="aggregate_daily_costs")
    log.info("cost_aggregation_started")

    try:
        # TODO Phase 5:
        # yesterday = date.today() - timedelta(days=1)
        # rows = await db.fetch_requests_for_date(yesterday)
        # aggregate = compute_daily_aggregate(rows)
        # await db.upsert_cost_aggregate(aggregate)
        # update_prometheus_gauges(aggregate)

        result: dict[str, str] = {
            "status": "not_implemented",
            "message": "Phase 5 placeholder — full implementation in Phase 5",
        }

        celery_tasks_total.labels(task_name="aggregate_daily_costs", status="success").inc()
        log.info("cost_aggregation_complete", **result)
        return result

    except Exception as exc:
        celery_tasks_total.labels(task_name="aggregate_daily_costs", status="failure").inc()
        log.error("cost_aggregation_error", error=str(exc), exc_info=True)
        raise
