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
    Phase 5: rolls up the previous UTC day's requests/responses/verifications
    into a single cost_aggregates row (upserted, so reruns for the same day
    stay idempotent) — see GET /v1/stats (apps/api/.../routers/stats.py),
    which reads only from this table rather than aggregating the raw tables
    live on every call.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
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
from llm_autopilot_core.models import (
    ClassifierVersion,
    CostAggregate,
    Request,
    Response,
    Verification,
)
from llm_autopilot_core.registry import compute_cost, get_model
from llm_autopilot_core.routing import get_routing_config
from llm_autopilot_core.schemas import ModelConfig, VerificationStatus
from sklearn.model_selection import train_test_split
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

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


# ── Daily cost aggregation (Phase 5) ─────────────────────────────────────────────


@dataclass(frozen=True)
class _DailyRollupInputs:
    """Raw per-day rows, already unwrapped from SQLAlchemy Row objects into
    plain Python values, so _compute_daily_aggregate() below can be unit
    tested without a database."""

    total_requests: int
    cache_hits: int
    # (cost_usd, input_tokens, output_tokens, complexity_tier, provider)
    response_rows: list[tuple[float, int, int, str, str]]
    # (status, quality_score)
    verification_rows: list[tuple[str, float]]


def _compute_daily_aggregate(
    inputs: _DailyRollupInputs, *, baseline_model: ModelConfig | None
) -> dict[str, Any]:
    """
    Pure rollup logic — no I/O. Mirrors the Prometheus recording rules'
    conventions (infra/prometheus/rules/recording_rules.yml) so this
    table and the live dashboards agree on what "escalation rate" and
    "cache hit rate" mean: a percentage of *all requests* that day, not
    just the sampled/verified subset.
    """
    total_cost_usd = sum(row[0] for row in inputs.response_rows)
    hypothetical_cost_usd = (
        sum(compute_cost(baseline_model, row[1], row[2]) for row in inputs.response_rows)
        if baseline_model is not None
        else 0.0
    )
    cost_savings_usd = max(0.0, hypothetical_cost_usd - total_cost_usd)

    requests_by_tier: dict[str, int] = {}
    requests_by_provider: dict[str, int] = {}
    for _cost, _in_tok, _out_tok, tier, provider in inputs.response_rows:
        requests_by_tier[tier] = requests_by_tier.get(tier, 0) + 1
        requests_by_provider[provider] = requests_by_provider.get(provider, 0) + 1

    escalated_count = sum(
        1
        for status, _score in inputs.verification_rows
        if status == VerificationStatus.ESCALATED.value
    )
    quality_scores = [score for _status, score in inputs.verification_rows]

    total_requests = inputs.total_requests
    cache_hit_rate = (inputs.cache_hits / total_requests * 100) if total_requests else 0.0
    escalation_rate = (escalated_count / total_requests * 100) if total_requests else 0.0
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return {
        "total_requests": total_requests,
        "total_cost_usd": total_cost_usd,
        "hypothetical_cost_usd": hypothetical_cost_usd,
        "cost_savings_usd": cost_savings_usd,
        "cache_hit_rate": cache_hit_rate,
        "escalation_rate": escalation_rate,
        "avg_quality_score": avg_quality_score,
        "requests_by_tier": requests_by_tier,
        "requests_by_provider": requests_by_provider,
    }


async def _fetch_daily_rollup_inputs(target_date: date) -> _DailyRollupInputs:
    start = datetime.combine(target_date, time.min, tzinfo=UTC)
    end = start + timedelta(days=1)

    async with managed_session() as session:
        total_requests = (
            await session.scalar(
                select(func.count())
                .select_from(Request)
                .where(Request.created_at >= start, Request.created_at < end)
            )
        ) or 0
        cache_hits = (
            await session.scalar(
                select(func.count())
                .select_from(Request)
                .where(
                    Request.created_at >= start,
                    Request.created_at < end,
                    Request.cache_hit.is_(True),
                )
            )
        ) or 0

        response_rows_raw = (
            await session.execute(
                select(
                    Response.cost_usd,
                    Response.input_tokens,
                    Response.output_tokens,
                    Response.complexity_tier,
                    Response.provider,
                ).where(Response.created_at >= start, Response.created_at < end)
            )
        ).all()
        response_rows = [
            (row[0], row[1], row[2], row[3].value, row[4].value) for row in response_rows_raw
        ]

        verification_rows_raw = (
            await session.execute(
                select(Verification.status, Verification.quality_score).where(
                    Verification.created_at >= start, Verification.created_at < end
                )
            )
        ).all()
        verification_rows = [(row[0].value, row[1]) for row in verification_rows_raw]

    return _DailyRollupInputs(
        total_requests=total_requests,
        cache_hits=cache_hits,
        response_rows=response_rows,
        verification_rows=verification_rows,
    )


async def _upsert_cost_aggregate(target_date: date, aggregate: dict[str, Any]) -> None:
    async with managed_session() as session:
        stmt = pg_insert(CostAggregate).values(date=target_date, **aggregate)
        update_cols = {key: getattr(stmt.excluded, key) for key in aggregate}
        update_cols["updated_at"] = func.now()
        stmt = stmt.on_conflict_do_update(index_elements=["date"], set_=update_cols)
        await session.execute(stmt)


async def _aggregate_daily_costs_async(target_date: date) -> dict[str, Any]:
    routing_config = get_routing_config()
    baseline_model = get_model(routing_config.cost_baseline.model)

    inputs = await _fetch_daily_rollup_inputs(target_date)
    aggregate = _compute_daily_aggregate(inputs, baseline_model=baseline_model)
    await _upsert_cost_aggregate(target_date, aggregate)

    return {"date": target_date.isoformat(), **aggregate}


@celery_app.task(
    name="llm_autopilot_worker.tasks.retraining.aggregate_daily_costs",
    queue="retraining",
    max_retries=2,
    default_retry_delay=300,
)
def aggregate_daily_costs() -> dict[str, Any]:
    """
    Daily cost aggregation.

    Summarises yesterday's request data into the cost_aggregates table.
    GET /v1/stats reads from this table; Grafana's cost_overview
    dashboard reads live Prometheus counters instead and is unaffected
    by this task.
    """
    log = logger.bind(task="aggregate_daily_costs")
    log.info("cost_aggregation_started")

    target_date = datetime.now(UTC).date() - timedelta(days=1)

    try:
        result = asyncio.run(_aggregate_daily_costs_async(target_date))
        celery_tasks_total.labels(task_name="aggregate_daily_costs", status="success").inc()
        log.info("cost_aggregation_complete", **result)
        return result

    except Exception as exc:
        celery_tasks_total.labels(task_name="aggregate_daily_costs", status="failure").inc()
        log.error("cost_aggregation_error", error=str(exc), exc_info=True)
        raise
