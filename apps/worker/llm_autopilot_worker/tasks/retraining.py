"""
Scheduled Celery tasks.

retrain_classifier     — weekly, Monday 00:00 UTC
    1. Fetch all verification failures since last retrain
    2. Merge with original training dataset
    3. Re-fit scikit-learn classifier
    4. Shadow-test new model vs current on held-out set
    5. Promote if accuracy improves; log either way
    6. Version the new model artifact in DB

aggregate_daily_costs  — daily, 01:00 UTC
    1. Sum cost/savings/request counts for the previous calendar day
    2. Write to cost_aggregates table
    3. Update Prometheus gauges so Grafana shows trailing 30-day trend

Phase 4 / Phase 5 TODO:
    - Full implementation of both tasks
    - Model versioning via classifier_versions table
    - Shadow testing logic
"""

from __future__ import annotations

import structlog
from llm_autopilot_core.metrics import celery_tasks_total

from llm_autopilot_worker.main import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(
    name="llm_autopilot_worker.tasks.retraining.retrain_classifier",
    queue="retraining",
    max_retries=1,
)
def retrain_classifier() -> dict[str, str]:
    """
    Weekly classifier retraining job.

    Pulls accumulated verification failures from PostgreSQL, merges them
    with the seed training set, retrains the scikit-learn complexity
    classifier, shadow-tests it, and promotes if metrics improve.

    Phase 4 implementation placeholder.
    """
    log = logger.bind(task="retrain_classifier")
    log.info("retraining_started")

    try:
        # TODO Phase 4:
        # failures = await fetch_verification_failures_since_last_retrain()
        # X_new, y_new = extract_features(failures)
        # X_all = np.vstack([X_seed, X_new])
        # y_all = np.concatenate([y_seed, y_new])
        # new_model = train_classifier(X_all, y_all)
        # metrics = shadow_test(new_model, current_model, X_holdout, y_holdout)
        # if metrics["accuracy"] > current_metrics["accuracy"]:
        #     promote(new_model)
        # log_classifier_version(metrics)

        result: dict[str, str] = {
            "status": "not_implemented",
            "message": "Phase 4 placeholder — full implementation in Phase 4",
        }

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

    Phase 5 implementation placeholder.
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
