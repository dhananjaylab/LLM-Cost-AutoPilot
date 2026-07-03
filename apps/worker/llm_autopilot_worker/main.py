"""
Celery application factory for LLM Cost Autopilot.

Two queues:
  - verification  : async quality checks after every sampled response
  - retraining    : weekly classifier retraining + daily cost aggregation

Beat schedule (run via `celery beat`):
  - Monday 00:00 UTC — retrain classifier from accumulated failures
  - Daily  01:00 UTC — aggregate cost metrics into cost_aggregates table

Start the worker:
    celery -A llm_autopilot_worker.main worker -Q verification,retraining -c 4

Start the beat scheduler (one instance only):
    celery -A llm_autopilot_worker.main beat --scheduler celery.beat:PersistentScheduler

Flower monitoring dashboard:
    celery -A llm_autopilot_worker.main flower --port=5555
"""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready, worker_shutdown
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.logging import configure_logging

# Configure logging immediately on import so tasks and workers get the configured logger factory
configure_logging()

settings = get_settings()

# ── Application ───────────────────────────────────────────────────────────────

celery_app = Celery(
    "llm_autopilot",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "llm_autopilot_worker.tasks.verification",
        "llm_autopilot_worker.tasks.retraining",
    ],
)

# ── Configuration ─────────────────────────────────────────────────────────────

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Reliability
    task_track_started=True,
    task_acks_late=True,  # ack only after task succeeds
    worker_prefetch_multiplier=1,  # one task at a time per worker slot
    task_reject_on_worker_lost=True,
    # Timeouts
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_time_limit=settings.celery_task_time_limit,
    # Connection Limits (Defensive against Redis Cloud free-tier connection limits)
    broker_pool_limit=2,
    redis_max_connections=5,
    # Result expiry
    result_expires=86_400,  # 24 hours
    # Queue routing
    task_routes={
        "llm_autopilot_worker.tasks.verification.*": {"queue": "verification"},
        "llm_autopilot_worker.tasks.retraining.*": {"queue": "retraining"},
    },
    task_default_queue="verification",
    task_queues={
        "verification": {"exchange": "verification", "routing_key": "verification"},
        "retraining": {"exchange": "retraining", "routing_key": "retraining"},
    },
    # Beat schedule
    beat_schedule={
        "retrain-classifier-weekly": {
            "task": "llm_autopilot_worker.tasks.retraining.retrain_classifier",
            "schedule": crontab(hour=0, minute=0, day_of_week=1),  # Monday midnight UTC
            "options": {"queue": "retraining"},
        },
        "aggregate-costs-daily": {
            "task": "llm_autopilot_worker.tasks.retraining.aggregate_daily_costs",
            "schedule": crontab(hour=1, minute=0),  # 01:00 UTC daily
            "options": {"queue": "retraining"},
        },
    },
)


# ── Signals ───────────────────────────────────────────────────────────────────


@worker_ready.connect
def on_worker_ready(**kwargs: object) -> None:  # noqa: ARG001
    import structlog

    structlog.get_logger(__name__).info(
        "worker_ready",
        broker=settings.celery_broker_url,
    )


@worker_shutdown.connect
def on_worker_shutdown(**kwargs: object) -> None:  # noqa: ARG001
    import structlog

    structlog.get_logger(__name__).info("worker_shutdown")
