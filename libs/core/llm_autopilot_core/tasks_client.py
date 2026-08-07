"""
Lightweight Celery client for enqueueing tasks from the API process.

Deliberately NOT llm_autopilot_worker.main.celery_app. Importing that
module would pull in Celery's full worker configuration (beat schedule,
task module imports) and, transitively, apps/worker's task modules —
which depend on scikit-learn/joblib for classifier retraining, installed
in Dockerfile.worker's image, not Dockerfile.api's. This client only
needs enough Celery to publish a task by name onto the broker; it never
imports task code, so the API image stays lean and the two apps stay
deployable independently.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from celery import Celery

from llm_autopilot_core.config import get_settings

VERIFY_RESPONSE_TASK_NAME = "llm_autopilot_worker.tasks.verification.verify_response"


@lru_cache(maxsize=1)
def get_task_producer() -> Celery:
    settings = get_settings()
    app = Celery("llm_autopilot_producer", broker=settings.celery_broker_url)
    app.conf.task_serializer = "json"
    app.conf.accept_content = ["json"]
    return app


def enqueue_verify_response(**kwargs: Any) -> None:
    """Fire-and-forget: publish onto the 'verification' queue, don't wait on it."""
    get_task_producer().send_task(VERIFY_RESPONSE_TASK_NAME, kwargs=kwargs, queue="verification")
