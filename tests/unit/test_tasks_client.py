from __future__ import annotations

from unittest.mock import MagicMock, patch

from llm_autopilot_core.tasks_client import (
    VERIFY_RESPONSE_TASK_NAME,
    enqueue_verify_response,
    get_task_producer,
)


class TestTaskProducer:
    def test_is_a_cached_singleton(self) -> None:
        get_task_producer.cache_clear()
        try:
            first = get_task_producer()
            second = get_task_producer()
            assert first is second
        finally:
            get_task_producer.cache_clear()

    def test_does_not_import_worker_task_modules(self) -> None:
        # The whole point of this client: constructing it must never pull
        # in llm_autopilot_worker.tasks.* (and therefore never require
        # scikit-learn/joblib in the API image). A successful import of
        # this test module without those packages installed is itself
        # part of the proof; this assertion documents the intent and is
        # robust against other tests that may already have imported worker
        # task modules.
        import sys

        before = set(sys.modules)
        get_task_producer.cache_clear()
        try:
            get_task_producer()
            assert "llm_autopilot_worker.tasks.verification" not in (set(sys.modules) - before)
        finally:
            get_task_producer.cache_clear()


class TestEnqueueVerifyResponse:
    def test_sends_correct_task_name_and_queue(self) -> None:
        fake_producer = MagicMock()
        with patch("llm_autopilot_core.tasks_client.get_task_producer", return_value=fake_producer):
            enqueue_verify_response(request_id="abc-123", prompt="hi", quality_score=0.9)

        fake_producer.send_task.assert_called_once_with(
            VERIFY_RESPONSE_TASK_NAME,
            kwargs={"request_id": "abc-123", "prompt": "hi", "quality_score": 0.9},
            queue="verification",
        )
