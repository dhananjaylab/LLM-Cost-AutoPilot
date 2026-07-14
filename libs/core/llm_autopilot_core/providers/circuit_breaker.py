"""
Minimal native-asyncio circuit breaker.

pybreaker (still a declared project dependency) IS a closed/open/half-open
circuit breaker library, but its `call_async` path unconditionally
references `tornado.gen` — and `tornado` is guarded behind a silent
`try/except ImportError` at import time (see pybreaker/__init__.py:
`HAS_TORNADO_SUPPORT`), so calling `call_async` without tornado installed
raises `NameError: name 'gen' is not defined` rather than failing fast or
falling back. `tornado` is not a dependency anywhere in this project, and
pulling in a legacy pre-async/await web framework just for one coroutine
decorator isn't worth it in an all-asyncio FastAPI/Celery codebase.

This module reimplements the same closed -> open -> half-open state
machine natively for asyncio, with no new dependencies. pybreaker stays
declared in pyproject.toml in case its *synchronous* `call()` path is
useful elsewhere later (e.g. a sync Celery task).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the breaker is OPEN."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"circuit breaker '{name}' is open")


class AsyncCircuitBreaker:
    """
    Per-provider circuit breaker.

    After `fail_max` consecutive failures, trips OPEN and rejects calls
    for `reset_timeout` seconds. Once that window elapses, the next call
    is let through as a HALF_OPEN probe: success closes the breaker,
    failure re-opens it (and restarts the timeout).
    """

    def __init__(self, *, name: str, fail_max: int = 5, reset_timeout: float = 30.0) -> None:
        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._fail_count = 0
        self._state = BreakerState.CLOSED
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def current_state(self) -> BreakerState:
        if (
            self._state == BreakerState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.reset_timeout
        ):
            return BreakerState.HALF_OPEN
        return self._state

    async def call(self, func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        if self.current_state == BreakerState.OPEN:
            raise CircuitOpenError(self.name)

        try:
            result = await func(*args, **kwargs)
        except Exception:
            await self._on_failure()
            raise
        else:
            await self._on_success()
            return result

    async def _on_success(self) -> None:
        async with self._lock:
            self._fail_count = 0
            self._state = BreakerState.CLOSED
            self._opened_at = None

    async def _on_failure(self) -> None:
        async with self._lock:
            self._fail_count += 1
            if self._fail_count >= self.fail_max:
                self._state = BreakerState.OPEN
                self._opened_at = time.monotonic()
