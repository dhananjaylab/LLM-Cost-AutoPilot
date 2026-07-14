"""Direct tests for the native asyncio circuit breaker (no mocked SDKs needed)."""

from __future__ import annotations

import asyncio

import pytest
from llm_autopilot_core.providers.circuit_breaker import (
    AsyncCircuitBreaker,
    BreakerState,
    CircuitOpenError,
)


class TestAsyncCircuitBreaker:
    async def test_starts_closed(self) -> None:
        breaker = AsyncCircuitBreaker(name="test", fail_max=3, reset_timeout=30.0)
        assert breaker.current_state == BreakerState.CLOSED

    async def test_successful_calls_keep_it_closed(self) -> None:
        breaker = AsyncCircuitBreaker(name="test", fail_max=2, reset_timeout=30.0)

        async def ok() -> str:
            return "fine"

        for _ in range(5):
            assert await breaker.call(ok) == "fine"
        assert breaker.current_state == BreakerState.CLOSED

    async def test_trips_open_after_fail_max_consecutive_failures(self) -> None:
        breaker = AsyncCircuitBreaker(name="test", fail_max=3, reset_timeout=30.0)

        async def boom() -> None:
            raise RuntimeError("provider down")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(boom)

        assert breaker.current_state == BreakerState.OPEN

    async def test_open_breaker_rejects_calls_without_invoking_func(self) -> None:
        breaker = AsyncCircuitBreaker(name="test", fail_max=1, reset_timeout=30.0)
        calls = 0

        async def boom() -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("provider down")

        with pytest.raises(RuntimeError):
            await breaker.call(boom)
        assert breaker.current_state == BreakerState.OPEN
        assert calls == 1

        # Second call should be rejected by the breaker itself — func must not run again.
        with pytest.raises(CircuitOpenError):
            await breaker.call(boom)
        assert calls == 1

    async def test_half_open_after_reset_timeout_then_closes_on_success(self) -> None:
        breaker = AsyncCircuitBreaker(name="test", fail_max=1, reset_timeout=0.05)

        async def boom() -> None:
            raise RuntimeError("down")

        async def ok() -> str:
            return "recovered"

        with pytest.raises(RuntimeError):
            await breaker.call(boom)
        assert breaker.current_state == BreakerState.OPEN

        await asyncio.sleep(0.06)
        assert breaker.current_state == BreakerState.HALF_OPEN

        result = await breaker.call(ok)
        assert result == "recovered"
        assert breaker.current_state == BreakerState.CLOSED

    async def test_half_open_probe_failure_reopens_the_breaker(self) -> None:
        breaker = AsyncCircuitBreaker(name="test", fail_max=1, reset_timeout=0.05)

        async def boom() -> None:
            raise RuntimeError("still down")

        with pytest.raises(RuntimeError):
            await breaker.call(boom)
        await asyncio.sleep(0.06)
        assert breaker.current_state == BreakerState.HALF_OPEN

        with pytest.raises(RuntimeError):
            await breaker.call(boom)
        assert breaker.current_state == BreakerState.OPEN

    async def test_error_message_includes_breaker_name(self) -> None:
        err = CircuitOpenError("anthropic")
        assert "anthropic" in str(err)
