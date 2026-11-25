"""Phase 6 / Workstream B — prompt caching across adapters."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from llm_autopilot_core.providers.caching import mark_cache_breakpoints
from llm_autopilot_core.providers.dispatcher import _ADAPTERS
from llm_autopilot_core.registry import MODEL_REGISTRY, compute_cost
from llm_autopilot_core.schemas import Message, Provider

SONNET = MODEL_REGISTRY["anthropic/claude-sonnet-5-5"]
LONG = "You are a meticulous analyst. " * 200  # ~1500 tokens, above every minimum but Haiku's


class TestMarkBreakpoints:
    def test_flags_last_system_message(self) -> None:
        out = mark_cache_breakpoints(
            [Message(role="system", content="s"), Message(role="user", content="u")]
        )
        assert [m.cache for m in out] == [True, False]

    def test_caller_flags_are_respected(self) -> None:
        msgs = [Message(role="user", content="u", cache=True), Message(role="system", content="s")]
        assert mark_cache_breakpoints(msgs) is msgs

    def test_multi_turn_marks_history_prefix(self) -> None:
        msgs = [Message(role=r, content="x") for r in ("user", "assistant", "user", "assistant")]
        assert [m.cache for m in mark_cache_breakpoints(msgs)] == [False, False, True, False]

    def test_no_system_short_chat_unchanged(self) -> None:
        out = mark_cache_breakpoints([Message(role="user", content="hi")])
        assert not any(m.cache for m in out)


class TestCostWithCacheBuckets:
    def test_write_and_read_rates(self) -> None:
        cost = compute_cost(SONNET, 10_000, 0, cached_input_tokens=6_000, cache_write_tokens=2_000)
        assert cost == pytest.approx(2_000 * 2e-6 + 6_000 * 2e-7 + 2_000 * 2.5e-6)

    def test_registry_invariants(self) -> None:
        for key, m in MODEL_REGISTRY.items():
            if m.cache_write_cost_per_token is not None:
                assert m.cache_write_cost_per_token >= m.cost_per_input_token, key


def _anthropic_response(**usage: int) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg",
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        usage=SimpleNamespace(output_tokens=5, **usage),
    )


class TestAnthropicAdapter:
    async def _send(self, messages: list[Message], **usage: int):
        captured: dict = {}

        async def _create(**kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return _anthropic_response(**usage)

        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=_create)))
        adapter = _ADAPTERS[Provider.ANTHROPIC]
        with patch.object(adapter, "_get_client", return_value=client):
            resp = await adapter.send(messages, SONNET, max_tokens=64, temperature=0.0)
        return resp, captured

    async def test_long_flagged_system_gets_cache_control(self) -> None:
        resp, sent = await self._send(
            mark_cache_breakpoints(
                [Message(role="system", content=LONG), Message(role="user", content="q")]
            ),
            input_tokens=10,
            cache_read_input_tokens=1_000,
            cache_creation_input_tokens=200,
        )
        assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert resp.input_tokens == 1_210
        assert resp.cached_input_tokens == 1_000
        assert resp.cache_write_tokens == 200

    async def test_short_prefix_below_minimum_is_not_marked(self) -> None:
        _, sent = await self._send(
            mark_cache_breakpoints(
                [Message(role="system", content="Be terse."), Message(role="user", content="q")]
            ),
            input_tokens=10,
        )
        assert sent["system"] == "Be terse."  # legacy string shape preserved

    async def test_missing_cache_usage_fields_default_to_zero(self) -> None:
        resp, _ = await self._send([Message(role="user", content="q")], input_tokens=7)
        assert (resp.cached_input_tokens, resp.cache_write_tokens) == (0, 0)


class TestOpenAIAdapter:
    async def test_cached_tokens_parsed(self) -> None:
        completion = SimpleNamespace(
            id="c",
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=SimpleNamespace(
                prompt_tokens=2_000,
                completion_tokens=10,
                prompt_tokens_details=SimpleNamespace(cached_tokens=1_536),
            ),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=completion))
            )
        )
        adapter = _ADAPTERS[Provider.OPENAI]
        with patch.object(adapter, "_get_client", return_value=client):
            resp = await adapter.send(
                [Message(role="user", content="q")],
                MODEL_REGISTRY["openai/gpt-4o"],
                max_tokens=16,
                temperature=0.0,
            )
        assert resp.input_tokens == 2_000
        assert resp.cached_input_tokens == 1_536
