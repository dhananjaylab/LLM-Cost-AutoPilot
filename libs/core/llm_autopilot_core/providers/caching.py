"""
Prompt-cache breakpoint marking (Phase 6 / Workstream B).

Only Anthropic needs explicit breakpoints (`cache_control`). OpenAI, Google
(implicit caching) and Groq cache matching prefixes automatically, so for them
the only requirement is a *stable prefix* — static system prompt first,
volatile content last. Adapters that don't use the `cache` flag ignore it.
"""

from __future__ import annotations

from llm_autopilot_core.schemas import Message

MAX_BREAKPOINTS = 4  # Anthropic limit


def estimate_tokens(text: str) -> int:
    """Cheap ~4 chars/token estimate; only used to skip sub-minimum prefixes."""
    return len(text) // 4


def mark_cache_breakpoints(messages: list[Message]) -> list[Message]:
    """
    Return messages with `cache=True` set where a breakpoint is useful.
    Caller-supplied flags win: if any message is already flagged, nothing changes.

    Heuristic: cache the last system message (static instructions / RAG context),
    and, in multi-turn chats (>=4 messages), the message before the latest user
    turn so the growing history prefix is reused.
    """
    if any(m.cache for m in messages):
        return messages
    targets: set[int] = set()
    system_idx = [i for i, m in enumerate(messages) if m.role == "system"]
    if system_idx:
        targets.add(system_idx[-1])
    if len(messages) >= 4:
        targets.add(len(messages) - 2)
    return [
        m.model_copy(update={"cache": True}) if i in targets else m for i, m in enumerate(messages)
    ][:]
