"""
Per-task-category scoring strategies for the async verification loop.

Each strategy calls the judge model (and, for pairwise categories, a
comparison model) via an injectable `send_fn` — defaults to
providers.dispatcher.send_request, but tests substitute a fake async
callable so this logic is testable without touching real provider SDKs.

All strategies return a ScoringResult with quality_score normalized to
[0, 1]. `escalation_candidate_*` is populated only by the pairwise
strategy (creative/reasoning): that comparison call already used a
top-tier model, so verify_response reuses its output directly instead of
paying for a second, redundant escalation rerun.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog

from llm_autopilot_core.providers.dispatcher import send_request as _default_send_request
from llm_autopilot_core.registry import get_model
from llm_autopilot_core.routing import RoutingConfig, select_model_for_tier
from llm_autopilot_core.schemas import (
    ComplexityTier,
    Message,
    ModelConfig,
    Provider,
    ProviderResponse,
)
from llm_autopilot_core.verification.task_category import TaskCategory

logger = structlog.get_logger(__name__)

SendRequestFn = Callable[..., Awaitable[ProviderResponse]]

# Per-category overrides of config.escalation_quality_threshold.
#   - EXTRACTION / CLASSIFICATION: no partial credit for a missed field
#     or a wrong label, so the bar is high (0.9).
#   - CREATIVE / REASONING: pairwise scoring maps win=1.0/tie=0.6/loss=0.0
#     (see _score_pairwise) — threshold must sit strictly between 0.0 and
#     0.6 so a tie passes and only a clear loss escalates.
# Any category not listed falls back to the global config default, which
# fits SUMMARIZATION: a judge score of 4/5 maps to 0.75, matching the
# phase spec's "score above 4/5" bar almost exactly.
CATEGORY_THRESHOLDS: dict[TaskCategory, float] = {
    TaskCategory.EXTRACTION: 0.9,
    TaskCategory.CLASSIFICATION: 0.9,
    TaskCategory.CREATIVE: 0.5,
    TaskCategory.REASONING: 0.5,
}


def get_threshold_for_category(category: TaskCategory, default_threshold: float) -> float:
    return CATEGORY_THRESHOLDS.get(category, default_threshold)


def is_self_judge(
    original_provider: Provider, original_model_id: str, judge_config: ModelConfig
) -> bool:
    """
    True when the model being verified IS the judge model.

    MODERATE tier's fallback chain currently ends on the same model
    configured as judge (anthropic/claude-haiku-4-5) — if every earlier
    provider in that chain is circuit-tripped, a request can legitimately
    land there, and letting it self-judge would reintroduce the exact
    same-family bias this guard exists to avoid.
    """
    return original_provider == judge_config.provider and original_model_id == judge_config.model_id


@dataclass(frozen=True)
class ScoringResult:
    quality_score: float
    judge_output: str
    escalation_candidate_content: str | None = None
    escalation_candidate_model_id: str | None = None
    escalation_candidate_provider: Provider | None = None
    escalation_candidate_cost_usd: float = 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip().lower()


# ── Extraction ────────────────────────────────────────────────────────────

_EXTRACTION_JUDGE_PROMPT = (
    "Extract every discrete factual value the request below is asking "
    "for (names, numbers, dates, emails, labels, etc). Respond with one "
    "'key: value' pair per line and nothing else.\n\nRequest:\n{prompt}"
)


async def _score_extraction(
    *,
    prompt: str,
    original_response: str,
    judge_config: ModelConfig,
    send_fn: SendRequestFn,
) -> ScoringResult:
    judge_resp = await send_fn(
        _EXTRACTION_JUDGE_PROMPT.format(prompt=prompt),
        judge_config,
        max_tokens=256,
        temperature=0.0,
    )
    judge_values = [
        line.split(":", 1)[1].strip() for line in judge_resp.content.splitlines() if ":" in line
    ]
    if not judge_values:
        # Judge extracted nothing to compare against — inconclusive, not
        # a failure the original model should be penalized for.
        return ScoringResult(quality_score=1.0, judge_output=judge_resp.content)

    original_norm = _normalize(original_response)
    matched = sum(1 for v in judge_values if _normalize(v) and _normalize(v) in original_norm)
    return ScoringResult(
        quality_score=_clamp01(matched / len(judge_values)), judge_output=judge_resp.content
    )


# ── Classification ────────────────────────────────────────────────────────

_CLASSIFICATION_JUDGE_PROMPT = (
    "Answer the classification request below with ONLY the label — no "
    "explanation, no punctuation.\n\nRequest:\n{prompt}"
)


async def _score_classification(
    *,
    prompt: str,
    original_response: str,
    judge_config: ModelConfig,
    send_fn: SendRequestFn,
) -> ScoringResult:
    judge_resp = await send_fn(
        _CLASSIFICATION_JUDGE_PROMPT.format(prompt=prompt),
        judge_config,
        max_tokens=16,
        temperature=0.0,
    )
    judge_label = _normalize(judge_resp.content)
    original_norm = _normalize(original_response)
    matched = bool(judge_label) and judge_label in original_norm
    return ScoringResult(quality_score=1.0 if matched else 0.0, judge_output=judge_resp.content)


# ── Summarization / structured analysis (G-Eval style) ──────────────────────

_SUMMARY_JUDGE_PROMPT = (
    "You are grading a response for faithfulness and coverage against "
    "the request below. Consider what the request needed, then finish "
    "with a single line in the exact form 'SCORE: <1-5>' (5 = fully "
    "correct and complete, 1 = wrong or missing the point).\n\n"
    "Request:\n{prompt}\n\nResponse to grade:\n{response}"
)
_SCORE_RE = re.compile(r"SCORE:\s*(\d)", re.IGNORECASE)


async def _score_summary(
    *,
    prompt: str,
    original_response: str,
    judge_config: ModelConfig,
    send_fn: SendRequestFn,
) -> ScoringResult:
    judge_resp = await send_fn(
        _SUMMARY_JUDGE_PROMPT.format(prompt=prompt, response=original_response),
        judge_config,
        max_tokens=256,
        temperature=0.0,
    )
    match = _SCORE_RE.search(judge_resp.content)
    if match is None:
        logger.warning("judge_score_unparseable", judge_output=judge_resp.content[:200])
        return ScoringResult(quality_score=0.5, judge_output=judge_resp.content)
    raw_score = int(match.group(1))
    return ScoringResult(
        quality_score=_clamp01((raw_score - 1) / 4), judge_output=judge_resp.content
    )


# ── Creative / reasoning — true pairwise ─────────────────────────────────────

_COMPARISON_SYSTEM_PROMPT = "Respond directly and completely to the user's request."
_PAIRWISE_JUDGE_PROMPT = (
    "Compare Response A and Response B for the request below and decide "
    "which better satisfies it. Respond with exactly one word: 'A', 'B', "
    "or 'TIE'.\n\nRequest:\n{prompt}\n\nResponse A:\n{a}\n\nResponse B:\n{b}"
)


async def _judge_pairwise_once(
    *, prompt: str, a: str, b: str, judge_config: ModelConfig, send_fn: SendRequestFn
) -> str:
    judge_resp = await send_fn(
        _PAIRWISE_JUDGE_PROMPT.format(prompt=prompt, a=a, b=b),
        judge_config,
        max_tokens=8,
        temperature=0.0,
    )
    verdict = _normalize(judge_resp.content)
    if verdict.startswith("a"):
        return "A"
    if verdict.startswith("b"):
        return "B"
    return "TIE"


async def _score_pairwise(
    *,
    prompt: str,
    original_response: str,
    judge_config: ModelConfig,
    routing_config: RoutingConfig,
    send_fn: SendRequestFn,
) -> ScoringResult:
    comparison_decision = select_model_for_tier(ComplexityTier.COMPLEX, 1.0, routing_config)
    comparison_config = get_model(
        f"{comparison_decision.selected_provider.value}/{comparison_decision.selected_model_id}"
    )
    if comparison_config is None:
        # Registry/routing drift — fail safe rather than crash the task.
        return ScoringResult(quality_score=1.0, judge_output="comparison model unavailable")

    comparison_resp = await send_fn(
        [
            Message(role="system", content=_COMPARISON_SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ],
        comparison_config,
        max_tokens=1024,
        temperature=0.7,
    )

    # Two passes with swapped position labels, to cancel judge position bias.
    verdict_1 = await _judge_pairwise_once(
        prompt=prompt,
        a=original_response,
        b=comparison_resp.content,
        judge_config=judge_config,
        send_fn=send_fn,
    )
    verdict_2 = await _judge_pairwise_once(
        prompt=prompt,
        a=comparison_resp.content,
        b=original_response,
        judge_config=judge_config,
        send_fn=send_fn,
    )
    original_won = verdict_1 == "A" and verdict_2 == "B"
    comparison_won = verdict_1 == "B" and verdict_2 == "A"

    if original_won:
        quality_score = 1.0
    elif comparison_won:
        quality_score = 0.0
    else:
        # Orderings disagree, or either pass was a TIE — call it a tie.
        quality_score = 0.6

    judge_output = f"pass1={verdict_1} pass2={verdict_2}"

    if quality_score == 0.0:
        return ScoringResult(
            quality_score=quality_score,
            judge_output=judge_output,
            escalation_candidate_content=comparison_resp.content,
            escalation_candidate_model_id=comparison_config.model_id,
            escalation_candidate_provider=comparison_config.provider,
            escalation_candidate_cost_usd=comparison_resp.cost_usd,
        )
    return ScoringResult(quality_score=quality_score, judge_output=judge_output)


# ── Dispatcher ────────────────────────────────────────────────────────────


async def score_response(
    *,
    task_category: TaskCategory,
    prompt: str,
    original_response: str,
    judge_config: ModelConfig,
    routing_config: RoutingConfig,
    send_fn: SendRequestFn = _default_send_request,
) -> ScoringResult:
    if task_category is TaskCategory.EXTRACTION:
        return await _score_extraction(
            prompt=prompt,
            original_response=original_response,
            judge_config=judge_config,
            send_fn=send_fn,
        )
    if task_category is TaskCategory.CLASSIFICATION:
        return await _score_classification(
            prompt=prompt,
            original_response=original_response,
            judge_config=judge_config,
            send_fn=send_fn,
        )
    if task_category is TaskCategory.SUMMARIZATION:
        return await _score_summary(
            prompt=prompt,
            original_response=original_response,
            judge_config=judge_config,
            send_fn=send_fn,
        )
    # CREATIVE and REASONING both use true pairwise scoring.
    return await _score_pairwise(
        prompt=prompt,
        original_response=original_response,
        judge_config=judge_config,
        routing_config=routing_config,
        send_fn=send_fn,
    )
