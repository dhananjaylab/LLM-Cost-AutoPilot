"""
Lightweight task-category heuristic for the async verifier.

ComplexityTier alone doesn't tell verify_response *how* to score a
response — MODERATE covers both classification and summarization, which
need completely different rubrics. This buckets a prompt into a
finer-grained TaskCategory using the same keyword-signal style as
classifier/features.py, but deliberately doesn't import from it: this
bucketing only matters on the async, low-volume verification path, so
keeping it standalone avoids coupling the routing-time feature extractor
to a verification-time concern.
"""

from __future__ import annotations

from enum import StrEnum


class TaskCategory(StrEnum):
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    CREATIVE = "creative"
    REASONING = "reasoning"


_EXTRACTION_KEYWORDS = (
    "extract",
    "reformat",
    "convert",
    "translate",
    "rewrite",
    "pull out",
    "copy",
    "capitalize",
    "identify the",
)
_CLASSIFICATION_KEYWORDS = (
    "classify",
    "categorize",
    "categorise",
    "sentiment",
    "label this",
    "is this positive",
    "is this spam",
)
_SUMMARIZATION_KEYWORDS = (
    "summarize",
    "summarise",
    "condense",
    "key takeaways",
    "tl;dr",
    "outline",
    "brief summary",
    "in one sentence",
    "in a few words",
)
_CREATIVE_KEYWORDS = (
    "write a story",
    "write a poem",
    "poem about",
    "write a haiku",
    "imagine",
    "brainstorm",
    "invent",
    "compose",
    "creative",
    "fictional",
)
_REASONING_KEYWORDS = (
    "analyze",
    "analyse",
    "compare",
    "evaluate",
    "critique",
    "assess",
    "justify",
    "argue",
    "synthesize",
    "synthesise",
    "recommend",
    "pros and cons",
    "trade-off",
    "tradeoff",
    "diagnose",
    "root cause",
    "design a",
    "explain step by step",
    "walk through",
)

# Checked in this order — first match wins. Creative/reasoning take
# priority over classification/summarization/extraction: a prompt like
# "analyze this and summarize your findings" needs the harder reasoning
# rubric, not the summarization one.
_PRECEDENCE: tuple[tuple[TaskCategory, tuple[str, ...]], ...] = (
    (TaskCategory.CREATIVE, _CREATIVE_KEYWORDS),
    (TaskCategory.REASONING, _REASONING_KEYWORDS),
    (TaskCategory.CLASSIFICATION, _CLASSIFICATION_KEYWORDS),
    (TaskCategory.SUMMARIZATION, _SUMMARIZATION_KEYWORDS),
    (TaskCategory.EXTRACTION, _EXTRACTION_KEYWORDS),
)


def classify_task_category(prompt: str) -> TaskCategory:
    """
    Bucket a prompt for scoring-strategy selection.

    Falls back to EXTRACTION for plain factual Q&A with no matched
    keyword (e.g. "What is the capital of France?") — that's the same
    "basic Q&A" bucket the original phase spec groups with extraction
    under Tier 1, and extraction's field-presence scoring degrades
    gracefully to a single-value check for a bare factual answer.
    """
    text_lower = prompt.lower()
    for category, keywords in _PRECEDENCE:
        if any(kw in text_lower for kw in keywords):
            return category
    return TaskCategory.EXTRACTION
