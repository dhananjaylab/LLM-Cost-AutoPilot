"""
Heuristic feature extraction for the complexity classifier.

Phase 2 spec calls for: token count, presence of instructions like
"analyze" or "compare", number of constraints, whether context is
provided, and output format complexity. Everything here is regex/keyword
based on purpose — this is a routing skeleton, not an NLP research
project, and the model doesn't need perfect signals to beat "always route
to the biggest model."

`token_count` is deliberately an approximate word-based count, not a real
tokenizer (e.g. tiktoken). The classifier only needs a coarse length
signal to separate tiers; exact token accounting already happens
downstream via each provider's real `usage.input_tokens` in
ProviderResponse. Keeping this dependency-free avoids coupling the
classifier's vocabulary to any one provider's tokenizer.

FEATURE_NAMES is the single source of truth for feature order — both
scripts/train_classifier.py and ComplexityClassifier.predict() build
their numpy arrays from this same tuple, so a trained pipeline can never
silently drift out of sync with what's extracted at inference time.
"""

from __future__ import annotations

import re

# ── Keyword sets (case-insensitive) ────────────────────────────────────────────

# Tier 1 signal: reformatting / extraction / direct lookup work.
_EXTRACTION_KEYWORDS = (
    "extract",
    "reformat",
    "convert",
    "translate",
    "rewrite",
    "identify the",
    "pull out",
    "copy",
    "list the",
    "capitalize",
)

# Tier 2 signal: summarization / classification / structured analysis.
_STRUCTURE_KEYWORDS = (
    "summarize",
    "summarise",
    "classify",
    "categorize",
    "categorise",
    "organize",
    "organise",
    "outline",
    "sentiment",
    "group",
)

# Tier 3 signal: multi-step reasoning / nuanced judgement.
_ANALYSIS_KEYWORDS = (
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
    "why",
    "root cause",
    "diagnose",
)

# Tier 3 signal: creative generation.
_CREATIVE_KEYWORDS = (
    "write a story",
    "write a poem",
    "poem about",
    "imagine",
    "brainstorm",
    "invent",
    "compose",
    "creative",
    "fictional",
)

# Constraint indicators — modal/requirement language plus quantity bounds.
_CONSTRAINT_PATTERNS = (
    r"\bmust\b",
    r"\bshould\b",
    r"\bat least\b",
    r"\bno more than\b",
    r"\bexactly\b",
    r"\bwithin\b",
    r"\bmake sure\b",
    r"\bensure\b",
    r"\bonly\b",
    r"\busing only\b",
    r"\bin \d+ words?\b",
    r"\bunder \d+\b",
    r"\bbetween \d+ and \d+\b",
)
_CONSTRAINT_RE = re.compile("|".join(_CONSTRAINT_PATTERNS), re.IGNORECASE)
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", re.MULTILINE)

# Context-provided indicators: an explicit pointer to appended material.
_CONTEXT_PHRASES = (
    "the following text",
    "the following",
    "given this",
    "given the text",
    "based on the text",
    "based on this",
    "here is the",
    "below:",
    "text below",
)
_MIN_QUOTED_CONTEXT_CHARS = 80

# Output-format complexity keyword sets.
_SIMPLE_FORMAT_KEYWORDS = ("list", "bullet", "one sentence", "short", "table", "in one word")
_COMPLEX_FORMAT_KEYWORDS = (
    "json",
    "```",
    "schema",
    "format:",
    "xml",
    "yaml",
    "fields:",
    "columns:",
    "code block",
)

_WORD_RE = re.compile(r"\S+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")

FEATURE_NAMES: tuple[str, ...] = (
    "char_count",
    "token_count",
    "sentence_count",
    "question_count",
    "extraction_signal_count",
    "structure_signal_count",
    "analysis_signal_count",
    "creative_signal_count",
    "constraint_count",
    "context_provided",
    "output_format_complexity",
)


def _count_hits(text_lower: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for kw in keywords if kw in text_lower)


def _has_quoted_context(text: str) -> bool:
    for quote_char in ('"', "'"):
        parts = text.split(quote_char)
        # Any span strictly between a pair of quote marks that's long enough.
        for span in parts[1::2]:
            if len(span) >= _MIN_QUOTED_CONTEXT_CHARS:
                return True
    if ":" in text:
        tail = text.rsplit(":", 1)[1]
        if len(tail.strip()) >= _MIN_QUOTED_CONTEXT_CHARS:
            return True
    return False


def extract_features(prompt: str) -> dict[str, float]:
    """
    Extract the fixed-order feature vector for a single prompt.

    Returns a dict keyed by FEATURE_NAMES (all values float, even counts,
    so callers can build a numpy array uniformly). Order is guaranteed to
    match FEATURE_NAMES; use `feature_vector()` if you want the raw list.
    """
    text = prompt.strip()
    text_lower = text.lower()

    char_count = float(len(text))
    token_count = float(len(_WORD_RE.findall(text)))
    sentence_count = float(len(_SENTENCE_SPLIT_RE.split(text)) - 1) if text else 0.0
    sentence_count = max(sentence_count, 1.0 if text else 0.0)
    question_count = float(text.count("?"))

    extraction_signal_count = float(_count_hits(text_lower, _EXTRACTION_KEYWORDS))
    structure_signal_count = float(_count_hits(text_lower, _STRUCTURE_KEYWORDS))
    analysis_signal_count = float(_count_hits(text_lower, _ANALYSIS_KEYWORDS))
    creative_signal_count = float(_count_hits(text_lower, _CREATIVE_KEYWORDS))

    constraint_count = float(len(_CONSTRAINT_RE.findall(text)) + len(_BULLET_LINE_RE.findall(text)))

    context_provided = float(
        any(phrase in text_lower for phrase in _CONTEXT_PHRASES) or _has_quoted_context(text)
    )

    has_complex_format = any(kw in text_lower for kw in _COMPLEX_FORMAT_KEYWORDS)
    has_simple_format = any(kw in text_lower for kw in _SIMPLE_FORMAT_KEYWORDS)
    output_format_complexity = 2.0 if has_complex_format else (1.0 if has_simple_format else 0.0)

    return {
        "char_count": char_count,
        "token_count": token_count,
        "sentence_count": sentence_count,
        "question_count": question_count,
        "extraction_signal_count": extraction_signal_count,
        "structure_signal_count": structure_signal_count,
        "analysis_signal_count": analysis_signal_count,
        "creative_signal_count": creative_signal_count,
        "constraint_count": constraint_count,
        "context_provided": context_provided,
        "output_format_complexity": output_format_complexity,
    }


def feature_vector(prompt: str) -> list[float]:
    """Same as extract_features(), but as an ordered list matching FEATURE_NAMES."""
    features = extract_features(prompt)
    return [features[name] for name in FEATURE_NAMES]
