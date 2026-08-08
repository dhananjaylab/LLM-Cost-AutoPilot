"""
Phase 4 — async quality verification loop.

Public API:
    from llm_autopilot_core.verification import (
        TaskCategory, classify_task_category,
        ScoringResult, score_response, get_threshold_for_category, is_self_judge,
    )
"""

from __future__ import annotations

from llm_autopilot_core.verification.scoring import (
    CATEGORY_THRESHOLDS,
    ScoringResult,
    get_threshold_for_category,
    is_self_judge,
    score_response,
)
from llm_autopilot_core.verification.task_category import TaskCategory, classify_task_category

__all__ = [
    "CATEGORY_THRESHOLDS",
    "ScoringResult",
    "TaskCategory",
    "classify_task_category",
    "get_threshold_for_category",
    "is_self_judge",
    "score_response",
]
