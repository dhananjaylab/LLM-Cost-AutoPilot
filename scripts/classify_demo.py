#!/usr/bin/env python3
"""
Phase 2 — end-to-end demo: prompt -> classifier -> routing decision.

Deliberately standalone, same spirit as scripts/baseline_test.py: no API,
no database, no live provider calls required. Just proves the library
surface Phase 2 adds — get_classifier() and select_model_for_tier() — works
together correctly, and gives a quick eyeball check of routing behavior
across a spread of prompts.

Usage:
    uv run python scripts/classify_demo.py
    uv run python scripts/classify_demo.py --trip-breaker groq   # exercise the fallback path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libs" / "core"))

from llm_autopilot_core.classifier import get_classifier  # noqa: E402
from llm_autopilot_core.providers.circuit_breaker import BreakerState  # noqa: E402
from llm_autopilot_core.providers.dispatcher import _BREAKERS  # noqa: E402
from llm_autopilot_core.routing import get_routing_config, select_model_for_tier  # noqa: E402
from llm_autopilot_core.schemas import Provider  # noqa: E402

_DEMO_PROMPTS = [
    "What is the capital of Brazil?",
    "Extract the phone number from this text: 'Call Jane at 555-234-6600 to schedule.'",
    "Summarize this paragraph in two sentences: Our quarterly results show steady growth in "
    "the enterprise segment, offset by softer consumer demand, with overall revenue roughly "
    "flat compared to last quarter.",
    "Classify this support ticket by urgency: 'Checkout is completely broken for all users.'",
    "Analyze the trade-offs of migrating a monolith to microservices for a 12-engineer team, "
    "and justify a recommendation.",
    "Write a short poem about the first snowfall of winter.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trip-breaker",
        choices=[p.value for p in Provider],
        help="Force this provider's circuit breaker OPEN first, to demo the fallback path",
    )
    args = parser.parse_args()

    if args.trip_breaker:
        provider = Provider(args.trip_breaker)
        breaker = _BREAKERS[provider]
        breaker._state = BreakerState.OPEN  # noqa: SLF001 — demo-only, not real traffic
        breaker._opened_at = float("inf")  # never looks half-open during this run
        print(f"(demo) forced {provider.value}'s circuit breaker OPEN\n")

    classifier = get_classifier()
    routing_config = get_routing_config()

    if not classifier.is_trained:
        print(
            "No trained classifier artifact found — run scripts/train_classifier.py first.\n"
            "Falling back to the untrained default (MODERATE, confidence 0.0) for every prompt.\n"
        )

    header = f"{'tier':<10}{'conf':>7}  {'model':<32}{'reason'}"
    print(header)
    print("-" * len(header))
    for prompt in _DEMO_PROMPTS:
        result = classifier.predict(prompt)
        decision = select_model_for_tier(result.tier, result.confidence, routing_config)
        override_flag = " [breaker override]" if decision.circuit_breaker_overrides else ""
        model_label = f"{decision.selected_provider.value}/{decision.selected_model_id}"
        print(
            f"{result.tier.value:<10}{result.confidence:>7.2f}  "
            f"{model_label:<32}{decision.reason}{override_flag}"
        )
        print(f"           prompt: {prompt[:78]}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
