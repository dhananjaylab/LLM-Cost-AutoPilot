#!/usr/bin/env python3
"""
Phase 1, Task 3 — baseline test script.

Sends the same 10 prompts to every enabled model in MODEL_REGISTRY whose
provider is actually configured, logs outputs/costs/latencies for each
(model, prompt) pair, and prints a per-model summary table. This is what
validates the Task 2 abstraction layer actually works end-to-end, and it
gives Phase 2's classifier/router real baseline data to compare against.

Usage:
    # See what would run and roughly what it would cost, without spending anything
    uv run python scripts/baseline_test.py --dry-run

    # Try only the free local model first (no API keys needed, needs `ollama serve`)
    uv run python scripts/baseline_test.py --providers ollama --yes

    # Full run across every provider with a configured key (asks for confirmation
    # first unless --yes is passed, since this spends real money on paid providers)
    uv run python scripts/baseline_test.py

    # Narrow to specific models by registry key
    uv run python scripts/baseline_test.py \
        --models openai/gpt-4o-mini,anthropic/claude-3-5-haiku-20241022

Results are written under <output-dir>/<timestamp>/ (default: var/baseline_results/):
    results.jsonl   — one line per (model, prompt) call: content, tokens, cost, latency, error
    summary.csv     — per-model aggregate: call count, success rate, avg latency, total cost
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from llm_autopilot_core.config import get_settings
from llm_autopilot_core.providers import ProviderError, send_request
from llm_autopilot_core.registry import MODEL_REGISTRY, compute_cost
from llm_autopilot_core.schemas import ModelConfig

# ── The 10 baseline prompts ────────────────────────────────────────────────────
# Deliberately varied across task types (factual QA, summarization, extraction,
# creative writing, classification, technical explanation, translation, basic
# reasoning, lexical, storytelling) since Phase 2's complexity classifier will
# need real examples across the spectrum it's meant to distinguish.
BASELINE_PROMPTS: list[str] = [
    "What is the capital of France?",
    "Summarize the following text in one sentence: The quick brown fox jumps "
    "over the lazy dog while the sun sets behind the mountains, casting long "
    "shadows across the quiet meadow.",
    "Extract the person's name and email from this text: 'Hi, this is Sarah "
    "Connor, you can reach me at sarah.connor@example.com for any questions.'",
    "Write a haiku about autumn leaves.",
    "Classify the sentiment of this review as positive, negative, or neutral: "
    "'The food was okay but the service was painfully slow.'",
    "Explain the difference between a list and a tuple in Python in two sentences.",
    "Translate 'Good morning, how are you?' into French.",
    "What is 17 multiplied by 24?",
    "Give me three synonyms for the word 'happy'.",
    "Write a one-paragraph story about a robot who discovers music for the first time.",
]

DEFAULT_MAX_TOKENS = 300  # kept low deliberately — this is a baseline smoke test, not a benchmark
DEFAULT_TEMPERATURE = 0.3  # low temperature for more comparable, reproducible baseline output


@dataclass
class CallResult:
    registry_key: str
    provider: str
    model_id: str
    prompt_index: int
    prompt: str
    success: bool
    content: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    retryable: bool | None = None


def _key_for(model: ModelConfig) -> str:
    return f"{model.provider.value}/{model.model_id}"


def _select_models(
    providers_filter: list[str] | None, models_filter: list[str] | None
) -> list[ModelConfig]:
    settings = get_settings()
    available = set(settings.available_providers)

    selected: list[ModelConfig] = []
    for key, model in MODEL_REGISTRY.items():
        if not model.enabled:
            continue
        if models_filter and key not in models_filter:
            continue
        if providers_filter and model.provider.value not in providers_filter:
            continue
        if model.provider.value not in available:
            print(
                f"  skipping {key}: provider '{model.provider.value}' has no API key "
                f"configured (set it in .env)",
                file=sys.stderr,
            )
            continue
        selected.append(model)
    return selected


def _estimate_cost(models: list[ModelConfig]) -> float:
    """Rough pre-flight estimate: ~40 input tokens/prompt, worst case max_tokens output."""
    avg_input_tokens = 40
    total = 0.0
    for model in models:
        for _ in BASELINE_PROMPTS:
            total += compute_cost(model, avg_input_tokens, DEFAULT_MAX_TOKENS)
    return total


async def _run_one(
    model: ModelConfig,
    registry_key: str,
    prompt_index: int,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> CallResult:
    async with semaphore:
        try:
            response = await send_request(
                prompt, model, max_tokens=max_tokens, temperature=temperature
            )
        except ProviderError as exc:
            return CallResult(
                registry_key=registry_key,
                provider=model.provider.value,
                model_id=model.model_id,
                prompt_index=prompt_index,
                prompt=prompt,
                success=False,
                error=str(exc),
                retryable=exc.retryable,
            )
        return CallResult(
            registry_key=registry_key,
            provider=model.provider.value,
            model_id=model.model_id,
            prompt_index=prompt_index,
            prompt=prompt,
            success=True,
            content=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        )


def _write_jsonl(results: list[CallResult], path: Path) -> None:
    with path.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def _write_summary_csv(results: list[CallResult], path: Path) -> None:
    by_model: dict[str, list[CallResult]] = {}
    for r in results:
        by_model.setdefault(r.registry_key, []).append(r)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "registry_key",
                "provider",
                "model_id",
                "total_calls",
                "successes",
                "failures",
                "success_rate_pct",
                "avg_latency_ms",
                "total_cost_usd",
                "avg_cost_usd_per_call",
            ]
        )
        for key, calls in sorted(by_model.items()):
            successes = [c for c in calls if c.success]
            n = len(calls)
            avg_latency = (
                sum(c.latency_ms for c in successes) / len(successes) if successes else 0.0
            )
            total_cost = sum(c.cost_usd for c in calls)
            writer.writerow(
                [
                    key,
                    calls[0].provider,
                    calls[0].model_id,
                    n,
                    len(successes),
                    n - len(successes),
                    round(100 * len(successes) / n, 1) if n else 0.0,
                    round(avg_latency, 1),
                    round(total_cost, 6),
                    round(total_cost / n, 6) if n else 0.0,
                ]
            )


def _print_summary_table(results: list[CallResult]) -> None:
    by_model: dict[str, list[CallResult]] = {}
    for r in results:
        by_model.setdefault(r.registry_key, []).append(r)

    header = f"{'model':<45} {'ok/total':>9} {'avg latency':>13} {'total cost':>12}"
    print("\n" + header)
    print("-" * len(header))
    grand_total_cost = 0.0
    for key, calls in sorted(by_model.items()):
        successes = [c for c in calls if c.success]
        avg_latency = sum(c.latency_ms for c in successes) / len(successes) if successes else 0.0
        total_cost = sum(c.cost_usd for c in calls)
        grand_total_cost += total_cost
        print(
            f"{key:<45} {len(successes)}/{len(calls):>7} "
            f"{avg_latency:>10.0f} ms {total_cost:>11.6f}$"
        )
    print("-" * len(header))
    print(f"{'TOTAL':<45} {'':>9} {'':>13} {grand_total_cost:>11.6f}$\n")


async def _run(args: argparse.Namespace) -> int:
    providers_filter = args.providers.split(",") if args.providers else None
    models_filter = args.models.split(",") if args.models else None
    models = _select_models(providers_filter, models_filter)

    if not models:
        print(
            "No models selected — either every candidate provider is missing an API "
            "key, or --providers/--models filtered everything out. Nothing to do.",
            file=sys.stderr,
        )
        return 1

    total_calls = len(models) * len(BASELINE_PROMPTS)
    estimated_cost = _estimate_cost(models)

    print(f"Models selected ({len(models)}):")
    for m in models:
        print(f"  - {m.provider.value}/{m.model_id}")
    print(f"\nPrompts: {len(BASELINE_PROMPTS)}")
    print(f"Total calls: {total_calls}")
    print(f"Rough cost estimate (worst case, max_tokens output every time): ${estimated_cost:.4f}")

    if args.dry_run:
        print("\n--dry-run set: not making any actual calls.")
        return 0

    if not args.yes:
        reply = input("\nProceed? [y/N] ").strip().lower()  # noqa: ASYNC250
        if reply != "y":
            print("Aborted.")
            return 1

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        _run_one(
            model,
            _key_for(model),
            i,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            semaphore=semaphore,
        )
        for model in models
        for i, prompt in enumerate(BASELINE_PROMPTS)
    ]

    print(f"\nRunning {len(tasks)} calls (concurrency={args.concurrency})...")
    results = await asyncio.gather(*tasks)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(list(results), output_dir / "results.jsonl")
    _write_summary_csv(list(results), output_dir / "summary.csv")

    _print_summary_table(list(results))
    print(f"Full results: {output_dir / 'results.jsonl'}")
    print(f"Summary CSV:  {output_dir / 'summary.csv'}")

    failures = [r for r in results if not r.success]
    if failures:
        print(f"\n{len(failures)} call(s) failed — see results.jsonl for details:")
        for f in failures[:10]:
            print(f"  - {f.registry_key} prompt#{f.prompt_index}: {f.error}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--providers", help="Comma-separated provider names to include (default: all configured)"
    )
    parser.add_argument(
        "--models", help="Comma-separated registry keys to include, overrides --providers"
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--concurrency", type=int, default=3, help="Max in-flight calls at once")
    parser.add_argument("--output-dir", default="var/baseline_results")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan, spend nothing")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
