from __future__ import annotations

import pytest
from llm_autopilot_core.routing import (
    CostBaselineConfig,
    RoutingConfig,
    TierRoute,
    VerificationRoutingConfig,
)
from llm_autopilot_core.schemas import (
    ComplexityTier,
    ModelConfig,
    Provider,
    ProviderResponse,
    QualityTier,
)
from llm_autopilot_core.verification.scoring import (
    get_threshold_for_category,
    is_self_judge,
    score_response,
)
from llm_autopilot_core.verification.task_category import TaskCategory


def _model(provider: Provider, model_id: str) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model_id=model_id,
        display_name=model_id,
        cost_per_input_token=0.000_001,
        cost_per_output_token=0.000_002,
        avg_latency_ms=500,
        quality_tier=QualityTier.MEDIUM,
        context_window=100_000,
        max_output_tokens=4_096,
    )


JUDGE_CONFIG = _model(Provider.ANTHROPIC, "claude-haiku-4-5")


def _routing_config() -> RoutingConfig:
    return RoutingConfig(
        version="1",
        tiers={
            ComplexityTier.SIMPLE: TierRoute(
                description="", models=["groq/llama-3.1-8b-instant"], max_latency_ms=3000
            ),
            ComplexityTier.MODERATE: TierRoute(
                description="", models=["openai/gpt-4o-mini"], max_latency_ms=5000
            ),
            ComplexityTier.COMPLEX: TierRoute(
                description="", models=["anthropic/claude-sonnet-4-6"], max_latency_ms=15000
            ),
        },
        verification=VerificationRoutingConfig(
            judge_model="anthropic/claude-haiku-4-5", judge_max_tokens=512
        ),
        cost_baseline=CostBaselineConfig(model="openai/gpt-4o"),
    )


def _response(content: str) -> ProviderResponse:
    return ProviderResponse(
        content=content,
        input_tokens=10,
        output_tokens=10,
        latency_ms=100.0,
        cost_usd=0.001,
        model_id="x",
        provider=Provider.ANTHROPIC,
    )


class FakeSendRequest:
    """Returns canned responses in call order — lets pairwise tests script
    the comparison-generation call and both judge passes independently."""

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = responses
        self.call_count = 0

    async def __call__(
        self,
        prompt: object,
        model_config: ModelConfig,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ProviderResponse:
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


class TestSelfJudgeGuard:
    def test_true_when_same_provider_and_model(self) -> None:
        assert is_self_judge(Provider.ANTHROPIC, "claude-haiku-4-5", JUDGE_CONFIG) is True

    def test_false_when_different_model(self) -> None:
        assert is_self_judge(Provider.ANTHROPIC, "claude-sonnet-4-6", JUDGE_CONFIG) is False

    def test_false_when_different_provider_same_model_id(self) -> None:
        assert is_self_judge(Provider.OPENAI, "claude-haiku-4-5", JUDGE_CONFIG) is False


class TestThresholds:
    def test_extraction_and_classification_use_high_bar(self) -> None:
        assert get_threshold_for_category(TaskCategory.EXTRACTION, 0.75) == 0.9
        assert get_threshold_for_category(TaskCategory.CLASSIFICATION, 0.75) == 0.9

    def test_creative_and_reasoning_use_low_bar(self) -> None:
        assert get_threshold_for_category(TaskCategory.CREATIVE, 0.75) == 0.5
        assert get_threshold_for_category(TaskCategory.REASONING, 0.75) == 0.5

    def test_summarization_falls_back_to_default(self) -> None:
        assert get_threshold_for_category(TaskCategory.SUMMARIZATION, 0.75) == 0.75


class TestScoreExtraction:
    async def test_all_fields_matched_scores_one(self) -> None:
        fake = FakeSendRequest([_response("name: Sarah\nemail: sarah@example.com")])
        result = await score_response(
            task_category=TaskCategory.EXTRACTION,
            prompt="Extract name and email",
            original_response="Sarah, sarah@example.com",
            judge_config=JUDGE_CONFIG,
            routing_config=_routing_config(),
            send_fn=fake,
        )
        assert result.quality_score == 1.0

    async def test_partial_match_scores_fraction(self) -> None:
        fake = FakeSendRequest([_response("name: Sarah\nemail: sarah@example.com")])
        result = await score_response(
            task_category=TaskCategory.EXTRACTION,
            prompt="Extract name and email",
            original_response="The name is Sarah but no email was given",
            judge_config=JUDGE_CONFIG,
            routing_config=_routing_config(),
            send_fn=fake,
        )
        assert result.quality_score == pytest.approx(0.5)

    async def test_no_extractable_fields_is_inconclusive(self) -> None:
        fake = FakeSendRequest([_response("nothing to extract here")])
        result = await score_response(
            task_category=TaskCategory.EXTRACTION,
            prompt="...",
            original_response="...",
            judge_config=JUDGE_CONFIG,
            routing_config=_routing_config(),
            send_fn=fake,
        )
        assert result.quality_score == 1.0


class TestScoreClassification:
    async def test_matching_label_scores_one(self) -> None:
        fake = FakeSendRequest([_response("positive")])
        result = await score_response(
            task_category=TaskCategory.CLASSIFICATION,
            prompt="Classify sentiment",
            original_response="This is a positive review overall.",
            judge_config=JUDGE_CONFIG,
            routing_config=_routing_config(),
            send_fn=fake,
        )
        assert result.quality_score == 1.0

    async def test_mismatched_label_scores_zero(self) -> None:
        fake = FakeSendRequest([_response("negative")])
        result = await score_response(
            task_category=TaskCategory.CLASSIFICATION,
            prompt="Classify sentiment",
            original_response="This is a positive review overall.",
            judge_config=JUDGE_CONFIG,
            routing_config=_routing_config(),
            send_fn=fake,
        )
        assert result.quality_score == 0.0


class TestScoreSummary:
    async def test_parses_score_line(self) -> None:
        fake = FakeSendRequest([_response("The summary covers the key points.\nSCORE: 4")])
        result = await score_response(
            task_category=TaskCategory.SUMMARIZATION,
            prompt="Summarize",
            original_response="...",
            judge_config=JUDGE_CONFIG,
            routing_config=_routing_config(),
            send_fn=fake,
        )
        assert result.quality_score == pytest.approx(0.75)

    async def test_unparseable_score_defaults_to_midpoint(self) -> None:
        fake = FakeSendRequest([_response("I cannot grade this.")])
        result = await score_response(
            task_category=TaskCategory.SUMMARIZATION,
            prompt="Summarize",
            original_response="...",
            judge_config=JUDGE_CONFIG,
            routing_config=_routing_config(),
            send_fn=fake,
        )
        assert result.quality_score == 0.5


class TestScorePairwise:
    async def test_original_wins_both_passes(self) -> None:
        # Call order: comparison generation, pass1 (A=original,B=comparison),
        # pass2 (A=comparison,B=original).
        fake = FakeSendRequest(
            [
                _response("comparison model output"),
                _response("A"),  # pass1: original (A) wins
                _response("B"),  # pass2: original (B) wins
            ]
        )
        result = await score_response(
            task_category=TaskCategory.CREATIVE,
            prompt="Write a poem",
            original_response="original poem text",
            judge_config=JUDGE_CONFIG,
            routing_config=_routing_config(),
            send_fn=fake,
        )
        assert result.quality_score == 1.0
        assert result.escalation_candidate_content is None

    async def test_comparison_wins_both_passes_populates_escalation_candidate(self) -> None:
        fake = FakeSendRequest(
            [
                _response("comparison model output"),
                _response("B"),  # pass1: comparison (B) wins
                _response("A"),  # pass2: comparison (A) wins
            ]
        )
        result = await score_response(
            task_category=TaskCategory.REASONING,
            prompt="Analyze this",
            original_response="original analysis",
            judge_config=JUDGE_CONFIG,
            routing_config=_routing_config(),
            send_fn=fake,
        )
        assert result.quality_score == 0.0
        assert result.escalation_candidate_content == "comparison model output"
        assert result.escalation_candidate_model_id == "claude-sonnet-4-6"
        assert result.escalation_candidate_provider == Provider.ANTHROPIC

    async def test_disagreeing_passes_scored_as_tie(self) -> None:
        fake = FakeSendRequest(
            [
                _response("comparison model output"),
                _response("A"),  # pass1: original wins
                _response("A"),  # pass2: comparison wins — inconsistent with pass1
            ]
        )
        result = await score_response(
            task_category=TaskCategory.CREATIVE,
            prompt="Write a poem",
            original_response="original poem text",
            judge_config=JUDGE_CONFIG,
            routing_config=_routing_config(),
            send_fn=fake,
        )
        assert result.quality_score == 0.6
        assert result.escalation_candidate_content is None

    async def test_tie_score_sits_below_low_threshold_pass(self) -> None:
        # Confirms the CATEGORY_THRESHOLDS design intent directly: a tie
        # (0.6) must clear the creative/reasoning threshold (0.5), and a
        # clear loss (0.0) must not.
        assert get_threshold_for_category(TaskCategory.CREATIVE, 0.75) <= 0.6
        assert get_threshold_for_category(TaskCategory.CREATIVE, 0.75) > 0.0
