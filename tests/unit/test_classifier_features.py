"""
Unit tests for llm_autopilot_core.classifier.features.

No trained model needed — these test the heuristic extraction logic in
isolation, the same way tests/unit/test_core.py tests registry.py without
needing a live database.
"""

from __future__ import annotations

from llm_autopilot_core.classifier.features import (
    FEATURE_NAMES,
    extract_features,
    feature_vector,
)


class TestFeatureNames:
    def test_feature_names_and_extract_features_agree(self) -> None:
        features = extract_features("What is the capital of France?")
        assert set(features.keys()) == set(FEATURE_NAMES)

    def test_feature_vector_matches_feature_names_order(self) -> None:
        prompt = "Summarize this: some text here."
        features = extract_features(prompt)
        vector = feature_vector(prompt)
        assert vector == [features[name] for name in FEATURE_NAMES]

    def test_all_values_are_floats(self) -> None:
        features = extract_features("Test prompt")
        assert all(isinstance(v, float) for v in features.values())


class TestExtractionSignal:
    def test_extraction_keyword_detected(self) -> None:
        features = extract_features("Extract the email address from this text: a@b.com")
        assert features["extraction_signal_count"] >= 1

    def test_translation_counts_as_extraction_signal(self) -> None:
        features = extract_features("Translate 'hello' into Spanish.")
        assert features["extraction_signal_count"] >= 1


class TestStructureSignal:
    def test_summarize_keyword_detected(self) -> None:
        features = extract_features("Summarize this paragraph in one sentence.")
        assert features["structure_signal_count"] >= 1

    def test_classify_keyword_detected(self) -> None:
        features = extract_features("Classify this review as positive or negative.")
        assert features["structure_signal_count"] >= 1


class TestAnalysisSignal:
    def test_analyze_keyword_detected(self) -> None:
        features = extract_features("Analyze the pros and cons of remote work.")
        assert features["analysis_signal_count"] >= 2  # "analyze" + "pros and cons"

    def test_compare_and_justify_detected(self) -> None:
        features = extract_features("Compare these two options and justify your recommendation.")
        assert features["analysis_signal_count"] >= 2


class TestCreativeSignal:
    def test_write_a_story_detected(self) -> None:
        features = extract_features("Write a story about a dragon who loves gardening.")
        assert features["creative_signal_count"] >= 1

    def test_non_creative_prompt_has_zero_creative_signal(self) -> None:
        features = extract_features("What is 12 plus 7?")
        assert features["creative_signal_count"] == 0.0


class TestConstraintCount:
    def test_modal_constraint_words_detected(self) -> None:
        prompt = "The summary must be under 50 words and should use only bullet points."
        features = extract_features(prompt)
        assert features["constraint_count"] >= 3

    def test_bullet_lines_count_as_constraints(self) -> None:
        prompt = "Follow these rules:\n- rule one\n- rule two\n- rule three"
        features = extract_features(prompt)
        assert features["constraint_count"] >= 3

    def test_no_constraints_in_plain_question(self) -> None:
        features = extract_features("What is the capital of Japan?")
        assert features["constraint_count"] == 0.0


class TestContextProvided:
    def test_quoted_context_detected(self) -> None:
        long_quote = "x" * 100
        features = extract_features(f"Summarize this: '{long_quote}'")
        assert features["context_provided"] == 1.0

    def test_short_quote_not_treated_as_context(self) -> None:
        features = extract_features("What does 'hello' mean?")
        assert features["context_provided"] == 0.0

    def test_context_phrase_detected(self) -> None:
        long_text = "some content here " * 6
        prompt = f"Based on the following text, answer the question. {long_text}"
        features = extract_features(prompt)
        assert features["context_provided"] == 1.0

    def test_no_context_in_plain_question(self) -> None:
        features = extract_features("What is 5 times 6?")
        assert features["context_provided"] == 0.0


class TestOutputFormatComplexity:
    def test_json_request_is_complex_format(self) -> None:
        features = extract_features("Respond in JSON format with fields: name, value.")
        assert features["output_format_complexity"] == 2.0

    def test_bullet_list_request_is_simple_format(self) -> None:
        features = extract_features("Give me the answer as a bullet list.")
        assert features["output_format_complexity"] == 1.0

    def test_free_text_has_zero_format_complexity(self) -> None:
        features = extract_features("What year did World War II end?")
        assert features["output_format_complexity"] == 0.0


class TestTokenAndCharCounts:
    def test_longer_prompt_has_higher_token_count(self) -> None:
        short = extract_features("Hi.")
        long_prompt = "This is a much longer prompt with many more words than the short one."
        long = extract_features(long_prompt)
        assert long["token_count"] > short["token_count"]
        assert long["char_count"] > short["char_count"]

    def test_question_mark_counted(self) -> None:
        features = extract_features("Is this true? And what about this?")
        assert features["question_count"] == 2.0
