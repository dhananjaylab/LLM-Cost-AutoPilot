from __future__ import annotations

from llm_autopilot_core.verification.task_category import TaskCategory, classify_task_category


class TestClassifyTaskCategory:
    def test_extraction_default_for_plain_qa(self) -> None:
        assert classify_task_category("What is the capital of France?") == TaskCategory.EXTRACTION

    def test_extraction_keyword(self) -> None:
        assert classify_task_category("Extract the email from this text") == TaskCategory.EXTRACTION

    def test_classification_keyword(self) -> None:
        prompt = "Classify this review as positive or negative"
        assert classify_task_category(prompt) == TaskCategory.CLASSIFICATION

    def test_summarization_keyword(self) -> None:
        prompt = "Summarize this article in one sentence"
        assert classify_task_category(prompt) == TaskCategory.SUMMARIZATION

    def test_creative_keyword(self) -> None:
        assert classify_task_category("Write a poem about the ocean") == TaskCategory.CREATIVE

    def test_reasoning_keyword(self) -> None:
        prompt = "Analyze the pros and cons of remote work"
        assert classify_task_category(prompt) == TaskCategory.REASONING

    def test_creative_takes_precedence_over_reasoning(self) -> None:
        # Both "poem" (creative) and "comparing" (reasoning) signals present.
        prompt = "Write a poem comparing summer and winter"
        assert classify_task_category(prompt) == TaskCategory.CREATIVE

    def test_reasoning_takes_precedence_over_summarization(self) -> None:
        prompt = "Analyze this and summarize your findings"
        assert classify_task_category(prompt) == TaskCategory.REASONING

    def test_case_insensitive(self) -> None:
        assert classify_task_category("CLASSIFY this as spam or not") == TaskCategory.CLASSIFICATION
