from __future__ import annotations

import pytest
from llm_autopilot_core.cache.semantic_cache import similarity_to_distance


class TestSimilarityToDistance:
    def test_default_config_value(self) -> None:
        # CACHE_SIMILARITY_THRESHOLD default is 0.92 → distance 0.08,
        # close to RedisVL's own default distance_threshold of 0.1.
        assert similarity_to_distance(0.92) == pytest.approx(0.08)

    def test_perfect_similarity_gives_zero_distance(self) -> None:
        assert similarity_to_distance(1.0) == 0.0

    def test_zero_similarity_gives_distance_one(self) -> None:
        assert similarity_to_distance(0.0) == 1.0

    def test_clamped_to_valid_zero_to_two_range(self) -> None:
        # Similarity is validated [0,1] at the Settings layer, but this
        # function stays defensive regardless of caller discipline.
        assert similarity_to_distance(-0.5) == 1.5
        assert similarity_to_distance(2.0) == 0.0
