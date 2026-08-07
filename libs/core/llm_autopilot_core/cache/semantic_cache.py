"""
Semantic cache — Phase 3.

Wraps RedisVL's SemanticCache with this project's Settings. RedisVL's
distance_threshold is a Redis COSINE *distance* in [0, 2] (0 = identical,
lower = stricter) — the opposite direction and a different scale from
this project's own CACHE_SIMILARITY_THRESHOLD (higher = stricter, [0,1]).
similarity_to_distance() makes that conversion explicit and testable
rather than passing the raw config value through, since RedisVL itself
has had sync/async scale-interpretation inconsistencies here (see
https://github.com/redis/redis-vl-python/issues/407) — worth being
deliberate about this rather than trusting a single numeric knob.
"""

from __future__ import annotations

from functools import lru_cache

from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

from llm_autopilot_core.config import get_settings


def similarity_to_distance(similarity_threshold: float) -> float:
    """
    Convert this project's similarity threshold (higher = stricter, [0,1])
    to RedisVL's native COSINE distance (lower = stricter, [0,2]).

    e.g. the default CACHE_SIMILARITY_THRESHOLD=0.92 → distance 0.08,
    close to RedisVL's own default of 0.1.
    """
    return max(0.0, min(2.0, 1.0 - similarity_threshold))


@lru_cache(maxsize=1)
def get_semantic_cache() -> SemanticCache:
    """
    Cached singleton — constructing SemanticCache builds/verifies the
    underlying Redis search index and loads the embedding model via
    HFTextVectorizer, both one-time costs worth paying once per process,
    the same reasoning as get_classifier() and get_routing_config().
    """
    settings = get_settings()
    return SemanticCache(
        name=settings.cache_index_name,
        redis_url=settings.redis_url,
        distance_threshold=similarity_to_distance(settings.cache_similarity_threshold),
        ttl=settings.redis_cache_ttl,
        vectorizer=HFTextVectorizer(settings.embedding_model),
    )
