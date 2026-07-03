"""
Shared pytest fixtures.

Fixtures here are available to all test modules without import.
Add service-level fixtures (DB session, Redis client) here as phases progress.
"""
from __future__ import annotations

import pytest
from llm_autopilot_core.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """
    Clear the lru_cache on get_settings() before each test so tests
    can override environment variables cleanly.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
