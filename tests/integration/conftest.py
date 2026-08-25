"""
Shared fixtures for integration tests.
"""

from __future__ import annotations

import pytest
from llm_autopilot_core.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache_integration():
    """Clear settings cache before each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
