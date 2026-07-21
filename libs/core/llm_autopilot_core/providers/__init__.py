"""
Unified provider abstraction layer (Phase 1, Task 2).

Public API:
    from llm_autopilot_core.providers import send_request
    response = await send_request("Hello!", model_config)
"""

from __future__ import annotations

from llm_autopilot_core.providers.base import BaseProviderAdapter, ProviderError
from llm_autopilot_core.providers.dispatcher import is_provider_available, send_request

__all__ = ["BaseProviderAdapter", "ProviderError", "is_provider_available", "send_request"]
