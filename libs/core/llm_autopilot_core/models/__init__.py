"""
ORM model package.

Import every model module here so `Base.metadata` is fully populated as
soon as `llm_autopilot_core.models` is imported — by alembic/env.py, by
the API/worker at startup, or by tests that inspect the schema.
"""

from __future__ import annotations

from llm_autopilot_core.models.classifier import ClassifierVersion
from llm_autopilot_core.models.costs import CostAggregate
from llm_autopilot_core.models.requests import Request
from llm_autopilot_core.models.responses import Response
from llm_autopilot_core.models.routing import RoutingDecision
from llm_autopilot_core.models.verification import Verification

__all__ = [
    "ClassifierVersion",
    "CostAggregate",
    "Request",
    "Response",
    "RoutingDecision",
    "Verification",
]
