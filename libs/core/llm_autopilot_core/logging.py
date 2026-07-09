"""
Logging configuration using structlog.

In development: colourised console output for readability.
In staging/production: JSON-formatted lines for log aggregators (Loki, CloudWatch, etc.)

Every log line automatically gets:
  - timestamp (ISO-8601)
  - log level
  - logger name
  - correlation IDs injected via structlog.contextvars (set in middleware)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def _add_app_context(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Inject static application context into every log line."""
    from llm_autopilot_core.config import get_settings

    s = get_settings()
    event_dict.setdefault("app", s.app_name)
    event_dict.setdefault("env", s.environment)
    event_dict.setdefault("version", s.app_version)
    return event_dict


def configure_logging(log_level: str | None = None) -> None:
    """
    Call once at application startup.

    Args:
        log_level: Override the level from settings (useful in tests).
    """
    from llm_autopilot_core.config import get_settings

    settings = get_settings()

    level_name = log_level or settings.log_level
    level = getattr(logging, level_name, logging.INFO)
    is_dev = settings.environment == "development"

    # Configure stdlib root logger so third-party libraries also route through
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    # Silence noisy loggers
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        _add_app_context,
    ]

    renderer: Any
    if is_dev:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
