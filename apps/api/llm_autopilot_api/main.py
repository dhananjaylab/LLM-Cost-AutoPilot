"""
LLM Cost Autopilot — FastAPI application.

Startup sequence (lifespan):
  1. Configure structured logging
  2. Apply YAML runtime overrides to model registry
  3. Verify database connection
  4. Initialise Prometheus app_info metric
  5. Configure OpenTelemetry tracing (if enabled)

POST /v1/completions  ← the main routing endpoint (Phase 3+)
GET  /v1/models       ← list available models + pricing (Phase 3+)
GET  /v1/stats        ← cost savings dashboard data (Phase 5+)
PUT  /v1/routing-config ← live routing policy update (Phase 5+)
GET  /v1/healthz      ← liveness probe
GET  /v1/readyz       ← readiness probe (checks DB + Redis)
GET  /metrics         ← Prometheus scrape endpoint
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.database import check_connection, dispose_engine
from llm_autopilot_core.logging import configure_logging
from llm_autopilot_core.metrics import initialise_info
from llm_autopilot_core.registry import load_yaml_overrides
from prometheus_client import REGISTRY, generate_latest

from llm_autopilot_api.routers import health

# Configure logging immediately on import so module-level loggers use the configured factory
configure_logging()

logger = structlog.get_logger(__name__)
settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ── Startup ───────────────────────────────────────────────────────────────

    logger.info(
        "startup",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        providers=settings.available_providers,
    )

    # Apply YAML runtime overrides (prices, enabled flags)
    load_yaml_overrides(settings.models_config_path)
    logger.info("model_registry_loaded")

    # Verify DB is reachable before accepting traffic
    if not await check_connection():
        logger.error("database_unreachable", url=settings.database_url)
        raise RuntimeError("Cannot connect to PostgreSQL — refusing to start")
    logger.info("database_connected")

    initialise_info()

    # ── OpenTelemetry tracing (optional) ──────────────────────────────────────
    if settings.enable_tracing and settings.otlp_endpoint:
        _configure_tracing(app)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("shutdown_initiated")
    await dispose_engine()
    logger.info("shutdown_complete")


def _configure_tracing(app: FastAPI | None = None) -> None:
    """Set up OTLP exporter for distributed tracing (Jaeger, Tempo, etc.)"""
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": settings.app_name,
                "service.version": settings.app_version,
                "deployment.environment": settings.environment,
            }
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
        )
        trace.set_tracer_provider(provider)

        fastapi_instrumentor = FastAPIInstrumentor()
        if app is not None:
            fastapi_instrumentor.instrument_app(app, tracer_provider=provider)
        else:
            fastapi_instrumentor.instrument()

        SQLAlchemyInstrumentor().instrument()
        logger.info("tracing_configured", endpoint=settings.otlp_endpoint)
    except ImportError:
        logger.warning(
            "tracing_deps_missing", hint="pip install opentelemetry-instrumentation-fastapi"
        )


# ── Application factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Intelligent LLM routing layer that analyses request complexity, "
            "routes to the cheapest capable model, and validates routing quality "
            "through an async verification loop."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS — tighten in production via environment variable
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Correlation ID + request logging middleware ───────────────────────────
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: object) -> Response:
        import uuid

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()
        response: Response = await call_next(request)  # type: ignore[operator]
        duration_ms = (time.perf_counter() - start) * 1_000

        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Prometheus metrics endpoint ───────────────────────────────────────────
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(REGISTRY),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router, prefix="/v1")
    # Phase 3+: app.include_router(completions.router, prefix="/v1")
    # Phase 5+: app.include_router(models.router, prefix="/v1")
    # Phase 5+: app.include_router(stats.router, prefix="/v1")

    return app


app = create_app()
