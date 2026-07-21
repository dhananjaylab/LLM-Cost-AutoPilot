"""
Central configuration module.

All settings are driven by environment variables (12-factor).
See .env.example for the full list. Load order:
  1. Defaults defined here
  2. .env file (dev only — ignored in prod containers)
  3. Actual environment variables (highest priority)

Usage:
    from llm_autopilot_core.config import get_settings
    settings = get_settings()          # cached singleton via lru_cache
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "LLM Cost Autopilot"
    app_version: str = "0.1.0"
    environment: str = Field(
        default="development",
        pattern="^(development|staging|production)$",
    )
    debug: bool = False
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://autopilot:autopilot@localhost:5432/autopilot"
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_pool_timeout: int = Field(default=30, ge=1)

    # ── Redis / RediSearch ────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_cache_ttl: int = Field(default=3600, ge=60)  # seconds
    cache_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    cache_index_name: str = "autopilot:semantic_cache"
    cache_vector_dim: int = 384  # all-MiniLM-L6-v2 output dimension

    # ── Celery ────────────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_task_soft_time_limit: int = 300  # seconds before SoftTimeLimitExceeded
    celery_task_time_limit: int = 600  # hard kill after 10 min

    # ── LLM Provider API Keys ─────────────────────────────────────────────────
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # ── Ollama (local) ────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 120  # local models can be slow

    # ── Routing config ────────────────────────────────────────────────────────
    routing_config_path: str = "configs/routing.yaml"
    models_config_path: str = "configs/models.yaml"

    # ── Classifier & verification ─────────────────────────────────────────────
    # Sampling rate when classifier confidence is LOW (≤ threshold)
    verification_sample_rate_low_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # Sampling rate when classifier confidence is HIGH (> threshold)
    verification_sample_rate_high_confidence: float = Field(default=0.05, ge=0.0, le=1.0)
    # Always verify this fraction regardless of confidence (drift detection)
    verification_random_baseline_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    # Confidence boundary between high/low
    classifier_confidence_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    # Quality score below which escalation is triggered
    escalation_quality_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    # Max latency (ms) to wait for escalation rerun
    escalation_max_latency_ms: int = Field(default=5000, ge=1000)

    # ── Phase 2: complexity classifier ────────────────────────────────────────
    # Where the trained sklearn pipeline artifact lives. train_classifier.py
    # writes here; ComplexityClassifier lazily loads from here.
    classifier_model_path: str = "var/classifier/model.joblib"

    # ── Observability ─────────────────────────────────────────────────────────
    enable_tracing: bool = True
    otlp_endpoint: str | None = None  # e.g. "http://jaeger:4318"
    metrics_namespace: str = "llm_autopilot"

    # ── Embedding model (local, no API cost) ─────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @field_validator("database_url")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v.startswith("postgresql"):
            raise ValueError("Only PostgreSQL is supported (use postgresql+asyncpg://)")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def available_providers(self) -> list[str]:
        """Return list of providers with valid API keys configured."""
        providers = ["ollama"]  # always available
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.google_api_key:
            providers.append("google")
        if self.groq_api_key:
            providers.append("groq")
        return providers


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call `get_settings.cache_clear()` in tests."""
    return Settings()
