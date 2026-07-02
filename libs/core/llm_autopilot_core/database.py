"""
Async database layer using SQLAlchemy 2.0 + asyncpg.

Usage in FastAPI:
    from llm_autopilot_core.database import get_async_session

    @router.get("/example")
    async def example(session: AsyncSession = Depends(get_async_session)):
        ...

Usage elsewhere (Celery tasks, scripts):
    from llm_autopilot_core.database import managed_session

    async with managed_session() as session:
        result = await session.execute(...)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from llm_autopilot_core.config import get_settings

settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_timeout=settings.database_pool_timeout,
    pool_pre_ping=True,          # detect stale connections before checkout
    echo=settings.debug,         # log SQL in dev only
    echo_pool=settings.debug,
)

# ── Session factory ───────────────────────────────────────────────────────────

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # attributes stay accessible after commit
    autoflush=False,
    autocommit=False,
)


# ── Declarative base (all ORM models inherit from this) ───────────────────────

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    Import this in each model module to register the table with Alembic:
        from llm_autopilot_core.database import Base
    """
    pass


# ── Session helpers ───────────────────────────────────────────────────────────

@asynccontextmanager
async def managed_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager that commits on success and rolls back on exception.
    Safe to use outside of FastAPI's dependency injection (Celery, scripts).
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields a session per request.

    FastAPI handles the finally cleanup via DI lifecycle, but we still
    flush + rollback on exceptions to keep the connection clean.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Startup / teardown helpers ────────────────────────────────────────────────

async def check_connection() -> bool:
    """Ping the database. Used by /readyz health endpoint."""
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception:
        return False


async def dispose_engine() -> None:
    """Gracefully close all pooled connections. Call on application shutdown."""
    await engine.dispose()
