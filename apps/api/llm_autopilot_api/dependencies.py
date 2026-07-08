"""
FastAPI dependency injection components.

Usage:
    from llm_autopilot_api.dependencies import get_session, get_redis

    @router.get("/example")
    async def example(
        session: AsyncSession = Depends(get_session),
        redis: Redis = Depends(get_redis),
    ):
        ...
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import redis.asyncio as aioredis
from llm_autopilot_core.config import get_settings
from llm_autopilot_core.database import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession

settings = get_settings()

# ── Database ──────────────────────────────────────────────────────────────────


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_async_session():
        yield session


# ── Redis ─────────────────────────────────────────────────────────────────────

_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Return a singleton async Redis client from the connection pool."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(  # type: ignore[no-untyped-call]
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False,  # raw bytes for vector storage
            max_connections=5,
        )
    return _redis_pool


async def close_redis() -> None:
    global _redis_pool
    if _redis_pool:
        await _redis_pool.close()
        _redis_pool = None
