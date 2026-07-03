"""
Alembic migration environment — async SQLAlchemy setup.

Reads DATABASE_URL from environment (overrides alembic.ini url).
Imports Base from llm_autopilot_core.database so Alembic can autogenerate
migrations by diffing ORM metadata against the live schema.

Add new ORM model modules to the import block below so Alembic sees them.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

# ── Make sure all models are imported before autogenerate ─────────────────────
# As you add ORM model files in Phase 5, add imports here:
#   from llm_autopilot_core.models.requests import Request          # noqa: F401
#   from llm_autopilot_core.models.responses import Response        # noqa: F401
#   from llm_autopilot_core.models.routing import RoutingDecision   # noqa: F401
#   from llm_autopilot_core.models.verification import Verification # noqa: F401
#   from llm_autopilot_core.models.costs import CostAggregate       # noqa: F401
from llm_autopilot_core.database import Base  # noqa: F401 — registers metadata
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Alembic config ────────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from llm_autopilot_core.config import get_settings

# Override URL from environment/settings — takes precedence over alembic.ini
settings = get_settings()
database_url = settings.database_url
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

target_metadata = Base.metadata


# ── Offline mode (generate SQL without a live DB) ─────────────────────────────
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online (async) mode ───────────────────────────────────────────────────────
def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # no pooling in migration scripts
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
