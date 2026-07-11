"""
Reusable SQLAlchemy column type helpers.

We store Python enums (ComplexityTier, Provider, VerificationStatus,
EscalationReason) as plain VARCHAR — `native_enum=False` — rather than
native Postgres ENUM types. Native enums require `ALTER TYPE ... ADD VALUE`
to add a new tier/status/provider later, which historically couldn't run
inside a transaction and complicates rollbacks. VARCHAR + a Python-side
StrEnum gives the same validation with painless migrations when a new
value shows up (e.g. a sixth provider).

`create_constraint=False` keeps the DB from adding a CHECK constraint we
didn't write into the migration by hand — this way the ORM model and the
hand-authored migration stay in exact agreement, so a future
`alembic check` / autogenerate diff won't propose "fixing" a mismatch.

`values_callable` stores each enum member's `.value` (e.g. "simple")
rather than SQLAlchemy's default of `.name` (e.g. "SIMPLE") — our
StrEnum values in schemas.py are lowercase, and this keeps DB rows
matching what the API actually sends/receives.
"""

from __future__ import annotations

from enum import Enum as PyEnum
from typing import TypeVar

from sqlalchemy import Enum as SAEnum

E = TypeVar("E", bound=PyEnum)


def enum_column(enum_cls: type[E], *, length: int = 32) -> SAEnum:
    return SAEnum(
        enum_cls,
        native_enum=False,
        create_constraint=False,
        length=length,
        values_callable=lambda obj: [e.value for e in obj],
    )
