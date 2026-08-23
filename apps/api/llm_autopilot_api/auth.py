"""
Minimal shared-secret admin authentication for mutating admin endpoints.

Not a full auth system — no per-user identity, no RBAC, no token
expiry. Just enough to stop an unauthenticated caller from silently
rewriting production routing behavior (and therefore cost) via
PUT /v1/admin/routing-config, which is the one endpoint this project
currently needs to protect. Upgrade to a real identity provider if/when
this project needs per-caller audit trails beyond the free-text
`updated_by` field on RoutingConfigUpdateRequest.

Read-only endpoints (GET /v1/models, GET /v1/stats, GET
/v1/admin/routing-config[/versions]) are intentionally NOT behind this —
they're operational visibility, same trust level as the Prometheus
/metrics endpoint.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status
from llm_autopilot_core.config import get_settings


async def require_admin_api_key(
    x_admin_api_key: str | None = Header(default=None),
) -> None:
    settings = get_settings()
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key is not configured on this server (set ADMIN_API_KEY)",
        )

    provided = x_admin_api_key or ""
    expected = settings.admin_api_key.get_secret_value()
    # Constant-time compare — this is the one place in the codebase that
    # compares a caller-supplied secret against a server-side one.
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-Admin-API-Key header",
        )
