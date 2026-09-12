"""API key authentication for mutating Recall endpoints."""

from __future__ import annotations

import secrets
from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def build_api_key_guard(required_api_key: str | None) -> Callable[..., None]:
    """Returns a FastAPI dependency that enforces ``RAG_API_KEY`` when configured.

    When ``required_api_key`` is unset or blank, the dependency is a no-op so local
    development remains zero-friction. When set, callers must provide either:
    - ``Authorization: Bearer <key>``
    - ``X-API-Key: <key>``
    """

    async def verify_api_key(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
        api_key_header: str | None = Depends(_api_key_header),
    ) -> None:
        if not required_api_key or not required_api_key.strip():
            return

        presented_key = _extract_presented_key(credentials, api_key_header)
        if presented_key is None or not secrets.compare_digest(presented_key, required_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return verify_api_key


def _extract_presented_key(
    credentials: HTTPAuthorizationCredentials | None,
    api_key_header: str | None,
) -> str | None:
    if credentials and credentials.scheme.lower() == "bearer" and credentials.credentials:
        return credentials.credentials
    if api_key_header:
        return api_key_header
    return None
