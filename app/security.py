"""
API-key security middleware for FastAPI.
"""

from __future__ import annotations

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from loguru import logger

from app.config import get_config

# Paths that skip auth (health-checks, docs)
_PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate x-api-key header on every request (if enabled in config)."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        cfg = get_config()

        # Skip if security disabled
        if not cfg.security.api_key_enabled:
            return await call_next(request)

        # Skip public endpoints
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("x-api-key")
        if not api_key or api_key not in cfg.security.api_keys:
            logger.warning(
                f"Unauthorized request from {request.client.host} to {request.url.path}"
            )
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
