# amifi_ai/api/middleware.py
"""
Request logging, timing, and exception middleware.
"""

import logging
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with:
    - Unique request ID
    - Method + path
    - Status code
    - Wall-clock latency
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        # Attach request ID for downstream use
        request.state.request_id = request_id

        logger.info(
            f"[REQ {request_id}] → {request.method} {request.url.path} "
            f"client={request.client.host if request.client else 'unknown'}"
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                f"[REQ {request_id}] ✗ UNHANDLED {type(exc).__name__}: {exc} "
                f"({elapsed:.1f}ms)"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(exc),
                    "request_id": request_id,
                },
            )

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"[REQ {request_id}] ← {response.status_code} ({elapsed:.1f}ms)"
        )

        # Inject request ID into response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
        return response
