"""
Request/response logging middleware for the Water Potability Prediction API.

Logs each incoming request method, path, and response status code with
timing information for observability and debugging.
"""

import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__, "api.log")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that logs every HTTP request with timing.

    Logs at INFO level:
        - HTTP method and URL path
        - Response status code
        - Wall-clock processing time in milliseconds

    Exceptions during request processing are logged at ERROR level
    and re-raised to allow FastAPI's exception handlers to process them.
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        method = request.method
        path   = request.url.path

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"{method} {path} → {response.status_code} "
                f"({elapsed_ms:.1f}ms)"
            )
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"{method} {path} → ERROR after {elapsed_ms:.1f}ms: {exc}"
            )
            raise
