"""Middleware — Request ID injection + access logging."""
import uuid
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("contentai")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"[{request_id}] Unhandled exception after {elapsed:.0f}ms: {exc}", exc_info=True)
            raise
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"[{request_id}] {request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)")
        response.headers["X-Request-ID"] = request_id
        return response
