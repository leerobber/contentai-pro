"""Rate Limiting Middleware for FastAPI."""
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("contentai")

# Paths that bypass rate limiting
_SKIP_PATHS = {"/api/health", "/api/rate-limit"}
_SKIP_PREFIXES = ("/static", "/favicon")


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10       # max concurrent requests per client


@dataclass
class ClientState:
    # Sliding-window timestamps (deque keeps only within the window)
    minute_window: deque = field(default_factory=deque)
    hour_window: deque = field(default_factory=deque)
    active_requests: int = 0


class RateLimiter:
    """Sliding-window rate limiter with per-client tracking."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._clients: Dict[str, ClientState] = {}

    def _get_state(self, client_ip: str) -> ClientState:
        if client_ip not in self._clients:
            self._clients[client_ip] = ClientState()
        return self._clients[client_ip]

    def _prune_window(self, window: deque, cutoff: float) -> None:
        while window and window[0] < cutoff:
            window.popleft()

    def check(self, client_ip: str) -> tuple[bool, str, int]:
        """
        Check if the request is within rate limits.

        Returns:
            (allowed, reason, retry_after_seconds)
        """
        now = time.time()
        state = self._get_state(client_ip)

        # Prune expired timestamps
        self._prune_window(state.minute_window, now - 60)
        self._prune_window(state.hour_window, now - 3600)

        if len(state.minute_window) >= self.config.requests_per_minute:
            retry_after = int(60 - (now - state.minute_window[0])) + 1
            return False, "per-minute limit exceeded", retry_after

        if len(state.hour_window) >= self.config.requests_per_hour:
            retry_after = int(3600 - (now - state.hour_window[0])) + 1
            return False, "per-hour limit exceeded", retry_after

        if state.active_requests >= self.config.burst_limit:
            return False, "burst limit exceeded", 5

        # Record this request
        state.minute_window.append(now)
        state.hour_window.append(now)
        state.active_requests += 1
        return True, "", 0

    def release(self, client_ip: str) -> None:
        """Decrement active request counter after request completes."""
        if client_ip in self._clients:
            state = self._clients[client_ip]
            if state.active_requests > 0:
                state.active_requests -= 1

    def get_stats(self, client_ip: str) -> dict:
        """Return current rate-limit statistics for a client IP."""
        now = time.time()
        state = self._get_state(client_ip)
        self._prune_window(state.minute_window, now - 60)
        self._prune_window(state.hour_window, now - 3600)
        return {
            "client_ip": client_ip,
            "requests_last_minute": len(state.minute_window),
            "requests_last_hour": len(state.hour_window),
            "active_requests": state.active_requests,
            "limits": {
                "per_minute": self.config.requests_per_minute,
                "per_hour": self.config.requests_per_hour,
                "burst": self.config.burst_limit,
            },
        }


# Global singleton
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces per-IP rate limits."""

    def __init__(self, app, limiter: Optional[RateLimiter] = None):
        super().__init__(app)
        self._limiter = limiter or rate_limiter

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Skip rate limiting for health/static routes
        if path in _SKIP_PATHS or any(path.startswith(p) for p in _SKIP_PREFIXES):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        allowed, reason, retry_after = self._limiter.check(client_ip)

        if not allowed:
            logger.warning("Rate limit exceeded for %s: %s", client_ip, reason)
            return JSONResponse(
                status_code=429,
                content={"detail": f"Too many requests: {reason}"},
                headers={"Retry-After": str(retry_after)},
            )

        try:
            response = await call_next(request)
        finally:
            self._limiter.release(client_ip)

        return response
