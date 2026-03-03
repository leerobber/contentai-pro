"""Semantic caching layer for expensive LLM operations.

Provides an in-memory cache with TTL support. Can be extended to Redis.
"""
import hashlib
import time
from typing import Any, Callable, Dict, Optional, Tuple


class SemanticCache:
    """In-memory semantic cache with TTL support. Can be extended to Redis."""

    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, Tuple[Any, float]] = {}  # hash -> (value, expiry)
        self.default_ttl = default_ttl

    def _hash_prompt(self, system: str, prompt: str) -> str:
        content = f"{system}|||{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get_or_generate(
        self,
        system: str,
        prompt: str,
        generator_fn: Callable,
        ttl: Optional[int] = None,
    ) -> Any:
        cache_key = self._hash_prompt(system, prompt)
        ttl = ttl if ttl is not None else self.default_ttl

        # Check cache
        if cache_key in self._cache:
            value, expiry = self._cache[cache_key]
            if time.time() < expiry:
                return value
            else:
                del self._cache[cache_key]  # Expired

        # Generate and cache
        result = await generator_fn()
        self._cache[cache_key] = (result, time.time() + ttl)
        return result

    def invalidate(self, system: str, prompt: str) -> None:
        cache_key = self._hash_prompt(system, prompt)
        self._cache.pop(cache_key, None)

    def clear(self) -> None:
        self._cache.clear()


# Singleton cache instance
semantic_cache = SemanticCache()
