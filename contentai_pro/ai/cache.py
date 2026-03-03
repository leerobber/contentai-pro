"""Semantic caching layer for expensive LLM operations.

Provides an in-memory cache with TTL support, LRU eviction, and concurrent
stampede protection. Can be extended to Redis.
"""
import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple


class SemanticCache:
    """In-memory semantic cache with TTL, LRU eviction, and stampede protection."""

    def __init__(self, default_ttl: int = 3600, max_size: int = 256):
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()  # hash -> (value, expiry)
        self.default_ttl = default_ttl
        self.max_size = max_size
        # Per-key locks to prevent concurrent cache stampedes
        self._inflight: Dict[str, asyncio.Lock] = {}

    def _hash_prompt(self, system: str, prompt: str) -> str:
        content = f"{system}|||{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _evict_expired(self) -> None:
        """Remove all expired entries from the cache."""
        now = time.time()
        expired = [k for k, (_, expiry) in self._cache.items() if expiry <= now]
        for k in expired:
            del self._cache[k]

    def _evict_lru(self) -> None:
        """Evict entries until the cache is within max_size. Expired entries first."""
        self._evict_expired()
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest (LRU) entry

    async def get_or_generate(
        self,
        system: str,
        prompt: str,
        generator_fn: Callable,
        ttl: Optional[int] = None,
    ) -> Any:
        cache_key = self._hash_prompt(system, prompt)
        ttl = ttl if ttl is not None else self.default_ttl

        # Fast path: cache hit (no lock needed for read)
        if cache_key in self._cache:
            value, expiry = self._cache[cache_key]
            if time.time() < expiry:
                # Move to end to mark as recently used
                self._cache.move_to_end(cache_key)
                return value
            else:
                del self._cache[cache_key]

        # Acquire a per-key lock so only one coroutine calls the generator
        if cache_key not in self._inflight:
            self._inflight[cache_key] = asyncio.Lock()
        lock = self._inflight[cache_key]

        async with lock:
            # Re-check after acquiring the lock (another coroutine may have populated it)
            if cache_key in self._cache:
                value, expiry = self._cache[cache_key]
                if time.time() < expiry:
                    self._cache.move_to_end(cache_key)
                    return value

            # Generate, evict if needed, then cache
            result = await generator_fn()
            self._evict_lru()
            self._cache[cache_key] = (result, time.time() + ttl)

        # Clean up the per-key lock once done
        self._inflight.pop(cache_key, None)
        return result

    def invalidate(self, system: str, prompt: str) -> None:
        cache_key = self._hash_prompt(system, prompt)
        self._cache.pop(cache_key, None)

    def clear(self) -> None:
        self._cache.clear()


# Singleton cache instance
semantic_cache = SemanticCache()
