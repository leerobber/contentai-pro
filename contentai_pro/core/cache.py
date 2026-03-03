"""In-memory LRU cache with TTL support."""
import time
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("contentai")


@dataclass
class CacheConfig:
    trend_ttl: int = 1800        # 30 minutes
    dna_profile_ttl: int = 3600  # 1 hour
    content_ttl: int = 900       # 15 minutes
    max_size: int = 512          # maximum total entries (LRU eviction)


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class LRUCache:
    """Thread-safe in-process LRU cache with per-entry TTL.

    Uses a ``threading.Lock`` to protect all mutations, making it safe for
    use from multiple threads or from sync code called via
    ``asyncio.run_in_executor``.  All public methods are synchronous.
    """

    def __init__(self, max_size: int = 512):
        self._max_size = max_size
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.time() > entry.expires_at:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: int) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = _CacheEntry(value=value, expires_at=time.time() + ttl)
            # Evict oldest entry if over capacity
            while len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("Cache evicted key: %s", evicted_key)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all entries whose key starts with prefix. Returns count removed."""
        with self._lock:
            keys_to_remove = [k for k in self._store if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._store[k]
            return len(keys_to_remove)

    def stats(self) -> dict:
        """Return a point-in-time snapshot of cache statistics. Values may be stale after return."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0


class AppCache:
    """Namespaced cache buckets for trends, DNA profiles, and content."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._cache = LRUCache(max_size=self.config.max_size)

    # --- Trend data ---

    def get_trends(self, niche: Optional[str] = None) -> Optional[Any]:
        return self._cache.get(f"trends:{niche or '__all__'}")

    def set_trends(self, data: Any, niche: Optional[str] = None) -> None:
        self._cache.set(f"trends:{niche or '__all__'}", data, self.config.trend_ttl)

    # --- DNA profiles ---

    def get_dna_profile(self, name: str) -> Optional[Any]:
        return self._cache.get(f"dna:{name}")

    def set_dna_profile(self, name: str, data: Any) -> None:
        self._cache.set(f"dna:{name}", data, self.config.dna_profile_ttl)

    def invalidate_dna_profile(self, name: str) -> None:
        self._cache.delete(f"dna:{name}")

    # --- Content by ID ---

    def get_content(self, content_id: str) -> Optional[Any]:
        return self._cache.get(f"content:{content_id}")

    def set_content(self, content_id: str, data: Any) -> None:
        self._cache.set(f"content:{content_id}", data, self.config.content_ttl)

    def invalidate_content(self, content_id: str) -> None:
        self._cache.delete(f"content:{content_id}")

    # --- Stats ---

    def stats(self) -> dict:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()


# Global singleton
app_cache = AppCache()
