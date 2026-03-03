"""Tests for the SemanticCache class."""
import asyncio
import time as _time

import pytest

from contentai_pro.ai.cache import SemanticCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _const(value):
    """Coroutine that returns a constant value."""
    return value


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_miss_calls_generator():
    """On a cache miss, generator_fn must be called once and its result returned."""
    cache = SemanticCache()
    calls = []

    async def gen():
        calls.append(1)
        return "result"

    result = await cache.get_or_generate("sys", "prompt", gen)
    assert result == "result"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_cache_hit_does_not_call_generator():
    """On a cache hit, generator_fn must NOT be called again."""
    cache = SemanticCache()
    calls = []

    async def gen():
        calls.append(1)
        return "cached"

    await cache.get_or_generate("sys", "prompt", gen)
    result = await cache.get_or_generate("sys", "prompt", gen)
    assert result == "cached"
    assert len(calls) == 1  # generator called only once


@pytest.mark.asyncio
async def test_different_prompts_are_independent():
    """Two different prompts should each call their respective generators."""
    cache = SemanticCache()

    result_a = await cache.get_or_generate("sys", "prompt_a", lambda: _const("a"))
    result_b = await cache.get_or_generate("sys", "prompt_b", lambda: _const("b"))
    assert result_a == "a"
    assert result_b == "b"


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expired_entry_triggers_regeneration(monkeypatch):
    """An expired cache entry should cause generator_fn to be called again."""
    cache = SemanticCache(default_ttl=1)
    calls = []

    async def gen():
        calls.append(len(calls))
        return f"v{len(calls)}"

    await cache.get_or_generate("sys", "p", gen, ttl=0)  # TTL=0 → expires immediately

    # Advance time slightly past expiry
    real_time = _time.time
    monkeypatch.setattr("contentai_pro.ai.cache.time.time", lambda: real_time() + 2)

    await cache.get_or_generate("sys", "p", gen)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lru_eviction_removes_oldest_entry():
    """When cache is full, inserting a new entry should evict the oldest one."""
    cache = SemanticCache(max_size=2)

    await cache.get_or_generate("s", "a", lambda: _const("va"), ttl=3600)
    await cache.get_or_generate("s", "b", lambda: _const("vb"), ttl=3600)

    # This should evict "a" (oldest)
    await cache.get_or_generate("s", "c", lambda: _const("vc"), ttl=3600)

    assert len(cache._cache) <= 2
    key_a = cache._hash_prompt("s", "a")
    assert key_a not in cache._cache


# ---------------------------------------------------------------------------
# Stampede protection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_cache_miss_calls_generator_once():
    """Concurrent misses for the same key should only call generator_fn once."""
    cache = SemanticCache()
    call_count = 0

    async def slow_gen():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)  # simulate latency
        return "generated"

    results = await asyncio.gather(
        cache.get_or_generate("sys", "prompt", slow_gen),
        cache.get_or_generate("sys", "prompt", slow_gen),
        cache.get_or_generate("sys", "prompt", slow_gen),
    )

    assert all(r == "generated" for r in results)
    assert call_count == 1


# ---------------------------------------------------------------------------
# invalidate / clear
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalidate_removes_entry():
    """invalidate() should remove a cached entry so the next call regenerates."""
    cache = SemanticCache()
    calls = []

    async def gen():
        calls.append(1)
        return "v"

    await cache.get_or_generate("s", "p", gen)
    cache.invalidate("s", "p")
    await cache.get_or_generate("s", "p", gen)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_clear_removes_all_entries():
    """clear() should empty the entire cache."""
    cache = SemanticCache()
    await cache.get_or_generate("s", "p1", lambda: _const("a"), ttl=3600)
    await cache.get_or_generate("s", "p2", lambda: _const("b"), ttl=3600)
    cache.clear()
    assert len(cache._cache) == 0
