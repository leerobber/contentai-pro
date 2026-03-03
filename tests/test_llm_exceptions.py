"""Tests for custom LLM exception classes and retry logic."""
from unittest.mock import patch

import pytest

from contentai_pro.ai.llm_adapter import LLMAdapter, LLMError, LLMTimeoutError, RateLimitError

# ---------- Exception hierarchy ----------

def test_rate_limit_is_llm_error():
    exc = RateLimitError("too many requests")
    assert isinstance(exc, LLMError)


def test_timeout_is_llm_error():
    exc = LLMTimeoutError("request timed out")
    assert isinstance(exc, LLMError)


def test_llm_error_message():
    exc = LLMError("something went wrong")
    assert "something went wrong" in str(exc)


# ---------- Retry logic ----------

@pytest.mark.asyncio
async def test_generate_retries_on_rate_limit():
    """generate() should retry on RateLimitError and succeed on the second attempt."""
    adapter = LLMAdapter()
    adapter._provider = "mock"

    call_count = 0

    async def flaky_mock(system, prompt, json_mode=False):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RateLimitError("rate limited")
        return "success"

    with patch.object(adapter, "_mock_generate", side_effect=flaky_mock):
        result = await adapter.generate("sys", "prompt", _retries=3, _backoff=0.0)

    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_raises_after_max_retries():
    """generate() should re-raise RateLimitError after exhausting retries."""
    adapter = LLMAdapter()
    adapter._provider = "mock"

    async def always_rate_limit(system, prompt, json_mode=False):
        raise RateLimitError("always limited")

    with patch.object(adapter, "_mock_generate", side_effect=always_rate_limit):
        with pytest.raises(RateLimitError):
            await adapter.generate("sys", "prompt", _retries=2, _backoff=0.0)


@pytest.mark.asyncio
async def test_generate_retries_on_timeout():
    """generate() should retry on LLMTimeoutError."""
    adapter = LLMAdapter()
    adapter._provider = "mock"

    call_count = 0

    async def timeout_then_ok(system, prompt, json_mode=False):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise LLMTimeoutError("timed out")
        return "ok"

    with patch.object(adapter, "_mock_generate", side_effect=timeout_then_ok):
        result = await adapter.generate("sys", "prompt", _retries=3, _backoff=0.0)

    assert result == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_does_not_retry_generic_llm_error():
    """Non-transient LLMError should propagate immediately without retrying."""
    adapter = LLMAdapter()
    adapter._provider = "mock"

    call_count = 0

    async def generic_error(system, prompt, json_mode=False):
        nonlocal call_count
        call_count += 1
        raise LLMError("unexpected error")

    with patch.object(adapter, "_mock_generate", side_effect=generic_error):
        with pytest.raises(LLMError):
            await adapter.generate("sys", "prompt", _retries=3, _backoff=0.0)

    # Should only be called once — no retry for generic LLMError
    assert call_count == 1
