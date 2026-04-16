"""
contentai_pro/ai/llm_adapter.py — Unified LLM Interface
=========================================================
Provider priority:
  0. Sovereign Core Gateway (local GPU cluster — RTX 5050 → Radeon 780M → Ryzen 7)
  1. Anthropic Claude
  2. OpenAI GPT-4o
  3. Mock (testing)

Token counting, cost estimation, retry with exponential backoff,
and per-run usage tracking via contextvars are all preserved from
the original implementation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from contentai_pro.core.config import settings

logger = logging.getLogger("contentai")

# Per-run usage context (set by Orchestrator.run())
_run_usage_var: ContextVar[Optional["LLMUsage"]] = ContextVar(
    "_run_usage_var", default=None
)


# ── Exceptions ──────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Base LLM exception."""

class RateLimitError(LLMError):
    """Rate limit hit."""

class LLMTimeoutError(LLMError):
    """Request timed out."""


# ── Cost table ───────────────────────────────────────────────────────────────

COST_TABLE: Dict[str, Dict[str, float]] = {
    "sovereign": {"input": 0.0, "output": 0.0},         # Free — local GPU
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "mock": {"input": 0.0, "output": 0.0},
}

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


# ── Usage tracking ────────────────────────────────────────────────────────────

@dataclass
class LLMUsage:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    sovereign_calls: int = 0
    fallback_calls: int = 0
    call_log: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: str = "unknown",
    ) -> None:
        rates = COST_TABLE.get(model, COST_TABLE.get("gpt-4o", {"input": 0, "output": 0}))
        cost = (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.total_calls += 1
        if provider == "sovereign":
            self.sovereign_calls += 1
        elif provider != "mock":
            self.fallback_calls += 1
        self.call_log.append({
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })

    def summary(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "sovereign_calls": self.sovereign_calls,
            "fallback_calls": self.fallback_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


# ── Core LLM function ─────────────────────────────────────────────────────────

async def llm(
    system: str,
    user: str,
    model: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    json_mode: bool = False,
) -> str:
    """
    Unified LLM call. Tries Sovereign Core first, falls back to cloud providers.
    Retries transient errors with exponential backoff.
    """
    resolved_model = model or settings.default_model

    for attempt in range(MAX_RETRIES):
        try:
            result, provider, in_tok, out_tok = await _dispatch(
                system=system,
                user=user,
                model=resolved_model,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
            )

            # Record usage
            usage = _run_usage_var.get()
            if usage is not None:
                usage.record(resolved_model, in_tok, out_tok, provider=provider)

            return result

        except RateLimitError:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("Rate limited — retrying in %.1fs (attempt %d)", delay, attempt + 1)
                await asyncio.sleep(delay)
            else:
                raise
        except LLMTimeoutError:
            if attempt < MAX_RETRIES - 1:
                logger.warning("LLM timeout — retrying (attempt %d)", attempt + 1)
                await asyncio.sleep(RETRY_BASE_DELAY)
            else:
                raise

    raise LLMError("All retry attempts exhausted")


async def _dispatch(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> tuple[str, str, int, int]:
    """
    Returns (response_text, provider_name, input_tokens, output_tokens).
    Tries Sovereign Core first, then cloud providers, then mock.
    """
    # ── 0. Sovereign Core Gateway ──────────────────────────────────────────
    if getattr(settings, "sovereign_enabled", True):
        try:
            from contentai_pro.ai.llm_sovereign import sovereign_llm
            result = await sovereign_llm.complete(
                prompt=user,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.debug(
                "Sovereign Core: backend=%s latency=%.0fms",
                result.backend_id, result.latency_ms
            )
            return result.text, "sovereign", result.prompt_tokens, result.completion_tokens
        except Exception as exc:
            logger.info("Sovereign Core unavailable (%s) — trying cloud", exc)

    # ── 1. Anthropic Claude ────────────────────────────────────────────────
    if settings.anthropic_api_key:
        try:
            return await _call_anthropic(system, user, model, max_tokens, temperature, json_mode)
        except RateLimitError:
            raise
        except Exception as exc:
            logger.warning("Anthropic error: %s — trying OpenAI", exc)

    # ── 2. OpenAI ──────────────────────────────────────────────────────────
    if settings.openai_api_key:
        try:
            return await _call_openai(system, user, model, max_tokens, temperature, json_mode)
        except RateLimitError:
            raise
        except Exception as exc:
            logger.warning("OpenAI error: %s — falling back to mock", exc)

    # ── 3. Mock (dev / testing) ────────────────────────────────────────────
    logger.warning("All LLM providers unavailable — using mock response")
    mock_text = f"[MOCK] system={system[:50]} | user={user[:100]}"
    return mock_text, "mock", len(system.split()) * 4, len(mock_text.split()) * 4


async def _call_anthropic(
    system, user, model, max_tokens, temperature, json_mode
) -> tuple[str, str, int, int]:
    import aiohttp
    claude_model = "claude-3-5-sonnet-20241022"
    payload: Dict[str, Any] = {
        "model": claude_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if json_mode:
        payload["system"] = system + "\nRespond with valid JSON only."

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as r:
            if r.status == 429:
                raise RateLimitError("Anthropic rate limit")
            if r.status >= 500:
                raise LLMError(f"Anthropic server error {r.status}")
            r.raise_for_status()
            data = await r.json()

    text = data["content"][0]["text"]
    return text, "anthropic", data["usage"]["input_tokens"], data["usage"]["output_tokens"]


async def _call_openai(
    system, user, model, max_tokens, temperature, json_mode
) -> tuple[str, str, int, int]:
    import aiohttp
    oai_model = "gpt-4o" if "gpt" not in model else model
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    payload: Dict[str, Any] = {
        "model": oai_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as r:
            if r.status == 429:
                raise RateLimitError("OpenAI rate limit")
            r.raise_for_status()
            data = await r.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, "openai", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
