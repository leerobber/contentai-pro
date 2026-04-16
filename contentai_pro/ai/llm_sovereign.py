"""
contentai_pro/ai/llm_sovereign.py — Sovereign Core Gateway Adapter

Replaces direct Anthropic/OpenAI calls with routing through the
Heterogeneous Compute Gateway. Falls back to cloud providers when
the local cluster is unavailable.

Priority order:
  1. Sovereign Core Gateway (localhost:8000) — RTX 5050 → Radeon 780M → Ryzen 7
  2. Anthropic Claude (cloud fallback)
  3. OpenAI GPT-4o (final fallback)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger("contentai.sovereign")

GATEWAY_URL = os.getenv("SOVEREIGN_GATEWAY_URL", "http://localhost:8000")
GATEWAY_TIMEOUT = float(os.getenv("SOVEREIGN_GATEWAY_TIMEOUT", "30"))
GATEWAY_ENABLED = os.getenv("SOVEREIGN_GATEWAY_ENABLED", "true").lower() == "true"


# ── Response model ──────────────────────────────────────────────────────────

@dataclass
class SovereignResponse:
    text: str
    model: str
    backend_id: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    routed_via: str = "sovereign"
    fallback_used: bool = False


# ── Gateway health probe ─────────────────────────────────────────────────────

_gateway_healthy: bool = True
_last_health_check: float = 0.0
_HEALTH_TTL = 30.0  # re-check every 30s


async def _check_gateway_health(client: httpx.AsyncClient) -> bool:
    global _gateway_healthy, _last_health_check
    now = time.time()
    if now - _last_health_check < _HEALTH_TTL:
        return _gateway_healthy
    try:
        r = await client.get(f"{GATEWAY_URL}/health", timeout=5.0)
        _gateway_healthy = r.status_code == 200
    except Exception:
        _gateway_healthy = False
    _last_health_check = now
    return _gateway_healthy


# ── Main adapter ────────────────────────────────────────────────────────────

class SovereignLLMAdapter:
    """
    Drop-in replacement for contentai_pro.ai.llm_adapter.LLMAdapter.
    Routes inference requests through the Sovereign Core gateway with
    automatic fallback to cloud providers.
    """

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=GATEWAY_TIMEOUT)

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str = "auto",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> SovereignResponse:
        """Send a completion request, preferring the local GPU cluster."""
        t0 = time.time()

        if GATEWAY_ENABLED and await _check_gateway_health(self._client):
            try:
                return await self._sovereign_complete(
                    prompt, system, model, max_tokens, temperature, t0
                )
            except Exception as exc:
                logger.warning("Sovereign gateway error — falling back to cloud: %s", exc)

        # Cloud fallback
        return await self._cloud_complete(prompt, system, max_tokens, temperature, t0)

    async def _sovereign_complete(
        self, prompt, system, model, max_tokens, temperature, t0
    ) -> SovereignResponse:
        payload = {
            "model": model,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        r = await self._client.post(
            f"{GATEWAY_URL}/inference",
            json=payload,
            timeout=GATEWAY_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return SovereignResponse(
            text=data.get("response", ""),
            model=data.get("model", model),
            backend_id=data.get("backend_id", "unknown"),
            latency_ms=(time.time() - t0) * 1000,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            routed_via="sovereign",
        )

    async def _cloud_complete(
        self, prompt, system, max_tokens, temperature, t0
    ) -> SovereignResponse:
        """Fallback: try Anthropic, then OpenAI."""
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                return await self._anthropic_complete(
                    prompt, system, max_tokens, temperature, t0, anthropic_key
                )
            except Exception as exc:
                logger.warning("Anthropic fallback failed: %s", exc)

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            return await self._openai_complete(
                prompt, system, max_tokens, temperature, t0, openai_key
            )

        raise RuntimeError("All LLM backends unavailable — no API keys configured")

    async def _anthropic_complete(
        self, prompt, system, max_tokens, temperature, t0, api_key
    ) -> SovereignResponse:
        messages = [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system

        r = await self._client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        r.raise_for_status()
        data = r.json()
        return SovereignResponse(
            text=data["content"][0]["text"],
            model=data["model"],
            backend_id="anthropic-cloud",
            latency_ms=(time.time() - t0) * 1000,
            prompt_tokens=data["usage"]["input_tokens"],
            completion_tokens=data["usage"]["output_tokens"],
            routed_via="anthropic",
            fallback_used=True,
        )

    async def _openai_complete(
        self, prompt, system, max_tokens, temperature, t0, api_key
    ) -> SovereignResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        r = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return SovereignResponse(
            text=choice["message"]["content"],
            model=data["model"],
            backend_id="openai-cloud",
            latency_ms=(time.time() - t0) * 1000,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            routed_via="openai",
            fallback_used=True,
        )

    async def close(self):
        await self._client.aclose()


# Module-level singleton
sovereign_llm = SovereignLLMAdapter()
