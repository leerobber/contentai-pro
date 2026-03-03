"""LLM Adapter — unified interface for Anthropic / OpenAI / Mock.

FIX: Token counting + cost estimation per call.
FIX: Retry logic with exponential backoff for transient failures.
"""
import json
import logging
import random
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from contentai_pro.core.config import settings

logger = logging.getLogger("contentai")

# Per-run usage tracking — set by Orchestrator.run() via contextvars so concurrent
# requests each accumulate their own token/cost totals without interfering.
_run_usage_var: ContextVar[Optional["LLMUsage"]] = ContextVar("_run_usage_var", default=None)


class LLMError(Exception):
    """Base exception for LLM errors."""


class RateLimitError(LLMError):
    """Raised when the LLM API returns a rate-limit response."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM API call times out (avoids shadowing built-in TimeoutError)."""


# Cost per 1K tokens (input/output) — updated as pricing changes
COST_TABLE = {
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "mock": {"input": 0.0, "output": 0.0},
}

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds


@dataclass
class LLMUsage:
    """Tracks cumulative token usage and costs across a session."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    call_log: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, input_tokens: int, output_tokens: int, model: str):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1
        costs = COST_TABLE.get(model, COST_TABLE.get("mock", {"input": 0.0, "output": 0.0}))
        call_cost = (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])
        self.total_cost_usd += call_cost
        self.call_log.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(call_cost, 6),
            "model": model,
        })

    def summary(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 4),
        }


class LLMAdapter:
    """Provider-agnostic LLM interface with prompt routing."""

    def __init__(self):
        self._provider = settings.LLM_PROVIDER
        self._client = None
        self.usage = LLMUsage()
        self._init_client()

    @property
    def provider(self) -> str:
        """Return the active provider name."""
        return self._provider

    def reset_usage(self):
        """Reset usage tracking (call at start of each pipeline run)."""
        self.usage = LLMUsage()

    def _init_client(self):
        if self._provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            except ImportError:
                logger.warning("anthropic package not installed; falling back to mock mode.")
                self._provider = "mock"
        elif self._provider == "openai" and settings.OPENAI_API_KEY:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            except ImportError:
                logger.warning("openai package not installed; falling back to mock mode.")
                self._provider = "mock"
        else:
            if self._provider != "mock":
                logger.warning(
                    f"LLM_PROVIDER='{self._provider}' but no API key is set; falling back to mock mode."
                )
            self._provider = "mock"

    async def generate(self, system: str, prompt: str, max_tokens: int = None,
                       temperature: float = None, json_mode: bool = False) -> str:
        max_tokens = max_tokens or settings.MAX_TOKENS
        temperature = temperature if temperature is not None else settings.TEMPERATURE

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                if self._provider == "anthropic":
                    return await self._anthropic_generate(system, prompt, max_tokens, temperature)
                elif self._provider == "openai":
                    return await self._openai_generate(system, prompt, max_tokens, temperature, json_mode)
                else:
                    return await self._mock_generate(system, prompt, json_mode)
            except Exception as e:
                last_error = e
                if self._provider == "mock":
                    raise  # Mock shouldn't fail; if it does, it's a code bug
                # Only retry transient errors (rate limits, timeouts, transient connectivity).
                # Unrecoverable errors (auth failures, bad requests) are re-raised immediately.
                error_name = type(e).__name__
                is_transient = (
                    isinstance(e, (RateLimitError, LLMTimeoutError))
                    or "RateLimit" in error_name
                    or "Timeout" in error_name
                    or "ServiceUnavailable" in error_name
                    or "APIConnection" in error_name
                )
                if not is_transient:
                    raise
                wait = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait:.1f}s")
                await asyncio.sleep(wait)

        raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries: {last_error}") from last_error

    def _record_usage(self, input_tokens: int, output_tokens: int, model: str) -> None:
        """Record token usage to the singleton tracker and, if set, the per-run tracker."""
        self.usage.record(input_tokens, output_tokens, model)
        run_usage = _run_usage_var.get()
        if run_usage is not None:
            run_usage.record(input_tokens, output_tokens, model)

    async def _anthropic_generate(self, system: str, prompt: str,
                                   max_tokens: int, temperature: float) -> str:
        response = await self._client.messages.create(
            model=settings.MODEL_NAME,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        # Track tokens
        input_tokens = getattr(response.usage, 'input_tokens', 0)
        output_tokens = getattr(response.usage, 'output_tokens', 0)
        self._record_usage(input_tokens, output_tokens, settings.MODEL_NAME)
        return response.content[0].text

    async def _openai_generate(self, system: str, prompt: str,
                                max_tokens: int, temperature: float,
                                json_mode: bool) -> str:
        kwargs = {
            "model": settings.MODEL_NAME,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)
        # Track tokens
        if response.usage:
            self._record_usage(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                settings.MODEL_NAME,
            )
        return response.choices[0].message.content

    async def _mock_generate(self, system: str, prompt: str, json_mode: bool = False) -> str:
        """Deterministic mock for UI testing without API keys."""
        # Estimate mock tokens for tracking
        input_tokens = len(prompt.split()) + len(system.split())
        prompt_lower = prompt.lower()

        # Check specific agent roles FIRST (some contain "json" in their prompts)
        if "advocate" in system.lower():
            result = self._mock_advocate()
        elif "critic" in system.lower():
            result = self._mock_critic()
        elif "judge" in system.lower():
            result = self._mock_judge()
        elif "research" in system.lower():
            result = self._mock_research(prompt_lower)
        elif "write" in system.lower() or "draft" in system.lower():
            result = self._mock_article(prompt_lower)
        elif "edit" in system.lower() or "revis" in system.lower():
            result = self._mock_edit(prompt_lower)
        elif "seo" in system.lower():
            result = self._mock_seo(prompt_lower)
        elif json_mode or "json" in system.lower():
            result = self._mock_json(prompt_lower)
        else:
            result = f"[Mock LLM] Response to: {prompt[:100]}..."

        output_tokens = len(result.split())
        self._record_usage(input_tokens, output_tokens, "mock")
        return result

    def _mock_research(self, prompt: str) -> str:
        return (
            "## Research Summary\n\n"
            "**Key Findings:**\n"
            "1. The market shows 34% YoY growth in AI-powered content tools.\n"
            "2. Enterprise adoption accelerated in Q3 2025 with 67% of Fortune 500 companies using AI writing.\n"
            "3. Multi-agent architectures outperform single-model approaches by 2.3x on content quality benchmarks.\n\n"
            "**Sources:** McKinsey Digital Report 2025, Gartner AI Hype Cycle, Stanford HAI Index.\n\n"
            "**Competitive Landscape:** Jasper, Copy.ai, Writer.com dominate SMB. Enterprise gap exists for "
            "multi-agent pipelines with voice consistency — our differentiation vector."
        )

    def _mock_article(self, prompt: str) -> str:
        topic = prompt[:60].strip(".,!? ").title() if prompt else "AI Content Generation"
        return (
            f"# {topic}\n\n"
            "The landscape of content creation is undergoing a fundamental transformation. "
            "Where traditional approaches relied on single-pass generation, modern multi-agent "
            "systems orchestrate specialized AI models through research, drafting, editing, and "
            "optimization stages — producing content that rivals expert human writers.\n\n"
            "## The Multi-Agent Advantage\n\n"
            "Think of it as an AI newsroom. A researcher gathers facts and context. A writer "
            "crafts the narrative. An editor refines clarity and flow. An SEO specialist ensures "
            "discoverability. Each agent is tuned for its role, and the orchestrator manages handoffs.\n\n"
            "## Voice Consistency at Scale\n\n"
            "The hardest problem in AI content isn't quality — it's consistency. Our Content DNA Engine "
            "fingerprints writing style across 14 dimensions (sentence rhythm, vocabulary tier, "
            "metaphor density, technical depth, etc.) and enforces it across every piece.\n\n"
            "## Quality Through Adversarial Debate\n\n"
            "Before any content ships, it faces an Adversarial Debate: an Advocate defends it, "
            "a Critic attacks it, and a Judge scores it. Content that survives this gauntlet "
            "consistently outperforms single-pass generation by 40% on human evaluation.\n\n"
            "---\n\n"
            "*Generated by ContentAI Pro — Multi-Agent Content Pipeline*"
        )

    def _mock_edit(self, prompt: str) -> str:
        return "[Edited version with improved clarity, tighter transitions, and stronger opening hook.]"

    def _mock_seo(self, prompt: str) -> str:
        return (
            "## SEO Optimizations Applied\n\n"
            "- **Primary keyword:** AI content generation (vol: 12K/mo, KD: 45)\n"
            "- **Secondary:** multi-agent AI writing, content automation platform\n"
            "- **Title tag:** AI Content Generation: How Multi-Agent Systems Outperform Single Models\n"
            "- **Meta description:** Discover how multi-agent AI pipelines produce 2.3x better content "
            "with voice consistency and adversarial quality control.\n"
            "- **H2 optimization:** ✅ All subheadings contain target keywords\n"
            "- **Internal links suggested:** 3 (content strategy, AI writing tools, voice branding)\n"
            "- **Readability:** Flesch-Kincaid 62 (ideal for B2B tech audience)"
        )

    def _mock_advocate(self) -> str:
        return (
            "This content excels in three areas: (1) Clear structure with logical flow from problem "
            "to solution. (2) Concrete data points — the 34% growth and 2.3x quality improvement "
            "provide credibility. (3) The newsroom metaphor makes complex multi-agent architecture "
            "accessible. The voice consistency section is the strongest differentiator."
        )

    def _mock_critic(self) -> str:
        return (
            "Weaknesses identified: (1) The opening paragraph uses passive voice — 'is undergoing' "
            "should be active. (2) No customer proof points or case studies cited. (3) The 2.3x claim "
            "needs a specific benchmark reference. (4) Missing CTA — what should the reader do next? "
            "(5) Paragraph 3 repeats information from the research summary."
        )

    def _mock_judge(self) -> str:
        score = round(random.uniform(7.0, 9.2), 1)
        return json.dumps({
            "score": score,
            "verdict": "pass" if score >= 7.5 else "revise",
            "strengths": ["Clear structure", "Good data integration", "Effective metaphors"],
            "weaknesses": ["Needs active voice", "Add social proof", "Strengthen CTA"],
            "revision_notes": "Convert passive constructions. Add 1-2 customer quotes. End with clear next step."
        })

    def _mock_json(self, prompt: str) -> str:
        return json.dumps({
            "result": "mock_json_response",
            "quality_score": 8.2,
            "dimensions": {f"dim_{i}": round(random.uniform(0.3, 0.95), 2) for i in range(14)},
        })


# Singleton
llm = LLMAdapter()
