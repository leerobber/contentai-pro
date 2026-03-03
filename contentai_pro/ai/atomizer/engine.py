"""Content Atomizer — transforms one piece into platform-native variants.

FIX: Batch API call generates all platform variants in a single LLM request.
FIX: Platform-aware truncation (sentence-boundary, not mid-word).
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from contentai_pro.ai.llm_adapter import llm
from contentai_pro.core.config import settings

logger = logging.getLogger("contentai")


PLATFORM_SPECS = {
    "twitter": {
        "max_chars": 280,
        "format": "thread",
        "style": "Punchy, hook-driven. Use line breaks. Max 6 tweets in thread. Include 2-3 relevant hashtags.",
        "prompt_hint": "Convert into a Twitter/X thread. Each tweet ≤280 chars. Start with a killer hook.",
    },
    "linkedin": {
        "max_chars": 3000,
        "format": "post",
        "style": "Professional storytelling. Opening hook + personal insight + industry takeaway. Use line spacing.",
        "prompt_hint": "Rewrite as a LinkedIn post. Professional tone, personal angle, actionable insight.",
    },
    "instagram": {
        "max_chars": 2200,
        "format": "caption",
        "style": "Visual-first caption. Conversational, emoji-accented (not excessive). Strong CTA. 15-20 hashtags at end.",
        "prompt_hint": "Write an Instagram caption. Engaging, visual language. Include relevant hashtags.",
    },
    "email": {
        "max_chars": None,
        "format": "newsletter",
        "style": "Subject line + preview text + personal opening + key insights + CTA. Scannable with bold key points.",
        "prompt_hint": "Convert into a newsletter email. Include subject line, preview text, and clear CTA.",
    },
    "reddit": {
        "max_chars": 40000,
        "format": "post",
        "style": "Informative, community-focused. No self-promotion feel. TL;DR at top. Detailed body. Invite discussion.",
        "prompt_hint": "Rewrite as a Reddit post. Community tone, TL;DR first, invite discussion.",
    },
    "youtube": {
        "max_chars": None,
        "format": "script",
        "style": "Hook (0-15s) → Problem → Solution → Examples → CTA. Conversational. Include suggested B-roll notes.",
        "prompt_hint": "Convert into a YouTube video script with timestamps, B-roll suggestions, and CTA.",
    },
    "tiktok": {
        "max_chars": None,
        "format": "script",
        "style": "15-60 second script. Immediate hook. Fast pace. Pattern interrupts. Trending audio suggestion.",
        "prompt_hint": "Write a TikTok script (30-60s). Instant hook, fast pace, trend-aware.",
    },
    "podcast": {
        "max_chars": None,
        "format": "show_notes",
        "style": "Episode title + 3 key discussion points + quotes to reference + listener CTA + timestamps.",
        "prompt_hint": "Create podcast show notes with episode title, discussion points, and timestamps.",
    },
}


@dataclass
class AtomizedVariant:
    platform: str
    content: str
    format: str
    char_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomizerResult:
    source_topic: str
    variants: List[AtomizedVariant]
    platforms_generated: int
    total_latency_ms: float = 0.0
    errors: List[Dict[str, str]] = field(default_factory=list)


ATOMIZER_SYSTEM = (
    "You are a platform-native content adaptation specialist. Transform content for the "
    "target platform while preserving core message, key data points, and brand voice. "
    "Follow platform conventions precisely — length, format, tone, and engagement patterns."
)


def _smart_truncate(text: str, max_chars: int, platform: str) -> str:
    """Truncate at sentence or tweet boundaries instead of mid-word."""
    if len(text) <= max_chars:
        return text

    # Twitter threads: truncate at tweet boundary (double newline)
    if platform == "twitter":
        tweets = re.split(r'\n\n+', text)
        result = ""
        for tweet in tweets:
            candidate = (result + "\n\n" + tweet).strip() if result else tweet
            if len(candidate) <= max_chars:
                result = candidate
            else:
                break
        if result:
            return result
        # Single tweet larger than max_chars — fall through to word-boundary logic below

    # All other platforms (and twitter overflow): truncate at sentence boundary
    if max_chars <= 3:
        return text[:max_chars]

    truncated = text[:max_chars]
    last_period = max(truncated.rfind('. '), truncated.rfind('.\n'), truncated.rfind('!'), truncated.rfind('?'))
    if last_period > max_chars * 0.5:
        return truncated[:last_period + 1]

    # Fall back to word boundary, ensuring we never exceed max_chars
    word_truncated = truncated.rsplit(' ', 1)[0]
    if not word_truncated:
        # No spaces found; return a hard cut within limit
        return text[:max_chars]
    if len(word_truncated) + 3 <= max_chars:
        return word_truncated + "..."
    # word_truncated is already near the limit; trim further to fit the ellipsis
    safe_cutoff = max_chars - 3
    return word_truncated[:safe_cutoff] + "..."


class AtomizerEngine:
    """Transforms source content into platform-optimized variants."""

    async def _atomize_single(self, platform: str, source_content: str,
                               topic: str) -> Optional[AtomizedVariant]:
        """Generate a single platform variant. Returns None on failure."""
        spec = PLATFORM_SPECS.get(platform)
        if not spec:
            return None

        prompt = (
            f"**Source Content:**\n{source_content}\n\n"
            f"**Target Platform:** {platform.upper()}\n"
            f"**Format:** {spec['format']}\n"
            f"**Style Guide:** {spec['style']}\n\n"
            f"**Instructions:** {spec['prompt_hint']}\n\n"
            f"Generate the {platform} variant now."
        )

        output = await llm.generate(ATOMIZER_SYSTEM, prompt, temperature=0.6)

        # Platform-aware truncation
        if spec["max_chars"] and len(output) > spec["max_chars"]:
            output = _smart_truncate(output, spec["max_chars"], platform)

        return AtomizedVariant(
            platform=platform,
            content=output,
            format=spec["format"],
            char_count=len(output),
            metadata={"max_chars": spec["max_chars"]},
        )

    async def atomize(self, source_content: str, topic: str,
                      platforms: List[str] = None) -> AtomizerResult:
        t0 = time.perf_counter()
        platforms = platforms or settings.ATOMIZER_PLATFORMS
        valid_platforms = [p for p in platforms if p in PLATFORM_SPECS]

        # Build a single prompt requesting all platform variants at once
        platform_instructions = "\n".join(
            f"- {p.upper()}: {PLATFORM_SPECS[p]['prompt_hint']}"
            for p in valid_platforms
        )
        prompt = (
            f"Generate content variants for ALL of the following platforms in one response.\n\n"
            f"Platforms and instructions:\n{platform_instructions}\n\n"
            f"Return as JSON with platform names as lowercase keys:\n"
            f"{json.dumps({p: f'{p} content here...' for p in valid_platforms})}\n\n"
            f"Original content to adapt:\n{source_content}\n\n"
            f"Topic: {topic}"
        )

        raw = await llm.generate(ATOMIZER_SYSTEM, prompt, json_mode=True, agent_role="atomizer")

        errors: List[Dict[str, Any]] = []
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.error("Failed to parse batch atomizer JSON response", exc_info=True)
            for p in valid_platforms:
                errors.append({
                    "platform": p,
                    "error": "Batch response JSON parse failure",
                })
            return AtomizerResult(
                source_topic=topic,
                variants=[],
                platforms_generated=0,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                errors=errors,
            )

        variants = []
        for platform in valid_platforms:
            content = data.get(platform)
            if content is None:
                errors.append({"platform": platform, "error": "Missing from batch response"})
                continue

            spec = PLATFORM_SPECS[platform]
            if spec["max_chars"] and len(content) > spec["max_chars"]:
                content = _smart_truncate(content, spec["max_chars"], platform)

            variants.append(AtomizedVariant(
                platform=platform,
                content=content,
                format=spec["format"],
                char_count=len(content),
                metadata={"max_chars": spec["max_chars"]},
            ))

        return AtomizerResult(
            source_topic=topic,
            variants=variants,
            platforms_generated=len(variants),
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            errors=errors,
        )


atomizer_engine = AtomizerEngine()
