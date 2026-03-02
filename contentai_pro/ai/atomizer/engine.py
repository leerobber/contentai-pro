"""Content Atomizer — transforms one piece into platform-native variants."""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any
from contentai_pro.ai.llm_adapter import llm
from contentai_pro.core.config import settings


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


ATOMIZER_SYSTEM = (
    "You are a platform-native content adaptation specialist. Transform content for the "
    "target platform while preserving core message, key data points, and brand voice. "
    "Follow platform conventions precisely — length, format, tone, and engagement patterns."
)


class AtomizerEngine:
    """Transforms source content into platform-optimized variants."""

    async def atomize(self, source_content: str, topic: str,
                      platforms: List[str] = None) -> AtomizerResult:
        t0 = time.perf_counter()
        platforms = platforms or settings.ATOMIZER_PLATFORMS
        variants = []

        for platform in platforms:
            spec = PLATFORM_SPECS.get(platform)
            if not spec:
                continue

            prompt = (
                f"**Source Content:**\n{source_content}\n\n"
                f"**Target Platform:** {platform.upper()}\n"
                f"**Format:** {spec['format']}\n"
                f"**Style Guide:** {spec['style']}\n\n"
                f"**Instructions:** {spec['prompt_hint']}\n\n"
                f"Generate the {platform} variant now."
            )

            output = await llm.generate(ATOMIZER_SYSTEM, prompt, temperature=0.6)

            # Truncate if platform has char limit
            if spec["max_chars"] and len(output) > spec["max_chars"]:
                output = output[:spec["max_chars"] - 3] + "..."

            variants.append(AtomizedVariant(
                platform=platform,
                content=output,
                format=spec["format"],
                char_count=len(output),
                metadata={"max_chars": spec["max_chars"]},
            ))

        return AtomizerResult(
            source_topic=topic,
            variants=variants,
            platforms_generated=len(variants),
            total_latency_ms=(time.perf_counter() - t0) * 1000,
        )


atomizer_engine = AtomizerEngine()
