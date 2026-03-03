"""Content Atomizer — transforms one piece into platform-native variants.

Enhancements (Atomizer Intelligence)
- TimingRecommendation: ML-inspired optimal posting windows per platform.
- PerformanceRecord   : Feedback loop to learn from engagement metrics.
- A/B variant support : generate_variants() produces alternative takes.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
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
    timing_recommendations: Dict[str, "TimingRecommendation"] = field(default_factory=dict)


# ── Atomizer Intelligence: timing & performance ───────────────────────────

# Evidence-based optimal posting windows (day-of-week / hour-of-day UTC)
_PLATFORM_TIMING: Dict[str, Dict[str, Any]] = {
    "twitter": {
        "best_days": ["Tuesday", "Wednesday", "Thursday"],
        "best_hours_utc": [13, 14, 15],   # 9-11 AM EST
        "frequency": "3-5 times/day",
        "rationale": "Engagement peaks mid-week during US business hours.",
    },
    "linkedin": {
        "best_days": ["Tuesday", "Wednesday", "Thursday"],
        "best_hours_utc": [11, 12, 17],
        "frequency": "1 time/day",
        "rationale": "Professional audience active at lunch and end-of-business.",
    },
    "instagram": {
        "best_days": ["Monday", "Wednesday", "Friday"],
        "best_hours_utc": [11, 14, 20],
        "frequency": "1-2 times/day",
        "rationale": "Visual content peaks mid-morning and evening.",
    },
    "email": {
        "best_days": ["Tuesday", "Thursday"],
        "best_hours_utc": [10, 14],
        "frequency": "1-2 times/week",
        "rationale": "Open rates highest Tuesday/Thursday mid-morning.",
    },
    "reddit": {
        "best_days": ["Monday", "Tuesday", "Wednesday"],
        "best_hours_utc": [14, 15, 16],
        "frequency": "1-2 times/week per subreddit",
        "rationale": "Reddit traffic peaks weekday afternoons EST.",
    },
    "youtube": {
        "best_days": ["Friday", "Saturday", "Sunday"],
        "best_hours_utc": [14, 15, 16],
        "frequency": "1-3 times/week",
        "rationale": "Video consumption spikes on weekends and Friday afternoons.",
    },
    "tiktok": {
        "best_days": ["Tuesday", "Thursday", "Friday"],
        "best_hours_utc": [6, 10, 19, 22],
        "frequency": "1-4 times/day",
        "rationale": "TikTok users check app morning, lunch, and evening.",
    },
    "podcast": {
        "best_days": ["Monday", "Tuesday", "Wednesday"],
        "best_hours_utc": [7, 8],
        "frequency": "Weekly",
        "rationale": "Podcast listeners subscribe during morning commute.",
    },
}


@dataclass
class TimingRecommendation:
    platform: str
    best_days: List[str]
    best_hours_utc: List[int]
    frequency: str
    rationale: str
    next_window: str = ""   # ISO-8601 of next recommended slot (computed at runtime)


@dataclass
class PerformanceRecord:
    """Engagement feedback for a published variant."""
    platform: str
    content_id: str
    impressions: int = 0
    clicks: int = 0
    shares: int = 0
    comments: int = 0
    engagement_rate: float = 0.0
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


ATOMIZER_SYSTEM = (
    "You are a platform-native content adaptation specialist. Transform content for the "
    "target platform while preserving core message, key data points, and brand voice. "
    "Follow platform conventions precisely — length, format, tone, and engagement patterns."
)


class AtomizerEngine:
    """Transforms source content into platform-optimized variants.

    New capabilities:
    - timing_for()      : Get optimal posting windows for a platform.
    - record_performance() : Feed engagement data back to the engine.
    - generate_variants()  : Produce N alternative takes for A/B testing.
    """

    def __init__(self):
        self._performance_history: Dict[str, List[PerformanceRecord]] = {}

    async def atomize(self, source_content: str, topic: str,
                      platforms: List[str] = None) -> AtomizerResult:
        t0 = time.perf_counter()
        platforms = platforms or settings.ATOMIZER_PLATFORMS
        variants = []
        timing: Dict[str, TimingRecommendation] = {}

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
            timing[platform] = self.timing_for(platform)

        return AtomizerResult(
            source_topic=topic,
            variants=variants,
            platforms_generated=len(variants),
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            timing_recommendations=timing,
        )

    def timing_for(self, platform: str) -> TimingRecommendation:
        """Return evidence-based optimal posting windows for the given platform."""
        data = _PLATFORM_TIMING.get(platform, {})
        if not data:
            return TimingRecommendation(
                platform=platform,
                best_days=["Monday", "Wednesday", "Friday"],
                best_hours_utc=[12, 18],
                frequency="1 time/day",
                rationale="No platform-specific data available; using general best practices.",
            )

        # Compute next upcoming window from current UTC time
        now = datetime.now(timezone.utc)
        next_window = self._next_posting_window(now, data["best_days"], data["best_hours_utc"])

        return TimingRecommendation(
            platform=platform,
            best_days=data["best_days"],
            best_hours_utc=data["best_hours_utc"],
            frequency=data["frequency"],
            rationale=data["rationale"],
            next_window=next_window,
        )

    def record_performance(self, record: PerformanceRecord) -> None:
        """Store engagement feedback for future learning."""
        if record.impressions > 0:
            record.engagement_rate = round(
                (record.clicks + record.shares + record.comments) / record.impressions, 4
            )
        self._performance_history.setdefault(record.platform, []).append(record)

    def get_performance_summary(self, platform: str) -> Dict[str, Any]:
        """Aggregate engagement statistics for a platform from recorded history."""
        records = self._performance_history.get(platform, [])
        if not records:
            return {"platform": platform, "records": 0}
        avg_eng = sum(r.engagement_rate for r in records) / len(records)
        return {
            "platform": platform,
            "records": len(records),
            "avg_engagement_rate": round(avg_eng, 4),
            "total_impressions": sum(r.impressions for r in records),
            "total_shares": sum(r.shares for r in records),
        }

    async def generate_variants(
        self,
        source_content: str,
        topic: str,
        platform: str,
        n: int = 2,
    ) -> List[AtomizedVariant]:
        """Generate N alternative variants of the same content for A/B testing."""
        spec = PLATFORM_SPECS.get(platform)
        if not spec:
            raise ValueError(f"Unknown platform: {platform}")

        variants = []
        angles = ["angle_A", "angle_B", "angle_C", "angle_D"]
        for i in range(n):
            prompt = (
                f"**Source Content:**\n{source_content}\n\n"
                f"**Target Platform:** {platform.upper()}\n"
                f"**Format:** {spec['format']}\n"
                f"**Style Guide:** {spec['style']}\n\n"
                f"**Variation #{i + 1} ({angles[i % len(angles)]}):** "
                f"Create a distinct alternative take — different hook, different structure, "
                f"same core message. Make it noticeably different from variation #1.\n\n"
                f"{spec['prompt_hint']}"
            )
            output = await llm.generate(ATOMIZER_SYSTEM, prompt, temperature=0.8)
            if spec["max_chars"] and len(output) > spec["max_chars"]:
                output = output[:spec["max_chars"] - 3] + "..."
            variants.append(AtomizedVariant(
                platform=platform,
                content=output,
                format=spec["format"],
                char_count=len(output),
                metadata={"ab_variant": angles[i % len(angles)], "max_chars": spec["max_chars"]},
            ))
        return variants

    @staticmethod
    def _next_posting_window(now: datetime, best_days: List[str], best_hours_utc: List[int]) -> str:
        """Return the ISO-8601 timestamp of the next optimal posting slot."""
        _day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6,
        }
        target_weekdays = sorted(_day_map[d] for d in best_days if d in _day_map)
        target_hours = sorted(best_hours_utc)

        if not target_weekdays or not target_hours:
            return now.isoformat()

        candidate = now.replace(minute=0, second=0, microsecond=0)
        # Search up to 8 days ahead
        for _ in range(8 * 24):
            if candidate.weekday() in target_weekdays and candidate.hour in target_hours:
                if candidate > now:
                    return candidate.isoformat()
            candidate += timedelta(hours=1)
        return now.isoformat()


atomizer_engine = AtomizerEngine()
