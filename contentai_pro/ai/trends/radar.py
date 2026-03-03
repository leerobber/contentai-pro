"""Trend Radar — live trending topics from HN, Reddit, Dev.to."""
import time
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timezone

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class TrendItem:
    title: str
    url: str
    source: str
    score: int = 0
    category: str = "general"
    description: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TrendResult:
    trends: List[TrendItem]
    sources_queried: List[str]
    total_found: int
    cache_hit: bool = False
    latency_ms: float = 0.0


class TrendRadar:
    """Scans HackerNews, Reddit, and Dev.to for trending topics."""

    def __init__(self):
        self._cache: Optional[TrendResult] = None
        self._cache_time: float = 0
        self._cache_ttl: int = 1800  # 30 min
        if HAS_HTTPX:
            self._client = httpx.AsyncClient(timeout=10.0)
        else:
            self._client = None

    async def scan(self, niche: Optional[str] = None, limit: int = 20) -> TrendResult:
        t0 = time.perf_counter()

        # Check cache
        if self._cache and (time.time() - self._cache_time) < self._cache_ttl:
            result = TrendResult(
                trends=self._filter(self._cache.trends, niche)[:limit],
                sources_queried=self._cache.sources_queried,
                total_found=self._cache.total_found,
                cache_hit=True,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return result

        # Fetch from sources
        if not self._client:
            return self._mock_trends(niche, limit, t0)

        tasks = [self._fetch_hn(), self._fetch_reddit(), self._fetch_devto()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_trends = []
        sources = []
        for r in results:
            if isinstance(r, list):
                all_trends.extend(r)
            if isinstance(r, list) and r:
                sources.append(r[0].source)

        # Sort by score
        all_trends.sort(key=lambda t: t.score, reverse=True)

        self._cache = TrendResult(
            trends=all_trends,
            sources_queried=sources or ["hackernews", "reddit", "devto"],
            total_found=len(all_trends),
        )
        self._cache_time = time.time()

        filtered = self._filter(all_trends, niche)[:limit]
        return TrendResult(
            trends=filtered,
            sources_queried=sources or ["hackernews", "reddit", "devto"],
            total_found=len(all_trends),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def _fetch_hn(self) -> List[TrendItem]:
        try:
            resp = await self._client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
            ids = resp.json()[:15]
            items = []
            for sid in ids[:10]:
                item_resp = await self._client.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
                data = item_resp.json()
                if data and data.get("title"):
                    items.append(TrendItem(
                        title=data["title"],
                        url=data.get("url", f"https://news.ycombinator.com/item?id={sid}"),
                        source="hackernews",
                        score=data.get("score", 0),
                        category="tech",
                    ))
            return items
        except Exception:
            return []

    async def _fetch_reddit(self) -> List[TrendItem]:
        try:
            headers = {"User-Agent": "ContentAI-Pro/2.0"}
            resp = await self._client.get("https://www.reddit.com/r/technology/hot.json?limit=10", headers=headers)
            data = resp.json()
            items = []
            for post in data.get("data", {}).get("children", []):
                d = post.get("data", {})
                items.append(TrendItem(
                    title=d.get("title", ""),
                    url=f"https://reddit.com{d.get('permalink', '')}",
                    source="reddit",
                    score=d.get("score", 0),
                    category="technology",
                ))
            return items
        except Exception:
            return []

    async def _fetch_devto(self) -> List[TrendItem]:
        try:
            resp = await self._client.get("https://dev.to/api/articles?top=1&per_page=10")
            articles = resp.json()
            return [
                TrendItem(
                    title=a.get("title", ""),
                    url=a.get("url", ""),
                    source="devto",
                    score=a.get("positive_reactions_count", 0),
                    category=a.get("tag_list", ["dev"])[0] if a.get("tag_list") else "dev",
                    description=a.get("description", ""),
                )
                for a in articles
            ]
        except Exception:
            return []

    def _filter(self, trends: List[TrendItem], niche: Optional[str]) -> List[TrendItem]:
        if not niche:
            return trends
        keywords = [k.strip().lower() for k in niche.split(",")]
        def matches(t):
            text = f"{t.title} {t.category} {t.description}".lower()
            return any(k in text for k in keywords)
        filtered = [t for t in trends if matches(t)]
        return filtered if len(filtered) >= 3 else trends

    def _mock_trends(self, niche, limit, t0) -> TrendResult:
        mock = [
            TrendItem(title="Claude 4.5 Released — Multi-Agent Benchmarks Shattered", url="https://example.com/1", source="hackernews", score=892, category="ai"),
            TrendItem(title="Why RAG is Being Replaced by Long-Context Models", url="https://example.com/2", source="hackernews", score=654, category="ai"),
            TrendItem(title="The Economics of AI Content at Scale", url="https://example.com/3", source="reddit", score=1247, category="technology"),
            TrendItem(title="Building Production Multi-Agent Systems: Lessons from 10M API Calls", url="https://example.com/4", source="devto", score=423, category="agents"),
            TrendItem(title="Content DNA: How Voice Fingerprinting Changes Brand Consistency", url="https://example.com/5", source="devto", score=318, category="content"),
            TrendItem(title="Adversarial AI Testing in Production — A Practical Guide", url="https://example.com/6", source="hackernews", score=567, category="ai"),
            TrendItem(title="From Blog to 8 Platforms: Content Atomization Strategies", url="https://example.com/7", source="reddit", score=891, category="content"),
            TrendItem(title="FastAPI + Async Agents: The Production Architecture Stack", url="https://example.com/8", source="devto", score=276, category="python"),
        ]
        filtered = self._filter(mock, niche)[:limit]
        return TrendResult(
            trends=filtered,
            sources_queried=["hackernews", "reddit", "devto"],
            total_found=len(mock),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def close(self):
        if self._client:
            await self._client.aclose()


trend_radar = TrendRadar()
