"""Specialist agents — Research, Write, Edit, SEO."""
import time
from typing import Dict, Any
from contentai_pro.ai.agents.base import BaseAgent, AgentResult
from contentai_pro.ai.llm_adapter import llm


class ResearchAgent(BaseAgent):
    name = "researcher"
    system_prompt = (
        "You are an expert research analyst. Given a topic, produce a structured research brief including: "
        "key facts, statistics, market data, competitive landscape, and source references. "
        "Format as markdown with clear sections. Be specific with numbers and citations."
    )

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        t0 = time.perf_counter()
        topic = context.get("topic", "")
        content_type = context.get("content_type", "blog_post")
        audience = context.get("audience", "tech professionals")

        prompt = (
            f"Research the following topic for a {content_type} targeting {audience}:\n\n"
            f"**Topic:** {topic}\n\n"
            f"Produce a comprehensive research brief with data points, trends, competitive analysis, "
            f"and recommended angles. Include at least 5 concrete statistics or data points."
        )
        output = await llm.generate(self.system_prompt, prompt)
        return AgentResult(
            agent=self.name, output=output,
            latency_ms=(time.perf_counter() - t0) * 1000,
            metadata={"topic": topic, "content_type": content_type}
        )


class WriterAgent(BaseAgent):
    name = "writer"
    system_prompt = (
        "You are a world-class content writer. Using the research provided, write compelling, "
        "well-structured content. Match the specified voice DNA profile if provided. "
        "Use clear headings, engaging hooks, concrete examples, and strong transitions."
    )

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        t0 = time.perf_counter()
        topic = context.get("topic", "")
        research = context.get("research", "")
        content_type = context.get("content_type", "blog_post")
        dna_profile = context.get("dna_profile", None)
        tone = context.get("tone", "professional yet approachable")
        word_count = context.get("word_count", 1200)

        dna_section = ""
        if dna_profile:
            dna_section = f"\n\n**Voice DNA Profile (match this style):**\n{dna_profile}\n"

        prompt = (
            f"Write a {content_type} (~{word_count} words) on the following topic.\n\n"
            f"**Topic:** {topic}\n"
            f"**Tone:** {tone}\n"
            f"{dna_section}\n"
            f"**Research Brief:**\n{research}\n\n"
            f"Produce publication-ready content with a compelling headline, strong opening hook, "
            f"clear structure, and actionable takeaways."
        )
        output = await llm.generate(self.system_prompt, prompt)
        return AgentResult(
            agent=self.name, output=output,
            latency_ms=(time.perf_counter() - t0) * 1000,
            metadata={"topic": topic, "word_count": word_count}
        )


class EditorAgent(BaseAgent):
    name = "editor"
    system_prompt = (
        "You are a senior editor at a top publication. Review and improve the given content. "
        "Fix: unclear sentences, passive voice, weak transitions, redundancy, factual inconsistencies. "
        "Preserve the author's voice while elevating clarity and impact. Return the full edited content."
    )

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        t0 = time.perf_counter()
        draft = context.get("draft", "")
        topic = context.get("topic", "")

        prompt = (
            f"Edit the following {context.get('content_type', 'article')} for publication quality.\n\n"
            f"**Topic:** {topic}\n\n"
            f"**Draft:**\n{draft}\n\n"
            f"Return the complete edited version. Maintain the original structure but improve "
            f"clarity, flow, and impact. Fix any factual or logical issues."
        )
        output = await llm.generate(self.system_prompt, prompt, temperature=0.3)
        return AgentResult(
            agent=self.name, output=output,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class SEOAgent(BaseAgent):
    name = "seo"
    system_prompt = (
        "You are an SEO specialist. Optimize the given content for search engines while "
        "maintaining readability and engagement. Apply keyword integration, meta tags, "
        "heading optimization, internal link suggestions, and readability scoring."
    )

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        t0 = time.perf_counter()
        content = context.get("content", "")
        topic = context.get("topic", "")
        keywords = context.get("keywords", [])

        kw_str = ", ".join(keywords) if keywords else "auto-detect from topic"

        prompt = (
            f"Optimize the following content for SEO.\n\n"
            f"**Topic:** {topic}\n"
            f"**Target Keywords:** {kw_str}\n\n"
            f"**Content:**\n{content}\n\n"
            f"Return the SEO-optimized version with:\n"
            f"1. Optimized title tag and meta description\n"
            f"2. Keyword-integrated headings\n"
            f"3. Natural keyword placement in body\n"
            f"4. Internal link suggestions\n"
            f"5. Readability score assessment"
        )
        output = await llm.generate(self.system_prompt, prompt, temperature=0.2)
        return AgentResult(
            agent=self.name, output=output,
            latency_ms=(time.perf_counter() - t0) * 1000,
            metadata={"keywords": keywords}
        )
