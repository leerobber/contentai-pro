"""Event bus — powers SSE streaming for pipeline progress."""
import asyncio
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, AsyncGenerator, Optional
from collections import defaultdict


@dataclass
class PipelineEvent:
    stage: str
    status: str  # "started" | "completed" | "failed" | "progress"
    data: dict = field(default_factory=dict)
    pipeline_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_sse(self) -> str:
        return f"data: {json.dumps(asdict(self))}\n\n"


class EventBus:
    """In-memory pub/sub for pipeline events."""

    def __init__(self):
        self._subscribers: Dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False
        # Drain all queues
        for queues in self._subscribers.values():
            for q in queues:
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass

    def new_pipeline_id(self) -> str:
        return str(uuid.uuid4())

    async def publish(self, pipeline_id: str, event: PipelineEvent):
        event.pipeline_id = pipeline_id
        for q in self._subscribers.get(pipeline_id, []):
            await q.put(event)

    async def subscribe(self, pipeline_id: str) -> AsyncGenerator[PipelineEvent, None]:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[pipeline_id].append(q)
        try:
            while self._running:
                event = await q.get()
                if event is None:
                    break
                yield event
                if event.status == "completed" and event.stage == "pipeline":
                    break
        finally:
            self._subscribers[pipeline_id].remove(q)
            if not self._subscribers[pipeline_id]:
                del self._subscribers[pipeline_id]

    async def emit_stage(self, pipeline_id: str, stage: str, status: str, data: Optional[dict] = None):
        await self.publish(pipeline_id, PipelineEvent(
            stage=stage, status=status, data=data or {}
        ))


# Singleton
event_bus = EventBus()
