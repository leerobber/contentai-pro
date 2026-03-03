"""Base agent — interface all specialist agents implement."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentResult:
    agent: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class BaseAgent(ABC):
    """Every pipeline agent implements this interface."""

    name: str = "base"
    system_prompt: str = ""

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Run agent on the given context. Returns AgentResult."""
        ...

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Override to customize prompt construction from context."""
        return str(context.get("input", ""))
