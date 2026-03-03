"""Webhook support — async delivery with retry for pipeline completion events."""
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("contentai")


class WebhookDeliveryError(Exception):
    """Raised when a webhook endpoint returns a retryable error (429 or 5xx)."""


@dataclass
class WebhookRegistration:
    id: str
    url: str
    events: List[str] = field(default_factory=lambda: ["pipeline.completed"])
    secret: Optional[str] = None
    active: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class WebhookPayload:
    event: str
    pipeline_id: str
    content_id: str
    topic: str
    stages_completed: List[str]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class WebhookManager:
    """Manages webhook registrations and async delivery with retry."""

    def __init__(self):
        self._registrations: Dict[str, WebhookRegistration] = {}

    # --- Registration ---

    def register(self, url: str, events: Optional[List[str]] = None,
                 secret: Optional[str] = None) -> WebhookRegistration:
        reg = WebhookRegistration(
            id=str(uuid.uuid4()),
            url=url,
            events=events or ["pipeline.completed"],
            secret=secret,
        )
        self._registrations[reg.id] = reg
        logger.info("Webhook registered: %s → %s", reg.id, url)
        return reg

    def unregister(self, webhook_id: str) -> bool:
        if webhook_id in self._registrations:
            del self._registrations[webhook_id]
            logger.info("Webhook unregistered: %s", webhook_id)
            return True
        return False

    def list_registrations(self) -> List[dict]:
        return [
            {
                "id": r.id,
                "url": r.url,
                "events": r.events,
                "active": r.active,
                "created_at": r.created_at,
            }
            for r in self._registrations.values()
        ]

    # --- Delivery ---

    async def deliver(self, payload: WebhookPayload) -> None:
        """Deliver a webhook payload to all active registrations matching the event."""
        matching = [
            r for r in self._registrations.values()
            if r.active and payload.event in r.events
        ]
        for reg in matching:
            await self._send_with_retry(reg, payload)

    async def _send_with_retry(self, reg: WebhookRegistration, payload: WebhookPayload) -> None:
        data = {
            "event": payload.event,
            "pipeline_id": payload.pipeline_id,
            "content_id": payload.content_id,
            "topic": payload.topic,
            "stages_completed": payload.stages_completed,
            "latency_ms": payload.latency_ms,
            "metadata": payload.metadata,
            "timestamp": payload.timestamp,
        }
        await self._post(reg.url, data)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, WebhookDeliveryError)),
        before=before_log(logger, logging.WARNING),
        reraise=False,
    )
    async def _post(self, url: str, data: dict) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                json=data,
                headers={"Content-Type": "application/json", "X-Source": "ContentAI-Pro"},
            )
            if response.status_code == 429 or response.status_code >= 500:
                # Retryable errors: rate-limited or server-side failure
                raise WebhookDeliveryError(
                    f"Webhook delivery failed with retryable status {response.status_code}: {url}"
                )
            if response.status_code >= 400:
                # Non-retryable client error — log and skip
                logger.warning("Webhook delivery failed (non-retryable %s): %s",
                               response.status_code, url)
            else:
                logger.debug("Webhook delivered to %s: %s", url, response.status_code)


# Global singleton
webhook_manager = WebhookManager()
