"""Application metrics — counters, histograms, and gauges."""
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

logger = logging.getLogger("contentai")

# Optional Prometheus support
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not installed; using built-in metrics only.")

# Maximum number of raw samples retained per histogram bucket to avoid unbounded memory growth
_MAX_RESERVOIR_SIZE = 1000


@dataclass
class _HistogramBucket:
    """Accumulates values for computing mean and percentiles using a bounded reservoir."""
    total: float = 0.0
    count: int = 0
    _reservoir: Deque = field(default_factory=lambda: deque(maxlen=_MAX_RESERVOIR_SIZE))

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        self._reservoir.append(value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def percentile(self, p: float) -> float:
        if not self._reservoir:
            return 0.0
        sorted_vals = sorted(self._reservoir)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


class Metrics:
    """In-process metrics store with optional Prometheus export."""

    def __init__(self):
        # Request counters: {endpoint: {status_code: count}}
        self._request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Stage latencies: {stage_name: HistogramBucket}
        self._stage_latencies: Dict[str, _HistogramBucket] = defaultdict(_HistogramBucket)
        # LLM call metrics
        self._llm_calls: int = 0
        self._llm_latency = _HistogramBucket()
        # Error counts: {error_type: count}
        self._error_counts: Dict[str, int] = defaultdict(int)
        # Active pipelines gauge
        self._active_pipelines: int = 0

        # Prometheus metrics (if available)
        if _PROMETHEUS_AVAILABLE:
            self._prom_requests = Counter(
                "contentai_requests_total",
                "Total HTTP requests",
                ["endpoint", "status_code"],
            )
            self._prom_stage_latency = Histogram(
                "contentai_stage_latency_seconds",
                "Pipeline stage latency",
                ["stage"],
            )
            self._prom_llm_calls = Counter(
                "contentai_llm_calls_total",
                "Total LLM API calls",
            )
            self._prom_llm_latency = Histogram(
                "contentai_llm_latency_seconds",
                "LLM call latency",
            )
            self._prom_active_pipelines = Gauge(
                "contentai_active_pipelines",
                "Currently active pipelines",
            )
            self._prom_errors = Counter(
                "contentai_errors_total",
                "Total errors by type",
                ["error_type"],
            )

    # --- Request tracking ---

    def record_request(self, endpoint: str, status_code: int) -> None:
        self._request_counts[endpoint][str(status_code)] += 1
        if _PROMETHEUS_AVAILABLE:
            self._prom_requests.labels(endpoint=endpoint, status_code=str(status_code)).inc()

    # --- Stage latency ---

    def record_stage_latency(self, stage: str, latency_ms: float) -> None:
        self._stage_latencies[stage].observe(latency_ms)
        if _PROMETHEUS_AVAILABLE:
            self._prom_stage_latency.labels(stage=stage).observe(latency_ms / 1000)

    # --- LLM calls ---

    def record_llm_call(self, latency_ms: float) -> None:
        self._llm_calls += 1
        self._llm_latency.observe(latency_ms)
        if _PROMETHEUS_AVAILABLE:
            self._prom_llm_calls.inc()
            self._prom_llm_latency.observe(latency_ms / 1000)

    # --- Active pipelines ---

    def pipeline_started(self) -> None:
        self._active_pipelines += 1
        if _PROMETHEUS_AVAILABLE:
            self._prom_active_pipelines.inc()

    def pipeline_finished(self) -> None:
        if self._active_pipelines > 0:
            self._active_pipelines -= 1
        if _PROMETHEUS_AVAILABLE:
            self._prom_active_pipelines.dec()

    # --- Errors ---

    def record_error(self, error_type: str) -> None:
        self._error_counts[error_type] += 1
        if _PROMETHEUS_AVAILABLE:
            self._prom_errors.labels(error_type=error_type).inc()

    # --- Summary ---

    def summary(self) -> dict:
        return {
            "requests": {ep: dict(codes) for ep, codes in self._request_counts.items()},
            "stage_latencies": {
                stage: {
                    "mean_ms": round(bucket.mean, 2),
                    "p95_ms": round(bucket.percentile(95), 2),
                    "count": bucket.count,
                }
                for stage, bucket in self._stage_latencies.items()
            },
            "llm": {
                "total_calls": self._llm_calls,
                "mean_latency_ms": round(self._llm_latency.mean, 2),
                "p95_latency_ms": round(self._llm_latency.percentile(95), 2),
            },
            "active_pipelines": self._active_pipelines,
            "errors": dict(self._error_counts),
        }

    def prometheus_export(self) -> Optional[bytes]:
        """Return Prometheus text format metrics, or None if not available."""
        if _PROMETHEUS_AVAILABLE:
            return generate_latest()
        return None

    def prometheus_content_type(self) -> str:
        if _PROMETHEUS_AVAILABLE:
            return CONTENT_TYPE_LATEST
        return "text/plain"


# Global singleton
metrics = Metrics()
