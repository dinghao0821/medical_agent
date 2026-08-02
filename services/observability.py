"""Observability (P4): Prometheus metrics, structured logging, LangSmith tracing.

All optional and degrade gracefully:
- Metrics require ``prometheus-client``; absent -> ``/metrics`` returns 501 and
  the middleware no-ops.
- JSON logging is opt-in; otherwise standard logging is used.
- LangSmith tracing is enabled by setting the standard LangChain env vars only
  when configured, so nothing changes by default.
"""

import os
import sys
import json
import logging
import time

logger = logging.getLogger(__name__)

# ---- Prometheus metrics (optional) ----
_PROM_AVAILABLE = False
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

    REQUEST_COUNT = Counter(
        "http_requests_total", "Total HTTP requests",
        ["method", "path", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds", "HTTP request latency (s)",
        ["method", "path"],
    )
    AGENT_ROUTED = Counter(
        "agent_routed_total", "Times a query was handled by an agent",
        ["agent"],
    )
    LLM_ERRORS = Counter(
        "llm_errors_total", "LLM/agent errors surfaced to the user",
    )
    _PROM_AVAILABLE = True
except Exception as e:  # pragma: no cover - optional dependency
    logger.info("prometheus-client not available (%s); metrics disabled.", e)
    _PROM_AVAILABLE = False


def metrics_available() -> bool:
    return _PROM_AVAILABLE


def render_metrics():
    """Return (payload_bytes, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST


def observe_request(method: str, path: str, status: int, duration: float):
    if not _PROM_AVAILABLE:
        return
    try:
        REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(duration)
    except Exception:
        pass


def record_agent(agent: str):
    if _PROM_AVAILABLE and agent:
        try:
            AGENT_ROUTED.labels(agent=str(agent)).inc()
        except Exception:
            pass


def record_llm_error():
    if _PROM_AVAILABLE:
        try:
            LLM_ERRORS.inc()
        except Exception:
            pass


# ---- Structured logging ----
class _JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(config):
    """Configure root logging level and (optionally) JSON formatting."""
    obs = getattr(config, "observability", None)
    level_name = getattr(obs, "log_level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    if bool(getattr(obs, "enable_json_logs", False)):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        root.handlers = [handler]


def setup_langsmith(config):
    """Enable LangSmith tracing by exporting standard LangChain env vars.

    Only acts when explicitly configured; otherwise leaves the environment
    untouched so tracing stays off by default.
    """
    obs = getattr(config, "observability", None)
    if not bool(getattr(obs, "langsmith_tracing", False)):
        return
    api_key = getattr(obs, "langsmith_api_key", "") or ""
    if not api_key:
        logger.warning("LANGSMITH_TRACING enabled but LANGSMITH_API_KEY is empty; skipping.")
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = getattr(obs, "langsmith_project", "medical-assistant")
    logger.info("LangSmith tracing enabled (project=%s).", os.environ["LANGCHAIN_PROJECT"])
