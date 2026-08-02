"""Structured per-request agent tracing (agent-harness observability).

Gives every request a lightweight, dependency-free trace so any answer can be
reconstructed: which agent was chosen, why (confidence), whether guardrails or
the injection filter fired, retrieval confidence, and per-step latency.

Usage
-----
    from services.agent_trace import start_trace, add_event, end_trace

    start_trace(session_id="abc")
    add_event("route", agent="RAG_AGENT", confidence=0.91)
    add_event("retrieval", retrieval_confidence=0.78, docs=5)
    end_trace(agent="RAG_AGENT", status="ok")

Design:
  * **opt-in**: no-op unless ``config.trace.enabled`` (ENABLE_AGENT_TRACE=true).
  * **degrade-safe**: uses contextvars so it's correct under async + threads,
    and every function swallows its own errors — tracing must never break a
    request.
  * **no new deps**: emits one JSON line per request via the standard logger;
    integrates with the JSON logging from services.observability.
"""

import time
import uuid
import logging
import contextvars

logger = logging.getLogger("agent.trace")

# Per-request trace context (safe across async tasks and threads).
_current_trace = contextvars.ContextVar("agent_trace", default=None)

_enabled_flag = False  # set once via configure()


def configure(config):
    """Enable/disable tracing from config (called once at startup)."""
    global _enabled_flag
    try:
        _enabled_flag = bool(getattr(getattr(config, "trace", None), "enabled", False))
    except Exception:
        _enabled_flag = False


def _active() -> bool:
    return _enabled_flag


def start_trace(session_id: str = None, query: str = None):
    """Begin a new trace for the current request/context."""
    if not _active():
        return None
    try:
        trace = {
            "trace_id": uuid.uuid4().hex[:12],
            "session_id": session_id,
            "t_start": time.time(),
            "events": [],
        }
        if query is not None:
            # Store only a short, non-sensitive preview.
            trace["query_preview"] = (query[:80] + "…") if len(query) > 80 else query
        _current_trace.set(trace)
        return trace
    except Exception:
        return None


def add_event(kind: str, **fields):
    """Record a step in the current trace (e.g. route / guardrail / retrieval)."""
    if not _active():
        return
    try:
        trace = _current_trace.get()
        if not trace:
            return
        event = {"kind": kind, "t": round(time.time() - trace["t_start"], 4)}
        event.update({k: v for k, v in fields.items() if v is not None})
        trace["events"].append(event)
    except Exception:
        pass


def end_trace(agent: str = None, status: str = "ok", **fields):
    """Finish the current trace and emit one structured JSON log line."""
    if not _active():
        return
    try:
        trace = _current_trace.get()
        if not trace:
            return
        trace["agent"] = agent
        trace["status"] = status
        trace["duration_s"] = round(time.time() - trace["t_start"], 4)
        trace.update(fields)
        trace.pop("t_start", None)
        logger.info("agent_trace %s", _safe_json(trace))
    except Exception:
        pass
    finally:
        try:
            _current_trace.set(None)
        except Exception:
            pass


def _safe_json(obj) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)
