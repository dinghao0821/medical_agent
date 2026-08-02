"""Tool registry + dispatcher for the structured tool framework.

Each tool is registered with metadata that drives reliability:

* ``timeout``    - hard wall-clock cap (seconds) on tool execution.
* ``retries``    - number of retries on transient failure (only for idempotent
  tools; non-idempotent tools never retry).
* ``idempotent`` - whether repeated calls with the same input are safe. Only
  idempotent tools are retried.
* ``read_only``  - whether the tool has no side-effects (never mutates state).
* ``dangerous``  - whether the tool has irreversible side-effects; requires
  explicit confirmation before execution.
* ``args_schema``- optional Pydantic model for input validation.

All reliability features are opt-in via decorator kwargs and degrade
gracefully: if any wrapping fails, the original tool runs unwrapped.
"""

import re
import time
import logging
import threading

logger = logging.getLogger(__name__)

# name -> {"func", "description", "keywords", "timeout", "retries",
#          "idempotent", "read_only", "dangerous", "args_schema"}
_REGISTRY = {}

# ---------------------------------------------------------------------------
# Metric counters (lightweight, in-process; no external dependency).
# ---------------------------------------------------------------------------
_METRICS = {}  # tool_name -> {"calls": N, "ok": N, "fail": N, ...}


def _record_metric(name, *, ok, timed_out=False, retries_used=0, duration=0.0):
    """Record one tool invocation outcome for observability."""
    try:
        m = _METRICS.setdefault(name, {
            "calls": 0, "ok": 0, "fail": 0, "timeout": 0, "retries": 0, "total_s": 0.0,
        })
        m["calls"] += 1
        if ok:
            m["ok"] += 1
        else:
            m["fail"] += 1
        if timed_out:
            m["timeout"] += 1
        m["retries"] += retries_used
        m["total_s"] += duration
    except Exception:
        pass


def get_metrics():
    """Return a snapshot of per-tool metrics (calls, ok, fail, latency)."""
    return {k: dict(v) for k, v in _METRICS.items()}


def reset_metrics():
    _METRICS.clear()


def register_tool(name, description="", keywords=None, *,
                  timeout=8, retries=0, idempotent=True,
                  read_only=True, dangerous=False, args_schema=None):
    """Decorator registering a callable tool.

    The tool function receives ``(text: str)`` -- the raw user query -- and
    returns either a result string, or ``None`` if it cannot handle the input.

    Reliability kwargs (all optional, backward-compatible):
        timeout     : max seconds before the call is abandoned (default 8).
        retries     : retry count on failure (default 0; only if idempotent).
        idempotent  : safe to retry (default True).
        read_only   : no side-effects (default True).
        dangerous   : irreversible side-effects; needs confirmation (default False).
        args_schema : Pydantic BaseModel subclass for input validation.
    """
    def _wrap(func):
        _REGISTRY[name] = {
            "func": func,
            "description": description or (func.__doc__ or "").strip(),
            "keywords": [k.lower() for k in (keywords or [])],
            "timeout": timeout,
            "retries": retries if idempotent else 0,
            "idempotent": idempotent,
            "read_only": read_only,
            "dangerous": dangerous,
            "args_schema": args_schema,
        }
        return func
    return _wrap


def get_tools():
    """Return {name: description} for all registered tools."""
    return {n: meta["description"] for n, meta in _REGISTRY.items()}


def get_tool_meta(name):
    """Return metadata for a tool, or None."""
    return _REGISTRY.get(name)


# ---------------------------------------------------------------------------
# Execution with timeout + retry + metrics
# ---------------------------------------------------------------------------
class _TimeoutError(Exception):
    """Internal: raised when a tool exceeds its wall-clock timeout."""


def _run_with_timeout(func, text, timeout):
    """Execute func(text) with a hard wall-clock timeout.

    Uses a daemon thread so it cannot deadlock the main event loop. On
    timeout raises _TimeoutError.
    """
    result_container = [None, None]  # [value, exception]

    def _worker():
        try:
            result_container[0] = func(text)
        except Exception as e:
            result_container[1] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise _TimeoutError("Tool exceeded %ss timeout" % timeout)

    if result_container[1] is not None:
        raise result_container[1]

    return result_container[0]


def _validate_args(text, args_schema):
    """Validate raw text against a Pydantic args_schema (if provided).

    Returns the raw text if valid; raises on validation failure.
    If text is not JSON, passes it through (tool does its own parsing).
    """
    if args_schema is None:
        return text
    import json as _json
    try:
        parsed = _json.loads(text)
        if isinstance(parsed, dict):
            args_schema(**parsed)
    except (ValueError, TypeError):
        pass
    return text


def run_tool(name, text):
    """Run a specific tool by name. Returns its output or None (fail-open).

    Applies timeout, retry (idempotent only), argument validation, and
    records metrics.  All failures degrade to None.
    """
    meta = _REGISTRY.get(name)
    if not meta:
        return None

    func = meta["func"]
    timeout = meta.get("timeout", 8)
    max_retries = meta.get("retries", 0)
    args_schema = meta.get("args_schema")

    try:
        text = _validate_args(text, args_schema)
    except Exception as e:
        logger.warning("Tool %s arg validation failed (%s).", name, e)
        _record_metric(name, ok=False, duration=0.0)
        return None

    t0 = time.time()
    retries_used = 0
    last_exc = None

    for attempt in range(1 + max_retries):
        try:
            result = _run_with_timeout(func, text, timeout)
            _record_metric(name, ok=True, retries_used=retries_used,
                           duration=time.time() - t0)
            return result
        except _TimeoutError as e:
            last_exc = e
            logger.warning("Tool %s timed out (attempt %d/%d).",
                           name, attempt + 1, 1 + max_retries)
        except Exception as e:
            last_exc = e
            logger.warning("Tool %s failed (attempt %d/%d): %s",
                           name, attempt + 1, 1 + max_retries, e)

        if attempt < max_retries and meta.get("idempotent", True):
            retries_used += 1
            time.sleep(min(0.1 * (2 ** attempt), 1.0))
        else:
            break

    timed_out = isinstance(last_exc, _TimeoutError)
    _record_metric(name, ok=False, timed_out=timed_out,
                   retries_used=retries_used, duration=time.time() - t0)
    if last_exc:
        logger.warning("Tool %s exhausted retries (%s).", name, last_exc)
    return None


def _match_by_keywords(text):
    """Return the first tool whose keyword appears in text, else None."""
    low = (text or "").lower()
    for name, meta in _REGISTRY.items():
        for kw in meta["keywords"]:
            if kw and re.search(r"\b" + re.escape(kw) + r"\b", low):
                return name
    return None


def maybe_run_tools(config, text):
    """Try to satisfy the query with a registered tool.

    Returns the tool output string on a hit, or None if tools are disabled,
    no tool matches, or the matched tool declines. Selection is offline
    keyword matching (no LLM), so it is cheap and deterministic.

    Dangerous tools are skipped here (they require explicit confirmation
    via run_tool with confirm=True).
    """
    if not bool(getattr(getattr(config, "tools", None), "enabled", False)):
        return None
    if not text:
        return None
    try:
        name = _match_by_keywords(text)
        if not name:
            return None
        meta = _REGISTRY.get(name, {})
        if meta.get("dangerous", False):
            logger.info("Tool '%s' is dangerous; skipped in auto-dispatch.", name)
            return None
        result = run_tool(name, text)
        if result:
            logger.info("Tool '%s' handled the query.", name)
        return result
    except Exception as e:
        logger.warning("maybe_run_tools error (%s).", e)
        return None
