"""LLM cost & token-budget governance (agent enhancement).

Gives the system **cost visibility and control**:
  * meter approximate token usage per user/session,
  * enforce optional per-user daily token budgets (reject/deny over-limit),
  * expose accumulated usage for observability.

Token counting uses ``tiktoken`` when available (accurate) and falls back to a
cheap char/4 heuristic otherwise — so it works even without the dependency.
Accounting is stored in Redis (shared across workers, auto-expiring daily) with
an in-process fallback. Entirely opt-in (``config.cost.enabled``) and fail-open:
a metering error never blocks a request.
"""

import time
import logging

logger = logging.getLogger(__name__)

_PREFIX = "cost:"          # cost:{scope}:{id}:{yyyymmdd} -> accumulated tokens
_mem_usage = {}           # fallback: {key: (tokens, expire_epoch)}
_encoder = None
_encoder_tried = False


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _cfg(config):
    return getattr(config, "cost", None)


def _enabled(config) -> bool:
    return bool(getattr(_cfg(config), "enabled", False))


def _redis(config):
    try:
        from services.redis_client import get_redis
        url = getattr(_cfg(config), "redis_url", "") or getattr(getattr(config, "api", None), "redis_url", "")
        return get_redis(url or None)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #
def count_tokens(text: str) -> int:
    """Approximate token count. Uses tiktoken if present, else char/4 heuristic."""
    global _encoder, _encoder_tried
    if not text:
        return 0
    if not _encoder_tried:
        _encoder_tried = True
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = None
    if _encoder is not None:
        try:
            return len(_encoder.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Usage accounting (per day)
# --------------------------------------------------------------------------- #
def _day_key(scope: str, ident: str) -> str:
    day = time.strftime("%Y%m%d", time.gmtime())
    return f"{_PREFIX}{scope}:{ident}:{day}"


def _mem_get(key: str) -> int:
    now = time.time()
    v = _mem_usage.get(key)
    if not v:
        return 0
    tokens, exp = v
    if exp <= now:
        _mem_usage.pop(key, None)
        return 0
    return tokens


def get_usage(config, user_id: str = None, session_id: str = None) -> int:
    """Return today's accumulated token usage for a user (or session)."""
    if not _enabled(config):
        return 0
    scope, ident = ("user", user_id) if user_id else ("session", session_id or "anon")
    key = _day_key(scope, ident)
    client = _redis(config)
    if client is not None:
        try:
            v = client.get(key)
            return int(v) if v else 0
        except Exception:
            pass
    return _mem_get(key)


def add_usage(config, tokens: int, user_id: str = None, session_id: str = None) -> None:
    """Add ``tokens`` to today's usage for a user (or session). Best-effort."""
    if not _enabled(config) or tokens <= 0:
        return
    scope, ident = ("user", user_id) if user_id else ("session", session_id or "anon")
    key = _day_key(scope, ident)
    client = _redis(config)
    if client is not None:
        try:
            new_total = client.incrby(key, int(tokens))
            if new_total == tokens:  # first write today -> expire at ~26h
                client.expire(key, 60 * 60 * 26)
            return
        except Exception as e:
            logger.warning("cost_tracker redis add failed (%s); using memory.", e)
    # Memory fallback
    now = time.time()
    cur = _mem_get(key)
    _mem_usage[key] = (cur + int(tokens), now + 60 * 60 * 26)


def record_interaction(config, prompt_text: str, completion_text: str,
                       user_id: str = None, session_id: str = None) -> int:
    """Meter one LLM interaction (prompt + completion). Returns tokens counted."""
    if not _enabled(config):
        return 0
    try:
        total = count_tokens(prompt_text) + count_tokens(completion_text)
        add_usage(config, total, user_id=user_id, session_id=session_id)
        try:
            from services.agent_trace import add_event
            add_event("cost", tokens=total)
        except Exception:
            pass
        return total
    except Exception as e:
        logger.warning("cost_tracker record failed (%s).", e)
        return 0


# --------------------------------------------------------------------------- #
# Budget enforcement
# --------------------------------------------------------------------------- #
def check_budget(config, user_id: str = None, session_id: str = None):
    """Return (allowed, info). ``allowed=False`` when the daily budget is exceeded.

    Fail-open: when disabled, no budget set, or any error -> allowed=True.
    """
    if not _enabled(config):
        return True, {}
    try:
        budget = int(getattr(_cfg(config), "daily_token_budget", 0))
        if budget <= 0:
            return True, {}  # no budget configured -> unlimited (metering only)
        used = get_usage(config, user_id=user_id, session_id=session_id)
        allowed = used < budget
        return allowed, {"used": used, "budget": budget, "remaining": max(0, budget - used)}
    except Exception as e:
        logger.warning("cost_tracker budget check failed (%s); allowing.", e)
        return True, {}
