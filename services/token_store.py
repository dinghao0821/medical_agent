"""Token revocation store (P3+): JWT ``jti`` denylist with graceful degradation.

Used to make logout and refresh-token rotation actually invalidate tokens
*before* their natural expiry — the piece plain stateless JWT can't do alone.

Backend selection (fully degradable, no hard dependency):
  * If Redis is reachable (``REDIS_URL`` set + ``redis`` installed), revoked
    ``jti`` values are stored as keys with a TTL equal to the token's remaining
    lifetime, so they auto-expire and the denylist is shared across workers /
    replicas.
  * Otherwise falls back to a process-local in-memory dict (single-process
    only). A one-time warning is logged; multi-worker deployments should run
    Redis for correct cross-worker revocation.

All functions are best-effort and never raise: a store failure must not take
the auth path down.
"""

import time
import logging

logger = logging.getLogger(__name__)

_PREFIX = "revoked_jti:"

# Process-local fallback: {jti: expire_epoch_seconds}
_mem_denylist = {}
_warned_memory = False


def _get_redis(config=None):
    """Return a shared Redis client or ``None`` (never raises)."""
    try:
        from services.redis_client import get_redis
        url = ""
        if config is not None:
            url = getattr(getattr(config, "auth", None), "redis_url", "") or ""
            if not url:
                # Reuse the app-wide REDIS_URL when auth-specific one is unset.
                url = getattr(getattr(config, "api", None), "redis_url", "") or ""
        return get_redis(url or None)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("token_store: redis lookup failed: %s", e)
        return None


def _mem_purge(now=None):
    now = now or time.time()
    expired = [k for k, exp in _mem_denylist.items() if exp <= now]
    for k in expired:
        _mem_denylist.pop(k, None)


def revoke(jti, ttl_seconds, config=None):
    """Add ``jti`` to the denylist for ``ttl_seconds`` (best-effort).

    ``ttl_seconds`` should be the token's remaining lifetime so the entry
    expires exactly when the token would have anyway.
    """
    global _warned_memory
    if not jti:
        return
    try:
        ttl = int(ttl_seconds)
    except (TypeError, ValueError):
        ttl = 0
    if ttl <= 0:
        return  # already expired; nothing to revoke

    client = _get_redis(config)
    if client is not None:
        try:
            client.setex(_PREFIX + str(jti), ttl, "1")
            return
        except Exception as e:
            logger.warning("token_store: redis revoke failed (%s); using memory.", e)

    # Fallback: in-memory (single process).
    if not _warned_memory:
        logger.warning(
            "token_store: revoking tokens in-memory only (no Redis). "
            "Revocation is NOT shared across workers/replicas; run Redis in prod."
        )
        _warned_memory = True
    _mem_purge()
    _mem_denylist[str(jti)] = time.time() + ttl


def is_revoked(jti, config=None):
    """Return True if ``jti`` is currently revoked (best-effort; fail-open)."""
    if not jti:
        return False
    client = _get_redis(config)
    if client is not None:
        try:
            return client.exists(_PREFIX + str(jti)) == 1
        except Exception as e:
            logger.warning("token_store: redis check failed (%s); using memory.", e)

    _mem_purge()
    return str(jti) in _mem_denylist


def reset_memory_store():
    """Clear the in-memory denylist (tests / reconfiguration)."""
    global _warned_memory
    _mem_denylist.clear()
    _warned_memory = False
