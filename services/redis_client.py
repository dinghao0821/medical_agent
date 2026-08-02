"""Shared Redis client with graceful degradation.

Provides a single, lazily-initialized Redis connection reused by the cache and
rate-limiter services. If ``REDIS_URL`` is unset, the ``redis`` package is
missing, or the server is unreachable, ``get_redis()`` returns ``None`` and the
dependent services fall back to no-op / in-memory behaviour. The connection is
probed once and the result cached, so failures don't repeatedly stall requests.
"""

import os
import logging

logger = logging.getLogger(__name__)

_client = None
_initialized = False


def get_redis(redis_url: str = None):
    """Return a shared Redis client, or ``None`` if unavailable.

    Args:
        redis_url: Optional connection string. Falls back to the ``REDIS_URL``
            environment variable. When empty, Redis-backed features are disabled.
    """
    global _client, _initialized
    if _initialized:
        return _client

    _initialized = True
    url = redis_url or os.getenv("REDIS_URL", "")
    if not url:
        logger.info("REDIS_URL not set; Redis-backed features (cache/rate-limit) disabled.")
        _client = None
        return None

    try:
        import redis  # local import so the app runs without the package installed

        client = redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        _client = client
        logger.info("Connected to Redis at %s", url)
    except Exception as e:
        logger.warning("Redis unavailable (%s); cache/rate-limit will fall back locally.", e)
        _client = None

    return _client


def reset_redis_client():
    """Reset the cached client (useful for tests / reconfiguration)."""
    global _client, _initialized
    _client = None
    _initialized = False
