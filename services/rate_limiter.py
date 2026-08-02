"""Distributed rate limiter (fixed-window) with in-memory fallback.

Replaces the previous unused ``config.api.rate_limit`` placeholder with a working
limiter:

- **Redis backend** (preferred): a fixed-window counter shared across all workers
  / replicas, so the limit is enforced globally.
- **In-memory fallback**: when Redis is unavailable, a per-process counter is used
  (best-effort; not shared across workers).

The limiter is **fail-open**: any backend error allows the request through rather
than blocking legitimate traffic on an infrastructure hiccup. Disabled by default
(``ENABLE_RATE_LIMIT=false``).
"""

import time
import logging

from services.redis_client import get_redis

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, config):
        rl_cfg = getattr(config, "rate_limit", None)
        self.enabled = bool(getattr(rl_cfg, "enabled", False))
        self.max_requests = int(getattr(rl_cfg, "max_requests", 60))
        self.window = int(getattr(rl_cfg, "window_seconds", 60))
        redis_url = getattr(getattr(config, "api", None), "redis_url", None)
        self.redis = get_redis(redis_url) if self.enabled else None
        # Per-process fallback state: {client_id: (count, window_start)}
        self._local = {}

    def check(self, client_id: str):
        """Register a request and decide whether it is allowed.

        Returns:
            (allowed: bool, info: dict) where info carries limit/remaining/reset.
        """
        if not self.enabled:
            return True, {}

        now = int(time.time())
        window_start = now - (now % self.window)
        reset_at = window_start + self.window

        # Redis-backed global counter.
        if self.redis:
            try:
                key = f"ratelimit:{client_id}:{window_start}"
                count = self.redis.incr(key)
                if count == 1:
                    self.redis.expire(key, self.window)
                allowed = count <= self.max_requests
                return allowed, {
                    "limit": self.max_requests,
                    "remaining": max(0, self.max_requests - count),
                    "reset": reset_at,
                }
            except Exception as e:
                logger.warning("Rate limiter Redis error (%s); failing open.", e)
                return True, {}

        # In-memory per-process fallback.
        count, ws = self._local.get(client_id, (0, window_start))
        if ws != window_start:
            count, ws = 0, window_start
        count += 1
        self._local[client_id] = (count, ws)
        allowed = count <= self.max_requests
        return allowed, {
            "limit": self.max_requests,
            "remaining": max(0, self.max_requests - count),
            "reset": reset_at,
        }
