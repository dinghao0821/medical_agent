"""Redis-backed response/result cache with graceful no-op fallback.

Used to cache deterministic, repeatable results (e.g. identical factual queries)
to cut latency and LLM cost. When Redis is unavailable the cache silently no-ops
(every ``get`` misses), so behaviour is identical to running without a cache.

Note on correctness: caching conversational responses ignores per-session
context, so this is disabled by default (``ENABLE_CACHE=false``) and best suited
to stateless factual lookups. Enable deliberately.
"""

import json
import hashlib
import logging

from services.redis_client import get_redis

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self, config):
        cache_cfg = getattr(config, "cache", None)
        self.enabled = bool(getattr(cache_cfg, "enabled", False))
        self.default_ttl = int(getattr(cache_cfg, "ttl", 3600))
        self.prefix = "cache:"
        redis_url = getattr(getattr(config, "api", None), "redis_url", None)
        self.redis = get_redis(redis_url) if self.enabled else None
        if self.enabled and self.redis is None:
            logger.info("Cache enabled but Redis unavailable; cache will no-op.")

    def _key(self, namespace: str, payload) -> str:
        if isinstance(payload, str):
            raw = payload
        else:
            raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{self.prefix}{namespace}:{digest}"

    def get(self, namespace: str, payload):
        """Return the cached value for (namespace, payload), or ``None`` on miss."""
        if not self.redis:
            return None
        try:
            value = self.redis.get(self._key(namespace, payload))
            return json.loads(value) if value is not None else None
        except Exception as e:
            logger.warning("Cache get failed (%s); treating as miss.", e)
            return None

    def set(self, namespace: str, payload, value, ttl: int = None):
        """Store ``value`` for (namespace, payload) with an optional TTL."""
        if not self.redis:
            return
        try:
            self.redis.setex(
                self._key(namespace, payload),
                int(ttl or self.default_ttl),
                json.dumps(value, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning("Cache set failed (%s); skipping.", e)
