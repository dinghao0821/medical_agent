from types import SimpleNamespace

from services.cache import CacheService
from tests.conftest import make_config


def test_cache_noop_without_redis():
    # Enabled but no Redis available -> every get misses, set is a no-op.
    cfg = make_config(cache=SimpleNamespace(enabled=True, ttl=60))
    c = CacheService(cfg)
    c.set("chat", "hello", {"response": "hi"})
    assert c.get("chat", "hello") is None


def test_cache_disabled():
    cfg = make_config(cache=SimpleNamespace(enabled=False, ttl=60))
    c = CacheService(cfg)
    assert c.redis is None
    assert c.get("chat", "hello") is None


def test_key_is_deterministic():
    cfg = make_config()
    c = CacheService(cfg)
    assert c._key("chat", "abc") == c._key("chat", "abc")
    assert c._key("chat", "abc") != c._key("chat", "abd")
