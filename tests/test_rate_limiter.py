from types import SimpleNamespace

from services.rate_limiter import RateLimiter
from tests.conftest import make_config


def test_disabled_allows_all():
    cfg = make_config(rate_limit=SimpleNamespace(enabled=False, max_requests=1, window_seconds=60))
    rl = RateLimiter(cfg)
    for _ in range(5):
        allowed, _info = rl.check("client-a")
        assert allowed is True


def test_in_memory_fallback_blocks_after_limit():
    # No REDIS_URL -> in-memory fallback.
    cfg = make_config(rate_limit=SimpleNamespace(enabled=True, max_requests=3, window_seconds=60))
    rl = RateLimiter(cfg)
    results = [rl.check("client-b")[0] for _ in range(4)]
    assert results == [True, True, True, False]


def test_per_client_isolation():
    cfg = make_config(rate_limit=SimpleNamespace(enabled=True, max_requests=1, window_seconds=60))
    rl = RateLimiter(cfg)
    assert rl.check("c1")[0] is True
    assert rl.check("c2")[0] is True   # different client not affected
    assert rl.check("c1")[0] is False  # c1 exhausted
