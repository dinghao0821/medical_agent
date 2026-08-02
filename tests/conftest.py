"""Shared test fixtures.

Tests use lightweight ``SimpleNamespace`` configs instead of the real ``Config``
so they run fast and without heavy deps (torch / langchain). External infra
(Redis) is intentionally absent, exercising the services' graceful fallbacks.
"""

import os
import sys
from types import SimpleNamespace

import pytest

# Ensure project root is importable when running pytest from anywhere.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(autouse=True)
def _reset_redis(monkeypatch):
    """Reset the shared Redis client and ensure no REDIS_URL leaks in."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    try:
        from services.redis_client import reset_redis_client
        reset_redis_client()
        yield
        reset_redis_client()
    except Exception:
        yield


def make_config(**overrides):
    """Build a minimal fake config for service tests."""
    cfg = SimpleNamespace(
        api=SimpleNamespace(redis_url=""),
        cache=SimpleNamespace(enabled=False, ttl=60),
        rate_limit=SimpleNamespace(enabled=False, max_requests=60, window_seconds=60),
        object_storage=SimpleNamespace(
            backend="local", endpoint_url="", bucket="test-bucket",
            access_key="", secret_key="", region="", public_base_url="",
        ),
        auth=SimpleNamespace(
            enabled=True, jwt_secret="test-secret", jwt_algorithm="HS256",
            token_expire_minutes=60,
        ),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
