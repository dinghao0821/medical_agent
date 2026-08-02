import pytest


def _memory_saver_cls():
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver


def test_redis_without_url_falls_back_to_memory():
    pytest.importorskip("langgraph")
    from agents.session.checkpointer_factory import build_checkpointer
    cp = build_checkpointer(backend="redis", redis_url=None)
    assert isinstance(cp, _memory_saver_cls())


def test_memory_backend():
    pytest.importorskip("langgraph")
    from agents.session.checkpointer_factory import build_checkpointer
    cp = build_checkpointer(backend="memory", redis_url=None)
    assert isinstance(cp, _memory_saver_cls())


def test_unknown_backend_falls_back():
    pytest.importorskip("langgraph")
    from agents.session.checkpointer_factory import build_checkpointer
    cp = build_checkpointer(backend="weird", redis_url=None)
    assert isinstance(cp, _memory_saver_cls())
