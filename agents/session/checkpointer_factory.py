"""LangGraph checkpointer factory.

Builds a LangGraph checkpointer according to the requested backend:

- ``memory``  -> ``MemorySaver`` (single-process, default for local dev)
- ``redis``   -> Redis-backed saver (shared across workers/replicas)

The Redis backend enables:
  * multi-worker / multi-replica deployments to share conversation state, and
  * native human-in-the-loop (``interrupt``/``Command(resume=...)``), whose
    pause/resume points must be persisted outside the request lifecycle.

If the Redis backend is requested but unavailable (missing dependency, no URL,
or connection failure), the factory logs a warning and gracefully falls back to
the in-memory saver so the application always remains runnable (e.g. Windows
local ``conda`` env without a Redis server).
"""

import logging

logger = logging.getLogger(__name__)


def _build_memory_checkpointer():
    """Return an in-memory checkpointer (single-process)."""
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def _build_redis_checkpointer(redis_url: str):
    """Return a Redis-backed checkpointer, or None if it cannot be constructed.

    Handles API differences across ``langgraph-checkpoint-redis`` versions,
    including the context-manager style ``from_conn_string`` factory.
    """
    try:
        from langgraph.checkpoint.redis import RedisSaver
    except Exception as e:  # dependency not installed
        logger.warning(
            "langgraph-checkpoint-redis not available (%s); falling back to MemorySaver.",
            e,
        )
        return None

    try:
        saver_or_cm = RedisSaver.from_conn_string(redis_url)
        # Some versions return a context manager that must be entered to obtain
        # the actual saver instance. Keep it open for the process lifetime.
        if hasattr(saver_or_cm, "__enter__"):
            checkpointer = saver_or_cm.__enter__()
        else:
            checkpointer = saver_or_cm

        # Best-effort schema/index setup (idempotent across versions).
        setup = getattr(checkpointer, "setup", None)
        if callable(setup):
            try:
                setup()
            except Exception as e:
                logger.debug("Redis checkpointer setup() skipped: %s", e)

        logger.info("Using Redis checkpointer at %s", redis_url)
        return checkpointer
    except Exception as e:
        logger.warning(
            "Failed to initialize Redis checkpointer (%s); falling back to MemorySaver.",
            e,
        )
        return None


def build_checkpointer(backend: str = "memory", redis_url: str = None):
    """Build a LangGraph checkpointer.

    Args:
        backend: ``"memory"`` or ``"redis"`` (case-insensitive).
        redis_url: Redis connection string, required when ``backend == "redis"``.

    Returns:
        A LangGraph ``BaseCheckpointSaver`` instance. Always returns a usable
        checkpointer, falling back to ``MemorySaver`` on any problem.
    """
    backend = (backend or "memory").strip().lower()

    if backend == "redis":
        if not redis_url:
            logger.warning(
                "CHECKPOINTER_BACKEND=redis but REDIS_URL is empty; "
                "falling back to MemorySaver."
            )
            return _build_memory_checkpointer()
        checkpointer = _build_redis_checkpointer(redis_url)
        if checkpointer is not None:
            return checkpointer
        return _build_memory_checkpointer()

    if backend not in ("memory", ""):
        logger.warning("Unknown CHECKPOINTER_BACKEND=%r; using MemorySaver.", backend)

    return _build_memory_checkpointer()
