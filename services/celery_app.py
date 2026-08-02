"""Celery application for async heavy tasks (P2).

Robust to a missing ``celery`` package or broker: if Celery cannot be imported,
``celery_app`` is ``None`` and ``CELERY_AVAILABLE`` is ``False``; callers then run
work synchronously (see ``services/task_queue.py``), so the app still runs.

Worker start (needs Redis broker):
    celery -A services.celery_app:celery_app worker --loglevel=info
"""

import os
import logging

logger = logging.getLogger(__name__)

celery_app = None
CELERY_AVAILABLE = False

try:
    from celery import Celery

    _broker = (
        os.getenv("CELERY_BROKER_URL")
        or os.getenv("REDIS_URL")
        or "redis://localhost:6379/0"
    )
    _backend = os.getenv("CELERY_RESULT_BACKEND") or _broker

    celery_app = Celery("medical_assistant", broker=_broker, backend=_backend)
    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        # Force import of task definitions on worker start.
        imports=("services.tasks",),
    )
    CELERY_AVAILABLE = True
except Exception as e:  # pragma: no cover - optional dependency / import-time issues
    logger.warning("Celery unavailable (%s); async task queue disabled.", e)
    celery_app = None
    CELERY_AVAILABLE = False
