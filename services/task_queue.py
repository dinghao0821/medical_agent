"""Submit facade for async/sync task execution (P2).

Chooses execution mode based on config + Celery availability:

- queue enabled AND Celery/broker available -> dispatch async (returns task_id)
- otherwise                                  -> run synchronously (returns result)

This lets endpoints call one function and remain fully functional whether or not
a broker/worker is deployed.
"""

import logging

logger = logging.getLogger(__name__)


def is_enabled(config) -> bool:
    return bool(getattr(getattr(config, "task_queue", None), "enabled", False))


def _submit(config, celery_task, sync_func, *args, **kwargs) -> dict:
    """Generic submit: async when enabled+available, else synchronous."""
    if is_enabled(config):
        try:
            if celery_task is not None:
                async_result = celery_task.delay(*args, **kwargs)
                return {"mode": "async", "task_id": async_result.id, "state": "PENDING"}
            logger.warning("Task queue enabled but Celery task unavailable; running synchronously.")
        except Exception as e:
            logger.warning("Task enqueue failed (%s); running synchronously.", e)

    result = sync_func(*args, **kwargs)
    return {"mode": "sync", "result": result}


def submit_ingest_directory(config, directory_path: str) -> dict:
    from services.tasks import ingest_directory_task, _ingest_directory
    return _submit(config, ingest_directory_task, _ingest_directory, directory_path)


def submit_ingest_file(config, document_path: str) -> dict:
    from services.tasks import ingest_file_task, _ingest_file
    return _submit(config, ingest_file_task, _ingest_file, document_path)


def get_task_status(task_id: str) -> dict:
    """Return the status/result of a previously submitted async task."""
    from services.celery_app import celery_app, CELERY_AVAILABLE

    if not CELERY_AVAILABLE or celery_app is None:
        return {"task_id": task_id, "state": "UNAVAILABLE",
                "error": "Async task queue is not available on this server."}
    try:
        res = celery_app.AsyncResult(task_id)
        payload = {"task_id": task_id, "state": res.state, "ready": res.ready()}
        if res.ready():
            if res.successful():
                payload["result"] = res.result
            else:
                payload["error"] = str(res.result)
        return payload
    except Exception as e:
        return {"task_id": task_id, "state": "ERROR", "error": str(e)}
