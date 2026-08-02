"""Background scheduler for the family-care companion channel.

Periodically scans ``CareReminderTask`` rows for due triggers, generates the
proactive check-in message, logs it to the care channel, and pushes it via
SSE to the elder if they currently have the channel open. Runs as a single
``asyncio`` background task started at app startup — additive, opt-in via
``config.family_care.enabled``, and fails soft: any error is logged and the
loop keeps running on its next tick.

Uses plain polling against the database rather than in-process timers, so a
server restart never causes a reminder to be permanently skipped: the next
scan after restart will pick up any task whose trigger time has passed.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

_task_handle = None


async def _run_once(config):
    from services.family_care_service import (
        find_due_tasks, mark_task_triggered, log_ai_checkin,
    )
    from services.care_message_generator import generate_checkin_message
    from services.care_sse import publish_care_event

    due = await asyncio.to_thread(find_due_tasks, config)
    for task in due:
        elder = task["elder_username"]
        task_type = task["task_type"]
        try:
            message = await asyncio.to_thread(
                generate_checkin_message, config, elder, task_type, task.get("custom_prompt")
            )
            log_entry = await asyncio.to_thread(
                log_ai_checkin, config, elder, message, task["id"], task_type
            )
            await asyncio.to_thread(mark_task_triggered, config, task["id"])
            publish_care_event(elder, {"type": "checkin", "log": log_entry})
            logger.info("[CareScheduler] Triggered task %s for %s", task["id"], elder)
        except Exception as e:
            logger.warning("[CareScheduler] Failed to trigger task %s: %s", task.get("id"), e)


async def _loop(config):
    interval = max(10, int(getattr(config.family_care, "scheduler_interval_seconds", 60)))
    logger.info("[CareScheduler] Started, interval=%ss", interval)
    while True:
        try:
            await _run_once(config)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[CareScheduler] Tick failed: %s", e)
        await asyncio.sleep(interval)


def start(config):
    """Start the scheduler loop as a background asyncio task. Idempotent."""
    global _task_handle
    if not bool(getattr(getattr(config, "family_care", None), "enabled", False)):
        return None
    if _task_handle is not None and not _task_handle.done():
        return _task_handle
    _task_handle = asyncio.create_task(_loop(config))
    return _task_handle
