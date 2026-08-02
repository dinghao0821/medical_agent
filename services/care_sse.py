"""In-process SSE pub/sub for the family-care companion channel.

A minimal per-username asyncio.Queue registry so the scheduler and the reply
endpoint can push events (new check-in message, new alert) to whichever
elder/caregiver browser tab currently holds an open SSE connection.

Single-process only (no Redis backing) — consistent with this project's
existing SSE endpoints (e.g. the doctor-review stream), which are likewise
in-process. When no subscriber is connected, events are simply dropped; the
underlying data is already persisted by ``family_care_service``, so the
client picks it up via polling/history on next load regardless.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict

logger = logging.getLogger(__name__)

# username -> set of asyncio.Queue subscribers
_subscribers: Dict[str, "set"] = defaultdict(set)


def subscribe(username: str) -> "asyncio.Queue":
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers[username].add(q)
    return q


def unsubscribe(username: str, q: "asyncio.Queue") -> None:
    try:
        _subscribers[username].discard(q)
    except Exception:
        pass


def _publish(username: str, event: Dict[str, Any]) -> None:
    for q in list(_subscribers.get(username, ())):
        try:
            q.put_nowait(event)
        except Exception:
            pass  # queue full/closed — drop, client will catch up via history


def publish_care_event(elder_username: str, event: Dict[str, Any]) -> None:
    """Push an event to the elder's open care-channel SSE connection, if any."""
    try:
        _publish(elder_username, event)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[CareSSE] publish_care_event failed: %s", e)


def publish_alert_event(caregiver_username: str, event: Dict[str, Any]) -> None:
    """Push an event to the caregiver's open alerts SSE connection, if any."""
    try:
        _publish(caregiver_username, event)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[CareSSE] publish_alert_event failed: %s", e)
