"""Browser Web Push notifications (optional enhancement).

Uses the standard Web Push protocol via VAPID keys — no third-party SMS/call
gateway required, it rides the browser vendor's own push infrastructure
(Chrome/FCM, Firefox/Mozilla push service, etc.), which is part of the open
web platform and free to use.

Fully optional and fail-open:
  * Without ``pywebpush`` installed, or without VAPID keys configured, this
    module silently no-ops — the in-app alert/care-channel history (persisted
    by ``family_care_service``) remains the reliable source of truth.
  * A subscription that the browser has revoked (410/404 response) is pruned
    automatically on next send.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _vapid_configured(config) -> bool:
    fc = getattr(config, "family_care", None)
    return bool(getattr(fc, "vapid_public_key", "") and getattr(fc, "vapid_private_key", ""))


def save_subscription(config, username: str, endpoint: str, p256dh: str, auth: str) -> None:
    from services.db import get_session, init_db, is_ready
    from services.models import PushSubscription

    if not is_ready():
        init_db(config)
    session = get_session()
    try:
        existing = session.query(PushSubscription).filter(
            PushSubscription.endpoint == endpoint
        ).first()
        if existing:
            existing.username = username
            existing.p256dh = p256dh
            existing.auth = auth
        else:
            session.add(PushSubscription(
                username=username, endpoint=endpoint, p256dh=p256dh, auth=auth,
            ))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def remove_subscription(config, endpoint: str) -> None:
    from services.db import get_session, init_db, is_ready
    from services.models import PushSubscription

    if not is_ready():
        init_db(config)
    session = get_session()
    try:
        session.query(PushSubscription).filter(PushSubscription.endpoint == endpoint).delete()
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def send_push_to_user(config, username: str, title: str, body: str) -> int:
    """Send a Web Push notification to every subscribed device of ``username``.

    Returns the number of successful sends. No-ops (returns 0) when VAPID
    keys are not configured or the ``pywebpush`` package is unavailable.
    """
    if not _vapid_configured(config):
        return 0
    try:
        from pywebpush import webpush, WebPushException
    except Exception:
        logger.debug("[WebPush] pywebpush not installed; skipping push send.")
        return 0

    from services.db import get_session, init_db, is_ready
    from services.models import PushSubscription

    if not is_ready():
        init_db(config)
    session = get_session()
    try:
        subs = session.query(PushSubscription).filter(
            PushSubscription.username == username
        ).all()
    finally:
        session.close()

    if not subs:
        return 0

    payload = json.dumps({"title": title, "body": body})
    sent = 0
    stale_endpoints = []
    for sub in subs:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub.endpoint,
                    "keys": {"p256dh": sub.p256dh, "auth": sub.auth},
                },
                data=payload,
                vapid_private_key=config.family_care.vapid_private_key,
                vapid_claims={"sub": f"mailto:{config.family_care.vapid_admin_email}"},
            )
            sent += 1
        except WebPushException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (404, 410):
                stale_endpoints.append(sub.endpoint)
            else:
                logger.warning("[WebPush] send failed for %s: %s", username, e)
        except Exception as e:
            logger.warning("[WebPush] unexpected error sending to %s: %s", username, e)

    for endpoint in stale_endpoints:
        try:
            remove_subscription(config, endpoint)
        except Exception:
            pass

    return sent
