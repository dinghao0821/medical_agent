"""Family-caregiver companionship service.

Implements:
  * Follow codes: an elder generates a short-lived numeric code; a caregiver
    redeems it to establish a many-to-many ``FamilyLink`` without ever needing
    the elder's credentials.
  * Reminder tasks: caregivers configure recurring/one-off proactive check-ins
    (medication, mood, meal, safety, follow-up, or fully custom).
  * The AI care channel: proactive check-in messages + the elder's replies,
    with local risk detection escalating to caregiver alerts.

Any ``patient``-role account may act as an elder (generates follow codes) or
as a caregiver (redeems follow codes, configures tasks) — the role is
contextual, not a separate account type.

All functions are additive and only touch the new tables; nothing here
changes existing behaviour when ``config.family_care.enabled`` is False.
"""

import json
import logging
import random
import string
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VALID_TASK_TYPES = ("medication", "mood", "meal", "safety_checkin", "follow_up", "custom")
_VALID_SCHEDULE_TYPES = ("daily", "weekly", "once")

# Preset opening prompts per task type — fed to the LLM as a style guide, not
# used verbatim, so it can weave in the elder's long-term-memory facts.
TASK_TYPE_GUIDANCE = {
    "medication": (
        "你是在提醒一位老年人按时服药。用温和、关心的语气问候，自然提到服药提醒，"
        "并询问是否已经服用、有无不适。不要说教，语气像家人。"
    ),
    "mood": (
        "你是在关心一位老年人今天的心情。用轻松自然的语气问候，"
        "鼓励对方说说今天的心情或发生的事，语气温暖、不说教。"
    ),
    "meal": (
        "你是在关心一位老年人有没有按时吃饭。用温和的语气询问用餐情况和食欲，"
        "语气自然像家人问候。"
    ),
    "safety_checkin": (
        "你是在关心一位老年人今天的活动和居家安全情况。用自然的语气问是否下楼走动、"
        "家里是否一切正常，不要显得像检查/审问。"
    ),
    "follow_up": (
        "你是在提醒一位老年人复诊/随访安排。用关心的语气提醒复诊时间，"
        "询问是否需要家人陪同，语气温和。"
    ),
    "custom": (
        "你是在向一位老年人传达家人设置的关心提醒。请在传达提醒内容后，"
        "自然加一句关心的问候，语气温暖、不生硬。"
    ),
}

TASK_TYPE_LABELS = {
    "medication": "服药提醒",
    "mood": "情绪问候",
    "meal": "饮食提醒",
    "safety_checkin": "安全巡查",
    "follow_up": "复诊/随访提醒",
    "custom": "自定义提醒",
}


# ---- Config / DB helpers ------------------------------------------------------

def _enabled(config) -> bool:
    return bool(getattr(getattr(config, "family_care", None), "enabled", False))


def _ensure_db(config):
    from services.db import init_db, is_ready
    if not is_ready():
        init_db(config)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---- Follow codes & family links ---------------------------------------------

def generate_follow_code(config, elder_username: str) -> Dict[str, Any]:
    """Create a short-lived numeric code the elder can hand to a caregiver."""
    from services.db import get_session
    from services.models import FollowCode

    _ensure_db(config)
    ttl_hours = int(getattr(config.family_care, "follow_code_ttl_hours", 24))
    session = get_session()
    try:
        for _attempt in range(5):
            code = "".join(random.choices(string.digits, k=6))
            exists = session.query(FollowCode.id).filter(FollowCode.code == code).first()
            if not exists:
                break
        else:
            raise RuntimeError("无法生成唯一关注码，请重试")

        row = FollowCode(
            elder_username=elder_username,
            code=code,
            expires_at=_now() + timedelta(hours=ttl_hours),
        )
        session.add(row)
        session.commit()
        return {"code": code, "expires_at": row.expires_at.isoformat()}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def redeem_follow_code(
    config, caregiver_username: str, code: str, relation_label: Optional[str] = None
) -> Dict[str, Any]:
    """Validate a follow code and create the FamilyLink. Raises ValueError on failure."""
    from services.db import get_session
    from services.models import FollowCode, FamilyLink

    _ensure_db(config)
    session = get_session()
    try:
        row = session.query(FollowCode).filter(FollowCode.code == code).first()
        if not row:
            raise ValueError("关注码不存在")
        if row.used_at is not None:
            raise ValueError("关注码已被使用")
        if row.expires_at < _now().replace(tzinfo=row.expires_at.tzinfo):
            raise ValueError("关注码已过期，请让老人重新生成")
        if row.elder_username == caregiver_username:
            raise ValueError("不能关注自己")

        existing = (
            session.query(FamilyLink)
            .filter(
                FamilyLink.elder_username == row.elder_username,
                FamilyLink.caregiver_username == caregiver_username,
                FamilyLink.status == "active",
            )
            .first()
        )
        if existing:
            row.used_at = _now()
            row.used_by = caregiver_username
            session.commit()
            return _link_to_dict(existing)

        link = FamilyLink(
            elder_username=row.elder_username,
            caregiver_username=caregiver_username,
            relation_label=(relation_label or "").strip()[:64] or None,
            status="active",
        )
        session.add(link)
        row.used_at = _now()
        row.used_by = caregiver_username
        session.commit()
        session.refresh(link)
        return _link_to_dict(link)
    except ValueError:
        session.rollback()
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def list_elders_for_caregiver(config, caregiver_username: str) -> List[Dict[str, Any]]:
    from services.db import get_session
    from services.models import FamilyLink

    _ensure_db(config)
    session = get_session()
    try:
        rows = (
            session.query(FamilyLink)
            .filter(
                FamilyLink.caregiver_username == caregiver_username,
                FamilyLink.status == "active",
            )
            .order_by(FamilyLink.created_at.desc())
            .all()
        )
        return [_link_to_dict(r) for r in rows]
    finally:
        session.close()


def list_caregivers_for_elder(config, elder_username: str) -> List[Dict[str, Any]]:
    from services.db import get_session
    from services.models import FamilyLink

    _ensure_db(config)
    session = get_session()
    try:
        rows = (
            session.query(FamilyLink)
            .filter(
                FamilyLink.elder_username == elder_username,
                FamilyLink.status == "active",
            )
            .order_by(FamilyLink.created_at.desc())
            .all()
        )
        return [_link_to_dict(r) for r in rows]
    finally:
        session.close()


def revoke_link(config, link_id: int, requester_username: str) -> bool:
    """Either party in the link may revoke it."""
    from services.db import get_session
    from services.models import FamilyLink

    _ensure_db(config)
    session = get_session()
    try:
        row = session.query(FamilyLink).filter(FamilyLink.id == link_id).first()
        if not row:
            return False
        if requester_username not in (row.elder_username, row.caregiver_username):
            raise PermissionError("无权解除该关系")
        row.status = "revoked"
        session.commit()
        return True
    except PermissionError:
        session.rollback()
        raise
    finally:
        session.close()


def _link_to_dict(row) -> Dict[str, Any]:
    return {
        "id": row.id,
        "elder_username": row.elder_username,
        "caregiver_username": row.caregiver_username,
        "relation_label": row.relation_label,
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _is_linked(config, elder_username: str, caregiver_username: str) -> bool:
    from services.db import get_session
    from services.models import FamilyLink

    _ensure_db(config)
    session = get_session()
    try:
        return (
            session.query(FamilyLink.id)
            .filter(
                FamilyLink.elder_username == elder_username,
                FamilyLink.caregiver_username == caregiver_username,
                FamilyLink.status == "active",
            )
            .first()
            is not None
        )
    finally:
        session.close()


# ---- Reminder tasks ------------------------------------------------------------

def create_reminder_task(
    config,
    caregiver_username: str,
    elder_username: str,
    task_type: str,
    schedule_type: str,
    schedule_time: Optional[str] = None,
    schedule_weekday: Optional[int] = None,
    schedule_datetime: Optional[datetime] = None,
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    from services.db import get_session
    from services.models import CareReminderTask

    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(f"不支持的任务类型: {task_type}")
    if schedule_type not in _VALID_SCHEDULE_TYPES:
        raise ValueError(f"不支持的调度类型: {schedule_type}")
    if task_type == "custom" and not (custom_prompt or "").strip():
        raise ValueError("自定义提醒必须填写提醒内容")
    if not _is_linked(config, elder_username, caregiver_username):
        raise PermissionError("尚未关注该老人，无法为其配置提醒")

    if schedule_type in ("daily", "weekly") and not schedule_time:
        raise ValueError("daily/weekly 任务必须指定 schedule_time（HH:MM）")
    if schedule_type == "weekly" and schedule_weekday is None:
        raise ValueError("weekly 任务必须指定 schedule_weekday（0-6）")
    if schedule_type == "once" and schedule_datetime is None:
        raise ValueError("once 任务必须指定 schedule_datetime")

    _ensure_db(config)
    session = get_session()
    try:
        row = CareReminderTask(
            elder_username=elder_username,
            created_by=caregiver_username,
            task_type=task_type,
            custom_prompt=(custom_prompt or "").strip()[:1000] or None,
            schedule_type=schedule_type,
            schedule_time=schedule_time or "",
            schedule_weekday=schedule_weekday,
            schedule_datetime=schedule_datetime,
            status="active",
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _task_to_dict(row)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def list_reminder_tasks(config, elder_username: str) -> List[Dict[str, Any]]:
    from services.db import get_session
    from services.models import CareReminderTask

    _ensure_db(config)
    session = get_session()
    try:
        rows = (
            session.query(CareReminderTask)
            .filter(CareReminderTask.elder_username == elder_username)
            .order_by(CareReminderTask.created_at.desc())
            .all()
        )
        return [_task_to_dict(r) for r in rows]
    finally:
        session.close()


def update_reminder_task(
    config, task_id: int, requester_username: str, status: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    from services.db import get_session
    from services.models import CareReminderTask

    _ensure_db(config)
    session = get_session()
    try:
        row = session.query(CareReminderTask).filter(CareReminderTask.id == task_id).first()
        if not row:
            return None
        if row.created_by != requester_username:
            raise PermissionError("只能由创建者修改该任务")
        if status:
            if status not in ("active", "paused"):
                raise ValueError("status 必须为 active 或 paused")
            row.status = status
        session.commit()
        session.refresh(row)
        return _task_to_dict(row)
    except (PermissionError, ValueError):
        session.rollback()
        raise
    finally:
        session.close()


def delete_reminder_task(config, task_id: int, requester_username: str) -> bool:
    from services.db import get_session
    from services.models import CareReminderTask

    _ensure_db(config)
    session = get_session()
    try:
        row = session.query(CareReminderTask).filter(CareReminderTask.id == task_id).first()
        if not row:
            return False
        if row.created_by != requester_username:
            raise PermissionError("只能由创建者删除该任务")
        session.delete(row)
        session.commit()
        return True
    except PermissionError:
        session.rollback()
        raise
    finally:
        session.close()


def _task_to_dict(row) -> Dict[str, Any]:
    return {
        "id": row.id,
        "elder_username": row.elder_username,
        "created_by": row.created_by,
        "task_type": row.task_type,
        "task_type_label": TASK_TYPE_LABELS.get(row.task_type, row.task_type),
        "custom_prompt": row.custom_prompt,
        "schedule_type": row.schedule_type,
        "schedule_time": row.schedule_time,
        "schedule_weekday": row.schedule_weekday,
        "schedule_datetime": row.schedule_datetime.isoformat() if row.schedule_datetime else None,
        "status": row.status,
        "last_triggered_at": row.last_triggered_at.isoformat() if row.last_triggered_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ---- Due-task scanning (called by the background scheduler) -----------------

def find_due_tasks(config) -> List[Dict[str, Any]]:
    """Return active tasks whose scheduled trigger time has arrived.

    Pure read + comparison against wall-clock time; the caller is responsible
    for generating the message and calling ``mark_task_triggered``.
    """
    from services.db import get_session
    from services.models import CareReminderTask

    _ensure_db(config)
    now = datetime.now()  # naive local time to match "HH:MM" schedule_time semantics
    today = now.date()
    session = get_session()
    try:
        rows = (
            session.query(CareReminderTask)
            .filter(CareReminderTask.status == "active")
            .all()
        )
        due = []
        for row in rows:
            if _is_due(row, now, today):
                due.append(_task_to_dict(row))
        return due
    finally:
        session.close()


def _is_due(row, now: datetime, today) -> bool:
    try:
        if row.schedule_type == "once":
            if row.schedule_datetime is None:
                return False
            if row.last_triggered_at is not None:
                return False
            return now >= row.schedule_datetime
        if not row.schedule_time or ":" not in row.schedule_time:
            return False
        hh, mm = row.schedule_time.split(":", 1)
        target_today = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        if row.schedule_type == "daily":
            already_today = (
                row.last_triggered_at is not None and row.last_triggered_at.date() == today
            )
            return now >= target_today and not already_today
        if row.schedule_type == "weekly":
            if row.schedule_weekday is None or now.weekday() != int(row.schedule_weekday):
                return False
            already_this_week = (
                row.last_triggered_at is not None
                and (today - row.last_triggered_at.date()).days < 7
            )
            return now >= target_today and not already_this_week
        return False
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[FamilyCare] _is_due error for task %s: %s", getattr(row, "id", "?"), e)
        return False


def mark_task_triggered(config, task_id: int) -> None:
    from services.db import get_session
    from services.models import CareReminderTask

    _ensure_db(config)
    session = get_session()
    try:
        row = session.query(CareReminderTask).filter(CareReminderTask.id == task_id).first()
        if row:
            row.last_triggered_at = datetime.now()
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---- Care conversation log + risk escalation ----------------------------------

def log_ai_checkin(
    config, elder_username: str, ai_message: str, task_id: Optional[int] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    from services.db import get_session
    from services.models import CareConversationLog

    _ensure_db(config)
    session = get_session()
    try:
        row = CareConversationLog(
            elder_username=elder_username,
            task_id=task_id,
            task_type=task_type,
            ai_message=ai_message,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _log_to_dict(row)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_care_messages(config, elder_username: str, limit: int = 50) -> List[Dict[str, Any]]:
    from services.db import get_session
    from services.models import CareConversationLog

    _ensure_db(config)
    limit = max(1, min(limit, 200))
    session = get_session()
    try:
        rows = (
            session.query(CareConversationLog)
            .filter(CareConversationLog.elder_username == elder_username)
            .order_by(CareConversationLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [_log_to_dict(r) for r in reversed(rows)]
    finally:
        session.close()


def record_elder_reply(
    config, log_id: int, elder_username: str, reply_text: str
) -> Tuple[Optional[Dict[str, Any]], Optional[dict]]:
    """Store the elder's reply and run local risk detection.

    Returns (updated_log_dict, concern_dict_or_None). Raises PermissionError if
    the log doesn't belong to the caller.
    """
    from services.care_risk_detector import detect_concern
    from services.db import get_session
    from services.models import CareConversationLog

    _ensure_db(config)
    session = get_session()
    try:
        row = session.query(CareConversationLog).filter(CareConversationLog.id == log_id).first()
        if not row:
            return None, None
        if row.elder_username != elder_username:
            raise PermissionError("无权回复该消息")

        row.elder_reply = (reply_text or "").strip()[:2000]
        row.replied_at = datetime.now()
        concern = detect_concern(row.elder_reply)
        if concern:
            row.risk_flag = "concern"
        session.commit()
        session.refresh(row)
        return _log_to_dict(row), concern
    except PermissionError:
        session.rollback()
        raise
    finally:
        session.close()


def mark_alert_generated(config, log_id: int) -> None:
    from services.db import get_session
    from services.models import CareConversationLog

    _ensure_db(config)
    session = get_session()
    try:
        row = session.query(CareConversationLog).filter(CareConversationLog.id == log_id).first()
        if row:
            row.alert_sent = True
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _log_to_dict(row) -> Dict[str, Any]:
    return {
        "id": row.id,
        "elder_username": row.elder_username,
        "task_id": row.task_id,
        "task_type": row.task_type,
        "task_type_label": TASK_TYPE_LABELS.get(row.task_type, row.task_type) if row.task_type else None,
        "ai_message": row.ai_message,
        "elder_reply": row.elder_reply,
        "replied_at": row.replied_at.isoformat() if row.replied_at else None,
        "risk_flag": row.risk_flag,
        "alert_sent": row.alert_sent,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ---- Family alerts -------------------------------------------------------------

def create_alerts_for_elder(
    config, elder_username: str, summary: str, source_log_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fan out one alert to every active caregiver following this elder."""
    from services.db import get_session
    from services.models import FamilyAlert

    caregivers = list_caregivers_for_elder(config, elder_username)
    if not caregivers:
        return []

    _ensure_db(config)
    session = get_session()
    created = []
    try:
        for link in caregivers:
            row = FamilyAlert(
                elder_username=elder_username,
                caregiver_username=link["caregiver_username"],
                source_log_id=source_log_id,
                summary=summary[:500],
            )
            session.add(row)
            created.append(row)
        session.commit()
        for row in created:
            session.refresh(row)
        return [_alert_to_dict(r) for r in created]
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def list_alerts(config, caregiver_username: str, unread_only: bool = False) -> List[Dict[str, Any]]:
    from services.db import get_session
    from services.models import FamilyAlert

    _ensure_db(config)
    session = get_session()
    try:
        query = session.query(FamilyAlert).filter(
            FamilyAlert.caregiver_username == caregiver_username
        )
        if unread_only:
            query = query.filter(FamilyAlert.is_read.is_(False))
        rows = query.order_by(FamilyAlert.created_at.desc()).limit(200).all()
        return [_alert_to_dict(r) for r in rows]
    finally:
        session.close()


def mark_alert_read(config, alert_id: int, caregiver_username: str) -> bool:
    from services.db import get_session
    from services.models import FamilyAlert

    _ensure_db(config)
    session = get_session()
    try:
        row = session.query(FamilyAlert).filter(FamilyAlert.id == alert_id).first()
        if not row:
            return False
        if row.caregiver_username != caregiver_username:
            raise PermissionError("无权操作该通知")
        row.is_read = True
        session.commit()
        return True
    except PermissionError:
        session.rollback()
        raise
    finally:
        session.close()


def _alert_to_dict(row) -> Dict[str, Any]:
    return {
        "id": row.id,
        "elder_username": row.elder_username,
        "caregiver_username": row.caregiver_username,
        "source_log_id": row.source_log_id,
        "summary": row.summary,
        "is_read": row.is_read,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
