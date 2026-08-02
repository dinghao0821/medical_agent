"""Admin user-management service: list users + change roles safely.

Design goals (security-first, minimal surface):
  * Only reachable via endpoints gated behind ``require_roles("admin")`` — this
    module itself does not re-check the caller's role, the API layer does.
  * Never allows the system to end up with zero admins ("lockout protection"):
    demoting the *last* remaining admin (self or anyone else) is rejected.
  * Promoting a user to ``doctor`` does NOT bypass the licence-verification
    workflow: their ``doctor_status`` is reset to ``unsubmitted`` so they still
    must upload a licence and get approved before touching patient data.
  * Every successful role change is meant to be paired with an audit-log write
    by the caller (``app.py``), which already has ``services.audit`` wired up.
  * Password hashes are never returned to the caller.
"""

import logging
from typing import Any, Dict, List, Optional

from services.auth import VALID_ROLES

logger = logging.getLogger(__name__)


def _ensure_db(config):
    from services.db import init_db, is_ready
    if not is_ready():
        init_db(config)


def _user_to_dict(user) -> Dict[str, Any]:
    return {
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "doctor_status": user.doctor_status,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def list_users(config, search: Optional[str] = None, role: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all users (no password hashes). Optional username substring / role filter."""
    from services.db import get_session
    from services.models import User

    _ensure_db(config)
    session = get_session()
    try:
        query = session.query(User)
        if search:
            query = query.filter(User.username.ilike(f"%{search.strip()}%"))
        if role:
            query = query.filter(User.role == role)
        rows = query.order_by(User.created_at.asc()).all()
        return [_user_to_dict(u) for u in rows]
    finally:
        session.close()


def count_admins(config) -> int:
    from services.db import get_session
    from services.models import User

    _ensure_db(config)
    session = get_session()
    try:
        return session.query(User).filter(User.role == "admin").count()
    finally:
        session.close()


def update_user_role(
    config, target_username: str, new_role: str, actor_username: str
) -> Optional[Dict[str, Any]]:
    """Change ``target_username``'s role. Returns the updated user dict, or
    ``None`` if the target doesn't exist.

    Raises:
        ValueError: ``new_role`` is not one of ``VALID_ROLES``.
        PermissionError: the change would remove the last remaining admin.
    """
    if new_role not in VALID_ROLES:
        raise ValueError(f"不支持的角色: {new_role}（必须是 {VALID_ROLES} 之一）")

    from services.db import get_session
    from services.models import User

    _ensure_db(config)
    session = get_session()
    try:
        target = session.query(User).filter(User.username == target_username).first()
        if not target:
            return None

        if target.role == "admin" and new_role != "admin":
            admin_count = session.query(User).filter(User.role == "admin").count()
            if admin_count <= 1:
                raise PermissionError("不能移除系统中唯一的管理员，请先设立另一位管理员。")

        old_role = target.role
        target.role = new_role

        # Promoting into "doctor" must still go through licence verification —
        # never let a role change alone grant access to the review queue.
        if new_role == "doctor" and old_role != "doctor":
            target.doctor_status = "unsubmitted"
            target.license_reviewed_by = None
            target.license_reviewed_at = None
            target.license_comments = None

        session.commit()
        logger.info(
            "Role change: %s -> %s (actor=%s, was=%s)",
            target_username, new_role, actor_username, old_role,
        )
        return _user_to_dict(target)
    except (ValueError, PermissionError):
        session.rollback()
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
