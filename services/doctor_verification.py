r"""Doctor licence verification service.

Registering with a valid invite code alone is NOT enough to become a trusted
doctor: an account must upload a practising-licence certificate and be approved
by an admin/reviewer. Only ``approved`` doctors may access the review queue and
receive patient medical-image diagnoses.

Doctor status lifecycle::

    unsubmitted --upload--> pending --approve--> approved
                                   \--reject--> rejected --re-upload--> pending
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def set_license(config, username: str, license_path: str) -> Optional[dict]:
    """Attach an uploaded licence to a doctor and move them to ``pending``."""
    from services.db import init_db, is_ready, get_session
    from services.models import User

    if not is_ready():
        init_db(config)

    session = get_session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if not user or user.role != "doctor":
            return None
        user.license_path = license_path
        user.doctor_status = "pending"
        user.license_reviewed_by = None
        user.license_reviewed_at = None
        user.license_comments = None
        session.commit()
        logger.info("Doctor %s submitted licence for review", username)
        return _user_to_dict(user)
    except Exception as e:
        session.rollback()
        logger.error("Failed to set licence for %s: %s", username, e)
        raise
    finally:
        session.close()


def get_status(config, username: str) -> Optional[dict]:
    """Return a doctor's verification status, or None if not found."""
    from services.db import init_db, is_ready, get_session
    from services.models import User

    if not is_ready():
        init_db(config)

    session = get_session()
    try:
        user = session.query(User).filter(User.username == username).first()
        return _user_to_dict(user) if user else None
    finally:
        session.close()


def list_pending(config) -> list:
    """List all doctors awaiting licence review (for the reviewer console)."""
    from services.db import init_db, is_ready, get_session
    from services.models import User

    if not is_ready():
        init_db(config)

    session = get_session()
    try:
        rows = (
            session.query(User)
            .filter(User.role == "doctor", User.doctor_status == "pending")
            .order_by(User.created_at.asc())
            .all()
        )
        return [_user_to_dict(u) for u in rows]
    finally:
        session.close()


def review_license(
    config,
    username: str,
    reviewer: str,
    verdict: str,
    comments: Optional[str] = None,
) -> Optional[dict]:
    """Admin approves/rejects a doctor's uploaded licence."""
    from services.db import init_db, is_ready, get_session
    from services.models import User

    if verdict not in ("approved", "rejected"):
        raise ValueError("verdict must be 'approved' or 'rejected'")

    if not is_ready():
        init_db(config)

    session = get_session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if not user or user.role != "doctor":
            return None
        user.doctor_status = verdict
        user.license_reviewed_by = reviewer
        user.license_reviewed_at = datetime.now(timezone.utc)
        user.license_comments = comments
        session.commit()
        logger.info("Doctor %s licence %s by %s", username, verdict, reviewer)
        return _user_to_dict(user)
    except Exception as e:
        session.rollback()
        logger.error("Failed to review licence for %s: %s", username, e)
        raise
    finally:
        session.close()


def is_approved_doctor(config, username: str) -> bool:
    """True only when the account is a doctor with an approved licence."""
    info = get_status(config, username)
    return bool(info and info.get("role") == "doctor" and info.get("doctor_status") == "approved")


def _user_to_dict(user) -> dict:
    return {
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "doctor_status": user.doctor_status,
        "license_path": user.license_path,
        "license_reviewed_by": user.license_reviewed_by,
        "license_reviewed_at": user.license_reviewed_at.isoformat() if user.license_reviewed_at else None,
        "license_comments": user.license_comments,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }
