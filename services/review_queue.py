"""
Doctor review queue for human-in-the-loop medical image validation.

When a patient uploads a medical image (brain MRI, chest X-ray, skin lesion),
the AI analysis is held here until a licensed doctor approves or rejects it.
The paused LangGraph session is resumed with the doctor's verdict, and the
final result is pushed to the patient via SSE.

Architecture:
    Patient upload → AI analysis → create_case() → graph interrupt →
    Doctor reviews → review_case() → resume graph → SSE push to patient
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def create_case(
    config,
    patient_username: str,
    session_id: str,
    image_path: str,
    image_type: str,
    ai_agent: str,
    ai_diagnosis: str,
    result_image: Optional[str] = None,
) -> str:
    """Create a new pending review case, returns the case_uid."""
    from services.db import init_db, is_ready, get_session
    from services.models import ReviewCase

    if not is_ready():
        init_db(config)

    case_uid = uuid.uuid4().hex[:12]
    session = get_session()
    try:
        case = ReviewCase(
            case_uid=case_uid,
            patient_username=patient_username,
            session_id=session_id,
            image_path=image_path,
            image_type=image_type,
            ai_agent=ai_agent,
            ai_diagnosis=ai_diagnosis,
            result_image=result_image,
            status="pending",
        )
        session.add(case)
        session.commit()
        logger.info("Review case %s created for patient %s", case_uid, patient_username)
        return case_uid
    except Exception as e:
        session.rollback()
        logger.error("Failed to create review case: %s", e)
        raise
    finally:
        session.close()


def get_pending_cases(config) -> list:
    """Return all pending cases (ordered by newest first) as dicts."""
    from services.db import init_db, is_ready, get_session
    from services.models import ReviewCase

    if not is_ready():
        init_db(config)

    session = get_session()
    try:
        cases = (
            session.query(ReviewCase)
            .filter(ReviewCase.status == "pending")
            .order_by(ReviewCase.created_at.desc())
            .all()
        )
        return [_case_to_dict(c) for c in cases]
    finally:
        session.close()


def get_case(config, case_uid: str) -> Optional[dict]:
    """Get a single case by UID."""
    from services.db import init_db, is_ready, get_session
    from services.models import ReviewCase

    if not is_ready():
        init_db(config)

    session = get_session()
    try:
        case = session.query(ReviewCase).filter(ReviewCase.case_uid == case_uid).first()
        return _case_to_dict(case) if case else None
    finally:
        session.close()


def review_case(
    config,
    case_uid: str,
    doctor_username: str,
    verdict: str,
    comments: Optional[str] = None,
    final_result: Optional[str] = None,
) -> Optional[dict]:
    """Doctor reviews a case: approve or reject."""
    from services.db import init_db, is_ready, get_session
    from services.models import ReviewCase

    if verdict not in ("approved", "rejected"):
        raise ValueError("verdict must be 'approved' or 'rejected'")

    if not is_ready():
        init_db(config)

    session = get_session()
    try:
        case = session.query(ReviewCase).filter(ReviewCase.case_uid == case_uid).first()
        if not case:
            logger.warning("Review case %s not found", case_uid)
            return None
        if case.status != "pending":
            logger.warning("Review case %s already reviewed, status=%s", case_uid, case.status)
            return _case_to_dict(case)

        case.status = verdict
        case.doctor_username = doctor_username
        case.doctor_comments = comments
        case.final_result = final_result
        case.reviewed_at = datetime.now(timezone.utc)
        session.commit()

        logger.info(
            "Review case %s %s by doctor %s", case_uid, verdict, doctor_username
        )
        return _case_to_dict(case)
    except Exception as e:
        session.rollback()
        logger.error("Failed to review case %s: %s", case_uid, e)
        raise
    finally:
        session.close()


def get_case_status_for_patient(config, case_uid: str) -> Optional[dict]:
    """Return minimal status info for the patient (no doctor PII)."""
    from services.db import init_db, is_ready, get_session
    from services.models import ReviewCase

    if not is_ready():
        init_db(config)

    session = get_session()
    try:
        case = session.query(ReviewCase).filter(ReviewCase.case_uid == case_uid).first()
        if not case:
            return None
        return {
            "status": case.status,
            "final_result": case.final_result,
            "doctor_comments": case.doctor_comments,
            "reviewed_at": case.reviewed_at.isoformat() if case.reviewed_at else None,
        }
    finally:
        session.close()


def _case_to_dict(case) -> dict:
    """Convert ORM object to plain dict for JSON serialization."""
    return {
        "case_uid": case.case_uid,
        "patient_username": case.patient_username,
        "session_id": case.session_id,
        "image_path": case.image_path,
        "image_type": case.image_type,
        "ai_agent": case.ai_agent,
        "ai_diagnosis": case.ai_diagnosis,
        "result_image": case.result_image,
        "status": case.status,
        "doctor_username": case.doctor_username,
        "doctor_comments": case.doctor_comments,
        "final_result": case.final_result,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "reviewed_at": case.reviewed_at.isoformat() if case.reviewed_at else None,
    }
