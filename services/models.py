"""ORM models (P3): users and audit logs."""

from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean

from services.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    # Role-based access control: patient | doctor | admin
    role = Column(String(32), default="patient", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    doctor_status = Column(String(16), default="none", nullable=False)
    license_path = Column(String(512), nullable=True)
    license_reviewed_by = Column(String(64), nullable=True)
    license_reviewed_at = Column(DateTime, nullable=True)
    license_comments = Column(Text, nullable=True)


class AuditLog(Base):
    """Append-only audit trail for diagnostic / sensitive operations.

    ``detail`` stores a PII-masked (and optionally encrypted) summary of the
    request, supporting compliance traceability without leaking raw PII.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    username = Column(String(64), index=True, nullable=True)
    role = Column(String(32), nullable=True)
    action = Column(String(64), nullable=True)          # chat | upload | validate
    agent = Column(String(128), nullable=True)          # routed agent name
    session_id = Column(String(64), index=True, nullable=True)
    validation_result = Column(String(16), nullable=True)
    detail = Column(Text, nullable=True)                # PII-masked / encrypted


class ReviewCase(Base):
    """A medical-image diagnosis awaiting a licensed doctor's review (HITL).

    When a patient uploads a pathology/medical image, the AI produces a
    provisional diagnosis and the LangGraph pauses at its native ``interrupt()``
    (persisted by the checkpointer under ``session_id``). Instead of letting the
    patient self-confirm, a ``ReviewCase`` row is created (``pending``) and
    surfaced on the doctor console. A doctor approves/rejects it; the graph is
    then resumed with that verdict and the finalized answer is pushed back to the
    patient. This makes human validation cross-user, asynchronous and auditable.
    """
    __tablename__ = "review_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Public opaque id used by the patient (SSE/poll) and the doctor console.
    case_uid = Column(String(64), unique=True, index=True, nullable=False)
    patient_username = Column(String(64), index=True, nullable=True)
    # LangGraph thread id — used to resume the paused graph on the doctor's verdict.
    session_id = Column(String(64), index=True, nullable=True)
    # Retained pathology image (relative path under uploads/) for the doctor to inspect.
    image_path = Column(String(512), nullable=True)
    image_type = Column(String(64), nullable=True)      # BRAIN MRI / CHEST X-RAY / SKIN LESION
    ai_agent = Column(String(128), nullable=True)        # routed agent name
    ai_diagnosis = Column(Text, nullable=True)           # provisional AI diagnosis text
    result_image = Column(String(512), nullable=True)    # optional segmentation image URL
    status = Column(String(16), default="pending", index=True, nullable=False)  # pending|approved|rejected
    doctor_username = Column(String(64), nullable=True)
    doctor_comments = Column(Text, nullable=True)
    final_result = Column(Text, nullable=True)           # answer after the doctor's verdict
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)

class ElderlyAssessmentCase(Base):
    """Persistent older-adult screening, professional review and follow-up."""
    __tablename__ = "elderly_assessment_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_uid = Column(String(64), unique=True, index=True, nullable=False)
    patient_username = Column(String(64), index=True, nullable=True)
    subject_code = Column(String(64), index=True, nullable=True)
    assessment_type = Column(String(32), index=True, nullable=False)
    answers_json = Column(Text, nullable=False)
    rule_result_json = Column(Text, nullable=False)
    explanation = Column(Text, nullable=True)
    intervention_plan_json = Column(Text, nullable=True)
    attachments_json = Column(Text, nullable=True)
    status = Column(String(24), default="pending_review", index=True, nullable=False)
    reviewer_username = Column(String(64), nullable=True)
    reviewer_comments = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    parent_case_uid = Column(String(64), index=True, nullable=True)
    comparison_json = Column(Text, nullable=True)
    follow_up_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    updated_at = Column(DateTime, nullable=True)


# --------------------------------------------------------------------------- #
# Family-caregiver companionship module (proactive check-ins + alerts)
# --------------------------------------------------------------------------- #

class FamilyLink(Base):
    """Many-to-many relationship: a caregiver follows an elder (or vice versa).

    Any ``patient``-role account can act as either an elder or a caregiver —
    the role is contextual, not a separate account type. Established via a
    time-limited ``FollowCode`` so the caregiver never needs the elder's
    credentials.
    """
    __tablename__ = "family_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    elder_username = Column(String(64), index=True, nullable=False)
    caregiver_username = Column(String(64), index=True, nullable=False)
    relation_label = Column(String(64), nullable=True)  # e.g. "儿子", "护理员"
    status = Column(String(16), default="active", index=True, nullable=False)  # active|revoked
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class FollowCode(Base):
    """Short-lived numeric code an elder generates so a caregiver can follow them."""
    __tablename__ = "follow_codes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    elder_username = Column(String(64), index=True, nullable=False)
    code = Column(String(12), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    used_by = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CareReminderTask(Base):
    """A recurring/one-off proactive check-in configured by a caregiver."""
    __tablename__ = "care_reminder_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    elder_username = Column(String(64), index=True, nullable=False)
    created_by = Column(String(64), nullable=False)  # caregiver username
    task_type = Column(String(32), nullable=False)  # medication|mood|meal|safety_checkin|follow_up|custom
    custom_prompt = Column(Text, nullable=True)  # used only when task_type == custom
    schedule_type = Column(String(16), nullable=False)  # daily|weekly|once
    schedule_time = Column(String(8), nullable=False)  # "HH:MM" for daily/weekly
    schedule_weekday = Column(Integer, nullable=True)  # 0=Mon..6=Sun, weekly only
    schedule_datetime = Column(DateTime, nullable=True)  # full timestamp, once only
    status = Column(String(16), default="active", index=True, nullable=False)  # active|paused
    last_triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CareConversationLog(Base):
    """One proactive AI check-in turn: the prompt sent + the elder's reply."""
    __tablename__ = "care_conversation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    elder_username = Column(String(64), index=True, nullable=False)
    task_id = Column(Integer, nullable=True)  # nullable: manual/ad-hoc check-ins
    task_type = Column(String(32), nullable=True)
    ai_message = Column(Text, nullable=False)
    elder_reply = Column(Text, nullable=True)
    replied_at = Column(DateTime, nullable=True)
    risk_flag = Column(String(16), default="none", nullable=False)  # none|concern
    alert_sent = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)


class FamilyAlert(Base):
    """A caregiver-facing notification generated from a flagged check-in reply."""
    __tablename__ = "family_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    elder_username = Column(String(64), index=True, nullable=False)
    caregiver_username = Column(String(64), index=True, nullable=False)
    source_log_id = Column(Integer, nullable=True)
    summary = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)


class PushSubscription(Base):
    """Browser Web Push subscription for a user (optional enhancement)."""
    __tablename__ = "push_subscriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), index=True, nullable=False)
    endpoint = Column(Text, unique=True, nullable=False)
    p256dh = Column(String(255), nullable=False)
    auth = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
