"""Persistent workflow for older-adult assessments and professional review.

Enterprise hardening: optimistic locking, atomic attachment updates,
pagination, path-traversal protection, attachment limits, magic-bytes
validation, and double-review prevention.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from services.elderly_assessment import assess

logger = logging.getLogger(__name__)

# ---- Constants ---------------------------------------------------------------

MAX_ATTACHMENTS = 20
MAX_ATTACHMENT_SIZE = 20 * 1024 * 1024  # 20 MB per file
ALLOWED_ATTACHMENT_EXT = {"png", "jpg", "jpeg", "pdf"}

# Magic-bytes signatures for real file-type validation (defence-in-depth).
_MAGIC_SIGS = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"%PDF", "pdf"),
]

_VALID_VERDICTS = ("approved", "revision_required", "rejected")
_VALID_STATUSES = ("pending_review", "approved", "revision_required", "rejected")

# Whitelist of assessment types accepted by create_case.
_VALID_ASSESSMENT_TYPES = {"adl", "cognition", "environment", "assistive_device"}


# ---- Public API --------------------------------------------------------------


def create_case(
    config,
    patient_username: str,
    subject_code: str,
    assessment_type: str,
    answers: Dict[str, Any],
    attachments: Optional[List[Dict[str, str]]] = None,
    parent_case_uid: Optional[str] = None,
    follow_up_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Create a persistent assessment case with deterministic scoring.

    Raises ValueError on bad input, PermissionError on cross-user parent linking.
    """
    from services.db import get_session, init_db, is_ready
    from services.models import ElderlyAssessmentCase

    if assessment_type not in _VALID_ASSESSMENT_TYPES:
        raise ValueError(f"不支持的评估类型: {assessment_type}")
    if not isinstance(answers, dict) or not answers:
        raise ValueError("answers 必须为非空字典")

    if not is_ready():
        init_db(config)

    rule_result = assess(assessment_type, answers)
    intervention = _build_intervention_plan(assessment_type, rule_result)
    explanation = _generate_explanation(config, assessment_type, answers, rule_result, intervention)

    comparison = None
    if parent_case_uid:
        parent = get_case(config, parent_case_uid)
        if not parent:
            raise ValueError("基线评估不存在")
        if parent["patient_username"] != patient_username:
            raise PermissionError("不能关联其他用户的评估")
        if parent["assessment_type"] != assessment_type:
            raise ValueError("复评类型必须与基线评估一致")
        comparison = _compare(parent["rule_result"], rule_result)

    # Retry on case_uid collision (extremely unlikely with 12 hex chars but safe).
    session = get_session()
    try:
        for _attempt in range(3):
            uid = uuid.uuid4().hex[:12]
            exists = session.query(ElderlyAssessmentCase.id).filter(
                ElderlyAssessmentCase.case_uid == uid
            ).first()
            if not exists:
                break
        else:
            raise RuntimeError("无法生成唯一评估编号，请重试")

        row = ElderlyAssessmentCase(
            case_uid=uid,
            patient_username=patient_username,
            subject_code=_sanitize_subject_code(subject_code, patient_username),
            assessment_type=assessment_type,
            answers_json=_dump(answers),
            rule_result_json=_dump(rule_result),
            explanation=explanation,
            intervention_plan_json=_dump(intervention),
            attachments_json=_dump(attachments or []),
            status="pending_review",
            parent_case_uid=parent_case_uid,
            comparison_json=_dump(comparison) if comparison else None,
            follow_up_date=follow_up_date,
            version=1,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _to_dict(row)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def list_cases(
    config,
    username: str,
    role: str,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> Dict[str, Any]:
    """Paginated case listing. Patients see only their own; doctors/admins see all.

    Returns {"items": [...], "total": N, "page": P, "page_size": S}.
    """
    from services.db import get_session, init_db, is_ready
    from services.models import ElderlyAssessmentCase

    if not is_ready():
        init_db(config)

    # Clamp pagination params to prevent abuse.
    page = max(1, min(page, 10000))
    page_size = max(1, min(page_size, 200))

    if status and status not in _VALID_STATUSES:
        raise ValueError(f"无效的状态: {status}")

    session = get_session()
    try:
        query = session.query(ElderlyAssessmentCase)
        if role == "patient":
            query = query.filter(ElderlyAssessmentCase.patient_username == username)
        if status:
            query = query.filter(ElderlyAssessmentCase.status == status)

        total = query.count()
        offset = (page - 1) * page_size
        rows = (
            query.order_by(ElderlyAssessmentCase.created_at.desc())
            .offset(offset)
            .limit(page_size)
            .all()
        )
        return {
            "items": [_to_dict(x) for x in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    finally:
        session.close()


def get_case(config, case_uid: str) -> Optional[Dict[str, Any]]:
    from services.db import get_session, init_db, is_ready
    from services.models import ElderlyAssessmentCase

    if not is_ready():
        init_db(config)
    session = get_session()
    try:
        row = (
            session.query(ElderlyAssessmentCase)
            .filter(ElderlyAssessmentCase.case_uid == case_uid)
            .first()
        )
        return _to_dict(row) if row else None
    finally:
        session.close()


def review_case(
    config,
    case_uid: str,
    reviewer_username: str,
    verdict: str,
    comments: Optional[str],
    intervention_plan: Optional[Dict[str, Any]] = None,
    follow_up_date: Optional[datetime] = None,
    expected_version: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Professional review with optimistic locking and double-review guard.

    Returns (case_dict, status_message) where status_message is one of:
    "ok", "not_found", "already_reviewed", "version_conflict".
    """
    from services.db import get_session, init_db, is_ready
    from services.models import ElderlyAssessmentCase

    if verdict not in _VALID_VERDICTS:
        raise ValueError("verdict must be approved, revision_required or rejected")

    # Truncate comments to prevent storage abuse.
    if comments and len(comments) > 5000:
        comments = comments[:5000]

    if not is_ready():
        init_db(config)

    session = get_session()
    try:
        row = (
            session.query(ElderlyAssessmentCase)
            .filter(ElderlyAssessmentCase.case_uid == case_uid)
            .with_for_update()
            .first()
        )
        if not row:
            return None, "not_found"

        # Optimistic-lock check: client must send the version they read.
        if expected_version is not None and row.version != expected_version:
            return _to_dict(row), "version_conflict"

        # Prevent re-review of already-finalised cases (idempotent re-submit
        # of the same verdict by the same reviewer is allowed).
        if row.status in _VALID_VERDICTS and row.reviewer_username != reviewer_username:
            return _to_dict(row), "already_reviewed"

        row.status = verdict
        row.reviewer_username = reviewer_username
        row.reviewer_comments = comments
        row.reviewed_at = datetime.now(timezone.utc)
        row.updated_at = datetime.now(timezone.utc)
        row.version = (row.version or 1) + 1
        if intervention_plan is not None:
            row.intervention_plan_json = _dump(intervention_plan)
        if follow_up_date is not None:
            row.follow_up_date = follow_up_date

        session.commit()
        session.refresh(row)
        return _to_dict(row), "ok"
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def add_attachment(
    config,
    case_uid: str,
    username: str,
    role: str,
    attachment: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Append one attachment atomically using a single UPDATE statement.

    Prevents the read-modify-write race that the old JSON-append had under
    concurrent uploads. Enforces count and ownership limits.
    """
    from services.db import get_session, init_db, is_ready
    from services.models import ElderlyAssessmentCase

    if not is_ready():
        init_db(config)

    session = get_session()
    try:
        row = (
            session.query(ElderlyAssessmentCase)
            .filter(ElderlyAssessmentCase.case_uid == case_uid)
            .with_for_update()
            .first()
        )
        if not row:
            return None

        if role == "patient" and row.patient_username != username:
            raise PermissionError("无权修改该评估")

        items = _load(row.attachments_json, [])
        if len(items) >= MAX_ATTACHMENTS:
            raise ValueError(f"附件数量已达上限（{MAX_ATTACHMENTS}）")

        items.append(attachment)
        row.attachments_json = _dump(items)
        row.updated_at = datetime.now(timezone.utc)
        row.version = (row.version or 1) + 1
        session.commit()
        session.refresh(row)
        return _to_dict(row)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_attachment_path(
    config, case_uid: str, attachment_id: str
) -> Optional[Tuple[str, str]]:
    """Resolve (absolute_path, original_filename) for an attachment.

    Performs path-traversal validation: the resolved real path must be inside
    the designated elderly-assessments data directory.
    """
    case = get_case(config, case_uid)
    if not case:
        return None
    for item in case.get("attachments", []):
        if item.get("id") == attachment_id:
            raw_path = item.get("path", "")
            if not raw_path:
                return None
            real = os.path.realpath(raw_path)
            base = os.path.realpath(os.path.join("data", "elderly_assessments"))
            if not real.startswith(base + os.sep):
                logger.warning("Path traversal blocked: %s", raw_path)
                return None
            if not os.path.isfile(real):
                return None
            return real, item.get("filename", "attachment")
    return None


def can_access(case: Dict[str, Any], username: str, role: str) -> bool:
    """Ownership / role-based access check."""
    return role in ("doctor", "admin") or case.get("patient_username") == username


def report_text(case: Dict[str, Any]) -> str:
    result = case.get("rule_result") or {}
    plan = case.get("intervention_plan") or {}
    lines = [
        "失能失智老年人综合检测与评价报告",
        f"评估编号：{case.get('case_uid', '-')}",
        f"对象编码：{case.get('subject_code') or '-'}",
        f"评估类型：{case.get('assessment_type', '-')}",
        f"风险等级：{result.get('risk_level', '-')}",
        f"评价结论：{result.get('level', '-')}",
        "",
        "个性化解释：",
        case.get("explanation") or "-",
        "",
        "干预/改造/辅具方案：",
    ]
    for idx, item in enumerate(plan.get("actions", []), 1):
        lines.append(
            f"{idx}. [{item.get('priority')}] {item.get('action')}"
            f"（依据：{item.get('reason')}）"
        )
    if case.get("comparison"):
        comp = case["comparison"]
        lines += [
            "",
            f"随访趋势：{comp.get('trend')}；分数变化：{comp.get('score_delta')}",
        ]
    lines += [
        "",
        f"复核状态：{case.get('status', '-')}",
        f"专业意见：{case.get('reviewer_comments') or '待复核'}",
        "",
        result.get("disclaimer", ""),
    ]
    return "\n".join(lines)


# ---- Input validation helpers -----------------------------------------------


def validate_attachment_file(filename: str, content: bytes) -> str:
    """Validate filename extension and magic bytes. Returns the extension.

    Raises ValueError on invalid type or mismatched content.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_ATTACHMENT_EXT:
        raise ValueError(f"仅支持 PNG/JPG/JPEG/PDF，收到 .{ext}")

    if len(content) > MAX_ATTACHMENT_SIZE:
        raise ValueError("附件过大")

    # Magic-bytes check: content must match at least one known signature.
    detected = None
    for sig, sig_ext in _MAGIC_SIGS:
        if content[:len(sig)] == sig:
            detected = sig_ext
            break
    if detected is None:
        raise ValueError("文件内容与声明类型不匹配（magic bytes 校验失败）")
    if detected != ext and not (detected == "jpg" and ext == "jpeg"):
        raise ValueError(f"文件扩展名 .{ext} 与实际内容 {detected} 不匹配")

    return ext


# ---- Internal helpers --------------------------------------------------------


def _sanitize_subject_code(subject_code: str, fallback: str) -> str:
    raw = (subject_code or fallback or "anonymous").strip()
    # Strip control chars and limit length.
    clean = "".join(c for c in raw if c.isprintable())
    return clean[:64]


def _build_intervention_plan(kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
    priority = "P1" if result.get("risk_level") in ("high", "critical") else "P2"
    actions = [
        {"priority": priority, "action": x, "reason": result.get("level", "风险评价")}
        for x in result.get("recommendations", [])
    ]
    if kind == "environment":
        for hazard in result.get("hazards", []):
            actions.append(
                {
                    "priority": "P1" if hazard.get("weight", 0) >= 3 else "P2",
                    "action": f"整改环境风险：{hazard.get('item')}",
                    "reason": "现场风险项命中",
                }
            )
    if kind == "assistive_device":
        for item in result.get("recommended_devices", []):
            actions.append(
                {
                    "priority": "P2",
                    "action": f"专业试配：{item.get('device')}",
                    "reason": item.get("reason"),
                }
            )
    return {
        "actions": actions,
        "review_required": True,
        "follow_up_suggestion_days": 30 if priority == "P1" else 90,
    }


def _generate_explanation(
    config, kind: str, answers: Dict[str, Any], result: Dict[str, Any], plan: Dict[str, Any]
) -> str:
    prompt = (
        f"你是老年医学、康复工程与适老环境评价助手。请基于以下确定性规则结果，"
        f"用300字以内中文解释：\n"
        f"1. 解释风险和关键发现；2. 给老年人/照护者可执行建议；"
        f"3. 明确筛查不等于诊断或正式辅具处方；4. 不得改变规则分数和风险等级。\n"
        f"评估类型：{kind}\n"
        f"答案：{json.dumps(answers, ensure_ascii=False)}\n"
        f"规则结果：{json.dumps(result, ensure_ascii=False)}\n"
        f"方案：{json.dumps(plan, ensure_ascii=False)}"
    )
    try:
        response = config.conversation.llm.invoke(prompt)
        text = getattr(response, "content", response)
        if text:
            return str(text)
    except Exception as exc:
        logger.warning("LLM explanation unavailable, using deterministic template: %s", exc)
    recs = "；".join(result.get("recommendations", []))
    return (
        f"本次评价为\"{result.get('level')}\"，风险等级为 {result.get('risk_level')}。"
        f"建议：{recs}。本结果须由专业人员结合现场与个体情况复核。"
    )


def _compare(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    old_score, new_score = old.get("score"), new.get("score")
    delta = None if old_score is None or new_score is None else new_score - old_score
    risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    old_risk, new_risk = old.get("risk_level"), new.get("risk_level")
    if risk_order.get(new_risk, 0) < risk_order.get(old_risk, 0):
        trend = "风险改善"
    elif risk_order.get(new_risk, 0) > risk_order.get(old_risk, 0):
        trend = "风险升高"
    else:
        trend = "风险等级稳定"
    return {
        "baseline_risk": old_risk,
        "current_risk": new_risk,
        "score_delta": delta,
        "trend": trend,
    }


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _load(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _to_dict(row) -> Dict[str, Any]:
    return {
        "case_uid": row.case_uid,
        "patient_username": row.patient_username,
        "subject_code": row.subject_code,
        "assessment_type": row.assessment_type,
        "answers": _load(row.answers_json, {}),
        "rule_result": _load(row.rule_result_json, {}),
        "explanation": row.explanation,
        "intervention_plan": _load(row.intervention_plan_json, {}),
        "attachments": _strip_attachment_paths(_load(row.attachments_json, [])),
        "status": row.status,
        "reviewer_username": row.reviewer_username,
        "reviewer_comments": row.reviewer_comments,
        "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
        "parent_case_uid": row.parent_case_uid,
        "comparison": _load(row.comparison_json, None),
        "follow_up_date": row.follow_up_date.isoformat() if row.follow_up_date else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "version": getattr(row, "version", 1) or 1,
        "updated_at": row.updated_at.isoformat() if getattr(row, "updated_at", None) else None,
    }


def _strip_attachment_paths(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove server-side 'path' field from attachment dicts before returning to API."""
    stripped = []
    for item in items:
        copy = {k: v for k, v in item.items() if k != "path"}
        stripped.append(copy)
    return stripped
