import os
import json
import uuid
import asyncio
import tempfile
from typing import Any, Dict, Union, Optional, List
import glob
import threading
from datetime import datetime, timezone
import time
import traceback
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie, Header, Body
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

from config import Config
from agents.agent_decision import process_query, resume_after_validation
from services.cache import CacheService
from services.rate_limiter import RateLimiter

# SSE (Server-Sent Events) is an optional dependency. When absent the streaming
# endpoint returns 501 and all classic endpoints keep working unchanged.
try:
    from sse_starlette.sse import EventSourceResponse
    _SSE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    EventSourceResponse = None
    _SSE_AVAILABLE = False

import logging

# Load configuration
config = Config()
logger = logging.getLogger(__name__)

# P2 infrastructure services (gracefully degrade without Redis).
cache = CacheService(config)
rate_limiter = RateLimiter(config)


async def rate_limit_dep(request: Request, session_id: Optional[str] = Cookie(None)):
    """FastAPI dependency enforcing distributed rate limiting.

    Keyed by session cookie when present, else client IP. No-op when disabled;
    fail-open on backend errors. Raises 429 when the limit is exceeded.
    """
    client_id = session_id or (request.client.host if request.client else "anonymous")
    allowed, info = rate_limiter.check(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({info.get('limit')}/{rate_limiter.window}s). Please slow down.",
        )


# ---- P3: authentication / authorization dependencies ----

def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """Resolve the current user from a Bearer JWT.

    When auth is disabled (default), returns an anonymous user so all endpoints
    behave exactly as before. When enabled, a valid token is required.
    """
    if not config.auth.enabled:
        return {"username": "anonymous", "role": "patient", "anonymous": True}

    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from services.auth import decode_access_token
    try:
        payload = decode_access_token(config, token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    # Revocation check: a token whose jti is on the denylist (logout / refresh
    # rotation) is rejected before natural expiry. Fail-open on store errors so
    # a transient Redis blip never locks everyone out.
    jti = payload.get("jti")
    if jti:
        try:
            from services.token_store import is_revoked
            if is_revoked(jti, config):
                raise HTTPException(status_code=401, detail="Token has been revoked")
        except HTTPException:
            raise
        except Exception:
            pass
    return {
        "username": username,
        "role": payload.get("role", "patient"),
        "jti": jti,
        "exp": payload.get("exp"),
    }


def require_roles(*roles):
    """Dependency factory enforcing that the user holds one of ``roles``.

    No enforcement when auth is disabled.
    """
    async def _dep(user: dict = Depends(get_current_user)) -> dict:
        if config.auth.enabled and user.get("role") not in roles:
            raise HTTPException(status_code=403, detail=f"Requires role in {roles}")
        return user
    return _dep


# P4: observability setup (structured logging + optional LangSmith tracing).
try:
    from services.observability import setup_logging, setup_langsmith
    setup_logging(config)
    setup_langsmith(config)
except Exception as e:
    logger.warning("Observability setup failed: %s", e)

# Agent tracing (opt-in; no-op unless ENABLE_AGENT_TRACE=true).
try:
    from services.agent_trace import configure as configure_trace
    configure_trace(config)
except Exception as e:
    logger.warning("Agent trace setup failed: %s", e)

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Medical Chatbot", version="3.0")

# ---- Security middleware -----

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# CORS: restrict to frontend origins in production
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if not _cors_origins:
    _cors_origins = ["*"]  # dev-friendly default; lock down in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if not config.api.debug:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Initialize the database when auth or audit is enabled (SQLite by default).
if config.auth.enabled or config.auth.enable_audit:
    try:
        from services.db import init_db
        init_db(config)
    except Exception as e:
        logger.warning("Database initialization at startup failed: %s", e)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record per-request Prometheus metrics (no-op if metrics unavailable)."""
    if not config.observability.enable_metrics:
        return await call_next(request)
    start = time.time()
    response = await call_next(request)
    try:
        from services.observability import observe_request
        # Use the matched route template to avoid high-cardinality path labels.
        route = request.scope.get("route")
        path = getattr(route, "path", None) or request.url.path
        observe_request(request.method, path, response.status_code, time.time() - start)
    except Exception:
        pass
    return response


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint (501 when prometheus-client is unavailable)."""
    from services.observability import metrics_available, render_metrics
    if not (config.observability.enable_metrics and metrics_available()):
        return JSONResponse(status_code=501, content={"error": "metrics not available"})
    data, content_type = render_metrics()
    return Response(content=data, media_type=content_type)

# Set up directories
UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
BRAIN_TUMOR_OUTPUT = "uploads/brain_tumor_output"
SPEECH_DIR = "uploads/speech"

# Create directories if they don't exist
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, BRAIN_TUMOR_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static files directory
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize ElevenLabs client
client = ElevenLabs(
    api_key=config.speech.eleven_labs_api_key,
)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_audio():
    """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
            print("Cleaned up old speech files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)  # Runs every 5 minutes

# Start background cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()


@app.on_event("startup")
async def _start_family_care_scheduler():
    """Start the proactive check-in scheduler loop (no-op unless enabled)."""
    if getattr(config.family_care, "enabled", False):
        try:
            from services.db import init_db, is_ready
            if not is_ready():
                init_db(config)
            from services.care_scheduler import start as start_care_scheduler
            start_care_scheduler(config)
            logger.info("Family-care scheduler started.")
        except Exception as e:
            logger.warning("Failed to start family-care scheduler: %s", e)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    conversation_history: List = []

class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "EXAMPLE_VOICE_ID"  # Default voice ID


class ElderlyAssessmentRequest(BaseModel):
    assessment_type: str
    answers: Dict[str, Any]


@app.get("/elderly/assessments/catalog")
async def elderly_assessment_catalog(user: dict = Depends(get_current_user)):
    """Return the available disability, cognition, environment and AT screens."""
    from services.elderly_assessment import assessment_catalog
    return {"status": "success", "assessments": assessment_catalog()}


@app.post("/elderly/assessments")
async def run_elderly_assessment(
    req: ElderlyAssessmentRequest,
    user: dict = Depends(require_roles("patient", "doctor", "admin")),
):
    """Run a structured older-adult screening/evaluation (not a diagnosis)."""
    from services.elderly_assessment import assess
    try:
        result = await run_in_threadpool(assess, req.assessment_type, req.answers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        from services.audit import write_audit
        await run_in_threadpool(
            write_audit, config, username=user.get("username"), role=user.get("role"),
            action="elderly_assessment", detail=f"type={req.assessment_type}; risk={result.get('risk_level')}",
        )
    except Exception:
        pass
    return {"status": "success", "result": result}


# ---- Doctor licence helpers (needed before elderly case routes) -------------
LICENSE_FOLDER = "uploads/licenses"
os.makedirs(LICENSE_FOLDER, exist_ok=True)
LICENSE_ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}


def _license_allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in LICENSE_ALLOWED_EXTENSIONS


def require_approved_doctor(*roles):
    """Dependency: allow admins plus doctors whose licence is APPROVED.

    Prevents a doctor who merely knows the invite code (but has not been
    verified by a reviewer) from touching the review queue or patient images.
    No enforcement when auth is disabled (dev / legacy behaviour).
    """
    async def _dep(user: dict = Depends(get_current_user)) -> dict:
        if not config.auth.enabled:
            return user
        role = user.get("role")
        if role == "admin":
            return user
        if role == "doctor":
            from services.doctor_verification import is_approved_doctor
            ok = await run_in_threadpool(is_approved_doctor, config, user.get("username"))
            if ok:
                return user
            raise HTTPException(status_code=403, detail="医生资质尚未通过审核，无法访问审核队列。")
        raise HTTPException(status_code=403, detail="Requires an approved doctor or admin.")
    return _dep


# ---- Elderly assessment case lifecycle endpoints ----------------------------
# Enterprise-grade workflow: submit -> score -> review queue -> professional
# review -> follow-up re-assessment -> text report download. All endpoints are
# auth-gated, rate-limited, audit-logged, and hardened against concurrency and
# path-traversal attacks.


class ElderlyCaseRequest(BaseModel):
    assessment_type: str
    subject_code: str = ""
    answers: Dict[str, Any]
    parent_case_uid: Optional[str] = None
    follow_up_date: Optional[str] = None


class ElderlyReviewRequest(BaseModel):
    verdict: str
    comments: Optional[str] = None
    intervention_plan: Optional[Dict[str, Any]] = None
    follow_up_date: Optional[str] = None
    expected_version: Optional[int] = None


def _parse_optional_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="ISO-8601")


def _elderly_case_or_404(case_uid: str):
    from services.elderly_case_service import get_case
    case = get_case(config, case_uid)
    if not case:
        raise HTTPException(status_code=404, detail="case not found")
    return case


async def _elderly_audit(user: dict, action: str, detail: str):
    """Best-effort audit log; never blocks the request."""
    try:
        from services.audit import write_audit
        await run_in_threadpool(
            write_audit, config,
            username=user.get("username"), role=user.get("role"),
            action=action, detail=detail,
        )
    except Exception:
        pass


@app.post("/elderly/cases")
async def create_elderly_case(
    req: ElderlyCaseRequest,
    user: dict = Depends(require_roles("patient", "admin")),
    _rl: None = Depends(rate_limit_dep),
):
    """Submit questionnaire -> deterministic score -> review queue."""
    from services.elderly_case_service import create_case
    try:
        case = await run_in_threadpool(
            create_case, config, user.get("username"), req.subject_code,
            req.assessment_type, req.answers, [], req.parent_case_uid,
            _parse_optional_datetime(req.follow_up_date),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    await _elderly_audit(
        user, "elderly_case_create",
        f"type={req.assessment_type}; risk={case.get('rule_result', {}).get('risk_level')}; "
        f"follow_up={'yes' if req.parent_case_uid else 'no'}",
    )
    return {"status": "success", "case": case}


@app.get("/elderly/cases")
async def list_elderly_cases(
    status: str = None,
    page: int = 1,
    page_size: int = 50,
    user: dict = Depends(get_current_user),
):
    """List assessment cases (paginated). Patients see only their own."""
    from services.elderly_case_service import list_cases
    if config.auth.enabled and user.get("role") == "doctor":
        from services.doctor_verification import is_approved_doctor
        if not await run_in_threadpool(is_approved_doctor, config, user.get("username")):
            raise HTTPException(status_code=403, detail="doctor not approved")
    try:
        result = await run_in_threadpool(
            list_cases, config, user.get("username"), user.get("role"), status, page, page_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success", **result}


@app.get("/elderly/cases/{case_uid}")
async def get_elderly_case(
    case_uid: str,
    user: dict = Depends(get_current_user),
):
    """Get full detail of a single assessment case."""
    from services.elderly_case_service import can_access
    case = await run_in_threadpool(_elderly_case_or_404, case_uid)
    if not can_access(case, user.get("username"), user.get("role")):
        raise HTTPException(status_code=403, detail="no access")
    if config.auth.enabled and user.get("role") == "doctor":
        from services.doctor_verification import is_approved_doctor
        if not await run_in_threadpool(is_approved_doctor, config, user.get("username")):
            raise HTTPException(status_code=403, detail="doctor not approved")
    return {"status": "success", "case": case}


@app.post("/elderly/cases/{case_uid}/attachments")
async def upload_elderly_attachment(
    case_uid: str,
    file: UploadFile = File(...),
    category: str = Form("environment"),
    user: dict = Depends(require_roles("patient", "admin")),
    _rl: None = Depends(rate_limit_dep),
):
    """Upload evidence with ownership, magic-bytes, size and count checks."""
    from services.elderly_case_service import (
        add_attachment, validate_attachment_file,
    )
    content = await file.read()
    try:
        ext = validate_attachment_file(file.filename or "", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    case = await run_in_threadpool(_elderly_case_or_404, case_uid)
    if user.get("role") == "patient" and case.get("patient_username") != user.get("username"):
        raise HTTPException(status_code=403, detail="no access")
    folder = os.path.join("data", "elderly_assessments", case_uid)
    os.makedirs(folder, exist_ok=True)
    safe_name = secure_filename(f"{uuid.uuid4().hex[:8]}_{file.filename}")
    path = os.path.join(folder, safe_name)
    with open(path, "wb") as fp:
        fp.write(content)
    attachment = {
        "id": uuid.uuid4().hex[:12],
        "filename": file.filename,
        "category": category if category in ("environment", "device", "clinical", "other") else "other",
        "ext": ext,
        "size": len(content),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "path": path,
    }
    try:
        updated = await run_in_threadpool(
            add_attachment, config, case_uid, user.get("username"), user.get("role"), attachment,
        )
    except ValueError as exc:
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=403, detail=str(exc))
    if not updated:
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=404, detail="case not found")
    await _elderly_audit(user, "elderly_attachment_upload", f"case={case_uid}; file={safe_name}")
    return {"status": "success", "case": updated}


@app.get("/elderly/cases/{case_uid}/attachments/{attachment_id}")
async def get_elderly_attachment(
    case_uid: str,
    attachment_id: str,
    user: dict = Depends(get_current_user),
):
    """Download attachment with auth + path-traversal protection."""
    from services.elderly_case_service import can_access, get_attachment_path
    case = await run_in_threadpool(_elderly_case_or_404, case_uid)
    if not can_access(case, user.get("username"), user.get("role")):
        raise HTTPException(status_code=403, detail="no access")
    if config.auth.enabled and user.get("role") == "doctor":
        from services.doctor_verification import is_approved_doctor
        if not await run_in_threadpool(is_approved_doctor, config, user.get("username")):
            raise HTTPException(status_code=403, detail="doctor not approved")
    result = await run_in_threadpool(get_attachment_path, config, case_uid, attachment_id)
    if not result:
        raise HTTPException(status_code=404, detail="attachment not found")
    real_path, filename = result
    await _elderly_audit(user, "elderly_attachment_download", f"case={case_uid}; att={attachment_id}")
    return FileResponse(real_path, filename=filename)


@app.post("/elderly/cases/{case_uid}/review")
async def review_elderly_case(
    case_uid: str,
    req: ElderlyReviewRequest,
    user: dict = Depends(require_approved_doctor()),
    _rl: None = Depends(rate_limit_dep),
):
    """Professional review with optimistic locking and double-review guard."""
    from services.elderly_case_service import review_case
    try:
        case, status_msg = await run_in_threadpool(
            review_case, config, case_uid, user.get("username"), req.verdict,
            req.comments, req.intervention_plan,
            _parse_optional_datetime(req.follow_up_date), req.expected_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if status_msg == "not_found":
        raise HTTPException(status_code=404, detail="case not found")
    if status_msg == "already_reviewed":
        raise HTTPException(status_code=409, detail="already reviewed by another doctor")
    if status_msg == "version_conflict":
        raise HTTPException(status_code=409, detail="version conflict, refresh and retry")
    await _elderly_audit(
        user, "elderly_case_review",
        f"case={case_uid}; verdict={req.verdict}; reviewer={user.get('username')}",
    )
    return {"status": "success", "case": case}


@app.get("/elderly/cases/{case_uid}/report")
async def download_elderly_report(
    case_uid: str,
    user: dict = Depends(get_current_user),
):
    """Download plain-text assessment report with ownership check."""
    from services.elderly_case_service import can_access, report_text
    case = await run_in_threadpool(_elderly_case_or_404, case_uid)
    if not can_access(case, user.get("username"), user.get("role")):
        raise HTTPException(status_code=403, detail="no access")
    if config.auth.enabled and user.get("role") == "doctor":
        from services.doctor_verification import is_approved_doctor
        if not await run_in_threadpool(is_approved_doctor, config, user.get("username")):
            raise HTTPException(status_code=403, detail="doctor not approved")
    report = await run_in_threadpool(report_text, case)
    await _elderly_audit(user, "elderly_report_download", f"case={case_uid}")
    return Response(
        content=report.encode("utf-8-sig"),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="elderly_assessment_{case_uid}.txt"',
        },
    )


# ---- Family-caregiver companionship module ----

def _family_care_dep(user: dict = Depends(require_roles("patient", "admin"))) -> dict:
    """Gate all family-care endpoints behind the feature flag (404 when off)."""
    if not getattr(config.family_care, "enabled", False):
        raise HTTPException(status_code=404, detail="Family-care module is disabled")
    return user


class FollowCodeRedeemRequest(BaseModel):
    code: str = Field(min_length=4, max_length=12)
    relation_label: Optional[str] = Field(default=None, max_length=64)


class ReminderTaskRequest(BaseModel):
    task_type: str
    schedule_type: str
    schedule_time: Optional[str] = None
    schedule_weekday: Optional[int] = None
    schedule_datetime: Optional[str] = None
    custom_prompt: Optional[str] = None


class ReminderTaskUpdateRequest(BaseModel):
    status: str


class CareReplyRequest(BaseModel):
    log_id: int
    text: str = Field(min_length=1, max_length=2000)


class PushSubscribeRequest(BaseModel):
    endpoint: str
    p256dh: str
    auth: str


@app.post("/family/follow-code")
async def create_follow_code(user: dict = Depends(_family_care_dep)):
    """An elder generates a short-lived numeric code for a caregiver to redeem."""
    from services.family_care_service import generate_follow_code
    result = await run_in_threadpool(generate_follow_code, config, user.get("username"))
    return {"status": "success", **result}


@app.post("/family/follow")
async def redeem_follow_code_endpoint(
    req: FollowCodeRedeemRequest, user: dict = Depends(_family_care_dep)
):
    """A caregiver redeems an elder's follow code to establish the relationship."""
    from services.family_care_service import redeem_follow_code
    try:
        link = await run_in_threadpool(
            redeem_follow_code, config, user.get("username"), req.code.strip(), req.relation_label
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success", "link": link}


@app.get("/family/elders")
async def list_my_elders(user: dict = Depends(_family_care_dep)):
    """List elders the current user (as caregiver) follows."""
    from services.family_care_service import list_elders_for_caregiver
    links = await run_in_threadpool(list_elders_for_caregiver, config, user.get("username"))
    return {"status": "success", "elders": links}


@app.get("/family/caregivers")
async def list_my_caregivers(user: dict = Depends(_family_care_dep)):
    """List caregivers following the current user (as elder)."""
    from services.family_care_service import list_caregivers_for_elder
    links = await run_in_threadpool(list_caregivers_for_elder, config, user.get("username"))
    return {"status": "success", "caregivers": links}


@app.delete("/family/link/{link_id}")
async def revoke_family_link(link_id: int, user: dict = Depends(_family_care_dep)):
    """Either party may revoke a family link."""
    from services.family_care_service import revoke_link
    try:
        ok = await run_in_threadpool(revoke_link, config, link_id, user.get("username"))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if not ok:
        raise HTTPException(status_code=404, detail="link not found")
    return {"status": "success"}


def _parse_reminder_datetime(value: Optional[str]):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=400, detail="schedule_datetime 必须为 ISO8601 格式")


@app.post("/family/elders/{elder_username}/reminders")
async def create_reminder(
    elder_username: str, req: ReminderTaskRequest, user: dict = Depends(_family_care_dep)
):
    """A caregiver configures a proactive check-in task for a followed elder."""
    from services.family_care_service import create_reminder_task
    schedule_dt = _parse_reminder_datetime(req.schedule_datetime)
    try:
        task = await run_in_threadpool(
            create_reminder_task, config, user.get("username"), elder_username,
            req.task_type, req.schedule_type, req.schedule_time, req.schedule_weekday,
            schedule_dt, req.custom_prompt,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success", "task": task}


@app.get("/family/elders/{elder_username}/reminders")
async def list_reminders(elder_username: str, user: dict = Depends(_family_care_dep)):
    """List reminder tasks for a given elder (caller must be linked or be the elder)."""
    from services.family_care_service import list_reminder_tasks, _is_linked
    is_self = user.get("username") == elder_username
    if not is_self and not await run_in_threadpool(_is_linked, config, elder_username, user.get("username")):
        raise HTTPException(status_code=403, detail="not linked to this elder")
    tasks = await run_in_threadpool(list_reminder_tasks, config, elder_username)
    return {"status": "success", "tasks": tasks}


@app.patch("/family/reminders/{task_id}")
async def update_reminder(
    task_id: int, req: ReminderTaskUpdateRequest, user: dict = Depends(_family_care_dep)
):
    """Pause/resume a reminder task (creator only)."""
    from services.family_care_service import update_reminder_task
    try:
        task = await run_in_threadpool(
            update_reminder_task, config, task_id, user.get("username"), req.status
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return {"status": "success", "task": task}


@app.delete("/family/reminders/{task_id}")
async def delete_reminder(task_id: int, user: dict = Depends(_family_care_dep)):
    """Delete a reminder task (creator only)."""
    from services.family_care_service import delete_reminder_task
    try:
        ok = await run_in_threadpool(delete_reminder_task, config, task_id, user.get("username"))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if not ok:
        raise HTTPException(status_code=404, detail="task not found")
    return {"status": "success"}


@app.get("/care/messages")
async def get_care_channel_messages(user: dict = Depends(_family_care_dep)):
    """An elder fetches their AI-companion channel history."""
    from services.family_care_service import get_care_messages
    messages = await run_in_threadpool(get_care_messages, config, user.get("username"))
    return {"status": "success", "messages": messages}


@app.post("/care/reply")
async def reply_to_care_message(req: CareReplyRequest, user: dict = Depends(_family_care_dep)):
    """An elder replies to a proactive check-in; triggers AI response + risk escalation."""
    from services.family_care_service import (
        record_elder_reply, create_alerts_for_elder, mark_alert_generated, get_care_messages,
    )
    from services.care_message_generator import generate_reply_response
    from services.care_sse import publish_alert_event

    try:
        updated_log, concern = await run_in_threadpool(
            record_elder_reply, config, req.log_id, user.get("username"), req.text
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if not updated_log:
        raise HTTPException(status_code=404, detail="check-in message not found")

    ai_reply = await run_in_threadpool(
        generate_reply_response, config, user.get("username"), updated_log.get("task_type"),
        updated_log.get("ai_message", ""), req.text,
    )
    from services.family_care_service import log_ai_checkin
    reply_log = await run_in_threadpool(
        log_ai_checkin, config, user.get("username"), ai_reply,
        updated_log.get("task_id"), updated_log.get("task_type"),
    )

    if concern:
        summary = f"{user.get('username')} {concern['summary']}：“{req.text[:100]}”"
        alerts = await run_in_threadpool(
            create_alerts_for_elder, config, user.get("username"), summary, req.log_id
        )
        await run_in_threadpool(mark_alert_generated, config, req.log_id)
        for alert in alerts:
            publish_alert_event(alert["caregiver_username"], {"type": "alert", "alert": alert})
            try:
                from services.web_push import send_push_to_user
                await run_in_threadpool(
                    send_push_to_user, config, alert["caregiver_username"],
                    "家人需要关注", summary,
                )
            except Exception:
                pass

    return {"status": "success", "reply_log": reply_log, "concern_detected": bool(concern)}


@app.get("/care/stream")
async def care_channel_stream(user: dict = Depends(_family_care_dep)):
    """SSE: push new proactive check-in messages to the elder's open tab."""
    if not _SSE_AVAILABLE:
        return JSONResponse(status_code=501, content={"error": "SSE not available"})
    from services.care_sse import subscribe, unsubscribe

    username = user.get("username")
    queue = subscribe(username)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25)
                    yield {"event": event.get("type", "message"), "data": json.dumps(event, ensure_ascii=False)}
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": json.dumps({"status": "ok"})}
        finally:
            unsubscribe(username, queue)

    return EventSourceResponse(event_generator())


@app.get("/family/alerts")
async def list_family_alerts(unread_only: bool = False, user: dict = Depends(_family_care_dep)):
    """A caregiver lists risk alerts raised for elders they follow."""
    from services.family_care_service import list_alerts
    alerts = await run_in_threadpool(list_alerts, config, user.get("username"), unread_only)
    return {"status": "success", "alerts": alerts}


@app.post("/family/alerts/{alert_id}/read")
async def mark_family_alert_read(alert_id: int, user: dict = Depends(_family_care_dep)):
    """A caregiver marks an alert as read."""
    from services.family_care_service import mark_alert_read
    try:
        ok = await run_in_threadpool(mark_alert_read, config, alert_id, user.get("username"))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if not ok:
        raise HTTPException(status_code=404, detail="alert not found")
    return {"status": "success"}


@app.get("/family/alerts/stream")
async def family_alerts_stream(user: dict = Depends(_family_care_dep)):
    """SSE: push new alerts to the caregiver's open tab in real time."""
    if not _SSE_AVAILABLE:
        return JSONResponse(status_code=501, content={"error": "SSE not available"})
    from services.care_sse import subscribe, unsubscribe

    username = user.get("username")
    queue = subscribe(username)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25)
                    yield {"event": event.get("type", "message"), "data": json.dumps(event, ensure_ascii=False)}
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": json.dumps({"status": "ok"})}
        finally:
            unsubscribe(username, queue)

    return EventSourceResponse(event_generator())


@app.post("/push/subscribe")
async def push_subscribe(req: PushSubscribeRequest, user: dict = Depends(_family_care_dep)):
    """Register a browser Web Push subscription for the current user."""
    from services.web_push import save_subscription
    await run_in_threadpool(
        save_subscription, config, user.get("username"), req.endpoint, req.p256dh, req.auth
    )
    return {"status": "success"}


@app.post("/push/unsubscribe")
async def push_unsubscribe(req: PushSubscribeRequest, user: dict = Depends(_family_care_dep)):
    """Remove a browser Web Push subscription."""
    from services.web_push import remove_subscription
    await run_in_threadpool(remove_subscription, config, req.endpoint)
    return {"status": "success"}


@app.get("/push/vapid-public-key")
async def push_vapid_public_key():
    """Expose the VAPID public key so the frontend can subscribe to Web Push."""
    return {"status": "success", "public_key": getattr(config.family_care, "vapid_public_key", "")}


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    """Serve the public marketing/registration landing page."""
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/app", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the authenticated patient chat app."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/doctor", response_class=HTMLResponse)
async def doctor_console(request: Request):
    """Serve the doctor review console page."""
    return templates.TemplateResponse("doctor.html", {"request": request})


@app.get("/reviewer", response_class=HTMLResponse)
async def reviewer_console(request: Request):
    """Serve the admin/reviewer console for verifying doctor licences."""
    return templates.TemplateResponse("reviewer.html", {"request": request})


@app.get("/health")
def health_check():
    """Health check endpoint for Docker health checks"""
    return {"status": "healthy"}


# ---- P3: authentication endpoints ----

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_\u4e00-\u9fff]+$")
    password: str
    email: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/register")
async def register(req: RegisterRequest):
    """Register a user. The first registered user becomes ``admin``; others are
    ``patient`` (roles are changed by an admin out-of-band). Requires auth enabled.
    """
    if not config.auth.enabled:
        return JSONResponse(status_code=400, content={"status": "error", "response": "Authentication is disabled on this server."})

    # Password strength check
    from services.auth import validate_password_strength
    ok, msg = validate_password_strength(req.password)
    if not ok:
        return JSONResponse(status_code=422, content={"status": "error", "response": msg})

    def _register():
        from services.db import init_db, is_ready, get_session
        from services.models import User
        from services.auth import hash_password
        if not is_ready():
            init_db(config)
        session = get_session()
        try:
            if session.query(User).filter(User.username == req.username).first():
                return {"_status": 409, "response": "Username already exists."}
            role = "admin" if session.query(User).count() == 0 else "patient"
            session.add(User(
                username=req.username,
                email=req.email,
                hashed_password=hash_password(req.password),
                role=role,
            ))
            session.commit()
            return {"_status": 200, "username": req.username, "role": role}
        finally:
            session.close()

    result = await run_in_threadpool(_register)
    status_code = result.pop("_status", 200)
    if status_code != 200:
        return JSONResponse(status_code=status_code, content={"status": "error", **result})
    return {"status": "success", **result}


class DoctorRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_\u4e00-\u9fff]+$")
    password: str
    email: Optional[str] = None
    invite_code: str


@app.post("/auth/register/doctor")
async def register_doctor(req: DoctorRegisterRequest):
    """Register as a doctor (requires a valid DOCTOR_INVITE_CODE)."""
    if not config.auth.enabled:
        return JSONResponse(status_code=400, content={"status": "error", "response": "Authentication is disabled on this server."})

    # Password strength check
    from services.auth import validate_password_strength
    ok, msg = validate_password_strength(req.password)
    if not ok:
        return JSONResponse(status_code=422, content={"status": "error", "response": msg})

    # Backward-compatible fallback: an already-running process may have loaded
    # an older Config class without `.review`; read env directly in that case.
    expected = getattr(getattr(config, "review", None), "doctor_invite_code", None) or os.getenv("DOCTOR_INVITE_CODE", "")
    if not expected:
        return JSONResponse(status_code=400, content={"status": "error", "response": "Doctor registration is not enabled (DOCTOR_INVITE_CODE not set)."})
    if req.invite_code != expected:
        return JSONResponse(status_code=403, content={"status": "error", "response": "邀请码不正确，无法注册为医生。"})

    def _register_doctor():
        from services.db import init_db, is_ready, get_session
        from services.models import User
        from services.auth import hash_password
        if not is_ready():
            init_db(config)
        session = get_session()
        try:
            if session.query(User).filter(User.username == req.username).first():
                return {"_status": 409, "response": "用户名已存在。"}
            session.add(User(
                username=req.username,
                email=req.email,
                hashed_password=hash_password(req.password),
                role="doctor",
                doctor_status="unsubmitted",
            ))
            session.commit()
            return {"_status": 200, "username": req.username, "role": "doctor", "doctor_status": "unsubmitted"}
        finally:
            session.close()

    result = await run_in_threadpool(_register_doctor)
    status_code = result.pop("_status", 200)
    if status_code != 200:
        return JSONResponse(status_code=status_code, content={"status": "error", **result})
    return {"status": "success", **result}


# ---- Doctor licence verification (anti-abuse for the invite code) ----



@app.post("/auth/doctor/license")
async def doctor_upload_license(
    license: UploadFile = File(...),
    user: dict = Depends(require_roles("doctor", "admin")),
):
    """A doctor uploads their practising-licence certificate for review."""
    if not config.auth.enabled:
        return JSONResponse(status_code=400, content={"status": "error", "response": "Authentication is disabled on this server."})
    if user.get("role") != "doctor":
        return JSONResponse(status_code=403, content={"status": "error", "response": "仅医生账号可上传执业资格证。"})
    if not _license_allowed(license.filename):
        return JSONResponse(status_code=400, content={"status": "error", "response": "不支持的文件类型。允许: PNG, JPG, JPEG, PDF"})

    content = await license.read()
    if len(content) > config.api.max_image_upload_size * 1024 * 1024:
        return JSONResponse(status_code=413, content={"status": "error", "response": f"文件过大，最大 {config.api.max_image_upload_size}MB"})

    filename = secure_filename(f"{user.get('username')}_{uuid.uuid4().hex[:8]}_{license.filename}")
    file_path = os.path.join(LICENSE_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(content)

    from services.doctor_verification import set_license
    info = await run_in_threadpool(set_license, config, user.get("username"), file_path)
    if info is None:
        return JSONResponse(status_code=404, content={"status": "error", "response": "医生账号不存在。"})

    from services.audit import write_audit
    await run_in_threadpool(
        write_audit, config,
        username=user.get("username"), role=user.get("role"),
        action="license_upload", detail=f"license={license.filename}",
    )
    return {"status": "success", "doctor_status": info.get("doctor_status")}


@app.get("/auth/doctor/status")
async def doctor_license_status(user: dict = Depends(require_roles("doctor", "admin"))):
    """A doctor checks their own licence-verification status."""
    if not config.auth.enabled:
        return {"status": "success", "doctor_status": "approved"}
    from services.doctor_verification import get_status
    info = await run_in_threadpool(get_status, config, user.get("username"))
    if not info:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "status": "success",
        "doctor_status": info.get("doctor_status"),
        "license_comments": info.get("license_comments"),
        "has_license": bool(info.get("license_path")),
    }


@app.get("/admin/doctors/pending")
async def admin_pending_doctors(user: dict = Depends(require_roles("admin"))):
    """Reviewer console: list doctors awaiting licence verification."""
    from services.doctor_verification import list_pending
    rows = await run_in_threadpool(list_pending, config)
    # Never leak the raw filesystem path to the client.
    for r in rows:
        r.pop("license_path", None)
    return {"status": "success", "doctors": rows}


@app.get("/admin/doctors/license/{username}")
async def admin_get_license(username: str, user: dict = Depends(require_roles("admin"))):
    """Reviewer streams a doctor's uploaded licence file to inspect it."""
    from services.doctor_verification import get_status
    info = await run_in_threadpool(get_status, config, username)
    if not info or not info.get("license_path"):
        raise HTTPException(status_code=404, detail="License not found")
    path = info["license_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="License file missing")
    return FileResponse(path)


class DoctorLicenseDecision(BaseModel):
    verdict: str  # "approved" or "rejected"
    comments: Optional[str] = None


@app.post("/admin/doctors/{username}/review")
async def admin_review_doctor(
    username: str,
    decision: DoctorLicenseDecision,
    user: dict = Depends(require_roles("admin")),
):
    """Reviewer approves or rejects a doctor's licence."""
    if decision.verdict not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="verdict must be approved or rejected")
    from services.doctor_verification import review_license
    info = await run_in_threadpool(
        review_license, config, username, user.get("username"),
        decision.verdict, decision.comments,
    )
    if info is None:
        raise HTTPException(status_code=404, detail="Doctor not found")

    from services.audit import write_audit
    await run_in_threadpool(
        write_audit, config,
        username=user.get("username"), role=user.get("role"),
        action="doctor_license_review", validation_result=decision.verdict,
        detail=f"doctor={username}; {decision.comments or ''}",
    )
    info.pop("license_path", None)
    return {"status": "success", "doctor": info}


# ---- Admin: user & role management ----
#
# Safety model:
#   * Gated behind require_roles("admin") — only an existing admin may act.
#   * Cannot demote the last remaining admin (lockout protection).
#   * Promoting to "doctor" resets doctor_status so the licence-verification
#     workflow still applies; it is never bypassed by a role change alone.
#   * Every change is written to the append-only audit log.

@app.get("/admin/users")
async def admin_list_users(
    search: Optional[str] = None, role: Optional[str] = None,
    user: dict = Depends(require_roles("admin")),
):
    """List all user accounts (admin console). No password hashes returned."""
    from services.user_admin import list_users
    users = await run_in_threadpool(list_users, config, search, role)
    return {"status": "success", "users": users}


class UserRoleUpdateRequest(BaseModel):
    role: str


@app.post("/admin/users/{username}/role")
async def admin_update_user_role(
    username: str, req: UserRoleUpdateRequest,
    user: dict = Depends(require_roles("admin")),
):
    """Change a user's role (patient/doctor/admin). Refuses to remove the last admin."""
    from services.user_admin import update_user_role
    try:
        updated = await run_in_threadpool(
            update_user_role, config, username, req.role, user.get("username")
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found")

    from services.audit import write_audit
    await run_in_threadpool(
        write_audit, config,
        username=user.get("username"), role=user.get("role"),
        action="user_role_change", validation_result=req.role,
        detail=f"target={username}; new_role={req.role}",
    )
    return {"status": "success", "user": updated}


@app.post("/auth/login")
async def login(req: LoginRequest):
    """Authenticate and return a JWT access token."""
    if not config.auth.enabled:
        return JSONResponse(status_code=400, content={"status": "error", "response": "Authentication is disabled on this server."})

    def _login():
        from services.db import init_db, is_ready, get_session
        from services.models import User
        from services.auth import verify_password, create_access_token, create_refresh_token
        if not is_ready():
            init_db(config)
        session = get_session()
        try:
            user = session.query(User).filter(User.username == req.username).first()
            if not user or not verify_password(req.password, user.hashed_password):
                return None
            token = create_access_token(config, subject=user.username, role=user.role)
            resp = {"access_token": token, "token_type": "bearer", "role": user.role}
            # Only issue a refresh token when the feature is enabled; otherwise
            # the response is byte-for-byte the legacy shape.
            if config.auth.refresh_token_enabled:
                resp["refresh_token"] = create_refresh_token(
                    config, subject=user.username, role=user.role
                )
            return resp
        finally:
            session.close()

    result = await run_in_threadpool(_login)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return result


@app.get("/auth/me")
async def auth_me(user: dict = Depends(get_current_user)):
    """Return the current authenticated user (anonymous when auth disabled)."""
    # Don't leak internal token metadata to the client.
    return {k: v for k, v in user.items() if k not in ("jti", "exp")}


class MemoryItemRequest(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    kind: str = Field(default="manual", max_length=50)


@app.get("/memory")
async def list_memory(user: dict = Depends(get_current_user)):
    """List the caller's opt-in long-term memories/health profile facts."""
    if not getattr(config.memory, "enabled", False):
        return {"status": "disabled", "memories": []}
    from services.long_term_memory import get_memories
    memories = await run_in_threadpool(get_memories, config, user.get("username"))
    return {"status": "success", "memories": memories}


@app.post("/memory")
async def add_memory_item(req: MemoryItemRequest, user: dict = Depends(get_current_user)):
    """Manually add a durable health-profile memory for the current user."""
    if not getattr(config.memory, "enabled", False):
        return JSONResponse(status_code=400, content={"status": "disabled", "response": "Long-term memory is disabled."})
    from services.long_term_memory import add_memory
    added = await run_in_threadpool(add_memory, config, user.get("username"), req.text, req.kind)
    return {"status": "success", "added": added}


@app.delete("/memory")
async def clear_memory(user: dict = Depends(get_current_user)):
    """Delete all long-term memories for the current user."""
    from services.long_term_memory import clear_memories
    await run_in_threadpool(clear_memories, config, user.get("username"))
    return {"status": "success"}


class RefreshRequest(BaseModel):
    refresh_token: str


@app.post("/auth/refresh")
async def refresh_token(req: RefreshRequest):
    """Exchange a valid refresh token for a new access + refresh token pair.

    Implements refresh-token *rotation*: the presented refresh token is revoked
    and a brand-new one is issued, so a leaked/replayed refresh token is
    single-use. Requires both auth and the refresh-token feature to be enabled.
    """
    if not config.auth.enabled:
        return JSONResponse(status_code=400, content={"status": "error", "response": "Authentication is disabled on this server."})
    if not config.auth.refresh_token_enabled:
        return JSONResponse(status_code=400, content={"status": "error", "response": "Refresh tokens are not enabled on this server."})

    from services.auth import (
        decode_token, create_access_token, create_refresh_token,
        REFRESH, remaining_ttl_seconds,
    )
    from services.token_store import is_revoked, revoke

    try:
        payload = decode_token(config, req.refresh_token, expected_type=REFRESH)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    jti = payload.get("jti")
    if jti and is_revoked(jti, config):
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")
    username = payload.get("sub")
    role = payload.get("role", "patient")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    # Rotate: revoke the presented refresh token, then mint a fresh pair.
    if jti:
        revoke(jti, remaining_ttl_seconds(payload), config)
    access = create_access_token(config, subject=username, role=role)
    new_refresh = create_refresh_token(config, subject=username, role=role)
    return {
        "access_token": access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
        "role": role,
    }


@app.post("/auth/logout")
async def logout(
    refresh_token_body: Optional[str] = Body(None, embed=True, alias="refresh_token"),
    user: dict = Depends(get_current_user),
):
    """Revoke the caller's current access token (and refresh token if supplied).

    After logout the tokens are added to the jti denylist and rejected on
    subsequent requests, even though the raw JWTs have not yet expired.
    """
    if not config.auth.enabled:
        return {"status": "success", "response": "Auth disabled; nothing to revoke."}

    from services.token_store import revoke
    from services.auth import decode_token, remaining_ttl_seconds

    # Revoke the current access token. We don't have its decoded exp on the
    # dependency result beyond 'exp', so compute the remaining TTL from it.
    jti = user.get("jti")
    if jti:
        ttl = remaining_ttl_seconds({"exp": user.get("exp")})
        if ttl <= 0:
            # Fallback upper bound so the entry still covers the token lifetime.
            ttl = int(config.auth.access_token_expire_minutes) * 60
        revoke(jti, ttl, config)

    # Optionally revoke the refresh token too (recommended on explicit logout).
    if refresh_token_body:
        try:
            payload = decode_token(config, refresh_token_body)
            rjti = payload.get("jti")
            if rjti:
                revoke(rjti, remaining_ttl_seconds(payload), config)
        except Exception:
            pass

    return {"status": "success", "response": "Logged out"}


class IngestRequest(BaseModel):
    directory: Optional[str] = None
    file: Optional[str] = None


@app.post("/ingest")
async def ingest(req: IngestRequest, _rl: None = Depends(rate_limit_dep)):
    """Ingest documents into the RAG store.

    Offloads to a Celery worker when the task queue is enabled (returns a
    task_id to poll via /tasks/{id}); otherwise runs synchronously off the event
    loop. Heavy, non-interactive job -> ideal for async decoupling.
    """
    from services.task_queue import submit_ingest_directory, submit_ingest_file

    if not req.directory and not req.file:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "response": "Provide either 'directory' or 'file'."},
        )
    try:
        if req.directory:
            info = await run_in_threadpool(submit_ingest_directory, config, req.directory)
        else:
            info = await run_in_threadpool(submit_ingest_file, config, req.file)
        return {"status": "submitted", **info}
    except Exception as e:
        logger.error("Error in /ingest: %s", e)
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"status": "error", "response": str(e)})


@app.get("/tasks/{task_id}")
async def task_status(task_id: str):
    """Poll the status/result of an async ingestion task."""
    from services.task_queue import get_task_status
    return await run_in_threadpool(get_task_status, task_id)

def _build_chat_result(response_data: dict) -> dict:
    """Assemble the JSON payload returned by /chat and /upload from graph state.

    Uses the per-request unique ``result_image`` URL produced by the agents
    (avoids the old fixed ``segmentation_plot.png`` shared file) and surfaces the
    native-interrupt ``awaiting_validation`` flag for HITL-aware clients.
    """
    result = {
        "status": "success",
        "response": response_data["messages"][-1].content,
        "agent": response_data["agent_name"],
    }
    if response_data.get("awaiting_validation"):
        result["awaiting_validation"] = True
    result_image = response_data.get("result_image")
    if result_image:
        result["result_image"] = result_image
    safety_verdict = response_data.get("safety_verdict")
    if safety_verdict:
        result["safety_verdict"] = {
            "verdict": safety_verdict.get("verdict"),
            "reason": safety_verdict.get("reason"),
            "needs_human_review": safety_verdict.get("needs_human_review", False),
        }
    return result


@app.post("/chat")
async def chat(
    request: QueryRequest, 
    response: Response, 
    session_id: Optional[str] = Cookie(None),
    x_session_id: Optional[str] = Header(None),
    _rl: None = Depends(rate_limit_dep),
    user: dict = Depends(require_roles("patient", "admin"))
):
    """Process user text query through the multi-agent system."""
    # An explicit X-Session-Id header (sent by the multi-conversation UI) takes
    # precedence over the cookie, so the client can isolate several chats.
    session_id = x_session_id or session_id
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Token-budget governance (opt-in): reject when the user's daily budget
        # is exhausted. No-op / fail-open when disabled or no budget configured.
        try:
            from services.cost_tracker import check_budget
            allowed, budget_info = check_budget(config, user_id=user.get("username"), session_id=session_id)
            if not allowed:
                return JSONResponse(status_code=429, content={
                    "status": "error", "agent": "System",
                    "response": "You've reached today's usage budget. Please try again tomorrow.",
                    "budget": budget_info,
                })
        except Exception:
            pass

        # Optional semantic cache (opt-in): return a prior answer for a
        # semantically-similar query. Falls back to the exact-match cache below.
        try:
            from services.semantic_cache import semantic_get
            sem = semantic_get(config, request.query)
            if sem is not None:
                response.set_cookie(key="session_id", value=session_id)
                sem_result = dict(sem)
                sem_result["cached"] = "semantic"
                return sem_result
        except Exception:
            pass

        # Optional response cache (opt-in, keyed by raw query text). Returns a
        # previously computed answer for identical queries within the TTL. No-ops
        # when disabled or Redis is unavailable.
        cached = cache.get("chat", request.query)
        if cached is not None:
            response.set_cookie(key="session_id", value=session_id)
            cached_result = dict(cached)
            cached_result["cached"] = True
            return cached_result

        # Run the (blocking) agent graph off the event loop so a single slow
        # LLM/inference call cannot stall other concurrent requests. The
        # session_id is threaded through as the LangGraph thread_id for per-user
        # conversation isolation.
        response_data = await run_in_threadpool(process_query, request.query, session_id, None, user.get("username"))

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        result = _build_chat_result(response_data)

        # Cache only clean, non-interrupt answers (never cache HITL pauses).
        if not result.get("awaiting_validation"):
            cache.set("chat", request.query, result)
            # Populate the semantic cache too (opt-in / no-op otherwise).
            try:
                from services.semantic_cache import semantic_set
                semantic_set(config, request.query, result)
            except Exception:
                pass

        # Meter token usage for cost governance (opt-in / no-op otherwise).
        try:
            from services.cost_tracker import record_interaction
            record_interaction(
                config, request.query, result.get("response", "") or "",
                user_id=user.get("username"), session_id=session_id,
            )
        except Exception:
            pass

        # Observability: record which agent handled the query.
        try:
            from services.observability import record_agent
            record_agent(result.get("agent"))
        except Exception:
            pass

        # Compliance audit trail (PII-masked; no-op when disabled).
        from services.audit import write_audit
        await run_in_threadpool(
            write_audit, config,
            username=user.get("username"), role=user.get("role"),
            action="chat", agent=result.get("agent"), session_id=session_id,
            detail=request.query,
        )

        return result
    except Exception as e:
        try:
            from services.observability import record_llm_error
            record_llm_error()
        except Exception:
            pass
        err = str(e)
        logger.error("Error in /chat: %s", err)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "agent": "System",
                "response": f"LLM request failed: {err}"
            }
        )

@app.post("/upload")
async def upload_image(
    response: Response,
    image: UploadFile = File(...), 
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None),
    x_session_id: Optional[str] = Header(None),
    _rl: None = Depends(rate_limit_dep),
    user: dict = Depends(require_roles("patient", "admin"))
):
    """Process medical image uploads with optional text input."""
    session_id = x_session_id or session_id
    # Validate file type
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400, 
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG"
            }
        )
    
    # Check file size before saving
    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:  # Convert MB to bytes
        return JSONResponse(
            status_code=413, 
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
            }
        )
    
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Save file securely
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    try:
        query = {"text": text, "image": file_path}
        # Offload the blocking graph invocation and pass session_id for isolation.
        response_data = await run_in_threadpool(process_query, query, session_id, None, user.get("username"))

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        result = _build_chat_result(response_data)

        # Compliance audit trail for a diagnostic image operation.
        from services.audit import write_audit
        await run_in_threadpool(
            write_audit, config,
            username=user.get("username"), role=user.get("role"),
            action="upload", agent=result.get("agent"), session_id=session_id,
            detail=f"image={image.filename}; text={text}",
        )

        # ---- Doctor review queue: when graph paused for validation, keep the
        #      image and create a pending ReviewCase instead of deleting it. ----
        if result.get("awaiting_validation"):
            # Extract image type from agent name.
            agent = result.get("agent", "")
            image_type = "MEDICAL_IMAGE"
            if "BRAIN" in agent:
                image_type = "BRAIN_MRI"
            elif "CHEST" in agent or "XRAY" in agent:
                image_type = "CHEST_XRAY"
            elif "SKIN" in agent:
                image_type = "SKIN_LESION"

            from services.review_queue import create_case
            case_uid = create_case(
                config,
                patient_username=user.get("username"),
                session_id=session_id,
                image_path=file_path,
                image_type=image_type,
                ai_agent=agent,
                ai_diagnosis=result.get("response", ""),
                result_image=result.get("result_image"),
            )
            # Replace the validation prompt with a "submitted" message.
            result["response"] = (
                "您的医学影像已提交至执业医生审核队列，请稍候。\n\n"
                "医生审核完成后，您将实时收到诊断结论。\n\n"
                "**重要提示：** AI 初步分析结果仅供参考，最终诊断需由执业医生确认。"
            )
            result["case_uid"] = case_uid
            result["awaiting_review"] = True
            # Do NOT delete the image — the doctor needs to inspect it.
            return result

        # Remove temporary file after sending (text-only or non-image agent).
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove temporary file: {str(e)}")
        
        return result
    except Exception as e:
        err = str(e)
        logger.error("Error in /upload: %s", err)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "agent": "System",
                "response": f"Image query failed: {err}"
            }
        )

@app.post("/validate")
async def validate_medical_output(
    response: Response,
    validation_result: str = Form(...), 
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None),
    _rl: None = Depends(rate_limit_dep),
    user: dict = Depends(require_approved_doctor())
):
    """Handle human validation for medical AI outputs.

    This resumes the LangGraph paused at the native ``interrupt()`` in the
    human-validation node (keyed on the session's thread_id) using
    ``Command(resume=...)`` — replacing the old approach of re-running the whole
    graph on a synthetic "Validation result: ..." text query.
    """
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Resume the paused graph for this session with the human's decision.
        response_data = await run_in_threadpool(
            resume_after_validation, session_id, validation_result, comments
        )

        final_text = response_data["messages"][-1].content

        # Audit the human-validation decision (who validated, outcome, comments).
        from services.audit import write_audit
        await run_in_threadpool(
            write_audit, config,
            username=user.get("username"), role=user.get("role"),
            action="validate", agent=response_data.get("agent_name"),
            session_id=session_id, validation_result=validation_result,
            detail=comments or "",
        )

        if validation_result.strip().lower().startswith("yes"):
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": final_text
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": final_text
            }
    except Exception as e:
        err = str(e)
        logger.error("Error in /validate: %s", err)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "Validation failed",
                "response": err
            }
        )


# ---- Doctor review queue endpoints ----


@app.get("/review/pending")
async def review_pending_list(user: dict = Depends(require_approved_doctor())):
    """List all pending review cases for the doctor console."""
    from services.review_queue import get_pending_cases
    cases = await run_in_threadpool(get_pending_cases, config)
    return {"status": "success", "cases": cases}


@app.get("/review/{case_uid}")
async def review_get_case(case_uid: str, user: dict = Depends(require_approved_doctor())):
    """Get full detail of a single review case."""
    from services.review_queue import get_case
    case = await run_in_threadpool(get_case, config, case_uid)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"status": "success", "case": case}


class ReviewDecision(BaseModel):
    verdict: str  # "approved" or "rejected"
    comments: Optional[str] = None


@app.post("/review/{case_uid}")
async def review_decide(
    case_uid: str,
    decision: ReviewDecision,
    user: dict = Depends(require_approved_doctor()),
):
    """Doctor approves or rejects a pending review case."""
    from services.review_queue import get_case, review_case

    case = await run_in_threadpool(get_case, config, case_uid)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if case.get("status") != "pending":
        return {"status": "already_reviewed", "case": case}

    validation_result = "yes" if decision.verdict == "approved" else "no"
    try:
        response_data = await run_in_threadpool(
            resume_after_validation, case["session_id"], validation_result, decision.comments
        )
        final_text = str(response_data["messages"][-1].content) if response_data.get("messages") else ""
    except Exception as e:
        logger.error("Failed to resume graph for case %s: %s", case_uid, e)
        raise HTTPException(status_code=500, detail=f"Failed to resume: {str(e)}")

    updated = await run_in_threadpool(
        review_case, config, case_uid, user.get("username"),
        verdict=decision.verdict, comments=decision.comments, final_result=final_text,
    )

    from services.audit import write_audit
    await run_in_threadpool(
        write_audit, config,
        username=user.get("username"), role=user.get("role"),
        action="doctor_review", agent=case.get("ai_agent"),
        session_id=case.get("session_id"),
        validation_result=decision.verdict, detail=decision.comments or "",
    )

    return {"status": "success", "case": updated}


@app.get("/review/status/{case_uid}")
async def review_patient_status(case_uid: str):
    """Patient checks whether their case has been reviewed. No auth required."""
    from services.review_queue import get_case_status_for_patient
    info = await run_in_threadpool(get_case_status_for_patient, config, case_uid)
    if not info:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"status": "success", **info}


@app.get("/review/image/{case_uid}")
async def review_get_image(case_uid: str, user: dict = Depends(require_approved_doctor())):
    """Serve the original pathology image for the doctor to inspect."""
    from services.review_queue import get_case
    case = await run_in_threadpool(get_case, config, case_uid)
    if not case or not case.get("image_path"):
        raise HTTPException(status_code=404, detail="Image not found")
    path = case["image_path"]
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image file missing")
    return FileResponse(path)


@app.get("/review/stream/{case_uid}")
async def review_sse_stream(case_uid: str):
    """SSE endpoint: patient subscribes to real-time review result."""
    if not _SSE_AVAILABLE:
        return JSONResponse(status_code=501, content={"error": "SSE not available"})

    from services.review_queue import get_case_status_for_patient

    async def event_generator():
        max_wait = 600
        waited = 0
        while waited < max_wait:
            info = await run_in_threadpool(get_case_status_for_patient, config, case_uid)
            if not info:
                yield {"event": "error", "data": json.dumps({"error": "Case not found"})}
                return
            if info.get("status") != "pending":
                yield {
                    "event": "result",
                    "data": json.dumps({
                        "status": info["status"],
                        "final_result": info.get("final_result", ""),
                        "doctor_comments": info.get("doctor_comments", ""),
                        "reviewed_at": info.get("reviewed_at"),
                    }),
                }
                return
            yield {"event": "heartbeat", "data": json.dumps({"status": "pending"})}
            await asyncio.sleep(1)
            waited += 1
        yield {"event": "timeout", "data": json.dumps({"error": "Review timed out"})}

    return EventSourceResponse(event_generator())


@app.post("/chat/stream")
async def chat_stream(
    request: QueryRequest,
    response: Response,
    session_id: Optional[str] = Cookie(None),
    x_session_id: Optional[str] = Header(None),
    _rl: None = Depends(rate_limit_dep),
    user: dict = Depends(require_roles("patient", "admin"))
):
    """Stream the assistant response as Server-Sent Events (SSE).

    Progressive enhancement over ``/chat``: the classic endpoint is untouched
    and remains the default. Clients that opt in receive incremental ``token``
    events followed by a terminal ``done`` event carrying agent metadata and any
    ``result_image``. Requires the optional ``sse-starlette`` dependency and the
    ``ENABLE_STREAMING`` feature flag; otherwise returns 501 so callers can fall
    back to ``/chat``.
    """
    if not _SSE_AVAILABLE or not config.features.enable_streaming:
        return JSONResponse(
            status_code=501,
            content={
                "status": "error",
                "agent": "System",
                "response": "Streaming is not enabled on this server. Use /chat instead."
            }
        )

    session_id = x_session_id or session_id
    if not session_id:
        session_id = str(uuid.uuid4())

    query = request.query

    async def event_generator():
        try:
            yield {"event": "message", "data": json.dumps({"type": "start"})}

            # --- Token-level real streaming (opt-in) ---
            # For plain conversational queries, stream the model's native tokens
            # as they are generated. Declines (returns None) for RAG/web/vision/
            # guardrail paths, which fall back to the graph below.
            streamed = False
            if config.features.stream_token_level:
                try:
                    from agents.agent_decision import stream_conversation_tokens
                    gen = await run_in_threadpool(stream_conversation_tokens, query, user.get("username"))
                    if gen is not None:
                        collected = []
                        # Pull chunks off the (blocking) generator without stalling the loop.
                        def _next(g):
                            try:
                                return next(g)
                            except StopIteration:
                                return None
                        while True:
                            chunk = await run_in_threadpool(_next, gen)
                            if chunk is None:
                                break
                            collected.append(chunk.get("data", ""))
                            yield {"event": "message", "data": json.dumps(chunk)}
                        streamed = True
                        # Meter cost for the streamed answer.
                        try:
                            from services.cost_tracker import record_interaction
                            record_interaction(config, query, "".join(collected),
                                                user_id=user.get("username"), session_id=session_id)
                        except Exception:
                            pass
                        yield {"event": "message", "data": json.dumps({"type": "done", "agent": "CONVERSATION_AGENT"})}
                except Exception as e:
                    logger.warning("Token-level streaming failed (%s); falling back to graph.", e)
                    streamed = False

            if not streamed:
                # Run the blocking multi-agent graph off the event loop.
                response_data = await run_in_threadpool(process_query, query, session_id, None, user.get("username"))
                response_text = response_data["messages"][-1].content or ""

                # Word-chunk the fully-formed answer for a typing effect (fallback
                # path: RAG/web/vision answers or when token streaming is off).
                for token in response_text.split(" "):
                    yield {"event": "message", "data": json.dumps({"type": "token", "data": token + " "})}
                    await asyncio.sleep(0)  # yield control to the event loop

                done_payload = {
                    "type": "done",
                    "agent": response_data.get("agent_name"),
                    "awaiting_validation": bool(response_data.get("awaiting_validation")),
                }
                result_image = response_data.get("result_image")
                if result_image:
                    done_payload["result_image"] = result_image
                yield {"event": "message", "data": json.dumps(done_payload)}
        except Exception as e:
            logger.error("Error in /chat/stream: %s", e)
            logger.error(traceback.format_exc())
            yield {"event": "message", "data": json.dumps({"type": "error", "data": str(e)})}

    sse_response = EventSourceResponse(event_generator())
    # Ensure the session cookie is set on the streaming response too.
    sse_response.set_cookie(key="session_id", value=session_id)
    return sse_response


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Endpoint to transcribe speech using ElevenLabs API"""
    if not audio.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "No audio file selected"}
        )
    
    try:
        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"
        
        # Read and save the file
        audio_content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(audio_content)
        
        # Debug: Print file size to check if it's empty
        file_size = os.path.getsize(temp_audio)
        print(f"Received audio file size: {file_size} bytes")
        
        if file_size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Received empty audio file"}
            )
        
        # Convert to MP3
        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"
        
        try:
            # Use pydub with format detection
            audio = AudioSegment.from_file(temp_audio)
            audio.export(mp3_path, format="mp3")
            
            # Debug: Print MP3 file size
            mp3_size = os.path.getsize(mp3_path)
            print(f"Converted MP3 file size: {mp3_size} bytes")

            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()
            print(f"Converted audio file into byte array successfully!")

            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )
            
            # Clean up temp files
            try:
                os.remove(temp_audio)
                os.remove(mp3_path)
                print(f"Deleted temp files: {temp_audio}, {mp3_path}")
            except Exception as e:
                print(f"Could not delete file: {e}")
            
            if transcription.text:
                return {"transcript": transcription.text}
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"API error: {transcription}", "details": transcription.text}
                )

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing audio: {str(e)}"}
            )
                
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """Endpoint to generate speech using ElevenLabs API"""
    try:
        text = request.text
        selected_voice_id = request.voice_id
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required"}
            )
        
        # Define API request to ElevenLabs
        elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.speech.eleven_labs_api_key
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        # Send request to ElevenLabs API
        response = requests.post(elevenlabs_url, headers=headers, json=payload)

        if response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to generate speech, status: {response.status_code}", "details": response.text}
            )
        
        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)

        # Return the generated audio file
        return FileResponse(
            path=temp_audio_path,
            media_type="audio/mpeg",
            filename="generated_speech.mp3"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Add exception handler for request entity too large
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)
