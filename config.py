"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class. 
"""

import os
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables from .env file
load_dotenv()

def _clean_env_value(value: str) -> str:
    if value is None:
        return ""
    cleaned = value.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned

def _extract_default_from_os_getenv_expr(value: str) -> str:
    """
    Support mistakenly copied Python expressions in .env, e.g.:
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    """
    pattern = r"""os\.getenv\(\s*['"][^'"]+['"]\s*,\s*['"]([^'"]+)['"]\s*\)"""
    match = re.search(pattern, value)
    return match.group(1).strip() if match else value

def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    cleaned = _clean_env_value(raw)
    if cleaned.startswith("os.getenv("):
        cleaned = _extract_default_from_os_getenv_expr(cleaned)
    return _clean_env_value(cleaned) or default

def _get_api_key() -> str:
    # Keep backward compatibility with projects that only define DASHSCOPE_API_KEY.
    return _env("OPENAI_API_KEY") or _env("DASHSCOPE_API_KEY")

def _get_base_url() -> str:
    base_url = _env("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if base_url and not base_url.startswith(("http://", "https://")):
        base_url = f"https://{base_url.lstrip('/')}"
    return base_url.rstrip("/")

def _get_model_name() -> str:
    return _env("MODEL_NAME", "qwen3-max")

def _get_vision_model_name() -> str:
    return _env("VISION_MODEL", _get_model_name())

def _get_embedding_model_name() -> str:
    return _env("EMBEDDING_MODEL", "text-embedding-v4")

def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable (1/true/yes/on -> True)."""
    raw = _env(name, "").strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default

def _build_chat_model(temperature: float, vision: bool = False) -> ChatOpenAI:
    model_name = _get_vision_model_name() if vision else _get_model_name()
    request_timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "45"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
    return ChatOpenAI(
        model=model_name,
        api_key=_get_api_key(),
        base_url=_get_base_url(),
        timeout=request_timeout,
        max_retries=max_retries,
        temperature=temperature,
    )

def _build_embedding_model() -> OpenAIEmbeddings:
    request_timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "45"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
    # Explicitly request the embedding dimensionality so it matches the existing
    # Qdrant collection. DashScope's text-embedding-v4 defaults to 1024 dims but
    # supports 1536/2048; without pinning this it silently returns 1024 and
    # Qdrant rejects it against a 1536-dim collection.
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
    return OpenAIEmbeddings(
        model=_get_embedding_model_name(),
        api_key=_get_api_key(),
        base_url=_get_base_url(),
        request_timeout=request_timeout,
        max_retries=max_retries,
        dimensions=embedding_dim,
        # DashScope/OpenAI-compatible embedding endpoints expect string inputs.
        # Keep text as raw strings instead of token-id lists.
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = _build_chat_model(temperature=0.1)  # Deterministic

class ConversationConfig:
    def __init__(self):
        self.llm = _build_chat_model(temperature=0.7)  # Creative but factual

class WebSearchConfig:
    def __init__(self):
        self.llm = _build_chat_model(temperature=0.3)  # Slightly creative but factual
        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
        self.distance_metric = "Cosine"  # Add this with a default value
        self.use_local = True  # Add this with a default value
        self.vector_local_path = "./data/qdrant_db"  # Add this with a default value
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"  # Ensure a valid name
        self.chunk_size = 512  # Modify based on documents and performance
        self.chunk_overlap = 50  # Modify based on documents and performance
        self.embedding_model = _build_embedding_model()
        self.llm = _build_chat_model(temperature=0.3)  # Slightly creative but factual
        self.summarizer_model = _build_chat_model(temperature=0.5, vision=True)  # Vision-capable summarization model
        self.chunker_model = _build_chat_model(temperature=0.0)  # factual
        self.response_generator_model = _build_chat_model(temperature=0.3)  # Slightly creative but factual
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        # Reranker (cross-encoder). Env-configurable so you can swap in a
        # medical-domain model, e.g. "pritamdeka/S-PubMedBert-MS-MARCO".
        self.reranker_model = _env("RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L-6")
        self.reranker_top_k = _env_int("RERANKER_TOP_K", 3)

        # Parent-child (small-to-big) retrieval: retrieve on small precise chunks,
        # then expand each hit to its larger parent chunk for richer generation
        # context. Off by default (keeps current behaviour).
        self.parent_child_enabled = _env_bool("ENABLE_PARENT_CHILD_RETRIEVAL", False)
        self.parent_chunk_size = _env_int("PARENT_CHUNK_SIZE", 2048)

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.40  # The auto routing from RAG agent to WEB_SEARCH agent is dependent on this value

        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

        # ----- Agentic RAG (CRAG / Self-RAG) switches -----
        # enable_crag: document-relevance grading + post-generation hallucination
        # & answer self-reflection. When any step judges the context insufficient,
        # confidence is dropped to 0 so the existing routing falls back to web search.
        # Fully backward compatible: disable to restore the original RAG behaviour.
        self.enable_crag = _env_bool("ENABLE_CRAG", True)
        # Minimum ratio of relevant retrieved docs before falling back to web search.
        self.crag_relevance_threshold = _env_float("CRAG_RELEVANCE_THRESHOLD", 0.5)

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.brain_tumor_segmentation_output_path = "./uploads/brain_tumor_output/segmentation_plot.png"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = _build_chat_model(temperature=0.1)  # Keep deterministic for classification tasks

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Replace with your actual key
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        # Externalized via environment for containerized / multi-worker deploys.
        self.host = _env("HOST", "0.0.0.0")
        self.port = _env_int("PORT", 8000)
        self.debug = _env_bool("DEBUG", False)  # default off for production safety
        self.rate_limit = _env_int("RATE_LIMIT", 10)
        self.max_image_upload_size = _env_int("MAX_IMAGE_UPLOAD_SIZE", 5)  # max upload size in MB

        # ----- Scale-out / session-sharing infrastructure -----
        # Number of gunicorn/uvicorn workers (used by process manager, not by
        # the embedded uvicorn.run dev server).
        self.workers = _env_int("WORKERS", 1)
        # LangGraph checkpointer backend: "memory" (default, single process) or
        # "redis" (shared across workers/replicas + native HITL persistence).
        self.checkpointer_backend = _env("CHECKPOINTER_BACKEND", "memory")
        self.redis_url = _env("REDIS_URL", "")

class FeatureConfig:
    """Feature switches for progressively-enhanced, degradable capabilities.

    Every switch defaults to a value that preserves the original behaviour so
    existing clients and the local dev environment keep working unchanged.
    """
    def __init__(self):
        # Token/step-level SSE streaming endpoint (/chat/stream). The classic
        # /chat endpoint is unaffected regardless of this flag.
        self.enable_streaming = _env_bool("ENABLE_STREAMING", True)
        # True token-level streaming for conversational answers (native LLM
        # .stream()). Off by default -> keeps the word-chunking behaviour.
        self.stream_token_level = _env_bool("STREAM_TOKEN_LEVEL", False)
        # Deep Research Agent (Plan-and-Execute + Reflection). Off by default:
        # it is a heavier multi-step flow and an opt-in routing target.
        self.enable_deep_research = _env_bool("ENABLE_DEEP_RESEARCH", False)
        # Max plan steps / reflection rounds for the deep research agent.
        self.deep_research_max_steps = _env_int("DEEP_RESEARCH_MAX_STEPS", 4)
        self.deep_research_max_reflections = _env_int("DEEP_RESEARCH_MAX_REFLECTIONS", 1)

class DeliberationConfig:
    """Adaptive test-time multi-agent review for high-stakes medical answers."""
    def __init__(self):
        self.enabled = _env_bool("ENABLE_AGENT_DELIBERATION", False)
        self.max_reviewers = _env_int("AGENT_DELIBERATION_MAX_REVIEWERS", 3)
        raw_roles = _env(
            "AGENT_DELIBERATION_ROLES",
            "geriatric medicine safety reviewer,clinical pharmacist and contraindication reviewer,evidence and uncertainty reviewer",
        )
        self.roles = [role.strip() for role in raw_roles.split(",") if role.strip()]


class ResearchConfig:
    """Parallel fan-out and evidence governance for deep medical research."""
    def __init__(self):
        self.parallel_enabled = _env_bool("DEEP_RESEARCH_PARALLEL", True)
        self.max_workers = _env_int("DEEP_RESEARCH_MAX_WORKERS", 4)
        self.evidence_min_coverage = _env_float("DEEP_RESEARCH_EVIDENCE_MIN_COVERAGE", 0.6)


class CacheConfig:
    """Redis-backed response/result cache (P2). Off by default.

    Caching conversational answers ignores per-session context, so enable
    deliberately (best for stateless factual queries). No-ops without Redis.
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_CACHE", False)
        self.ttl = _env_int("CACHE_TTL", 3600)  # seconds

class RateLimitConfig:
    """Distributed rate limiting (P2). Off by default.

    Redis-backed fixed window shared across workers; in-memory fallback and
    fail-open when Redis is unavailable.
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_RATE_LIMIT", False)
        self.max_requests = _env_int("RATE_LIMIT_MAX_REQUESTS", 60)
        self.window_seconds = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)

class ObservabilityConfig:
    """Metrics / logging / tracing (P4). Metrics on by default (cheap); tracing off."""
    def __init__(self):
        self.enable_metrics = _env_bool("ENABLE_METRICS", True)
        self.enable_json_logs = _env_bool("ENABLE_JSON_LOGS", False)
        self.log_level = _env("LOG_LEVEL", "INFO")
        # LangSmith agent tracing (uses standard LangChain env vars when on).
        self.langsmith_tracing = _env_bool("LANGSMITH_TRACING", False)
        self.langsmith_api_key = _env("LANGSMITH_API_KEY", "")
        self.langsmith_project = _env("LANGSMITH_PROJECT", "medical-assistant")

class AuthConfig:
    """Authentication, authorization & compliance (P3). All off/degrade-safe.

    When ``enabled`` is False (default), endpoints are NOT protected and behave
    exactly as before. Database defaults to local SQLite so no infra is required.
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_AUTH", False)
        self.jwt_secret = _env("JWT_SECRET", "change-me-in-production")
        self.jwt_algorithm = _env("JWT_ALGORITHM", "HS256")
        self.token_expire_minutes = _env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 1440)
        # Empty -> SQLite at data/app.db; set to a postgresql:// DSN for prod.
        self.database_url = _env("DATABASE_URL", "")
        # ----- Refresh tokens + revocation (opt-in; off => legacy behaviour) -----
        # When True: /auth/login returns a short-lived access token plus a
        # long-lived refresh token, /auth/refresh rotates them, and /auth/logout
        # revokes the current token(s) via the jti denylist (services.token_store).
        # When False (default): login behaves exactly as before (single access
        # token valid for ``token_expire_minutes``, no refresh, no revocation).
        self.refresh_token_enabled = _env_bool("ENABLE_REFRESH_TOKEN", False)
        # Access-token lifetime used ONLY when refresh tokens are enabled.
        self.access_token_expire_minutes = _env_int("ACCESS_TOKEN_TTL_MINUTES", 15)
        # Refresh-token lifetime (default 7 days).
        self.refresh_token_expire_minutes = _env_int("REFRESH_TOKEN_TTL_MINUTES", 10080)
        # Redis URL for the revocation denylist; falls back to the app-wide
        # REDIS_URL (APIConfig.redis_url) when unset. Without Redis, revocation
        # degrades to an in-process memory store (single-worker only).
        self.redis_url = _env("AUTH_REDIS_URL", "") or _env("REDIS_URL", "")
        # Compliance
        self.enable_audit = _env_bool("ENABLE_AUDIT", True)
        self.enable_pii_masking = _env_bool("ENABLE_PII_MASKING", True)
        # Optional Fernet key to encrypt audit 'detail' at rest.
        self.encryption_key = _env("AUDIT_ENCRYPTION_KEY", "")

class TaskQueueConfig:
    """Async task queue (Celery + Redis) for heavy jobs (P2).

    When disabled (default), submitted jobs run synchronously in-process, so the
    app works without a broker/worker. When enabled, jobs are dispatched to
    Celery workers via the Redis broker.
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_TASK_QUEUE", False)
        self.broker_url = _env("CELERY_BROKER_URL", "") or _env("REDIS_URL", "")
        self.result_backend = _env("CELERY_RESULT_BACKEND", "") or self.broker_url

class ObjectStorageConfig:
    """Object storage for uploads / segmentation outputs (P2).

    Backend "local" (default) keeps the existing on-disk + /uploads behaviour.
    Backend "s3" targets MinIO/S3; any failure falls back to local disk.
    """
    def __init__(self):
        self.backend = _env("OBJECT_STORAGE_BACKEND", "local")  # local | s3
        self.endpoint_url = _env("S3_ENDPOINT_URL", "")          # e.g. http://minio:9000
        self.bucket = _env("S3_BUCKET", "medical-assistant")
        self.access_key = _env("S3_ACCESS_KEY", "")
        self.secret_key = _env("S3_SECRET_KEY", "")
        self.region = _env("S3_REGION", "")
        self.public_base_url = _env("S3_PUBLIC_BASE_URL", "")    # optional CDN/base for URLs

class SecurityConfig:
    """Agent-hardening: prompt-injection defence & output leak protection.

    All local/heuristic (no LLM cost, no new deps). Enabled by default because
    it is fail-open and low-risk, but every sub-check can be turned off and the
    whole layer disabled with ENABLE_AGENT_SECURITY=false to restore the exact
    previous behaviour.
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_AGENT_SECURITY", True)
        # Fence untrusted retrieved/web content as data (spotlighting).
        self.wrap_untrusted = _env_bool("SECURITY_WRAP_UNTRUSTED", True)
        # Redact leaked system prompt / secrets from responses.
        self.scan_output = _env_bool("SECURITY_SCAN_OUTPUT", True)

class TraceConfig:
    """Structured per-request agent tracing (routing / guardrail / retrieval).

    Emits JSON span logs (and optional Prometheus counters). Off by default;
    when disabled all trace calls are no-ops.
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_AGENT_TRACE", False)

class MemoryConfig:
    """Cross-session long-term memory (user_id-keyed durable facts).

    Off by default. When on, stored facts are injected into prompts and
    (optionally) auto-extracted from each turn by the LLM. Degrades to a local
    JSON store when Redis is unavailable.
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_LONG_TERM_MEMORY", False)
        self.auto_extract = _env_bool("MEMORY_AUTO_EXTRACT", True)
        self.redis_url = _env("MEMORY_REDIS_URL", "") or _env("REDIS_URL", "")

class ClarificationConfig:
    """Proactive multi-turn clarification for vague symptom descriptions."""
    def __init__(self):
        self.enabled = _env_bool("ENABLE_CLARIFICATION", False)

class ToolsConfig:
    """Structured tool-calling framework (drug info, calculators, ...).

    ``enabled``       -> fast keyword dispatcher.
    ``agent_enabled`` -> LLM-driven ReAct tool calling (autonomous, multi-step),
                         with graceful fallback to the keyword dispatcher.
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_TOOLS", False)
        self.agent_enabled = _env_bool("ENABLE_TOOL_AGENT", False)

class CitationConfig:
    """Sentence-level citation grounding + confidence labelling for RAG."""
    def __init__(self):
        self.enabled = _env_bool("ENABLE_CITATIONS", False)

class MedicalSafetyConfig:
    """Final-answer medical safety critic.

    Adds a deterministic, fail-open review step that softens unsafe prescription
    or diagnosis language and injects emergency guidance for red-flag content.
    Disabled by default to preserve existing behavior.
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_MEDICAL_SAFETY_CRITIC", False)
        self.mode = _env("MEDICAL_SAFETY_CRITIC_MODE", "rules")

class CostConfig:
    """LLM cost / token-budget governance. Off by default.

    When enabled, meters token usage per user/session; if a positive
    ``daily_token_budget`` is set, requests over budget are rejected (429).
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_COST_TRACKING", False)
        self.daily_token_budget = _env_int("DAILY_TOKEN_BUDGET", 0)  # 0 => metering only
        self.redis_url = _env("COST_REDIS_URL", "") or _env("REDIS_URL", "")

class SemanticCacheConfig:
    """Embedding-similarity response cache (semantic cache). Off by default."""
    def __init__(self):
        self.enabled = _env_bool("ENABLE_SEMANTIC_CACHE", False)
        self.threshold = _env_float("SEMANTIC_CACHE_THRESHOLD", 0.92)
        self.max_entries = _env_int("SEMANTIC_CACHE_MAX_ENTRIES", 500)

class SummaryConfig:
    """Semantic conversation summarisation to compress long histories."""
    def __init__(self):
        self.enabled = _env_bool("ENABLE_CONVERSATION_SUMMARY", False)
        # Summarise once history exceeds this many messages.
        self.trigger_messages = _env_int("SUMMARY_TRIGGER_MESSAGES", 20)

class TriageConfig:
    """Emergency red-flag triage (detect critical symptoms -> urge care)."""
    def __init__(self):
        self.enabled = _env_bool("ENABLE_EMERGENCY_TRIAGE", True)

class ReviewConfig:
    """Doctor review queue (HITL): patient-uploaded medical images are held for
    a licensed doctor to approve/reject before the diagnosis is finalized.

    When a doctor validates, the paused LangGraph is resumed with the verdict
    and the final answer is pushed to the patient via SSE. Patients cannot
    self-confirm image diagnoses — the old ``patient clicks Yes`` flow is
    replaced entirely.
    """
    def __init__(self):
        # Invite code required to register as a ``doctor`` role. This is only the
        # FIRST gate: after registering, a doctor account starts as
        # ``unsubmitted`` and MUST upload a practising-licence certificate that an
        # admin/reviewer approves before it can access the review queue or receive
        # patient medical-image diagnoses. So a leaked invite code alone can never
        # grant real doctor privileges. Must match DOCTOR_INVITE_CODE.
        self.doctor_invite_code = _env("DOCTOR_INVITE_CODE", "")


class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True


class FamilyCareConfig:
    """Family-caregiver companionship module: follow codes, reminder tasks,
    an AI-driven "care channel" for proactive check-ins, and risk alerts.

    Fully opt-in and additive: disabled by default so existing behaviour is
    unchanged. The background scheduler thread is only started when enabled.
    """
    def __init__(self):
        self.enabled = _env_bool("ENABLE_FAMILY_CARE", False)
        # How often (seconds) the background scheduler scans due reminder tasks.
        self.scheduler_interval_seconds = _env_int("FAMILY_CARE_SCHEDULER_INTERVAL", 60)
        # Follow-code validity window (hours) before it expires unused.
        self.follow_code_ttl_hours = _env_int("FAMILY_CARE_FOLLOW_CODE_TTL_HOURS", 24)
        # Web Push (VAPID) is optional; without keys, alerts degrade to in-app only.
        self.vapid_public_key = _env("VAPID_PUBLIC_KEY", "")
        self.vapid_private_key = _env("VAPID_PRIVATE_KEY", "")
        self.vapid_admin_email = _env("VAPID_ADMIN_EMAIL", "admin@example.com")


class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.review = ReviewConfig()
        self.family_care = FamilyCareConfig()
        self.ui = UIConfig()
        self.features = FeatureConfig()
        self.deliberation = DeliberationConfig()
        self.research = ResearchConfig()
        self.cache = CacheConfig()
        self.rate_limit = RateLimitConfig()
        self.object_storage = ObjectStorageConfig()
        self.task_queue = TaskQueueConfig()
        self.auth = AuthConfig()
        self.observability = ObservabilityConfig()
        self.security = SecurityConfig()
        self.trace = TraceConfig()
        self.memory = MemoryConfig()
        self.clarification = ClarificationConfig()
        self.tools = ToolsConfig()
        self.summary = SummaryConfig()
        self.triage = TriageConfig()
        self.citation = CitationConfig()
        self.medical_safety = MedicalSafetyConfig()
        self.cost = CostConfig()
        self.semantic_cache = SemanticCacheConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history

# # Example usage
# config = Config()
