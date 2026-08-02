"""Cross-session long-term memory for the medical assistant (agent enhancement).

The LangGraph checkpointer persists *within-session* conversation state (keyed
on thread_id = session_id). This module adds **cross-session** memory keyed on a
stable ``user_id``: durable facts such as chronic conditions, allergies,
medications and stated preferences, so the assistant "remembers" a returning
user across different chat sessions.

Storage backends (degrade-safe, no hard dependency):
  * Redis (shared across workers/replicas) when a URL is configured/available.
  * Otherwise a local JSON file under ``data/memory/`` (single node).

Everything is best-effort and fail-open: a memory failure must never break a
chat turn. The whole feature is gated by ``config.memory.enabled``.
"""

import os
import json
import time
import logging

logger = logging.getLogger(__name__)

_MEM_DIR = os.path.join("data", "memory")
_REDIS_PREFIX = "ltm:"
_MAX_ITEMS = 50  # cap stored facts per user to bound size/prompt cost


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _cfg(config):
    return getattr(config, "memory", None)


def _enabled(config) -> bool:
    return bool(getattr(_cfg(config), "enabled", False))


def _redis(config):
    try:
        from services.redis_client import get_redis
        url = getattr(_cfg(config), "redis_url", "") or getattr(getattr(config, "api", None), "redis_url", "")
        return get_redis(url or None)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Low-level store (Redis or local JSON), returns list[dict]
# --------------------------------------------------------------------------- #
def _local_path(user_id: str) -> str:
    safe = "".join(c for c in str(user_id) if c.isalnum() or c in ("-", "_")) or "anon"
    return os.path.join(_MEM_DIR, f"{safe}.json")


def _load(config, user_id: str):
    client = _redis(config)
    if client is not None:
        try:
            raw = client.get(_REDIS_PREFIX + str(user_id))
            return json.loads(raw) if raw else []
        except Exception as e:
            logger.warning("LTM redis load failed (%s); trying local file.", e)
    try:
        path = _local_path(user_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("LTM local load failed (%s).", e)
    return []


def _save(config, user_id: str, items) -> None:
    items = items[-_MAX_ITEMS:]
    client = _redis(config)
    if client is not None:
        try:
            client.set(_REDIS_PREFIX + str(user_id), json.dumps(items, ensure_ascii=False))
            return
        except Exception as e:
            logger.warning("LTM redis save failed (%s); trying local file.", e)
    try:
        os.makedirs(_MEM_DIR, exist_ok=True)
        with open(_local_path(user_id), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("LTM local save failed (%s).", e)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_memories(config, user_id: str):
    """Return the list of stored memory items for a user (newest last)."""
    if not _enabled(config) or not user_id:
        return []
    try:
        return _load(config, user_id)
    except Exception:
        return []


def add_memory(config, user_id: str, text: str, kind: str = "fact") -> bool:
    """Add a single memory item; de-duplicates on normalized text. Returns added?"""
    if not _enabled(config) or not user_id or not text:
        return False
    try:
        text = text.strip()
        items = _load(config, user_id)
        norm = text.lower()
        if any((it.get("text", "").lower() == norm) for it in items):
            return False  # already known
        items.append({"text": text, "kind": kind, "ts": int(time.time())})
        _save(config, user_id, items)
        return True
    except Exception as e:
        logger.warning("LTM add failed (%s).", e)
        return False


def format_for_prompt(config, user_id: str) -> str:
    """Render stored memories as a compact block to prepend to a prompt.

    Returns an empty string when disabled or no memory exists, so callers can
    unconditionally concatenate it without changing behaviour.
    """
    items = get_memories(config, user_id)
    if not items:
        return ""
    lines = [f"- {it.get('text','')}" for it in items[-_MAX_ITEMS:] if it.get("text")]
    if not lines:
        return ""
    return (
        "Known information about this returning user (from previous sessions; "
        "use only if relevant, and never assume beyond it):\n" + "\n".join(lines) + "\n"
    )


def clear_memories(config, user_id: str) -> None:
    """Delete all memories for a user (privacy / right-to-be-forgotten)."""
    if not user_id:
        return
    client = _redis(config)
    if client is not None:
        try:
            client.delete(_REDIS_PREFIX + str(user_id))
        except Exception:
            pass
    try:
        path = _local_path(user_id)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Optional LLM-based fact extraction (fail-open, opt-in via config)
# --------------------------------------------------------------------------- #
_EXTRACT_PROMPT = (
    "You extract durable, cross-session medical facts about a user from a chat "
    "turn. Return a JSON array of short strings capturing ONLY stable facts worth "
    "remembering long-term: chronic conditions, allergies, current medications, "
    "age/sex if stated, and explicit preferences. Do NOT include transient "
    "symptoms, questions, chit-chat, or anything uncertain. If nothing qualifies, "
    "return []. Keep each item under 100 characters.\n\n"
    "USER MESSAGE:\n{user}\n\nASSISTANT REPLY:\n{assistant}\n\nJSON:"
)


def extract_and_store(config, user_id: str, user_text: str, assistant_text: str, llm=None) -> int:
    """Extract durable facts from a turn and store them. Returns #added.

    Requires ``config.memory.auto_extract`` and an LLM. Entirely fail-open: any
    error results in zero additions and never disrupts the chat.
    """
    if not _enabled(config) or not user_id:
        return 0
    if not bool(getattr(_cfg(config), "auto_extract", False)):
        return 0
    if llm is None or not user_text:
        return 0
    try:
        prompt = _EXTRACT_PROMPT.format(user=user_text[:1500], assistant=(assistant_text or "")[:1500])
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", None) or str(resp)
        facts = _parse_json_array(content)
        added = 0
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                if add_memory(config, user_id, fact.strip(), kind="auto"):
                    added += 1
        if added:
            logger.info("LTM: stored %d fact(s) for user.", added)
        return added
    except Exception as e:
        logger.warning("LTM extract failed (%s); skipping.", e)
        return 0


def _parse_json_array(text: str):
    """Best-effort parse of a JSON array possibly wrapped in prose/markdown."""
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        pass
    return []
