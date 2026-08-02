"""Semantic (embedding-similarity) response cache.

The plain ``CacheService`` only hits on byte-identical queries. This cache hits
on **semantically similar** ones — "what helps a headache?" ≈ "how do I treat a
headache?" — reusing the prior answer to cut latency and LLM cost.

How it works:
  * embed the query (reusing the RAG embedding model),
  * compare (cosine) against recently cached query embeddings,
  * on similarity >= threshold, return the stored answer.

Storage: Redis (shared, capped list) with an in-process fallback. Entirely
opt-in (``config.semantic_cache.enabled``) and fail-open — any error is a cache
miss, never an exception to the caller.
"""

import json
import time
import logging

logger = logging.getLogger(__name__)

_REDIS_KEY = "semcache:entries"     # Redis list of JSON entries
_MEM = []                           # fallback: [{"vec":[...], "result":{...}, "ts":...}]
_embedder = None
_embedder_tried = False


def _cfg(config):
    return getattr(config, "semantic_cache", None)


def _enabled(config) -> bool:
    return bool(getattr(_cfg(config), "enabled", False))


def _threshold(config) -> float:
    return float(getattr(_cfg(config), "threshold", 0.92))


def _max_entries(config) -> int:
    return int(getattr(_cfg(config), "max_entries", 500))


def _redis(config):
    try:
        from services.redis_client import get_redis
        url = getattr(getattr(config, "api", None), "redis_url", "")
        return get_redis(url or None)
    except Exception:
        return None


def _get_embedder(config):
    """Reuse the RAG embedding model (lazy, cached)."""
    global _embedder, _embedder_tried
    if not _embedder_tried:
        _embedder_tried = True
        try:
            _embedder = config.rag.embedding_model
        except Exception as e:
            logger.warning("semantic_cache: no embedding model (%s); disabled.", e)
            _embedder = None
    return _embedder


def _embed(config, text):
    emb = _get_embedder(config)
    if emb is None:
        return None
    try:
        return emb.embed_query(text)
    except Exception as e:
        logger.warning("semantic_cache embed failed (%s).", e)
        return None


def _cosine(a, b):
    try:
        import numpy as np
        va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
        denom = (np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    except Exception:
        # Pure-python fallback
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0


def _load_entries(config):
    client = _redis(config)
    if client is not None:
        try:
            raw = client.lrange(_REDIS_KEY, 0, _max_entries(config) - 1)
            return [json.loads(r) for r in raw]
        except Exception as e:
            logger.warning("semantic_cache load failed (%s); using memory.", e)
    return list(_MEM)


def semantic_get(config, query):
    """Return a cached result for a semantically-similar query, or None."""
    if not _enabled(config) or not query:
        return None
    try:
        qv = _embed(config, query)
        if qv is None:
            return None
        thr = _threshold(config)
        best, best_sim = None, 0.0
        for e in _load_entries(config):
            sim = _cosine(qv, e.get("vec", []))
            if sim > best_sim:
                best_sim, best = sim, e
        if best is not None and best_sim >= thr:
            logger.info("Semantic cache HIT (sim=%.3f).", best_sim)
            try:
                from services.agent_trace import add_event
                add_event("semantic_cache", hit=True, similarity=round(best_sim, 3))
            except Exception:
                pass
            return best.get("result")
    except Exception as e:
        logger.warning("semantic_get failed (%s); miss.", e)
    return None


def semantic_set(config, query, result):
    """Store (query embedding, result) in the semantic cache. Best-effort."""
    if not _enabled(config) or not query or result is None:
        return
    try:
        qv = _embed(config, query)
        if qv is None:
            return
        entry = {"vec": qv, "result": result, "ts": int(time.time())}
        client = _redis(config)
        if client is not None:
            try:
                client.lpush(_REDIS_KEY, json.dumps(entry, ensure_ascii=False))
                client.ltrim(_REDIS_KEY, 0, _max_entries(config) - 1)
                return
            except Exception as e:
                logger.warning("semantic_cache store failed (%s); using memory.", e)
        _MEM.insert(0, entry)
        del _MEM[_max_entries(config):]
    except Exception as e:
        logger.warning("semantic_set failed (%s).", e)


def reset_memory():
    _MEM.clear()
