"""Parent-child (small-to-big) context expansion for RAG.

Rationale: retrieving on small, precise chunks maximises hit accuracy, but small
chunks give the generator too little context. "Small-to-big" retrieves on the
small chunks, then feeds the LLM a larger *parent* window.

This is a **query-time** implementation that needs no re-ingestion: after
retrieval + rerank, hits from the same source document are merged into one
larger context block (deduplicated, capped at ``parent_chunk_size`` chars),
preserving the top ranking. Opt-in (``config.rag.parent_child_enabled``) and
fail-open (returns the input unchanged on any error).
"""

import logging

logger = logging.getLogger(__name__)


def _enabled(config) -> bool:
    return bool(getattr(getattr(config, "rag", None), "parent_child_enabled", False))


def expand_to_parents(config, docs):
    """Merge same-source retrieved chunks into larger parent-context docs.

    Input/return: list of doc dicts with at least ``content`` and ``source``.
    Ranking is preserved by the order of first appearance; scores from the
    best chunk per source are kept.
    """
    if not _enabled(config) or not docs:
        return docs
    try:
        cap = int(getattr(config.rag, "parent_chunk_size", 2048))
        merged = {}       # source -> merged doc dict
        order = []        # preserve first-seen order (already rank-sorted)
        for d in docs:
            src = d.get("source") or d.get("id") or id(d)
            content = d.get("content", "") or ""
            if src not in merged:
                merged[src] = dict(d)
                order.append(src)
            else:
                existing = merged[src]
                # Append new content up to the cap, avoiding duplication.
                if content and content not in existing.get("content", ""):
                    combined = existing["content"] + "\n\n" + content
                    existing["content"] = combined[:cap]
                # Keep the best score/rank signals.
                for k in ("combined_score", "rerank_score", "score"):
                    if k in d and d[k] is not None:
                        existing[k] = max(existing.get(k, d[k]), d[k])
        result = [merged[s] for s in order]
        logger.info("Parent-child expansion: %d chunks -> %d parent blocks.", len(docs), len(result))
        return result
    except Exception as e:
        logger.warning("parent-child expansion failed (%s); using original docs.", e)
        return docs
