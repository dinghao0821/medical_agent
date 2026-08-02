"""Sentence-level citation grounding & confidence labelling (RAG trust layer).

After the RAG answer is generated, this module attaches **inline citations**
(``[1]``, ``[2]`` ...) mapping claims to the specific retrieved documents that
support them, and marks the answer's overall grounding confidence. This is the
trust backbone for a medical assistant: users (and reviewers) can see which
statements have a source vs. which are model inference.

Approach: an LLM rewrite pass that inserts citation markers keyed to numbered
sources, plus a lightweight grounded-ratio heuristic. Fully opt-in
(``config.citation.enabled``) and fail-open: any error returns the original
answer unchanged.
"""

import re
import logging

logger = logging.getLogger(__name__)


def _enabled(config) -> bool:
    return bool(getattr(getattr(config, "citation", None), "enabled", False))


_CITE_PROMPT = (
    "You add inline citations to a medical answer using ONLY the numbered sources "
    "below. For each factual claim that a source supports, append the matching "
    "marker like [1] or [2][3] right after the sentence. Do NOT add citations to "
    "sentences that no source supports (leave them uncited). Do not change the "
    "wording otherwise, and do not invent sources. Return only the cited answer.\n\n"
    "NUMBERED SOURCES:\n{sources}\n\nANSWER:\n{answer}\n\nCITED ANSWER:"
)


def _number_sources(retrieved_docs):
    """Return (numbered_text, list) of sources for citation."""
    numbered = []
    for i, doc in enumerate(retrieved_docs, start=1):
        content = (doc.get("content") or "")[:600]
        title = doc.get("source") or f"Document {i}"
        numbered.append({"n": i, "title": title, "content": content,
                         "path": doc.get("source_path")})
    text = "\n\n".join(f"[{s['n']}] {s['title']}: {s['content']}" for s in numbered)
    return text, numbered


def add_citations(config, answer_text, retrieved_docs, llm):
    """Return (cited_answer, meta). Fail-open to (answer_text, {}) on any issue.

    meta = {"grounded_ratio": float, "num_sources": int, "sources": [...]}.
    """
    if not _enabled(config) or not answer_text or not retrieved_docs or llm is None:
        return answer_text, {}

    try:
        sources_text, numbered = _number_sources(retrieved_docs)
        prompt = _CITE_PROMPT.format(sources=sources_text[:6000], answer=answer_text[:4000])
        resp = llm.invoke(prompt)
        cited = getattr(resp, "content", None) or str(resp)
        if not cited.strip():
            return answer_text, {}

        # Grounded ratio: fraction of sentences carrying at least one citation.
        sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+", cited) if s.strip()]
        if sentences:
            grounded = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
            ratio = round(grounded / len(sentences), 2)
        else:
            ratio = 0.0

        # Append a numbered reference list for the markers actually used.
        used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", cited)})
        if used:
            refs = "\n\n##### References\n" + "\n".join(
                f"[{s['n']}] "
                + (f"[{s['title']}]({s['path']})" if s.get("path") else s["title"])
                for s in numbered if s["n"] in used
            )
            cited = cited + refs

        # A short trust badge so users see grounding at a glance.
        badge = _confidence_badge(ratio)
        cited = f"{badge}\n\n{cited}"

        meta = {"grounded_ratio": ratio, "num_sources": len(used), "sources": numbered}
        logger.info("Citations added: grounded_ratio=%.2f, sources_used=%d", ratio, len(used))
        return cited, meta
    except Exception as e:
        logger.warning("add_citations failed (%s); returning original answer.", e)
        return answer_text, {}


def _confidence_badge(ratio: float) -> str:
    if ratio >= 0.7:
        return "> 🟢 **Well-sourced** — most statements are backed by retrieved references below."
    if ratio >= 0.3:
        return "> 🟡 **Partially sourced** — some statements are backed by references; others are general knowledge. Verify important details."
    return "> 🟠 **Limited sourcing** — few statements map to retrieved references. Treat as general guidance and consult a professional."
