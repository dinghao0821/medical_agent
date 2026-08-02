"""LLM-as-judge graders for Corrective RAG (CRAG) / Self-RAG.

Provides three binary graders used to make the RAG pipeline self-correcting:

1. ``grade_document_relevance(query, doc)`` — is a retrieved document relevant
   to the question? (filters noise, decides whether to fall back to web search)
2. ``grade_hallucination(context, answer)`` — is the answer grounded in the
   retrieved facts? (detects hallucination)
3. ``grade_answer(query, answer)`` — does the answer actually address the
   question? (answer quality)

All graders use structured output when available and degrade gracefully:
on any failure they return a *lenient* default so the RAG pipeline behaves
exactly like the original (non-CRAG) flow instead of breaking.
"""

import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _BinaryGrade(BaseModel):
    """Binary yes/no grade with a short reason."""
    binary_score: str = Field(description="Answer 'yes' or 'no'")
    reason: str = Field(default="", description="Brief justification")


class RAGGrader:
    """Bundles the CRAG graders around the shared RAG LLM."""

    def __init__(self, config):
        self.config = config
        self.llm = config.rag.llm
        try:
            self.structured = self.llm.with_structured_output(_BinaryGrade)
        except Exception as e:
            logger.warning(f"Grader structured output unavailable, using text fallback: {e}")
            self.structured = None

    def _grade(self, instruction: str, payload: str, default: bool) -> bool:
        """Run a single binary grade. Returns ``default`` on any failure."""
        prompt = (
            f"{instruction}\n\n{payload}\n\n"
            "Respond with a binary score 'yes' or 'no' only."
        )
        # 1) Preferred: structured output
        if self.structured is not None:
            try:
                res = self.structured.invoke(prompt)
                return str(res.binary_score).strip().lower().startswith("y")
            except Exception as e:
                logger.warning(f"Structured grade failed, falling back to text: {e}")
        # 2) Fallback: plain text
        try:
            res = self.llm.invoke(prompt)
            text = res.content if hasattr(res, "content") else str(res)
            return "yes" in text.strip().lower()[:12]
        except Exception as e:
            logger.warning(f"Grade failed entirely, defaulting to {default}: {e}")
            return default

    def grade_document_relevance(self, query: str, doc_content: str) -> bool:
        """True if the document is relevant to the query."""
        if not doc_content:
            return False
        return self._grade(
            "You are a grader assessing the relevance of a retrieved document to a "
            "user question. If the document contains keyword(s) or semantic meaning "
            "related to the question, grade it as relevant.",
            f"Retrieved document:\n{doc_content[:2000]}\n\nUser question: {query}",
            default=True,
        )

    def grade_hallucination(self, context: str, answer: str) -> bool:
        """True if the answer is grounded in (supported by) the context."""
        if not answer:
            return True
        return self._grade(
            "You are a grader assessing whether an LLM answer is grounded in and "
            "supported by a set of retrieved medical facts. Answer 'yes' if the "
            "answer is supported by the facts, 'no' if it makes unsupported claims.",
            f"Set of facts:\n{context[:4000]}\n\nLLM answer: {answer[:2000]}",
            default=True,
        )

    def grade_answer(self, query: str, answer: str) -> bool:
        """True if the answer addresses/resolves the question."""
        if not answer:
            return False
        return self._grade(
            "You are a grader assessing whether an answer addresses and resolves a "
            "user question.",
            f"User question: {query}\n\nLLM answer: {answer[:2000]}",
            default=True,
        )
