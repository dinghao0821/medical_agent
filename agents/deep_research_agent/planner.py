"""Research planner for the Deep Research Agent.

Decomposes a complex medical question into a small set of focused, individually
answerable sub-questions using structured output, with graceful text-parsing
and single-question fallbacks so it never hard-fails.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResearchPlan(BaseModel):
    """Structured research plan."""
    sub_questions: List[str] = Field(
        default_factory=list,
        description="3-5 focused, self-contained sub-questions that together "
                    "answer the user's medical question.",
    )


_PLAN_INSTRUCTION = (
    "You are a medical research planner. Decompose the user's question into 3-5 "
    "focused, self-contained sub-questions that can each be researched "
    "independently and together produce a comprehensive, evidence-based answer. "
    "Cover mechanism/definition, diagnosis/methods, evidence/outcomes, and "
    "limitations where relevant. Keep each sub-question concise."
)


class ResearchPlanner:
    def __init__(self, config):
        self.config = config
        self.llm = config.rag.llm
        try:
            self.structured = self.llm.with_structured_output(ResearchPlan)
        except Exception as e:
            logger.warning(f"Planner structured output unavailable, using text fallback: {e}")
            self.structured = None

    def _clean(self, lines: List[str]) -> List[str]:
        cleaned = []
        for ln in lines:
            s = ln.strip().lstrip("-*0123456789.) ").strip()
            if s:
                cleaned.append(s)
        return cleaned

    def plan(self, query: str, max_steps: int = 4) -> List[str]:
        """Return a list of sub-questions (bounded by ``max_steps``)."""
        prompt = f"{_PLAN_INSTRUCTION}\n\nUser question: {query}"

        # 1) Preferred: structured output
        if self.structured is not None:
            try:
                res = self.structured.invoke(prompt)
                subs = self._clean(res.sub_questions)
                if subs:
                    return subs[:max_steps]
            except Exception as e:
                logger.warning(f"Structured planning failed, falling back to text: {e}")

        # 2) Fallback: plain text, one sub-question per line
        try:
            res = self.llm.invoke(prompt + "\n\nReturn each sub-question on its own line.")
            text = res.content if hasattr(res, "content") else str(res)
            subs = self._clean(text.splitlines())
            if subs:
                return subs[:max_steps]
        except Exception as e:
            logger.warning(f"Text planning failed, using the original query as the sole step: {e}")

        # 3) Last resort: research the original question directly.
        return [query]
