"""Report composer for the Deep Research Agent.

Synthesizes the multi-step findings (each with its own sources) into a single
structured medical review with inline citations and a references section.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ReportComposer:
    def __init__(self, config):
        self.config = config
        self.llm = config.rag.llm

    @staticmethod
    def _collect_sources(findings: List[Dict[str, Any]]) -> List[str]:
        """De-duplicate source references across all findings, preserving order."""
        seen = set()
        ordered = []
        for f in findings:
            for src in f.get("sources", []) or []:
                key = str(src)
                if key and key not in seen:
                    seen.add(key)
                    ordered.append(key)
        return ordered

    def compose(self, query: str, findings: List[Dict[str, Any]]) -> str:
        """Compose the final referenced medical review."""
        if not findings:
            return ("I was unable to gather enough information to produce a deep "
                    "research report on this topic.")

        # Build the evidence block fed to the LLM.
        evidence_parts = []
        for i, f in enumerate(findings, 1):
            sub_q = f.get("sub_question", "")
            answer = f.get("answer", "")
            evidence_parts.append(f"[Finding {i}] Sub-question: {sub_q}\nEvidence: {answer}")
        evidence = "\n\n".join(evidence_parts)

        sources = self._collect_sources(findings)
        references_block = ""
        if sources:
            references_block = "\n".join(f"[{i}] {s}" for i, s in enumerate(sources, 1))

        prompt = (
            "You are a medical research writer. Using ONLY the evidence below, "
            "write a comprehensive, well-structured medical review that answers "
            "the user's question. Use clear section headings, be factual and "
            "concise, cite evidence inline where appropriate, and include an "
            "explicit note recommending consultation with a licensed healthcare "
            "professional. Do not fabricate facts beyond the evidence.\n\n"
            f"User question: {query}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Structure: Overview, Key Findings (grouped logically), Limitations, "
            "and a short Conclusion."
        )

        try:
            res = self.llm.invoke(prompt)
            report = res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            logger.warning(f"Report composition LLM call failed, using raw findings: {e}")
            # Degrade gracefully to a simple concatenation of findings.
            report = "\n\n".join(
                f"### {f.get('sub_question','')}\n{f.get('answer','')}" for f in findings
            )

        if references_block:
            report = f"{report}\n\n---\n**References**\n{references_block}"

        return report
