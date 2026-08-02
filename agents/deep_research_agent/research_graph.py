r"""LangGraph subgraph for the Deep Research Agent.

Implements a Plan-and-Execute + Reflection loop:

    plan -> research -> reflect --(gaps & budget left)--> research
                              \--(done)--> compose -> END

Each research step reuses the EXISTING tools (Medical RAG, and web search as a
fallback when the knowledge base is insufficient) so the agent adds no new
retrieval infrastructure. Every external call is defensive: failures degrade to
partial findings rather than crashing the flow.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)


class ResearchState(TypedDict, total=False):
    query: str
    chat_history: str
    pending: List[str]          # sub-questions still to research
    findings: List[Dict[str, Any]]
    reflection_count: int
    report: str
    evidence_report: Dict[str, Any]


def build_deep_research_graph(
    config,
    rag_agent,
    web_agent,
    planner,
    composer,
    max_steps: int = 4,
    max_reflections: int = 1,
):
    """Compile and return the deep-research StateGraph.

    Args:
        config: global Config.
        rag_agent: an instance exposing ``process_query(query, chat_history)``.
        web_agent: an instance exposing ``process_web_search_results(query, chat_history)``.
        planner: ResearchPlanner instance.
        composer: ReportComposer instance.
        max_steps: maximum number of sub-questions per planning round.
        max_reflections: maximum number of additional research rounds.
    """
    min_conf = getattr(config.rag, "min_retrieval_confidence", 0.4)
    llm = config.rag.llm
    research_config = getattr(config, "research", None)
    parallel_enabled = bool(getattr(research_config, "parallel_enabled", True))
    max_workers = max(1, min(int(getattr(research_config, "max_workers", 4)), max_steps))
    evidence_min_coverage = float(getattr(research_config, "evidence_min_coverage", 0.6))

    def _research_one(sub_q: str, chat_history: str) -> Dict[str, Any]:
        """Answer a single sub-question via RAG, falling back to web search."""
        answer = ""
        sources: List[str] = []
        try:
            rag_res = rag_agent.process_query(sub_q, chat_history=chat_history)
            if isinstance(rag_res, dict):
                resp = rag_res.get("response", "")
                answer = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
                sources = list(rag_res.get("sources", []) or [])
                confidence = float(rag_res.get("confidence", 0.0) or 0.0)
            else:
                answer = str(rag_res)
                confidence = 0.0
        except Exception as e:
            logger.warning(f"[DeepResearch] RAG failed for '{sub_q}': {e}")
            confidence = 0.0

        insufficient = (
            not answer
            or confidence < min_conf
            or "don't have enough information" in answer.lower()
            or "not enough information" in answer.lower()
        )

        if insufficient:
            try:
                web_answer = web_agent.process_web_search_results(query=sub_q, chat_history=chat_history)
                web_text = web_answer if isinstance(web_answer, str) else getattr(web_answer, "content", str(web_answer))
                if web_text:
                    answer = (answer + "\n\n" if answer else "") + f"(Web search) {web_text}"
                    sources.append("Web search (Tavily)")
            except Exception as e:
                logger.warning(f"[DeepResearch] Web search failed for '{sub_q}': {e}")

        return {"sub_question": sub_q, "answer": answer or "(no information found)", "sources": sources}

    def _find_gaps(query: str, findings: List[Dict[str, Any]], budget: int) -> List[str]:
        """Ask the LLM whether the findings leave important gaps; return up to
        ``budget`` new sub-questions, or an empty list."""
        if budget <= 0:
            return []
        covered = "\n".join(f"- {f['sub_question']}: {f['answer'][:300]}" for f in findings)
        prompt = (
            "You are reviewing research findings for completeness. Based on the "
            "original question and the findings so far, list any IMPORTANT missing "
            "sub-questions that still need answering to give a complete, accurate "
            "medical answer. If the findings are already sufficient, reply exactly "
            "with 'NONE'. Otherwise list each missing sub-question on its own line.\n\n"
            f"Original question: {query}\n\nFindings so far:\n{covered}"
        )
        try:
            res = llm.invoke(prompt)
            text = res.content if hasattr(res, "content") else str(res)
            if "none" in text.strip().lower()[:8]:
                return []
            gaps = []
            for ln in text.splitlines():
                s = ln.strip().lstrip("-*0123456789.) ").strip()
                if s:
                    gaps.append(s)
            return gaps[:budget]
        except Exception as e:
            logger.warning(f"[DeepResearch] Reflection gap-analysis failed: {e}")
            return []

    # ---- Graph nodes ----
    def plan_node(state: ResearchState) -> ResearchState:
        query = state["query"]
        pending = planner.plan(query, max_steps=max_steps)
        logger.info(f"[DeepResearch] Planned {len(pending)} sub-questions")
        return {**state, "pending": pending, "findings": [], "reflection_count": 0}

    def research_node(state: ResearchState) -> ResearchState:
        """Fan out independent research tasks and merge results deterministically."""
        chat_history = state.get("chat_history", "")
        findings = list(state.get("findings", []))
        pending = list(state.get("pending", []))
        if parallel_enabled and len(pending) > 1:
            ordered = [None] * len(pending)
            with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as executor:
                futures = {
                    executor.submit(_research_one, sub_q, chat_history): index
                    for index, sub_q in enumerate(pending)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    sub_q = pending[index]
                    try:
                        ordered[index] = future.result()
                    except Exception as exc:
                        logger.warning("[DeepResearch] Parallel task failed for '%s': %s", sub_q, exc)
                        ordered[index] = {
                            "sub_question": sub_q,
                            "answer": "(research task failed)",
                            "sources": [],
                        }
            findings.extend(item for item in ordered if item is not None)
        else:
            for sub_q in pending:
                logger.info(f"[DeepResearch] Researching: {sub_q}")
                findings.append(_research_one(sub_q, chat_history))
        return {**state, "findings": findings, "pending": []}

    def reflect_node(state: ResearchState) -> ResearchState:
        reflection_count = state.get("reflection_count", 0)
        remaining_budget = max_steps if reflection_count < max_reflections else 0
        gaps = _find_gaps(state["query"], state.get("findings", []), remaining_budget)
        if gaps:
            logger.info(f"[DeepResearch] Reflection found {len(gaps)} gap(s); researching further")
            return {**state, "pending": gaps, "reflection_count": reflection_count + 1}
        return {**state, "pending": []}

    def compose_node(state: ResearchState) -> ResearchState:
        """Compose only after a deterministic evidence-coverage critic runs."""
        findings = state.get("findings", [])
        supported = [f for f in findings if f.get("sources") and f.get("answer") not in {
            "(no information found)", "(research task failed)"
        }]
        coverage = len(supported) / len(findings) if findings else 0.0
        evidence_report = {
            "total_findings": len(findings),
            "supported_findings": len(supported),
            "coverage": round(coverage, 3),
            "threshold": evidence_min_coverage,
            "status": "grounded" if coverage >= evidence_min_coverage else "limited",
        }
        report = composer.compose(state["query"], findings)
        if coverage < evidence_min_coverage:
            report += (
                "\n\n> **证据完整性提示：** 当前可溯源证据覆盖率为 "
                f"{coverage:.0%}，低于设定阈值 {evidence_min_coverage:.0%}。"
                "请将结论视为初步研究摘要，并由专业人员核对原始来源。"
            )
        return {**state, "report": report, "evidence_report": evidence_report}

    def _after_reflect(state: ResearchState) -> str:
        return "research" if state.get("pending") else "compose"

    workflow = StateGraph(ResearchState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("research", research_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("compose", compose_node)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "research")
    workflow.add_edge("research", "reflect")
    workflow.add_conditional_edges("reflect", _after_reflect, {
        "research": "research",
        "compose": "compose",
    })
    workflow.add_edge("compose", END)

    return workflow.compile()
