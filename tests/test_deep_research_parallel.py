"""Tests for parallel Deep Research fan-out and evidence governance."""

import threading
import time

from agents.deep_research_agent.research_graph import build_deep_research_graph


class _Message:
    def __init__(self, content):
        self.content = content


class _LLM:
    def invoke(self, prompt):
        if "missing sub-questions" in str(prompt):
            return _Message("NONE")
        return _Message("Composed report")


class _RAG:
    def __init__(self):
        self.threads = set()

    def process_query(self, query, chat_history=""):
        self.threads.add(threading.get_ident())
        time.sleep(0.02)
        sources = [] if query == "unsupported" else [f"source:{query}"]
        return {"response": f"answer:{query}", "confidence": 1.0, "sources": sources}


class _Web:
    def process_web_search_results(self, query, chat_history=""):
        return ""


class _Planner:
    def plan(self, query, max_steps=4):
        return ["supported", "unsupported"]


class _Composer:
    def compose(self, query, findings):
        return "Composed report"


class _Config:
    class rag:
        min_retrieval_confidence = 0.4
        llm = _LLM()

    class research:
        parallel_enabled = True
        max_workers = 2
        evidence_min_coverage = 0.75


def test_parallel_research_preserves_order_and_warns_on_low_coverage():
    rag = _RAG()
    graph = build_deep_research_graph(
        _Config(), rag, _Web(), _Planner(), _Composer(), max_steps=2, max_reflections=0
    )

    result = graph.invoke({"query": "q", "chat_history": ""})

    assert [item["sub_question"] for item in result["findings"]] == [
        "supported", "unsupported"
    ]
    assert len(rag.threads) == 2
    assert result["evidence_report"]["coverage"] == 0.5
    assert result["evidence_report"]["status"] == "limited"
    assert "证据完整性提示" in result["report"]
