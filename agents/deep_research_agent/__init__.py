"""Deep Research Agent (new capability, main-line three).

A Plan-and-Execute + Reflection multi-step research agent that orchestrates the
EXISTING Medical RAG and web-search tools to produce a citation-backed medical
review. Exposed as a single ``run`` entry point so ``agent_decision`` can wire it
in as an opt-in routing target (``DEEP_RESEARCH_AGENT``) without touching the
behaviour of any existing agent.
"""

import logging

from .planner import ResearchPlanner
from .report_composer import ReportComposer
from .research_graph import build_deep_research_graph

logger = logging.getLogger(__name__)


class DeepResearchAgent:
    def __init__(self, config):
        self.config = config
        # Reuse existing retrieval tools (lazy import to avoid import cycles).
        from agents.rag_agent import MedicalRAG
        from agents.web_search_processor_agent import WebSearchProcessorAgent

        self.rag_agent = MedicalRAG(config)
        self.web_agent = WebSearchProcessorAgent(config)
        self.planner = ResearchPlanner(config)
        self.composer = ReportComposer(config)

        features = getattr(config, "features", None)
        self.max_steps = getattr(features, "deep_research_max_steps", 4)
        self.max_reflections = getattr(features, "deep_research_max_reflections", 1)

        self.graph = build_deep_research_graph(
            config=config,
            rag_agent=self.rag_agent,
            web_agent=self.web_agent,
            planner=self.planner,
            composer=self.composer,
            max_steps=self.max_steps,
            max_reflections=self.max_reflections,
        )

    def run(self, query: str, chat_history: str = "") -> str:
        """Run the full deep-research pipeline and return the composed report."""
        logger.info(f"[DeepResearch] Starting deep research for: {query}")
        try:
            result = self.graph.invoke({"query": query, "chat_history": chat_history})
            return result.get("report") or "No report could be generated."
        except Exception as e:
            logger.error(f"[DeepResearch] Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Deep research could not be completed due to an error: {e}"
