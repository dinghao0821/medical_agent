import os
import logging
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class WebSearchProcessor:
    """
    Processes web search results and routes them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        self.config = config
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Build the prompt for the web search.
        
        Args:
            query: User query
            chat_history: chat history
            
        Returns:
            Complete prompt string
        """
        # Add chat history if provided
        # print("Chat History:", chat_history)
            
        # Build the prompt
        prompt = f"""Here are the last few messages from our conversation:

        {chat_history}

        The user asked the following question:

        {query}

        Summarize them into a single, well-formed question only if the past conversation seems relevant to the current query so that it can be used for a web search.
        Keep it concise and ensure it captures the key intent behind the discussion.
        """

        return prompt
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Fetches web search results, processes them using LLM, and returns a user-friendly response.
        """
        # print(f"[WebSearchProcessor] Fetching web search results for: {query}")
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)

        # Step 1: refine the query for search. If the LLM call fails (e.g. the
        # provider's content-inspection rejects the input with a 400), degrade
        # gracefully to searching with the raw user query instead of crashing.
        try:
            web_search_query = self.llm.invoke(web_search_query_prompt)
            search_text = getattr(web_search_query, "content", None) or query
        except Exception as e:
            logger.warning(f"[WebSearchProcessor] Query refinement failed ({e}); using raw query")
            search_text = query

        # Retrieve web search results
        web_results = self.web_search_agent.search(search_text)

        # print(f"[WebSearchProcessor] Fetched results: {web_results}")

        # Security: web pages are the least-trusted source of all. Fence them so
        # the model treats them as data, not instructions (injection defence).
        fenced_results = web_results
        try:
            from agents.guardrails.injection_filter import wrap_untrusted
            fenced_results = wrap_untrusted(self.config, web_results, source="web_search_results")
        except Exception:
            pass

        # Construct prompt to LLM for processing the results
        llm_prompt = (
            "You are an AI assistant specialized in medical information. Below are web search results "
            "retrieved for a user query. Summarize and generate a helpful, concise response. "
            "Use reliable sources only and ensure medical accuracy.\n\n"
            f"Query: {query}\n\nWeb Search Results:\n{fenced_results}\n\nResponse:"
        )

        # Step 2: summarize the results. On LLM failure (including provider
        # content-inspection 400s that medical content can falsely trigger),
        # return the raw findings with a disclaimer so the user still gets an
        # answer rather than a 503.
        try:
            response = self.llm.invoke(llm_prompt)
            return response
        except Exception as e:
            logger.warning(f"[WebSearchProcessor] Result summarization failed ({e}); returning raw results")
            return (
                "I retrieved the following information from a web search but could not "
                "automatically summarize it. Please interpret it with care and consult a "
                "licensed healthcare professional:\n\n"
                f"{web_results}"
            )
