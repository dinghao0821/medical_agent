"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field
import os, getpass
import uuid
import logging
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails
from agents.session.checkpointer_factory import build_checkpointer

import cv2
import numpy as np

from config import Config

load_dotenv()

logger = logging.getLogger(__name__)

# Load configuration
config = Config()

# Compiled-graph singleton. The graph is stateless; per-conversation state is
# persisted by the checkpointer keyed on thread_id (= session_id), so a single
# compiled graph can safely serve all concurrent sessions across requests.
_compiled_graph = None


def _unique_segmentation_path(default_path: str) -> str:
    """Return a per-request unique output path in the same directory as
    ``default_path``.

    The original code wrote every segmentation to a single fixed file
    (``segmentation_plot.png``), which causes concurrent requests to overwrite
    each other. Here we keep the same directory/extension but inject a UUID so
    each request gets its own file, and we ensure the directory exists.
    """
    directory = os.path.dirname(default_path) or "."
    filename = os.path.basename(default_path)
    stem, ext = os.path.splitext(filename)
    ext = ext or ".png"
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{stem}_{uuid.uuid4().hex}{ext}")


# Lazily-built object storage singleton (P2). Local backend by default; when
# configured for S3/MinIO, segmentation outputs are pushed to the bucket and a
# remote URL is returned. Any failure degrades to the local /uploads URL.
_object_storage = None


def _get_object_storage():
    global _object_storage
    if _object_storage is None:
        try:
            from services.object_storage import ObjectStorage
            _object_storage = ObjectStorage(config)
        except Exception as e:
            logger.warning(f"Object storage unavailable ({e}); using local /uploads URLs.")
            _object_storage = False  # sentinel: tried and failed
    return _object_storage or None


def _to_public_url(fs_path: str) -> Optional[str]:
    """Return a public URL for a generated file under ``uploads/``.

    When object storage is configured for S3/MinIO the file is uploaded and its
    remote URL is returned; otherwise (local backend / any failure) it maps the
    on-disk path to the ``/uploads/...`` URL served by FastAPI's static mount.
    Returns ``None`` if the path is not under an ``uploads`` directory.
    """
    if not fs_path:
        return None

    store = _get_object_storage()
    if store is not None and getattr(store, "active_backend", "local") == "s3":
        try:
            return store.upload_file(fs_path)
        except Exception as e:
            logger.warning(f"Object storage upload failed ({e}); falling back to local URL.")

    normalized = fs_path.replace("\\", "/").lstrip("./")
    marker = "uploads/"
    idx = normalized.find(marker)
    if idx == -1:
        return None
    return "/" + normalized[idx:]


# Agent that takes the decision of routing the request further to correct task specific agent
class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Decision model
    DECISION_MODEL = os.getenv("MODEL_NAME", "qwen3-max")
    
    # Vision model for image analysis
    VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-plus")
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system that routes user queries to 
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - The primary geriatric-care agent for older-adult health conversations, disability/ADL screening, dementia-risk screening, home-environment safety evaluation, rehabilitation-assistive-device guidance, caregiving support, as well as general chat.
    2. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical literature. Currently ingested medical knowledge involves 'introduction to brain tumor', 'deep learning techniques to diagnose and detect brain tumors', 'deep learning techniques to diagnose and detect covid / covid-19 from chest x-ray'.
    3. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    4. BRAIN_TUMOR_AGENT - For analysis of brain MRI images to detect and segment tumors.
    5. CHEST_XRAY_AGENT - For analysis of chest X-ray images to detect abnormalities.
    6. SKIN_LESION_AGENT - For analysis of skin lesion images to classify them as benign or malignant.

    Make your decision based on these guidelines:
    - If the user has not uploaded an image, route older-adult disability, cognition, home-safety, caregiving or assistive-device questions to CONVERSATION_AGENT; use RAG/WEB only when authoritative evidence or current research is explicitly needed.
    - If the user uploads a medical image, decide which medical vision agent is appropriate based on the image type and the user's query. If the image is uploaded without a query, always route to the correct medical vision agent based on the image type.
    - If the user asks about recent medical developments or current health situations, use the web search pocessor agent.
    - If the user asks specific medical knowledge questions, use the RAG agent.
    - For general conversation, greetings, or non-medical questions, use the conversation agent. But if image is uploaded, always go to the medical vision agents first.

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """

    image_analyzer = ImageAnalysisAgent(config=config)


# Appended to the routing system prompt ONLY when the Deep Research Agent is
# enabled, so the default routing behaviour is unchanged when it is off.
DEEP_RESEARCH_PROMPT_ADDENDUM = """

    7. DEEP_RESEARCH_AGENT - For complex, open-ended medical research questions that
    require multi-step investigation and synthesis across multiple sources (e.g.
    "provide a comprehensive review of ...", "compare treatment approaches for ...",
    "summarize the current evidence on ..."). Prefer this only when the user
    explicitly wants an in-depth, comprehensive report/review rather than a short
    factual answer.
    """


class AgentState(MessagesState):
    """State maintained across the workflow."""
    # messages: List[BaseMessage]  # Conversation history
    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of medical image if present
    user_id: Optional[str]  # Stable user id for cross-session long-term memory
    user_memory: Optional[str]  # Rendered long-term memory block (may be empty)
    output: Optional[str]  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    insufficient_info: bool  # Flag indicating RAG response has insufficient information
    result_image: Optional[str]  # Public URL of a generated segmentation image (unique per request)
    safety_verdict: Optional[Dict[str, Any]]  # Medical safety critic metadata
    deliberation_report: Optional[Dict[str, Any]]  # Adaptive specialist debate metadata


class AgentDecision(BaseModel):
    """Structured output schema for the routing/decision agent."""
    agent: str = Field(description="Target specialized agent name")
    reasoning: str = Field(default="", description="Step-by-step reasoning for the routing decision")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")


def create_agent_graph(checkpointer=None):
    """Create and configure the LangGraph for agent orchestration.

    Args:
        checkpointer: Optional LangGraph checkpointer to inject. When None, one is
            built from environment configuration (CHECKPOINTER_BACKEND / REDIS_URL)
            with a safe in-memory fallback.
    """

    # Initialize guardrails with the same LLM used elsewhere
    guardrails = LocalGuardrails(config.rag.llm)

    # LLM
    decision_model = config.agent_decision.llm

    # Opt-in Deep Research routing target (new capability, default off).
    deep_research_enabled = getattr(getattr(config, "features", None), "enable_deep_research", False)

    # Create the decision prompt (extend the agent catalogue only when the Deep
    # Research Agent is enabled so default routing is unaffected).
    system_prompt = AgentConfig.DECISION_SYSTEM_PROMPT
    if deep_research_enabled:
        system_prompt = system_prompt + DEEP_RESEARCH_PROMPT_ADDENDUM
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Modern structured-output router (Tool-Calling / JSON mode) replacing the
    # fragile JSON-string parsing. Falls back gracefully to the legacy
    # JsonOutputParser (and finally to a safe default) if the model/endpoint does
    # not support structured output or a call fails.
    try:
        structured_router = decision_model.with_structured_output(AgentDecision)
    except Exception as e:
        logger.warning(f"with_structured_output unavailable, using JSON-parser fallback: {e}")
        structured_router = None

    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    fallback_decision_chain = decision_prompt | decision_model | json_parser

    def make_routing_decision(decision_input: str) -> Dict[str, Any]:
        """Resolve routing to {'agent','reasoning','confidence'}.

        Order: structured output -> legacy JSON parsing -> safe default.
        """
        # 1) Structured output (preferred, robust)
        if structured_router is not None:
            try:
                msgs = decision_prompt.format_messages(input=decision_input)
                result = structured_router.invoke(msgs)
                return {
                    "agent": result.agent,
                    "reasoning": result.reasoning,
                    "confidence": float(result.confidence),
                }
            except Exception as e:
                logger.warning(f"Structured routing failed, falling back to JSON parser: {e}")

        # 2) Legacy JSON string parsing
        try:
            result = fallback_decision_chain.invoke({"input": decision_input})
            return {
                "agent": result.get("agent"),
                "reasoning": result.get("reasoning", ""),
                "confidence": float(result.get("confidence", 0.0)),
            }
        except Exception as e:
            logger.error(f"Routing decision failed entirely, defaulting to CONVERSATION_AGENT: {e}")
            return {"agent": "CONVERSATION_AGENT", "reasoning": "fallback", "confidence": 1.0}
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Check input through guardrails if text is present
        if input_text:
            # Fast, local prompt-injection scan first (no LLM cost). Blocks
            # instruction-override / prompt-exfiltration attempts before routing.
            try:
                from agents.guardrails.injection_filter import scan_input as _scan_input
                inj_safe, inj_reason = _scan_input(config, input_text)
            except Exception:
                inj_safe, inj_reason = True, ""
            if not inj_safe:
                logger.warning("[Security] Blocked injection attempt (%s)", inj_reason)
                try:
                    from services.agent_trace import add_event
                    add_event("security_block", reason=inj_reason)
                except Exception:
                    pass
                block_msg = AIMessage(
                    content="I can't help with that request. Let's keep to medical questions."
                )
                return {
                    **state,
                    "messages": block_msg,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True
                }

            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                # If input is blocked, return early with guardrail message
                print(f"Selected agent: INPUT GUARDRAILS, Message: ", message)
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True  # flag to end flow
                }
        
        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)

        # --- Agent enhancements (all opt-in; skipped for image inputs) ---
        if input_text and not has_image:
            # 1) Emergency red-flag triage: critical symptoms short-circuit with an
            #    urgent care message (highest priority, safety-first).
            try:
                from agents.guardrails.emergency_triage import check_red_flags
                triage_msg = check_red_flags(config, input_text)
            except Exception:
                triage_msg = None
            if triage_msg:
                try:
                    from services.agent_trace import add_event
                    add_event("emergency_triage", triggered=True)
                except Exception:
                    pass
                return {
                    **state,
                    "messages": AIMessage(content=triage_msg),
                    "agent_name": "EMERGENCY_TRIAGE",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True,
                }

            # 2) Structured tools. Prefer the LLM-driven ReAct agent (autonomous,
            #    multi-step tool calling) when enabled; otherwise fall back to the
            #    fast keyword dispatcher. Both degrade to normal routing on miss.
            tool_out = None
            try:
                from agents.tools import run_tool_agent, agent_available, maybe_run_tools
                if agent_available(config):
                    tool_out = run_tool_agent(config, input_text, config.conversation.llm)
                if not tool_out:
                    tool_out = maybe_run_tools(config, input_text)
            except Exception:
                tool_out = None
            if tool_out:
                try:
                    from services.agent_trace import add_event
                    add_event("tool", handled=True)
                except Exception:
                    pass
                return {
                    **state,
                    "messages": AIMessage(content=tool_out),
                    "agent_name": "TOOL_AGENT",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True,
                }

            # 3) Proactive clarification: if a symptom description is too vague,
            #    ask a focused follow-up instead of guessing.
            try:
                from agents.guardrails.clarification import needs_clarification
                clarify_msg = needs_clarification(config, input_text)
            except Exception:
                clarify_msg = None
            if clarify_msg:
                try:
                    from services.agent_trace import add_event
                    add_event("clarification", asked=True)
                except Exception:
                    pass
                return {
                    **state,
                    "messages": AIMessage(content=clarify_msg),
                    "agent_name": "CLARIFICATION",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True,
                }

        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False  # Explicitly set to False for normal flow
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)  # Not provided control from config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        # Make the decision (structured output with graceful fallback)
        decision = make_routing_decision(decision_input)

        # Decided agent
        print(f"Decision: {decision['agent']}")

        # Trace the routing decision (no-op unless ENABLE_AGENT_TRACE=true).
        try:
            from services.agent_trace import add_event
            add_event("route", agent=decision.get("agent"),
                      confidence=decision.get("confidence"), has_image=has_image)
        except Exception:
            pass
        
        # Update state with decision
        updated_state = {
            **state,
            "agent_name": decision["agent"],
        }
        
        # Route based on agent name and confidence
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}
        
        return {"agent_state": updated_state, "next": decision["agent"]}

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation."""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history
        recent_context = ""
        for msg in messages:#[-20:]:  # Get last 10 exchanges (20 messages)  # currently considering complete history - limit control from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        # Long-term memory block (empty unless enabled + facts exist).
        user_memory = state.get("user_memory", "") or ""

        # Combine everything for the decision input
        conversation_prompt = f"""User query: {input_text}

        Recent conversation context: {recent_context}

        {user_memory}

You are an AI-powered Geriatric Rehabilitation and Age-Friendly Environment Assistant supporting the research programme "失能失智老年人居住环境及康复辅助器具检测与评价关键技术研究". Focus on older adults, especially people with functional disability or cognitive impairment, their family caregivers, clinicians and researchers. Respond naturally while ensuring medical accuracy and clarity.

        ### Priority Domain
        - Screen activities of daily living and care dependency without issuing disability certification.
        - Identify possible cognitive decline/dementia risk without diagnosing dementia; consider hearing, vision, education, mood, delirium and medication confounders.
        - Evaluate home risks: falls, bathroom/toilet accessibility, lighting, fire, wandering, medication and emergency-call safety.
        - Recommend rehabilitation assistive-device categories only after considering body function, measurements, environment, caregiver capacity, trial use and follow-up.
        - Prefer person-centred, dignity-preserving, accessible Chinese suitable for older adults and caregivers.
        - For structured assessment requests, explain that the platform offers ADL, cognition, environment and assistive-device screening modules.

        ### Role & Capabilities
        - Engage in **general conversation** while maintaining professionalism.
        - Answer **medical questions** using verified knowledge.
        - Route **complex queries** to RAG (retrieval-augmented generation) or web search if needed.
        - Handle **follow-up questions** while keeping track of conversation context.
        - Redirect **medical images** to the appropriate AI analysis agent.

        ### Guidelines for Responding:
        1. **General Conversations:**
        - If the user engages in casual talk (e.g., greetings, small talk), respond in a friendly, engaging manner.
        - Keep responses **concise and engaging**, unless a detailed answer is needed.

        2. **Medical Questions:**
        - If you have **high confidence** in answering, provide a medically accurate response.
        - Ensure responses are **clear, concise, and factual**.

        3. **Follow-Up & Clarifications:**
        - Maintain conversation history for better responses.
        - If a query is unclear, ask **follow-up questions** before answering.

        4. **Handling Medical Image Analysis:**
        - Do **not** attempt to analyze images yourself.
        - If user speaks about analyzing or processing or detecting or segmenting or classifying any disease from any image, ask the user to upload the image so that in the next turn it is routed to the appropriate medical vision agents.
        - If an image was uploaded, it would have been routed to the medical computer vision agents. Read the history to know about the diagnosis results and continue conversation if user asks anything regarding the diagnosis.
        - After processing, **help the user interpret the results**.

        5. **Uncertainty & Ethical Considerations:**
        - If unsure, **never assume** medical facts.
        - Recommend consulting a **licensed healthcare professional** for serious medical concerns.
        - Avoid providing **medical diagnoses** or **prescriptions**—stick to general knowledge.
        - Never present a cognition screen as a dementia diagnosis, or an assistive-device suggestion as a completed professional fitting.
        - Escalate sudden confusion, acute weakness, falls with injury, choking, wandering, abuse/neglect or caregiver collapse to urgent professional help.

        ### Response Format:
        - Maintain a **conversational yet professional tone**.
        - Use **bullet points or numbered lists** for clarity when needed.
        - If pulling from external sources (RAG/Web Search), mention **where the information is from** (e.g., "According to Mayo Clinic...").
        - If a user asks for a diagnosis, remind them to **seek medical consultation**.

        ### Example User Queries & Responses:

        **User:** "Hey, how's your day going?"
        **You:** "I'm here and ready to help! How can I assist you today?"

        **User:** "I have a headache and fever. What should I do?"
        **You:** "I'm not a doctor, but headaches and fever can have various causes, from infections to dehydration. If your symptoms persist, you should see a medical professional."

        Conversational LLM Response:"""

        # print("Conversation Prompt:", conversation_prompt)

        response = config.conversation.llm.invoke(conversation_prompt)

        # print("Conversation respone:", response)

        # response = AIMessage(content="This would be handled by the conversation agent.")

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using RAG."""
        # Initialize the RAG agent

        print(f"Selected agent: RAG_AGENT")

        rag_agent = MedicalRAG(config)
        
        messages = state["messages"]
        query = state["current_input"]
        rag_context_limit = config.rag.context_limit

        recent_context = ""
        for msg in messages[-rag_context_limit:]:# limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        response = rag_agent.process_query(query, chat_history=recent_context)
        retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # Check if response indicates insufficient information
        insufficient_info = False
        response_content = response["response"]
        
        # Extract the content properly based on type
        if isinstance(response_content, dict) and hasattr(response_content, 'content'):
            # If it's an AIMessage or similar object with a content attribute
            response_text = response_content.content
        else:
            # If it's already a string
            response_text = response_content
            
        print(f"Response text type: {type(response_text)}")
        print(f"Response text preview: {response_text[:100]}...")
        
        if isinstance(response_text, str) and (
            "I don't have enough information to answer this question based on the provided context" in response_text or 
            "I don't have enough information" in response_text or 
            "don't have enough information" in response_text.lower() or
            "not enough information" in response_text.lower() or
            "insufficient information" in response_text.lower() or
            "cannot answer" in response_text.lower() or
            "unable to answer" in response_text.lower()
            ):
            
            print("RAG response indicates insufficient information")
            print(f"Response text that triggered insufficient_info: {response_text[:100]}...")
            insufficient_info = True

        print(f"Insufficient info flag set to: {insufficient_info}")

        # Store RAG output ONLY if confidence is high
        if retrieval_confidence >= config.rag.min_retrieval_confidence:
            # response_output = response["response"]
            response_output = AIMessage(content=response_text)
        else:
            response_output = AIMessage(content="")
        
        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info
        }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)

        # print("######### DEBUG WEB SEARCH:", processed_response)
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents
        }

    # Deep Research Agent node (new capability). Reuses existing RAG/Web tools
    # via a Plan-and-Execute + Reflection subgraph and returns a cited report.
    def run_deep_research_agent(state: AgentState) -> AgentState:
        """Handle in-depth medical research via the Deep Research subgraph."""
        print(f"Selected agent: DEEP_RESEARCH_AGENT")

        from agents.deep_research_agent import DeepResearchAgent

        messages = state["messages"]
        current_input = state["current_input"]

        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        recent_context = ""
        for msg in messages[-config.rag.context_limit:]:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"

        try:
            deep_agent = DeepResearchAgent(config)
            report = deep_agent.run(input_text, chat_history=recent_context)
        except Exception as e:
            logger.error(f"Deep Research Agent failed: {e}")
            report = f"Deep research could not be completed due to an error: {e}"

        return {
            **state,
            "output": AIMessage(content=report),
            "needs_human_validation": False,
            "agent_name": "DEEP_RESEARCH_AGENT",
        }

    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> Dict[str, str]:
        """Route based on RAG confidence score and response content."""
        # Debug prints
        print(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        print(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")
        
        # Redirect if confidence is low or if response indicates insufficient info
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format
        return "check_validation"  # No transition needed if confidence is high and info is sufficient
    
    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """Handle brain MRI image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: BRAIN_TUMOR_AGENT")

        # Unique per-request output path to avoid concurrent overwrites.
        output_path = _unique_segmentation_path(
            AgentConfig.image_analyzer.brain_tumor_segmentation_output_path
        )

        # Segment tumor region from the uploaded brain MRI image
        segmentation_result = AgentConfig.image_analyzer.segment_brain_tumor(image_path, output_path)

        result_image = None
        if segmentation_result is True:
            response = AIMessage(content="Following is the analyzed **segmented** output of the uploaded brain MRI image, highlighting the detected tumor region:")
            result_image = _to_public_url(output_path)
        elif segmentation_result is None:
            response = AIMessage(content="The brain tumor analysis model is currently unavailable. Please ensure the model checkpoint is installed to enable MRI tumor segmentation.")
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "BRAIN_TUMOR_AGENT",
            "result_image": result_image
        }
    
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """Handle chest X-ray image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: CHEST_XRAY_AGENT")

        # classify chest x-ray into covid or normal
        predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)

        if predicted_class == "covid19":
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **POSITIVE** result for **COVID-19**.")
        elif predicted_class == "normal":
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **NEGATIVE** result for **COVID-19**, i.e., **NORMAL**.")
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        # response = AIMessage(content="This would be handled by the chest X-ray agent, analyzing the image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "CHEST_XRAY_AGENT"
        }
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: SKIN_LESION_AGENT")

        # Unique per-request output path to avoid concurrent overwrites.
        output_path = _unique_segmentation_path(
            AgentConfig.image_analyzer.skin_lesion_segmentation_output_path
        )

        # Segment the skin lesion region from the uploaded image
        predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path, output_path)

        result_image = None
        if predicted_mask:
            response = AIMessage(content="Following is the analyzed **segmented** output of the uploaded skin lesion image:")
            result_image = _to_public_url(output_path)
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT",
            "result_image": result_image
        }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation via LangGraph's native interrupt().

        Instead of faking validation by re-running the whole graph on a
        "yes/no" text query, this node PAUSES the graph with ``interrupt()`` and
        surfaces the diagnosis for review. The caller resumes execution with
        ``Command(resume={"validation_result": ..., "comments": ...})`` once a
        human (typically a clinician) has reviewed the output. Pause/resume
        state is persisted by the checkpointer keyed on the session's thread_id.
        """
        print(f"Selected agent: HUMAN_VALIDATION")

        diagnosis_text = state["output"].content if hasattr(state["output"], "content") else str(state["output"])

        validation_prompt = (
            f"{diagnosis_text}\n\n**Human Validation Required:**\n"
            "- If you're a healthcare professional: Please validate the output. "
            "Select **Yes** or **No**. If No, provide comments.\n"
            "- If you're a patient: Simply click Yes to confirm."
        )

        # Pause here; the payload is delivered to the caller (and the UI). The
        # returned value is whatever is passed to Command(resume=...).
        decision = interrupt({
            "type": "medical_validation",
            "agent": state["agent_name"],
            "diagnosis": diagnosis_text,
            "prompt": validation_prompt,
        })

        # ----- Execution resumes here after Command(resume=...) -----
        decision = decision or {}
        validation_result = str(decision.get("validation_result", "yes"))
        comments = decision.get("comments")

        if validation_result.strip().lower().startswith("no"):
            content = (
                "The previous medical analysis requires further review. "
                "A healthcare professional has flagged potential inaccuracies."
            )
            if comments:
                content += f"\n\n**Reviewer comments:** {comments}"
            final_message = AIMessage(content=content)
        else:
            # Confirmed by validator: keep the diagnosis as the final answer.
            final_message = AIMessage(content=diagnosis_text)

        return {
            **state,
            "output": final_message,
            "messages": final_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response."""
        output = state["output"]
        current_input = state["current_input"]

        # Check if output is valid
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
        
        # If the last message was a human validation message
        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")
            
            # If validation input exists
            if validation_input.lower().startswith(('yes', 'no')):
                # Add the validation result to the conversation history
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
                
                # If validation is 'No', modify the output
                if validation_input.lower().startswith('no'):
                    fallback_message = AIMessage(content="The previous medical analysis requires further review. A healthcare professional has flagged potential inaccuracies.")
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Apply output sanitization
        sanitized_output = guardrails.check_output(output_text, input_text)
        # sanitized_output = output_text

        # Security: redact any leaked system prompt / secrets before returning.
        try:
            from agents.guardrails.injection_filter import scan_output as _scan_output
            _text = sanitized_output if isinstance(sanitized_output, str) else getattr(sanitized_output, "content", str(sanitized_output))
            _text, _leaked = _scan_output(config, _text)
            if _leaked:
                sanitized_output = _text
        except Exception:
            pass

        # Medical safety critic: final-answer review for high-risk medical phrasing.
        safety_verdict = None
        try:
            from agents.guardrails.medical_safety_critic import review_response
            _text = sanitized_output if isinstance(sanitized_output, str) else getattr(sanitized_output, "content", str(sanitized_output))
            safety_verdict = review_response(config, input_text, _text)
            if safety_verdict.get("changed"):
                sanitized_output = safety_verdict.get("revised_response", _text)
        except Exception:
            safety_verdict = None

        # Adaptive test-time deliberation: high-stakes drafts are reviewed by
        # independent specialist personas in parallel and synthesized once. The
        # deterministic medical critic above remains the first safety layer.
        deliberation_report = None
        try:
            from agents.deliberation_agent import deliberate_response
            _text = sanitized_output if isinstance(sanitized_output, str) else getattr(sanitized_output, "content", str(sanitized_output))
            deliberation_report = deliberate_response(
                config, input_text, _text, safety_verdict=safety_verdict
            )
            if deliberation_report.get("triggered"):
                sanitized_output = deliberation_report.get("revised_response", _text)
                try:
                    from services.agent_trace import add_event
                    add_event(
                        "deliberation",
                        reviewers=len(deliberation_report.get("reviews", [])),
                        needs_human_review=deliberation_report.get("needs_human_review", False),
                        reason=deliberation_report.get("reason"),
                    )
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Adaptive deliberation failed open: %s", exc)
            deliberation_report = None

        # For non-validation cases, add the sanitized output to messages
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
        
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message,
            "safety_verdict": safety_verdict,
            "deliberation_report": deliberation_report,
        }

    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    if deep_research_enabled:
        workflow.add_node("DEEP_RESEARCH_AGENT", run_deep_research_agent)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # Connect decision router to agents
    route_targets = {
        "CONVERSATION_AGENT": "CONVERSATION_AGENT",
        "RAG_AGENT": "RAG_AGENT",
        "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
        "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
        "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
        "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
        "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
    }
    if deep_research_enabled:
        route_targets["DEEP_RESEARCH_AGENT"] = "DEEP_RESEARCH_AGENT"
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        route_targets
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")
    if deep_research_enabled:
        workflow.add_edge("DEEP_RESEARCH_AGENT", "check_validation")

    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  # Route to guardrails instead of END
        }
    )
    
    # workflow.add_edge("human_validation", END)
    
    # Compile the graph with an injectable, externalizable checkpointer.
    # Default backend/URL come from environment; falls back to in-memory safely.
    if checkpointer is None:
        checkpointer = build_checkpointer(
            backend=os.getenv("CHECKPOINTER_BACKEND", "memory"),
            redis_url=os.getenv("REDIS_URL"),
        )
    return workflow.compile(checkpointer=checkpointer)


def init_agent_state() -> AgentState:
    """Initialize the agent state with default values."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "user_id": None,
        "user_memory": "",
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False,
        "result_image": None,
        "safety_verdict": None,
        "deliberation_report": None
    }


def get_agent_graph():
    """Return the process-wide compiled graph singleton.

    The graph is stateless; per-conversation state is persisted by the
    checkpointer keyed on thread_id (= session_id). Reusing a single compiled
    graph avoids recompiling the LangGraph on every request (previously O(nodes)
    per call) and lets all concurrent sessions share the same graph safely.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_agent_graph()
    return _compiled_graph


def _finalize_result(result):
    """Post-process a graph invocation result.

    - If the graph paused for human validation (native interrupt), surface the
      validation prompt as the latest AI message and tag ``agent_name`` with a
      ", HUMAN_VALIDATION" suffix so existing callers/UI keep working unchanged.
    - Trim conversation history and echo messages to the console.
    """
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if interrupts:
        try:
            first = interrupts[0]
            payload = first.value if hasattr(first, "value") else (first or {})
        except Exception:
            payload = {}
        payload = payload or {}
        agent = payload.get("agent", "")
        prompt_msg = AIMessage(content=payload.get("prompt", "Human validation required."))
        result["messages"] = list(result.get("messages", [])) + [prompt_msg]
        result["agent_name"] = f"{agent}, HUMAN_VALIDATION" if agent else "HUMAN_VALIDATION"
        result["awaiting_validation"] = True

    # Semantic summarisation (opt-in): compress older history into a recap while
    # keeping recent turns verbatim. Falls back to plain truncation below.
    if isinstance(result, dict) and result.get("messages"):
        try:
            from services.conversation_summary import maybe_summarize
            compressed = maybe_summarize(config, result["messages"], config.conversation.llm)
            if compressed:
                result["messages"] = compressed
        except Exception:
            pass

    if isinstance(result, dict) and result.get("messages") and \
            len(result["messages"]) > config.max_conversation_history:
        result["messages"] = result["messages"][-config.max_conversation_history:]

    if isinstance(result, dict):
        for m in result.get("messages", []):
            if hasattr(m, "pretty_print"):
                m.pretty_print()

    return result


def process_query(query: Union[str, Dict], session_id: str = None, conversation_history: List[BaseMessage] = None, user_id: str = None) -> str:
    """
    Process a user query through the agent decision system.

    Args:
        query: User input (text string or dict with text and image)
        session_id: Conversation/session identifier used as the LangGraph
            thread_id to isolate concurrent users. When absent, a shared
            "default" thread is used (single-user / local behaviour).
        conversation_history: Deprecated; state is persisted by the checkpointer.

    Returns:
        Response from the appropriate agent
    """
    # Use the shared compiled graph singleton (no per-request recompilation).
    graph = get_agent_graph()

    # Initialize state
    state = init_agent_state()

    # Add the current query
    state["current_input"] = query

    # Cross-session long-term memory: attach the user's durable facts so agents
    # can personalise. No-op (empty string) when the feature is off.
    state["user_id"] = user_id
    try:
        from services.long_term_memory import format_for_prompt
        state["user_memory"] = format_for_prompt(config, user_id) if user_id else ""
    except Exception:
        state["user_memory"] = ""

    # To handle image upload case
    if isinstance(query, dict):
        query = query.get("text", "") + ", user uploaded an image for diagnosis."

    state["messages"] = [HumanMessage(content=query)]

    # Per-session thread isolation: each session_id maps to its own checkpointer
    # thread, so concurrent users never share conversation memory.
    thread_id = session_id or "default"
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Begin a structured trace for this request (no-op unless enabled).
    try:
        from services.agent_trace import start_trace, end_trace
        start_trace(session_id=thread_id, query=query if isinstance(query, str) else str(query))
    except Exception:
        end_trace = None

    # Last-resort safety net: an unhandled error in any node (e.g. the LLM
    # provider rejecting content with a 400 data-inspection error) should degrade
    # to a friendly message instead of surfacing a 503 to the user. Per-component
    # graceful degradation handles most cases before reaching here.
    try:
        result = graph.invoke(state, thread_config)
    except Exception as e:
        logger.error(f"process_query graph invocation failed: {e}")
        if end_trace:
            try:
                end_trace(agent="System", status="error")
            except Exception:
                pass
        fallback = AIMessage(content=(
            "I'm sorry, I couldn't complete this request due to a temporary issue "
            "(the content service rejected part of the request). Please rephrase "
            "your question or try again."
        ))
        return {
            "messages": [HumanMessage(content=query), fallback],
            "agent_name": "System",
            "output": fallback,
        }

    # Close the trace with the agent that handled the request.
    if end_trace:
        try:
            end_trace(agent=result.get("agent_name"), status="ok")
        except Exception:
            pass

    # Long-term memory: extract durable facts from this turn (opt-in; fail-open).
    if user_id:
        try:
            from services.long_term_memory import extract_and_store
            user_text = query if isinstance(query, str) else str(query)
            out = result.get("output") if isinstance(result, dict) else None
            assistant_text = getattr(out, "content", None) or (out if isinstance(out, str) else "")
            extract_and_store(config, user_id, user_text, assistant_text, llm=config.conversation.llm)
        except Exception:
            pass

    # Detect pending human-validation interrupt / trim history / echo to console.
    return _finalize_result(result)


def resume_after_validation(session_id: str = None, validation_result: str = "yes", comments: str = None):
    """Resume a graph paused at native human-validation ``interrupt()``.

    Args:
        session_id: Same session/thread used for the original diagnosis request.
        validation_result: "yes"/"no" (or free text starting with yes/no).
        comments: Optional reviewer comments (surfaced when rejected).

    Returns:
        The finalized graph result after the validation decision is applied.
    """
    graph = get_agent_graph()
    thread_id = session_id or "default"
    thread_config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        Command(resume={"validation_result": validation_result, "comments": comments}),
        thread_config,
    )
    return _finalize_result(result)


_standalone_router = None
_standalone_router_tried = False


def _standalone_route(query: str) -> str:
    """Lightweight module-level router used by the streaming path.

    Reuses the same decision model + structured output as the graph router, so
    the streaming decision stays consistent. Returns the agent name, defaulting
    to CONVERSATION_AGENT on any failure (safe: streaming handles that path).
    """
    global _standalone_router, _standalone_router_tried
    try:
        if not _standalone_router_tried:
            _standalone_router_tried = True
            try:
                _standalone_router = config.agent_decision.llm.with_structured_output(AgentDecision)
            except Exception:
                _standalone_router = None
        prompt = (
            AgentConfig.DECISION_SYSTEM_PROMPT
            + f"\n\nUser query: {query}\nHas image: False\n\nWhich agent should handle this query?"
        )
        if _standalone_router is not None:
            res = _standalone_router.invoke(prompt)
            return getattr(res, "agent", "CONVERSATION_AGENT") or "CONVERSATION_AGENT"
    except Exception as e:
        logger.warning("standalone route failed (%s); defaulting to conversation.", e)
    return "CONVERSATION_AGENT"


def stream_conversation_tokens(query: str, user_id: str = None):
    """Yield real, token-level chunks for a plain conversational query.

    This uses the chat model's native ``.stream()`` for true token streaming
    (as opposed to word-chunking a fully-formed answer). It is intentionally
    scoped to the conversation path: it returns ``None`` (no generator) when the
    query needs routing to RAG / web / a vision agent / guardrails, so the caller
    falls back to the non-streaming graph. Fully guarded + fail-open.

    Yields dicts: {"type": "token", "data": "..."} then nothing (caller closes).
    Returns None (not a generator) when it declines to handle the query.
    """
    if not isinstance(query, str) or not query.strip():
        return None

    # Reuse the same fast safety checks the graph's analyze_input runs, so
    # streaming can't bypass injection/emergency guards.
    try:
        from agents.guardrails.injection_filter import scan_input as _scan
        safe, _ = _scan(config, query)
        if not safe:
            return None  # let the graph handle the block message
    except Exception:
        pass
    try:
        from agents.guardrails.emergency_triage import check_red_flags
        if check_red_flags(config, query):
            return None  # emergency must go through the graph's short-circuit
    except Exception:
        pass

    # Only stream when routing says CONVERSATION_AGENT; otherwise decline.
    agent = _standalone_route(query)
    if agent != "CONVERSATION_AGENT":
        return None

    # Build the conversation prompt (memory-aware) and stream it.
    user_memory = ""
    try:
        from services.long_term_memory import format_for_prompt
        user_memory = format_for_prompt(config, user_id) if user_id else ""
    except Exception:
        user_memory = ""

    prompt = (
        f"User query: {query}\n\n{user_memory}\n\n"
        "You are an AI-powered Medical Conversation Assistant. Respond naturally, "
        "accurately and concisely. For serious concerns, advise consulting a "
        "licensed healthcare professional. Do not diagnose or prescribe."
    )

    def _gen():
        collected = []
        try:
            for chunk in config.conversation.llm.stream(prompt):
                piece = getattr(chunk, "content", None)
                if piece:
                    collected.append(piece)
                    yield {"type": "token", "data": piece}
        except Exception as e:
            logger.warning("Token streaming failed mid-way (%s).", e)
            if not collected:
                raise
        # Persist long-term memory from the streamed turn (best-effort).
        try:
            from services.long_term_memory import extract_and_store
            extract_and_store(config, user_id, query, "".join(collected), llm=config.conversation.llm)
        except Exception:
            pass

    return _gen()
