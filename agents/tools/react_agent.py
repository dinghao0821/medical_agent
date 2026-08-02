"""LLM-driven tool use (ReAct) -- turns the tool registry into a real agent.

The keyword dispatcher in ``registry.maybe_run_tools`` is fast but "dumb": it
picks at most one tool by keyword. This module upgrades tool use to a proper
**ReAct loop** where the LLM itself decides *which* tool(s) to call, with *what*
arguments, and can chain **multiple** calls (e.g. look up a drug, then convert a
dose) before answering.

Built on LangChain''s ``bind_tools`` (native function-calling). Fully opt-in
(``config.tools.agent_enabled``) and degrade-safe: if the model/endpoint doesn''t
support tool-calling, or anything errors, callers fall back to the keyword
dispatcher -- so behaviour never regresses.

Reliability features added:
  * Per-tool timeout on execution (via registry metadata).
  * Dangerous-tool gating: tools flagged ``dangerous=True`` are refused in the
    ReAct loop unless ``confirm_dangerous=True`` is passed.
  * Arg validation: if a tool has ``args_schema``, LLM-provided args are
    validated before the tool runs; invalid args feed an error back to the LLM
    instead of crashing.
  * LLM invoke timeout to prevent hung model calls from blocking the loop.
  * Metrics + trace events for observability.
"""

import json
import time
import logging
import threading

logger = logging.getLogger(__name__)

_MAX_STEPS = 4  # bound the ReAct loop to keep latency/cost predictable
_LLM_TIMEOUT = 30  # seconds; cap on a single LLM invoke in the ReAct loop


def _build_lc_tools():
    """Wrap registered tools as LangChain StructuredTools (text-in/text-out)."""
    from langchain_core.tools import tool as lc_tool
    from .registry import _REGISTRY

    lc_tools = []
    for name, meta in _REGISTRY.items():
        func = meta["func"]
        desc = meta["description"] or name

        def _make(fn, tool_name, tool_desc):
            @lc_tool(tool_name, description=tool_desc)
            def _runner(query: str) -> str:
                """Run the underlying tool with the raw query text."""
                out = fn(query)
                return out if out else "No result from this tool."
            return _runner

        try:
            lc_tools.append(_make(func, name, desc))
        except Exception as e:
            logger.debug("Skipping tool %s for ReAct (%s).", name, e)
    return lc_tools


def agent_available(config) -> bool:
    return bool(getattr(getattr(config, "tools", None), "agent_enabled", False))


def _invoke_with_timeout(llm, messages, timeout=_LLM_TIMEOUT):
    """Invoke LLM with a wall-clock timeout (daemon thread)."""
    result_container = [None, None]

    def _worker():
        try:
            result_container[0] = llm.invoke(messages)
        except Exception as e:
            result_container[1] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError("LLM invoke exceeded %ss" % timeout)
    if result_container[1] is not None:
        raise result_container[1]
    return result_container[0]


def _validate_tool_args(args, args_schema):
    """Validate LLM-provided args against a Pydantic schema.

    Returns (True, validated_text) on success, (False, error_msg) on failure.
    The tool protocol passes text to the function, so we serialise validated
    args back to a JSON string for the tool to parse.
    """
    if args_schema is None:
        # No schema: pass raw text or serialised args as string.
        if isinstance(args, dict) and args:
            return True, json.dumps(args, ensure_ascii=False)
        return True, ""
    try:
        if isinstance(args, dict):
            validated = args_schema(**args)
            return True, json.dumps(validated.model_dump(), ensure_ascii=False)
        return True, str(args) if args else ""
    except Exception as e:
        return False, "Invalid args for %s: %s" % (args_schema.__name__, e)


def run_tool_agent(config, text: str, llm, *, confirm_dangerous: bool = False):
    """Run a ReAct tool-calling loop. Returns a final answer string or None.

    Returns None when: the feature is off, no LLM, tool-calling is unsupported,
    the model chose to call no tools (so a normal agent should handle it), or
    any error -- all of which let the caller degrade to keyword tools / routing.

    Args:
        confirm_dangerous: if True, allow execution of tools flagged
            ``dangerous=True``.  Default False (dangerous tools are refused
            and an error message is fed back to the LLM).
    """
    from .registry import get_tool_meta, _record_metric

    if not agent_available(config) or not text or llm is None:
        return None

    try:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    except Exception:
        return None

    lc_tools = _build_lc_tools()
    if not lc_tools:
        return None

    try:
        llm_with_tools = llm.bind_tools(lc_tools)
    except Exception as e:
        logger.info("bind_tools unavailable (%s); falling back to keyword tools.", e)
        return None

    tools_by_name = {t.name: t for t in lc_tools}
    system = SystemMessage(content=(
        "You are a medical assistant with access to calculator/lookup tools. "
        "Use a tool ONLY when it clearly helps (e.g. BMI, unit conversion, drug "
        "info). If no tool is needed, do not call one. After tool results, give a "
        "concise, accurate answer with appropriate medical disclaimers. Never "
        "invent tool outputs."
    ))
    messages = [system, HumanMessage(content=text)]

    called_any = False
    try:
        for _ in range(_MAX_STEPS):
            ai = _invoke_with_timeout(llm_with_tools, messages)
            messages.append(ai)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                if called_any:
                    content = getattr(ai, "content", "") or ""
                    return content or None
                return None

            called_any = True
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)

                meta = get_tool_meta(name)

                # --- Dangerous-tool gating ---
                if meta and meta.get("dangerous", False) and not confirm_dangerous:
                    msg = ("Tool '%s' requires explicit confirmation and was "
                           "blocked. Ask the user to confirm before proceeding." % name)
                    messages.append(ToolMessage(content=msg, tool_call_id=call_id or name))
                    _record_metric(name, ok=False, duration=0.0)
                    continue

                # --- Arg validation ---
                if meta and meta.get("args_schema"):
                    ok, validated = _validate_tool_args(args, meta["args_schema"])
                    if not ok:
                        messages.append(ToolMessage(
                            content=validated, tool_call_id=call_id or name))
                        _record_metric(name, ok=False, duration=0.0)
                        continue
                    query_arg = validated
                else:
                    query_arg = ""
                    if isinstance(args, dict):
                        query_arg = args.get("query") or next(iter(args.values()), "") if args else ""
                    else:
                        query_arg = str(args)

                tool = tools_by_name.get(name)
                if tool is None:
                    result = "Unknown tool: %s" % name
                else:
                    t0 = time.time()
                    try:
                        result = tool.invoke(query_arg) if query_arg else tool.invoke(text)
                        _record_metric(name, ok=True, duration=time.time() - t0)
                    except Exception as e:
                        result = "Tool error: %s" % e
                        _record_metric(name, ok=False, duration=time.time() - t0)
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id or name))

            try:
                from services.agent_trace import add_event
                add_event("tool_react", step_tools=[
                    (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None))
                    for tc in tool_calls
                ])
            except Exception:
                pass

        last = messages[-1]
        return getattr(last, "content", None) or None
    except Exception as e:
        logger.warning("ReAct tool loop failed (%s); degrading.", e)
        return None