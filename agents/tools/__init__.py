"""Structured tool-calling framework for the medical assistant.

Replaces the previous "fixed set of agents" limitation with a small, extensible
registry of callable tools (calculators, drug-info lookup, ...). New tools are
added by writing a function and decorating it with ``@register_tool`` -- no
graph surgery required.

The framework is deliberately dependency-free and deterministic (no external
APIs, no LLM needed to *run* a tool). Tool *selection* can be done by simple
keyword matching (offline) or by an LLM if one is provided.

Opt-in via ``config.tools.enabled``; when disabled, ``maybe_run_tools`` returns
None and behaviour is unchanged.

Reliability: every tool call is wrapped with timeout, retry (idempotent only),
arg validation, and metric recording.  See ``registry.get_metrics`` for
observability and ``registry.get_tool_meta`` for per-tool metadata.
"""

from .registry import (
    register_tool, get_tools, get_tool_meta, run_tool, maybe_run_tools,
    get_metrics, reset_metrics,
)
from .react_agent import run_tool_agent, agent_available
from . import builtin  # noqa: F401  (import registers the built-in tools)

__all__ = [
    "register_tool", "get_tools", "get_tool_meta", "run_tool", "maybe_run_tools",
    "get_metrics", "reset_metrics",
    "run_tool_agent", "agent_available",
]