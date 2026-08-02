"""Session & state-management utilities for the multi-agent system.

Provides pluggable LangGraph checkpointer backends (in-memory / Redis) so that
conversation state can be shared across multiple worker processes / replicas,
which is a prerequisite for horizontal scaling and native human-in-the-loop.
"""

from .checkpointer_factory import build_checkpointer

__all__ = ["build_checkpointer"]
