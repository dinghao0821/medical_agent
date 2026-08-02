"""Semantic conversation summarisation (agent enhancement).

Long chats blow up token cost and eventually exceed context windows. The
existing code simply truncates to the last N messages, which silently *loses*
earlier context (e.g. a diagnosis stated 30 messages ago). This module instead
summarises the older part of the conversation into a compact recap and keeps the
recent messages verbatim — preserving continuity at a fraction of the tokens.

Opt-in via ``config.summary.enabled``; fail-open (falls back to plain truncation
on any error, exactly matching prior behaviour).
"""

import logging

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = (
    "Summarise the following medical assistant conversation into a concise recap "
    "(<=150 words) that preserves clinically relevant facts: the user's reported "
    "symptoms, any diagnoses/analysis results, advice already given, and open "
    "questions. Omit greetings and small talk. Write in third person.\n\n"
    "CONVERSATION:\n{conversation}\n\nRECAP:"
)


def _enabled(config) -> bool:
    return bool(getattr(getattr(config, "summary", None), "enabled", False))


def _render(messages) -> str:
    lines = []
    for m in messages:
        role = m.__class__.__name__.replace("Message", "")
        content = getattr(m, "content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def maybe_summarize(config, messages, llm):
    """Return a possibly-compressed message list.

    If summarisation is enabled and the history exceeds the trigger threshold,
    the older messages are replaced by a single ``SystemMessage`` recap while the
    most recent messages are kept verbatim. Otherwise returns ``messages``
    unchanged (caller keeps its existing truncation behaviour).
    """
    if not _enabled(config) or not messages or llm is None:
        return None
    try:
        cfg = config.summary
        trigger = int(getattr(cfg, "trigger_messages", 20))
        if len(messages) <= trigger:
            return None

        from langchain_core.messages import SystemMessage

        keep = max(4, trigger // 2)          # keep the most recent messages verbatim
        older = messages[:-keep]
        recent = messages[-keep:]

        recap = llm.invoke(_SUMMARY_PROMPT.format(conversation=_render(older)))
        recap_text = getattr(recap, "content", None) or str(recap)
        if not recap_text.strip():
            return None

        summary_msg = SystemMessage(content=f"[Conversation recap] {recap_text.strip()}")
        logger.info("Summarised %d older messages into a recap.", len(older))
        return [summary_msg] + list(recent)
    except Exception as e:
        logger.warning("Conversation summarisation failed (%s); keeping full history.", e)
        return None
