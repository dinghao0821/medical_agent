"""Proactive multi-turn clarification (agent enhancement).

When a user describes a symptom too vaguely to give a useful answer (e.g. "I feel
bad", "I have pain", "something's wrong"), a good clinician asks focused
follow-up questions before advising. This module detects such vague, under-
specified symptom messages and returns a short clarifying question instead of
letting the LLM guess.

Local heuristics only (no LLM, no deps). Opt-in via ``config.clarification``;
conservative by design to avoid interrogating clearly-specified queries.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Vague symptom cues that, when present *without* specifics, warrant a follow-up.
_VAGUE_PATTERNS = [
    r"\bi (?:feel|am feeling) (?:bad|unwell|sick|off|weird|strange|terrible|awful)\b",
    r"\bsomething(?:'s| is) wrong\b",
    r"\bi (?:have|got) (?:a )?pain\b",
    r"\bi(?:'m| am) in pain\b",
    r"\bi (?:don'?t feel|do not feel) (?:good|well|right)\b",
    r"\bi(?:'m| am) not feeling (?:good|well|right)\b",
    r"\bit hurts\b",
    r"\bi(?:'m| am) sick\b",
]
_COMPILED_VAGUE = [re.compile(p, re.IGNORECASE) for p in _VAGUE_PATTERNS]

# Specificity signals — if present, the message is probably detailed enough.
_SPECIFIC_CUES = [
    r"\b\d+\s*(?:day|week|month|year|hour|min)", r"\bsince\b", r"\bfor the (?:past|last)\b",
    r"\b(?:head|chest|stomach|abdomen|back|throat|leg|arm|knee|ear|eye|tooth)\b",
    r"\bfever\b", r"\bcough\b", r"\bnausea\b", r"\bvomit", r"\brash\b", r"\bdizz",
    r"\bsharp\b", r"\bdull\b", r"\bthrobbing\b", r"\bburning\b",
]
_COMPILED_SPECIFIC = [re.compile(p, re.IGNORECASE) for p in _SPECIFIC_CUES]


def _enabled(config) -> bool:
    return bool(getattr(getattr(config, "clarification", None), "enabled", False))


def needs_clarification(config, text: str):
    """Return a clarifying-question message if the input is too vague, else None."""
    if not text or not _enabled(config):
        return None
    try:
        low = text.strip()
        # Only consider short-ish messages; long ones usually carry detail.
        if len(low) > 200:
            return None
        is_vague = any(rx.search(low) for rx in _COMPILED_VAGUE)
        if not is_vague:
            return None
        # If the message already contains specific cues, don't interrogate.
        has_specifics = any(rx.search(low) for rx in _COMPILED_SPECIFIC)
        if has_specifics:
            return None
        logger.info("[Clarification] Vague symptom detected; asking follow-up.")
        return (
            "I want to help accurately. Could you share a few more details so I can "
            "give useful information?\n\n"
            "- **Where** exactly is the problem (body area)?\n"
            "- **How long** have you had it, and is it getting better or worse?\n"
            "- **How would you describe it** (e.g. sharp, dull, throbbing) and how severe (1–10)?\n"
            "- Any **other symptoms** (fever, nausea, dizziness, etc.)?\n\n"
            "_If your symptoms are severe or sudden, please seek medical care right away._"
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[Clarification] error (%s); skipping.", e)
        return None
