"""Lightweight, local risk-signal detection for the family-care companion channel.

When an elder replies to a proactive AI check-in, this module scans the reply
for concern signals (missed medication, low mood, physical discomfort, falls,
loneliness) using pure keyword/regex heuristics — no LLM call, instantaneous,
zero cost, and fail-open so a detector bug never blocks a normal reply.

Distinct from ``agents/guardrails/emergency_triage.py`` (which flags true
medical emergencies mid-conversation): this module flags *caregiver-relevant
concern* which is a lower bar — worth notifying a family member, not
necessarily worth calling emergency services.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# category -> (patterns, human-readable summary template)
_CONCERN_PATTERNS = {
    "missed_medication": (
        [r"没吃药", r"忘了吃药", r"没有吃药", r"忘记吃药", r"药吃完了", r"没有药了",
         r"\bforgot.*medic", r"\bdidn'?t take.*medic"],
        "可能未按时服药",
    ),
    "physical_discomfort": (
        [r"不舒服", r"疼", r"头晕", r"胸闷", r"喘不过气", r"没力气", r"发烧", r"呕吐",
         r"\bhurts?\b", r"\bpain\b", r"\bdizzy\b", r"\bnot feeling well\b"],
        "提到身体不适",
    ),
    "fall_or_injury": (
        [r"摔倒", r"摔了", r"跌倒", r"磕到", r"受伤",
         r"\bfell\b", r"\bfall(?:en)?\b.*(?:down|floor)"],
        "可能发生跌倒或受伤",
    ),
    "low_mood": (
        [r"不开心", r"难过", r"孤独", r"没人陪", r"想哭", r"心情不好", r"太累了",
         r"没意思", r"活着没意思",
         r"\blonely\b", r"\bsad\b", r"\bdepress"],
        "情绪低落或孤独感",
    ),
    "no_food": (
        [r"没吃饭", r"不想吃", r"没胃口", r"吃不下",
         r"\bnot eating\b", r"\bno appetite\b"],
        "食欲不佳或未进食",
    ),
}

_COMPILED = {
    cat: ([re.compile(p, re.IGNORECASE) for p in pats], summary)
    for cat, (pats, summary) in _CONCERN_PATTERNS.items()
}


def detect_concern(text: str) -> Optional[dict]:
    """Scan an elder's reply for concern signals.

    Returns ``{"category": str, "summary": str}`` for the first match, or
    ``None`` when no concern signal is found. Fail-open: any internal error
    is treated as "no concern detected" rather than raising.
    """
    if not text:
        return None
    try:
        for cat, (patterns, summary) in _COMPILED.items():
            for rx in patterns:
                if rx.search(text):
                    return {"category": cat, "summary": summary}
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[CareRisk] detect_concern error (%s); skipping.", e)
    return None
