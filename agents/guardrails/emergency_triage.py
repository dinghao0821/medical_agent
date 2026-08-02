"""Emergency red-flag triage (medical-safety agent enhancement).

Detects descriptions of potentially life-threatening ("red-flag") symptoms and
returns an urgent message advising immediate care, *before* the query is routed
to a normal informational agent. This is a safety-critical feature for a medical
assistant: some presentations (stroke, heart attack, anaphylaxis, suicidal
ideation) must never be met with a leisurely informational answer.

Pure local keyword/heuristic matching — no LLM, no deps, instantaneous.
Enabled by default (``config.triage.enabled``); fail-open on any error.
"""

import re
import logging

logger = logging.getLogger(__name__)

# category -> (compiled patterns, guidance)
_RED_FLAGS = {
    "cardiac": (
        [r"chest pain", r"crushing.*chest", r"pressure in (?:my )?chest",
         r"pain.*(?:radiat|spread).*(?:arm|jaw)", r"can'?t breathe.*chest",
         r"胸痛", r"胸闷", r"心绞痛", r"冷汗.*胸", r"胸.*冒冷汗"],
        "chest pain or suspected heart attack",
    ),
    "stroke": (
        [r"face.*droop", r"slurred speech", r"sudden.*(?:numb|weakness).*(?:one side|face|arm)",
         r"can'?t (?:speak|move) (?:one|my) (?:side|arm|leg)", r"sudden.*vision loss",
         r"中风", r"卒中", r"嘴角歪", r"半身.*(?:无力|麻木)", r"突然.*说不出话"],
        "signs of a possible stroke (FAST: Face, Arms, Speech, Time)",
    ),
    "breathing": (
        [r"can'?t breathe", r"cannot breathe", r"struggling to breathe",
         r"severe.*(?:short(?:ness)? of breath)", r"turning blue", r"choking",
         r"呼吸困难", r"喘不上气", r"透不过气"],
        "severe difficulty breathing",
    ),
    "anaphylaxis": (
        [r"throat.*(?:closing|swelling)", r"tongue.*swell", r"anaphylaxis",
         r"severe allergic reaction", r"hives.*(?:breath|swall)"],
        "a possible severe allergic reaction (anaphylaxis)",
    ),
    "bleeding": (
        [r"bleeding.*(?:won'?t|not) stop", r"heavy bleeding", r"coughing up blood",
         r"vomiting blood"],
        "severe or uncontrolled bleeding",
    ),
    "overdose": (
        [r"overdos", r"took too many.*(?:pill|tablet|medic)", r"too many.*(?:pill|sleeping pill)",
         r"poison", r"swallowed.*(?:bleach|chemical)",
         r"服药过量", r"吃.*太多.*(?:药|安眠)", r"中毒"],
        "a possible overdose or poisoning",
    ),
    "mental_health": (
        [r"kill myself", r"suicid", r"end my life", r"want to die", r"self[- ]harm",
         r"hurt myself"],
        "thoughts of self-harm or suicide",
    ),
    "neuro": (
        [r"worst headache of my life", r"sudden.*severe.*headache", r"unconscious",
         r"passed out", r"seizure", r"convulsion"],
        "a severe neurological emergency",
    ),
}

_COMPILED = {
    cat: ([re.compile(p, re.IGNORECASE) for p in pats], guidance)
    for cat, (pats, guidance) in _RED_FLAGS.items()
}

_CRISIS_NOTE = (
    "\n\nIf you are in immediate danger or thinking about harming yourself, please "
    "contact your local emergency number now, or reach a suicide-prevention hotline "
    "in your country. You are not alone and help is available."
)


def _enabled(config) -> bool:
    return bool(getattr(getattr(config, "triage", None), "enabled", False))


def check_red_flags(config, text: str):
    """Return an urgent-care message if a red-flag symptom is detected, else None."""
    if not text or not _enabled(config):
        return None
    try:
        for cat, (patterns, guidance) in _COMPILED.items():
            for rx in patterns:
                if rx.search(text):
                    logger.warning("[Triage] Red-flag detected: %s", cat)
                    msg = (
                        f"⚠️ **This may be a medical emergency.** Your message mentions "
                        f"{guidance}. Please **seek emergency medical help immediately** — "
                        f"call your local emergency number (e.g. 911/112/120) or go to the "
                        f"nearest emergency department now.\n\n"
                        f"I'm an AI assistant and cannot provide emergency care. Do not wait "
                        f"for an online answer if symptoms are severe or worsening."
                    )
                    if cat == "mental_health":
                        msg += _CRISIS_NOTE
                    return msg
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[Triage] check_red_flags error (%s); skipping.", e)
    return None
