"""Deterministic medical safety critic for final assistant responses.

This is an opt-in, fail-open guardrail that reviews generated medical answers
for high-risk language before they are returned to users. It does not replace a
licensed clinician; it simply makes risky model phrasing safer by adding clear
limitations, emergency guidance, and clinician-review language.
"""

import re
from typing import Dict


_EMERGENCY_PATTERNS = [
    r"\b(crushing|severe) chest pain\b",
    r"\b(chest pain).*(arm|jaw|shortness of breath|sweat)",
    r"\b(stroke|face droop|slurred speech|one-sided weakness)\b",
    r"\b(can'?t breathe|severe shortness of breath)\b",
    r"\b(anaphylaxis|throat swelling)\b",
    r"\b(suicidal|kill myself|self[- ]harm)\b",
    r"(剧烈胸痛|胸痛.*(左臂|下颌|呼吸困难)|中风|口齿不清|偏瘫|无法呼吸|自杀|轻生)",
]

# Imperative prescription / diagnosis patterns that are too authoritative for
# an AI assistant, especially in a general chat setting.
_UNSAFE_PATTERNS = [
    r"\bmust\s+take\b",
    r"\bshould\s+take\s+[a-zA-Z]+\s*\d+\s*mg\b",
    r"\btake\s+[a-zA-Z]+\s*\d+\s*mg\b",
    r"\bstart\s+(?:taking\s+)?[a-zA-Z]+\s*\d+\s*mg\b",
    r"\bincrease\s+your\s+dose\b",
    r"\bstop\s+taking\s+your\s+\w+\b",
    r"\bdefinitely\s+(?:is|have|has)\b",
    r"\bdiagnosis\s+is\b",
    r"(必须服用|直接服用|立即服用).*(mg|毫克|片)",
    r"(诊断就是|一定是|肯定是)",
]

_CAUTION_PATTERNS = [
    r"\bprobably\s+(?:cancer|tumou?r|stroke|heart attack)\b",
    r"\b(?:cancer|tumou?r|stroke|heart attack)\b",
    r"(癌症|肿瘤|中风|心梗|心肌梗死)",
]

_BASE_DISCLAIMER = (
    "\n\n---\n"
    "**医疗安全提示：** 以上内容仅供健康科普和就医沟通参考，不能替代执业医生的面对面诊断、处方或治疗方案。"
)

_CLINICIAN_REVIEW = (
    "如涉及药物剂量、抗生素、处方药调整、停药或明确诊断，请先咨询 licensed clinician / 执业医生或药师。"
)

_EMERGENCY_NOTICE = (
    "如果出现胸痛伴呼吸困难/放射痛、疑似中风、严重过敏、无法呼吸、自伤风险等急症信号，请立即拨打当地急救电话或前往急诊。"
)


def _enabled(config) -> bool:
    return bool(getattr(getattr(config, "medical_safety", None), "enabled", False))


def _matches_any(patterns, text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE | re.DOTALL) for p in patterns)


def review_response(config, user_text: str, assistant_text: str) -> Dict[str, object]:
    """Review a generated response and return a structured verdict.

    Return keys:
        verdict: skipped | safe | caution | unsafe | emergency
        changed: whether revised_response differs from assistant_text
        reason: short machine-readable reason
        revised_response: text to show to user
        needs_human_review: whether a clinician/doctor review is advisable
    """
    original = assistant_text or ""
    if not _enabled(config) or not original.strip():
        return {
            "verdict": "skipped",
            "changed": False,
            "reason": "disabled_or_empty",
            "revised_response": original,
            "needs_human_review": False,
        }

    combined = f"{user_text or ''}\n{original}"

    if _matches_any(_EMERGENCY_PATTERNS, combined):
        revised = original
        if _EMERGENCY_NOTICE not in revised:
            revised = f"{_EMERGENCY_NOTICE}\n\n{revised}"
        if "医疗安全提示" not in revised:
            revised += _BASE_DISCLAIMER
        return {
            "verdict": "emergency",
            "changed": revised != original,
            "reason": "emergency_red_flag",
            "revised_response": revised,
            "needs_human_review": True,
        }

    if _matches_any(_UNSAFE_PATTERNS, original):
        revised = original
        if _CLINICIAN_REVIEW not in revised:
            revised += f"\n\n{_CLINICIAN_REVIEW}"
        if "医疗安全提示" not in revised:
            revised += _BASE_DISCLAIMER
        return {
            "verdict": "unsafe",
            "changed": revised != original,
            "reason": "over_authoritative_prescription_or_diagnosis",
            "revised_response": revised,
            "needs_human_review": True,
        }

    if _matches_any(_CAUTION_PATTERNS, original):
        revised = original
        if "医疗安全提示" not in revised:
            revised += _BASE_DISCLAIMER
        return {
            "verdict": "caution",
            "changed": revised != original,
            "reason": "high_stakes_medical_topic",
            "revised_response": revised,
            "needs_human_review": False,
        }

    return {
        "verdict": "safe",
        "changed": False,
        "reason": "no_high_risk_pattern",
        "revised_response": original,
        "needs_human_review": False,
    }
