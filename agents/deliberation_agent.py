"""Adaptive multi-agent deliberation for high-stakes medical answers.

The workflow uses test-time compute only when a deterministic risk gate decides
that a draft is medically consequential. Independent specialist perspectives
then critique the draft in parallel before a synthesizer produces the final
answer. It is opt-in and fail-open: low-risk answers and provider failures keep
the original response unchanged.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field


_HIGH_STAKES_PATTERNS = (
    r"\b(diagnos(?:is|e)|prescri(?:be|ption)|dosage|dose|contraindicat|interaction|surgery|emergency)\b",
    r"诊断|处方|剂量|用药|停药|换药|手术|禁忌|相互作用|急诊|急救|胸痛|呼吸困难|意识不清|卒中|中风",
)


class SpecialistReview(BaseModel):
    role: str = Field(description="Specialist perspective used for the review")
    risk_level: str = Field(description="low, medium, high, or critical")
    concerns: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    recommended_changes: List[str] = Field(default_factory=list)


class DeliberationSynthesis(BaseModel):
    revised_response: str = Field(description="Final medically cautious response")
    uncertainty: str = Field(default="", description="Important residual uncertainty")
    needs_human_review: bool = Field(default=False)


def _content(value: Any) -> str:
    return str(getattr(value, "content", value) or "")


def should_deliberate(config, user_text: str, draft: str, safety_verdict=None) -> bool:
    cfg = getattr(config, "deliberation", None)
    if not cfg or not getattr(cfg, "enabled", False) or not (draft or "").strip():
        return False
    verdict = (safety_verdict or {}).get("verdict", "")
    if verdict in {"caution", "unsafe", "emergency"}:
        return True
    combined = f"{user_text or ''}\n{draft or ''}"
    return any(re.search(pattern, combined, re.IGNORECASE) for pattern in _HIGH_STAKES_PATTERNS)


def _review_one(llm, role: str, user_text: str, draft: str) -> SpecialistReview:
    prompt = f"""You are the {role} in a medical AI deliberation panel.
Review the draft without diagnosing the patient or inventing facts. Identify
unsafe certainty, missing context, evidence gaps, contraindications, and when
professional or urgent care is needed. Return a concise structured review.

User request:
{user_text}

Draft answer:
{draft}
"""
    try:
        structured = llm.with_structured_output(SpecialistReview)
        review = structured.invoke(prompt)
        review.role = review.role or role
        return review
    except Exception as exc:
        return SpecialistReview(
            role=role,
            risk_level="unknown",
            concerns=[f"review unavailable: {type(exc).__name__}"],
        )


def deliberate_response(config, user_text: str, draft: str, safety_verdict=None) -> Dict[str, Any]:
    """Run adaptive specialist debate and return a structured audit record."""
    original = draft or ""
    if not should_deliberate(config, user_text, original, safety_verdict):
        return {
            "triggered": False,
            "revised_response": original,
            "reviews": [],
            "needs_human_review": bool((safety_verdict or {}).get("needs_human_review", False)),
            "reason": "low_risk_or_disabled",
        }

    cfg = config.deliberation
    llm = getattr(config.agent_decision, "llm", None) or config.conversation.llm
    roles = list(getattr(cfg, "roles", [])) or [
        "geriatric medicine safety reviewer",
        "clinical pharmacist and contraindication reviewer",
        "evidence and uncertainty reviewer",
    ]
    max_reviewers = max(2, min(int(getattr(cfg, "max_reviewers", 3)), len(roles)))
    selected_roles = roles[:max_reviewers]
    ordered = [None] * len(selected_roles)

    with ThreadPoolExecutor(max_workers=max_reviewers) as executor:
        futures = {
            executor.submit(_review_one, llm, role, user_text, original): index
            for index, role in enumerate(selected_roles)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                ordered[index] = future.result()
            except Exception as exc:
                ordered[index] = SpecialistReview(
                    role=selected_roles[index], risk_level="unknown",
                    concerns=[f"review unavailable: {type(exc).__name__}"],
                )

    reviews = [review for review in ordered if review is not None]
    if not reviews or all(
        review.risk_level == "unknown"
        and review.concerns
        and all(str(concern).startswith("review unavailable:") for concern in review.concerns)
        for review in reviews
    ):
        return {
            "triggered": True,
            "revised_response": original,
            "reviews": [review.model_dump() for review in reviews],
            "uncertainty": "",
            "needs_human_review": bool((safety_verdict or {}).get("needs_human_review", False)),
            "reason": "all_reviews_unavailable",
        }

    review_text = "\n\n".join(
        f"[{r.role}] risk={r.risk_level}; concerns={r.concerns}; "
        f"missing={r.missing_information}; changes={r.recommended_changes}"
        for r in reviews
    )
    synthesis_prompt = f"""You are the chair of a medical AI review panel.
Revise the draft using the specialist reviews. Preserve useful content, remove
unsupported certainty, clearly separate general information from personalized
medical advice, state key uncertainty, and recommend clinician or emergency
care only when justified. Never invent patient facts or citations. Return a
structured result in the same language as the user.

User request:
{user_text}

Original draft:
{original}

Specialist reviews:
{review_text}
"""
    try:
        synthesis = llm.with_structured_output(DeliberationSynthesis).invoke(synthesis_prompt)
        revised = (synthesis.revised_response or original).strip()
        needs_human = bool(synthesis.needs_human_review)
        uncertainty = synthesis.uncertainty
        reason = "specialist_consensus"
    except Exception as exc:
        revised = original
        needs_human = bool((safety_verdict or {}).get("needs_human_review", False))
        uncertainty = ""
        reason = f"synthesis_failed:{type(exc).__name__}"

    return {
        "triggered": True,
        "revised_response": revised,
        "reviews": [review.model_dump() for review in reviews],
        "uncertainty": uncertainty,
        "needs_human_review": needs_human,
        "reason": reason,
    }
