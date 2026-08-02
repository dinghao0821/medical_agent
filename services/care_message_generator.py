"""LLM-driven message generation for the family-care companion channel.

Two responsibilities:
  1. Generate the proactive opening check-in message for a due reminder task,
     styled per ``task_type`` and optionally personalised with the elder's
     long-term-memory facts (e.g. known medications).
  2. Generate a warm, natural AI reply to the elder's response in the care
     channel — this is a plain conversational reply, NOT a diagnosis; risk
     detection is handled separately by ``services.care_risk_detector``.

Both functions are fail-open: any LLM error falls back to a simple templated
message so the companion feature never crashes a scheduler tick or a reply.
"""

import logging
from typing import Optional

from services.family_care_service import TASK_TYPE_GUIDANCE, TASK_TYPE_LABELS

logger = logging.getLogger(__name__)

_FALLBACK_OPENERS = {
    "medication": "到吃药的时间啦，您吃了吗？身体感觉怎么样？",
    "mood": "今天感觉怎么样呀？心情好吗？",
    "meal": "到饭点啦，今天吃了什么好吃的？",
    "safety_checkin": "今天有没有下楼走走呀？家里一切都还好吧？",
    "follow_up": "别忘了这次的复诊安排哦，需要家人陪您一起去吗？",
    "custom": "",
}


def generate_checkin_message(config, elder_username: str, task_type: str, custom_prompt: Optional[str] = None) -> str:
    """Generate the opening proactive message for a due reminder task."""
    guidance = TASK_TYPE_GUIDANCE.get(task_type, TASK_TYPE_GUIDANCE["custom"])
    memory_block = ""
    try:
        from services.long_term_memory import format_for_prompt
        memory_block = format_for_prompt(config, elder_username) or ""
    except Exception:
        memory_block = ""

    extra = f"\n家属设置的具体提醒内容：{custom_prompt}" if task_type == "custom" and custom_prompt else ""

    prompt = (
        f"{guidance}\n"
        f"{memory_block}\n"
        f"{extra}\n"
        f"请用一到两句自然、温暖的中文向这位老年人发起问候/提醒，不要使用书面语或列表，"
        f"像家人说话一样简短亲切，不要输出任何前缀或解释，只输出要说的话本身。"
    )
    try:
        llm = config.conversation.llm
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        text = (text or "").strip()
        if text:
            return text
    except Exception as e:
        logger.warning("[CareMessage] LLM opener generation failed (%s); using fallback.", e)
    return _FALLBACK_OPENERS.get(task_type, "") or (custom_prompt or "家人让我来问候您一下，最近还好吗？")


def generate_reply_response(
    config, elder_username: str, task_type: Optional[str], ai_message: str, elder_reply: str
) -> str:
    """Generate the AI's warm follow-up reply to what the elder said."""
    memory_block = ""
    try:
        from services.long_term_memory import format_for_prompt
        memory_block = format_for_prompt(config, elder_username) or ""
    except Exception:
        memory_block = ""

    label = TASK_TYPE_LABELS.get(task_type or "", "日常关怀")
    prompt = (
        f"你是一位对老年人非常有耐心、温暖体贴的AI陪伴助手，本轮话题是：{label}。\n"
        f"{memory_block}\n"
        f"你刚才对老人说：{ai_message}\n"
        f"老人回复：{elder_reply}\n\n"
        f"请用一到两句自然、口语化的中文回应老人，语气温暖、有耐心，"
        f"如果老人提到不适、负面情绪或异常情况，要表达关心并温和建议其告知家人或就医，"
        f"但不要说教、不要输出任何医疗诊断，也不要输出前缀或解释，只输出要说的话本身。"
    )
    try:
        llm = config.conversation.llm
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        text = (text or "").strip()
        if text:
            return text
    except Exception as e:
        logger.warning("[CareMessage] LLM reply generation failed (%s); using fallback.", e)
    return "谢谢您告诉我，我会记在心里的。要是感觉不舒服，一定要及时跟家人说哦。"
