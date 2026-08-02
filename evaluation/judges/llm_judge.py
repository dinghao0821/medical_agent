"""LLM-as-Judge 通用封装（用于开放文本质量评判：忠实度、有用性、引用支持度）。

设计要点（见 EVALUATION_DESIGN.md 第四节）：
  - 固定 rubric，结构化输出 {score, reason}
  - judge 模型经配置层，可与被测模型不同以减少自我偏好
  - 不可用时返回 score=None（标记 n/a），不污染门禁

judge 不是评测的强依赖：纯规则套件完全不需要它。
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("llm_judge")


# 各维度评分 rubric（1-5 分）
RUBRICS = {
    "faithfulness": (
        "评估回答是否完全基于提供的上下文，没有臆造。\n"
        "5=所有关键陈述都有上下文支持；3=部分陈述缺乏支持；1=大量无依据臆造。"
    ),
    "helpfulness": (
        "评估回答对用户问题的有用性和完整性。\n"
        "5=直接、完整、可操作；3=部分回答或含糊；1=答非所问或空洞。"
    ),
    "citation_support": (
        "评估被引用的来源是否确实支持其标注的陈述。\n"
        "5=所有引用都真实支持对应陈述；3=部分引用不匹配；1=引用与陈述无关。"
    ),
}


def _get_judge_llm(config):
    """取 judge 模型；优先独立配置，退回对话模型。judge 与被测模型可不同。"""
    ev = getattr(config, "evaluation", None)
    llm = getattr(ev, "judge_llm", None) if ev else None
    if llm is not None:
        return llm
    conv = getattr(config, "conversation", None)
    return getattr(conv, "llm", None)


def judge(config, dimension: str, question: str, answer: str,
          context: str = "") -> Dict[str, Any]:
    """对单个 (question, answer[, context]) 在指定维度上打分。

    返回 {"score": float|None, "reason": str, "dimension": str}
    score 为 None 表示 judge 不可用（n/a），不参与阈值判定。
    """
    rubric = RUBRICS.get(dimension)
    if rubric is None:
        return {"score": None, "reason": f"unknown dimension {dimension}", "dimension": dimension}

    llm = _get_judge_llm(config)
    if llm is None:
        return {"score": None, "reason": "judge llm unavailable", "dimension": dimension}

    prompt = f"""你是严格中立的医疗 AI 评审。请只依据给定信息按 rubric 打分（1-5 整数）。
不要偏好更长或更靠前的答案。只输出 JSON: {{"score": <1-5>, "reason": "<简短依据>"}}。

评分维度: {dimension}
Rubric:
{rubric}

用户问题:
{question}

{("参考上下文:\n" + context) if context else ""}

待评回答:
{answer}
"""
    try:
        # 优先结构化输出；失败则退回文本解析
        from pydantic import BaseModel

        class _JudgeOut(BaseModel):
            score: int
            reason: str = ""

        try:
            out = llm.with_structured_output(_JudgeOut).invoke(prompt)
            return {"score": float(out.score), "reason": out.reason, "dimension": dimension}
        except Exception:
            raw = llm.invoke(prompt)
            text = getattr(raw, "content", str(raw))
            import json
            import re
            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0)) if m else {}
            return {
                "score": float(data.get("score")) if data.get("score") is not None else None,
                "reason": str(data.get("reason", "")),
                "dimension": dimension,
            }
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("judge failed (%s)", e)
        return {"score": None, "reason": f"judge error: {e}", "dimension": dimension}
