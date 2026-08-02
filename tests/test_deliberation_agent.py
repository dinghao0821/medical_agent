"""Tests for adaptive high-stakes multi-agent deliberation."""

import threading
import time

from agents.deliberation_agent import (
    DeliberationSynthesis,
    SpecialistReview,
    deliberate_response,
    should_deliberate,
)


class _Structured:
    def __init__(self, schema, owner):
        self.schema = schema
        self.owner = owner

    def invoke(self, prompt):
        if self.schema is SpecialistReview:
            self.owner.threads.add(threading.get_ident())
            time.sleep(0.02)
            role = str(prompt).split("You are the ", 1)[1].split(" in a medical", 1)[0]
            return SpecialistReview(
                role=role,
                risk_level="high",
                concerns=["unsupported certainty"],
                recommended_changes=["state uncertainty"],
            )
        return DeliberationSynthesis(
            revised_response="这是一般医学信息，具体用药请由临床医生结合病史确认。",
            uncertainty="缺少完整病史和用药清单",
            needs_human_review=True,
        )


class _LLM:
    def __init__(self):
        self.threads = set()

    def with_structured_output(self, schema):
        return _Structured(schema, self)


class _Config:
    class deliberation:
        enabled = True
        max_reviewers = 3
        roles = ["geriatric reviewer", "pharmacist reviewer", "evidence reviewer"]

    class agent_decision:
        llm = _LLM()

    class conversation:
        llm = None


def test_low_risk_answer_does_not_spend_deliberation_budget():
    assert not should_deliberate(_Config(), "你好", "您好，有什么可以帮助？")
    result = deliberate_response(_Config(), "你好", "您好，有什么可以帮助？")
    assert result["triggered"] is False
    assert result["reviews"] == []


def test_high_stakes_answer_runs_parallel_specialists_and_synthesis():
    llm = _Config.agent_decision.llm
    llm.threads.clear()
    result = deliberate_response(
        _Config(),
        "老人同时服用多种药，应该停药吗？",
        "可以直接停药。",
        safety_verdict={"verdict": "unsafe", "needs_human_review": True},
    )
    assert result["triggered"] is True
    assert len(result["reviews"]) == 3
    assert len(llm.threads) == 3
    assert result["needs_human_review"] is True
    assert "临床医生" in result["revised_response"]
    assert result["reason"] == "specialist_consensus"
