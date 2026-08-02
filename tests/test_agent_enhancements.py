"""Unit tests for agent enhancements: tools, triage, clarification, memory."""

import os
import tempfile


# --------------------------------------------------------------------------- #
# Minimal config stubs (each feature reads its own sub-config)
# --------------------------------------------------------------------------- #
class _Cfg:
    class tools:      enabled = True
    class triage:     enabled = True
    class clarification: enabled = True
    class medical_safety:
        enabled = True
        mode = "rules"

    class memory:
        enabled = True
        auto_extract = False
        redis_url = ""

    class api:
        redis_url = ""


class _Off:
    class tools:      enabled = False
    class triage:     enabled = False
    class clarification: enabled = False
    class memory:     enabled = False
    class medical_safety:
        enabled = False
        mode = "rules"


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_bmi_tool():
    from agents.tools import maybe_run_tools
    out = maybe_run_tools(_Cfg(), "calculate my BMI, I'm 70 kg and 175 cm")
    assert out and "BMI" in out and "22.9" in out


def test_drug_info_tool():
    from agents.tools import maybe_run_tools
    out = maybe_run_tools(_Cfg(), "what is the dose of ibuprofen?")
    assert out and "NSAID" in out


def test_unit_converter_tool():
    from agents.tools import maybe_run_tools
    out = maybe_run_tools(_Cfg(), "convert 100 kg to pounds")
    assert out and "lb" in out


def test_egfr_tool_ckd_epi_2021():
    from agents.tools import maybe_run_tools
    out = maybe_run_tools(_Cfg(), "calculate egfr age 60 creatinine 1.2 female")
    assert out and "eGFR" in out and "CKD-EPI 2021" in out


def test_cha2ds2_vasc_tool():
    from agents.tools import maybe_run_tools
    out = maybe_run_tools(_Cfg(), "CHA2DS2-VASc age 76 female hypertension diabetes stroke")
    assert out and "CHA2DS2-VASc" in out and "Score: 7" in out


def test_tools_disabled_is_noop():
    from agents.tools import maybe_run_tools
    assert maybe_run_tools(_Off(), "calculate my BMI 70 kg 175 cm") is None


# --------------------------------------------------------------------------- #
# Emergency triage
# --------------------------------------------------------------------------- #
def test_triage_detects_cardiac():
    from agents.guardrails.emergency_triage import check_red_flags
    msg = check_red_flags(_Cfg(), "I have crushing chest pain spreading to my arm")
    assert msg and "emergency" in msg.lower()


def test_triage_mental_health_adds_crisis_note():
    from agents.guardrails.emergency_triage import check_red_flags
    msg = check_red_flags(_Cfg(), "I want to kill myself")
    assert msg and "hotline" in msg.lower()


def test_triage_ignores_normal_query():
    from agents.guardrails.emergency_triage import check_red_flags
    assert check_red_flags(_Cfg(), "what foods are good for a cold?") is None


# --------------------------------------------------------------------------- #
# Medical safety critic
# --------------------------------------------------------------------------- #
def test_medical_safety_critic_rewrites_prescription_language():
    from agents.guardrails.medical_safety_critic import review_response
    verdict = review_response(
        _Cfg(),
        user_text="I have fever. What should I take?",
        assistant_text="You must take amoxicillin 500mg three times daily.",
    )
    assert verdict["verdict"] == "unsafe"
    assert verdict["changed"] is True
    assert "licensed clinician" in verdict["revised_response"]


def test_medical_safety_critic_disabled_is_noop():
    from agents.guardrails.medical_safety_critic import review_response
    verdict = review_response(_Off(), "x", "Take ibuprofen if appropriate.")
    assert verdict["verdict"] == "skipped"
    assert verdict["revised_response"] == "Take ibuprofen if appropriate."


# --------------------------------------------------------------------------- #
# Clarification
# --------------------------------------------------------------------------- #
def test_clarification_triggers_on_vague():
    from agents.guardrails.clarification import needs_clarification
    msg = needs_clarification(_Cfg(), "I feel bad")
    assert msg and "how long" in msg.lower()


def test_clarification_skips_specific():
    from agents.guardrails.clarification import needs_clarification
    # Has body-area + duration specifics -> no interrogation.
    assert needs_clarification(_Cfg(), "I have a sharp headache for 3 days") is None


# --------------------------------------------------------------------------- #
# Long-term memory
# --------------------------------------------------------------------------- #
def test_memory_add_and_format(monkeypatch, tmp_path):
    import services.long_term_memory as ltm
    monkeypatch.setattr(ltm, "_MEM_DIR", str(tmp_path))

    cfg = _Cfg()
    assert ltm.add_memory(cfg, "user1", "Allergic to penicillin") is True
    assert ltm.add_memory(cfg, "user1", "Allergic to penicillin") is False  # dedup
    block = ltm.format_for_prompt(cfg, "user1")
    assert "penicillin" in block
    ltm.clear_memories(cfg, "user1")
    assert ltm.get_memories(cfg, "user1") == []


def test_memory_disabled_is_noop(tmp_path, monkeypatch):
    import services.long_term_memory as ltm
    monkeypatch.setattr(ltm, "_MEM_DIR", str(tmp_path))
    assert ltm.add_memory(_Off(), "u", "x") is False
    assert ltm.format_for_prompt(_Off(), "u") == ""
