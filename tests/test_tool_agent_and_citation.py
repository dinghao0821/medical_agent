"""Tests for the ReAct tool agent and RAG citation grounding."""


class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


# --------------------------------------------------------------------------- #
# ReAct tool agent
# --------------------------------------------------------------------------- #
class _ToolsOn:
    class tools:
        enabled = True
        agent_enabled = True


class _ToolsOff:
    class tools:
        enabled = True
        agent_enabled = False


class _FakeToolLLM:
    """Fake LLM: first call requests the bmi_calculator tool, then answers."""
    def __init__(self):
        self._step = 0
        self._bound = False

    def bind_tools(self, tools):
        self._bound = True
        return self

    def invoke(self, messages):
        self._step += 1
        if self._step == 1:
            return _Msg(tool_calls=[{"name": "bmi_calculator", "args": {"query": "70 kg 175 cm"}, "id": "t1"}])
        return _Msg(content="Your BMI is about 22.9 (normal). This is screening only.")


def test_react_agent_uses_tool_then_answers():
    from agents.tools.react_agent import run_tool_agent
    out = run_tool_agent(_ToolsOn(), "I'm 70 kg and 175 cm, what's my BMI?", _FakeToolLLM())
    assert out and "22.9" in out


def test_react_agent_disabled_returns_none():
    from agents.tools.react_agent import run_tool_agent
    assert run_tool_agent(_ToolsOff(), "bmi 70 kg 175 cm", _FakeToolLLM()) is None


def test_react_agent_no_bind_tools_degrades():
    from agents.tools.react_agent import run_tool_agent

    class _NoBind:
        def invoke(self, m): return _Msg(content="hi")
    # No bind_tools attr on a plain object -> AttributeError -> degrade to None.
    assert run_tool_agent(_ToolsOn(), "bmi 70 kg 175 cm", object()) is None


# --------------------------------------------------------------------------- #
# Citation grounding
# --------------------------------------------------------------------------- #
class _CiteOn:
    class citation:
        enabled = True


class _CiteOff:
    class citation:
        enabled = False


class _FakeCiteLLM:
    def invoke(self, prompt):
        # Pretend the model inserted a citation marker on the first sentence.
        return _Msg(content="A brain tumor is an abnormal growth [1]. See a doctor.")


def test_citation_adds_markers_and_refs():
    from agents.rag_agent.citation import add_citations
    docs = [{"content": "A brain tumor is an abnormal growth of cells.",
             "source": "Neuro Textbook", "source_path": "/data/neuro.pdf"}]
    cited, meta = add_citations(_CiteOn(), "A brain tumor is an abnormal growth. See a doctor.",
                                docs, _FakeCiteLLM())
    assert "[1]" in cited
    assert "References" in cited
    assert meta.get("grounded_ratio", 0) > 0


def test_citation_disabled_is_noop():
    from agents.rag_agent.citation import add_citations
    docs = [{"content": "x", "source": "s", "source_path": "/p"}]
    cited, meta = add_citations(_CiteOff(), "original answer", docs, _FakeCiteLLM())
    assert cited == "original answer"
    assert meta == {}
