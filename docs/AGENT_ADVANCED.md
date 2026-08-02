# Advanced Agent Capabilities — Tool ReAct & Citation Grounding

Two high-value upgrades, both **opt-in + degrade-safe + no regression**.

## 1. LLM-driven tool calling (ReAct)
`agents/tools/react_agent.py`

Upgrades tool use from the fast-but-dumb keyword dispatcher to a real **ReAct
loop**: the LLM itself decides *which* tool(s) to call, with *what* arguments,
and can **chain multiple steps** (e.g. drug lookup → dose conversion) before
answering.

- Built on LangChain `bind_tools` (native function-calling); wraps the existing
  tool registry — no need to rewrite tools.
- Bounded to `_MAX_STEPS = 4` for predictable latency/cost.
- Traced via `tool_react` events.
- **Selection in `analyze_input`**: if `ENABLE_TOOL_AGENT=true`, try ReAct first;
  on miss/unsupported/error, fall back to the keyword dispatcher, then to normal
  routing. So nothing regresses when the model lacks tool-calling.
- Config: `ENABLE_TOOLS` (keyword), `ENABLE_TOOL_AGENT` (ReAct).

## 2. Sentence-level citation grounding
`agents/rag_agent/citation.py`

After a RAG answer is generated, an LLM pass inserts inline `[n]` markers linking
claims to the numbered retrieved sources, appends a **References** list of the
markers actually used, and prepends a **confidence badge**:

- 🟢 Well-sourced (≥70% sentences cited)
- 🟡 Partially sourced (≥30%)
- 🟠 Limited sourcing (<30%)

- `grounded_ratio` = fraction of sentences carrying ≥1 citation — a quantitative
  trust signal, complementary to the existing CRAG hallucination grader.
- Hooked into `response_generator.generate_response` (opt-in); fail-open leaves
  the answer untouched on any error.
- Config: `ENABLE_CITATIONS`.

## Why these two
For a *medical* assistant these deliver the most trust/capability per unit of
work: ReAct turns the tool registry into genuine autonomous tool use, and
citation grounding makes every claim auditable — the foundation of clinical
trust and compliance.

## Tests
`tests/test_tool_agent_and_citation.py` — ReAct (tool-then-answer, disabled
no-op, unsupported-degrade) and citations (markers+refs, disabled no-op). All
green (full suite: 41 passed).

## Enabling
```bash
ENABLE_TOOLS=true
ENABLE_TOOL_AGENT=true     # LLM autonomous multi-step tool calling
ENABLE_CITATIONS=true      # sentence-level grounding for RAG answers
```
