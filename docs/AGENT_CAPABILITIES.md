# Agent Capability Enhancements

Five enhancements to the agent layer, all **opt-in + degrade-safe + no
regression**. Emergency triage is the only one on by default (safety-first);
the rest default off and are byte-for-byte no-ops when disabled.

Execution order inside `analyze_input` (before routing), so any of them can
short-circuit the normal flow with a direct answer:
**injection scan → emergency triage → tools → clarification → route**.

## 1. Cross-session long-term memory
`services/long_term_memory.py`

- Durable, **`user_id`-keyed** facts (chronic conditions, allergies, meds,
  preferences) that persist across sessions — unlike the checkpointer, which is
  per-session (`thread_id`).
- Storage: Redis if available, else local JSON under `data/memory/`.
- On each turn: stored facts are injected into the conversation prompt; after
  the turn, the LLM optionally extracts new durable facts (`MEMORY_AUTO_EXTRACT`).
- `user_id` flows from `get_current_user().username` through `process_query`.
- Privacy: `clear_memories()` supports right-to-be-forgotten.
- Config: `ENABLE_LONG_TERM_MEMORY`, `MEMORY_AUTO_EXTRACT`, `MEMORY_REDIS_URL`.

## 2. Structured tool framework
`agents/tools/` (`registry.py`, `builtin.py`)

- Extensible registry: add a tool by decorating a function with
  `@register_tool(name, description, keywords)` — no graph changes.
- Built-ins (offline, deterministic, no LLM cost): **BMI calculator**, **unit
  converter** (kg/lb, cm/in, °C/°F), **drug-info lookup** (curated local table).
- Matched queries are answered directly (`agent_name="TOOL_AGENT"`), with medical
  disclaimers baked in.
- Config: `ENABLE_TOOLS`.

## 3. Proactive multi-turn clarification
`agents/guardrails/clarification.py`

- Detects vague symptom messages ("I feel bad", "it hurts") that lack specifics
  and asks a focused follow-up (location / duration / character / severity /
  associated symptoms) instead of guessing.
- Conservative: skips messages that already contain specific cues, so it won't
  interrogate well-formed questions.
- Config: `ENABLE_CLARIFICATION`.

## 4. Semantic conversation summarisation
`services/conversation_summary.py`

- Replaces naive truncation: once history exceeds `SUMMARY_TRIGGER_MESSAGES`,
  older messages are summarised into a single recap `SystemMessage` (preserving
  symptoms/diagnoses/advice) while recent turns are kept verbatim.
- Fail-open: any error falls back to the original truncation behaviour.
- Config: `ENABLE_CONVERSATION_SUMMARY`, `SUMMARY_TRIGGER_MESSAGES`.

## 5. Emergency red-flag triage (safety-critical)
`agents/guardrails/emergency_triage.py`

- Detects potentially life-threatening presentations — cardiac (chest pain),
  stroke (FAST), breathing distress, anaphylaxis, severe bleeding, neuro
  emergencies, and **self-harm / suicidal ideation** — and returns an urgent
  "seek emergency care now" message before any informational routing.
- Mental-health hits append a crisis-support note.
- Local keyword heuristics (instant, no LLM). **On by default.**
- Config: `ENABLE_EMERGENCY_TRIAGE` (default true).

## Additional techniques considered

Beyond the three requested, these were evaluated as high-value agent
enhancements; #4 and #5 above were implemented as the best fit for a *medical*
assistant. Deferred / future candidates:

- **Cost & token-budget governance** (explicitly deferred earlier).
- **Tool selection via LLM function-calling** (current selection is offline
  keyword matching; an LLM router can be layered on the same registry).
- **Reflection/self-critique on final answers** (partially present via CRAG/
  Self-RAG in the RAG path; could be generalised to all agents).

## Tests

`tests/test_agent_enhancements.py` — 11 tests covering tools (BMI/drug/units),
triage (cardiac + mental-health + negative), clarification (vague vs specific),
and memory (add/dedup/format/clear + disabled no-op). All green.

## Enabling summary

```bash
ENABLE_LONG_TERM_MEMORY=true
MEMORY_AUTO_EXTRACT=true
ENABLE_TOOLS=true
ENABLE_CLARIFICATION=true
ENABLE_CONVERSATION_SUMMARY=true
ENABLE_EMERGENCY_TRIAGE=true   # already default on
```
