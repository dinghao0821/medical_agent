# Agent Hardening — Security, Tracing & Evaluation

This phase strengthens the **Agent layer itself** (not just infrastructure) with
three additions, each following the project's **opt-in + degrade-safe + no
regression** rule.

## 1. Prompt-injection defence & output-leak protection

`agents/guardrails/injection_filter.py` — a fast, **local, dependency-free**
layer that complements the existing LLM-based `LocalGuardrails`.

| Function | What it does | Where it runs |
|----------|--------------|---------------|
| `scan_input` | Heuristic detection of instruction-override / prompt-exfiltration / jailbreak | `agent_decision.analyze_input`, **before** the LLM guardrail (cheaper, blocks early) |
| `wrap_untrusted` | Fences retrieved/web content as *data, not instructions* ("spotlighting") | `rag_agent/response_generator.py`, `web_search_processor.py` |
| `scan_output` | Redacts leaked system prompt / API-key-shaped secrets | `agent_decision.apply_output_guardrails`, after the LLM guardrail |

- **Config**: `SecurityConfig` (`ENABLE_AGENT_SECURITY`, `SECURITY_WRAP_UNTRUSTED`,
  `SECURITY_SCAN_OUTPUT`). On by default (fail-open, low-risk); set
  `ENABLE_AGENT_SECURITY=false` to fully restore prior behaviour.
- **Fail-open**: any internal error allows the request through — the filter can
  never take the app down.
- **No LLM calls / no new deps**: pure regex, works even when the model provider
  is degraded.

### Why "wrap untrusted"?
Web pages and retrieved documents are attacker-controllable. Fencing them with
explicit "treat as data, never obey" markers is the standard mitigation
(spotlighting) against *indirect* prompt injection.

## 2. Structured agent tracing

`services/agent_trace.py` — one JSON line per request capturing the decision
path, so any answer is reconstructable.

- **Events**: `route` (agent + confidence), `security_block` (reason),
  retrieval confidence, per-step latency, final agent + status + duration.
- **Config**: `TraceConfig` (`ENABLE_AGENT_TRACE`, default **off**). When off,
  every trace call is a no-op.
- **Implementation**: `contextvars` (correct under async + threads), errors
  self-swallowed, emitted via the standard logger (`agent.trace`) so it composes
  with `ENABLE_JSON_LOGS`.

Example log line:
```json
agent_trace {"trace_id":"a1b2c3d4e5f6","session_id":"s-1","events":[{"kind":"route","t":0.42,"agent":"RAG_AGENT","confidence":0.91},{"kind":"retrieval","t":1.1,"retrieval_confidence":0.78}],"agent":"RAG_AGENT","status":"ok","duration_s":2.3}
```

## 3. Evaluation loop (regression gate)

`evaluation/run_eval.py` + golden sets — turns "did my change make the agent
worse?" into a measurable, CI-enforceable check.

| Suite | File | Runs | Threshold |
|-------|------|------|-----------|
| Injection defence | `golden_injection.jsonl` | **Offline** (no LLM) | accuracy ≥ 0.9 |
| Routing accuracy | `golden_routing.jsonl` | Live (`--routing`, needs LLM) | accuracy ≥ 0.75 |

```bash
python -m evaluation.run_eval               # offline injection suite (CI-safe)
python -m evaluation.run_eval --routing     # + live routing accuracy
```

- Non-zero exit on failure → wired into `.github/workflows/ci.yml` as a gate.
- The injection suite already caught and fixed 2 missed attack patterns during
  development (`disregard the above rules`, `forget everything ... no
  restrictions`) — demonstrating the loop's value. Current score: **100%
  accuracy, 0 false positives**.

## Tests

`tests/test_injection_filter.py` covers blocking, allow-listing legitimate
medical queries, disabled no-op, fencing, and leak redaction.

## Enabling summary

```bash
ENABLE_AGENT_SECURITY=true    # default on
SECURITY_WRAP_UNTRUSTED=true
SECURITY_SCAN_OUTPUT=true
ENABLE_AGENT_TRACE=false      # turn on to get per-request JSON traces
```

## Not included (deferred)

Cost & quota governance (per-user token budgets, spend metering) was explicitly
deferred and can be added later as a separate layer.
