# Performance & Cost Enhancements — Streaming / Cost / Semantic Cache / Retrieval

Four mainstream upgrades, all **opt-in + degrade-safe + no regression**.

## 1. Token-level real streaming
`agents/agent_decision.py:stream_conversation_tokens()` + `app.py:/chat/stream`

- For plain **conversational** queries, streams the chat model's **native tokens**
  (`llm.stream()`) as they're generated — real streaming, not word-chunking a
  finished answer.
- It re-runs the same fast safety checks (injection scan, emergency triage) and
  a standalone router; declines (returns None) for RAG/web/vision/guardrail paths
  → those fall back to the graph + word-chunking. So nothing regresses.
- Cost is metered for the streamed answer.
- Config: `STREAM_TOKEN_LEVEL` (default off).

## 2. LLM cost & token-budget governance
`services/cost_tracker.py`

- **Token counting**: `tiktoken` if available, else char/4 heuristic.
- **Metering**: per-user/session daily usage in Redis (auto-expiring) with an
  in-memory fallback; emits a `cost` trace event.
- **Budget enforcement**: `/chat` checks `check_budget` before processing; over
  `DAILY_TOKEN_BUDGET` (>0) → HTTP 429. `0` = metering only.
- Config: `ENABLE_COST_TRACKING`, `DAILY_TOKEN_BUDGET`, `COST_REDIS_URL`.

## 3. Semantic (embedding-similarity) cache
`services/semantic_cache.py`

- Embeds the query (reusing the RAG embedding model) and returns a prior answer
  when cosine similarity ≥ threshold — hits on paraphrases the exact-match cache
  misses ("what helps a headache?" ≈ "how to treat a headache?").
- Storage: capped Redis list with in-memory fallback. Checked before the
  exact-match cache in `/chat`; populated after successful answers.
- Config: `ENABLE_SEMANTIC_CACHE`, `SEMANTIC_CACHE_THRESHOLD`, `SEMANTIC_CACHE_MAX_ENTRIES`.

## 4. Reranker config + parent-child (small-to-big) retrieval
`config.py` + `agents/rag_agent/parent_child.py`

- **Reranker** was already a cross-encoder; now the model + top_k are
  env-configurable (`RERANKER_MODEL`, `RERANKER_TOP_K`) so you can swap in a
  medical model like `pritamdeka/S-PubMedBert-MS-MARCO`.
- **Parent-child**: a query-time small-to-big expansion (no re-ingestion) —
  after retrieval+rerank, same-source chunks are merged into a larger parent
  context (capped at `PARENT_CHUNK_SIZE`) for richer generation. Hooked in
  `MedicalRAG.process_query` before generation.
- Config: `RERANKER_MODEL`, `RERANKER_TOP_K`, `ENABLE_PARENT_CHILD_RETRIEVAL`,
  `PARENT_CHUNK_SIZE`.

## Tests
`tests/test_cost_semcache_parentchild.py` — token counting, budget enforcement,
cost disabled no-op; semantic cache hit/miss/disabled; parent-child merge/disabled.
7 passed.

## Enabling
```bash
STREAM_TOKEN_LEVEL=true
ENABLE_COST_TRACKING=true
DAILY_TOKEN_BUDGET=100000          # per-user daily cap (0 = metering only)
ENABLE_SEMANTIC_CACHE=true
ENABLE_PARENT_CHILD_RETRIEVAL=true
RERANKER_MODEL=pritamdeka/S-PubMedBert-MS-MARCO   # optional medical reranker
```
