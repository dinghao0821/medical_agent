"""L5 引用溯源评测（需 LLM + 知识库，离线自动 SKIP）。

  - 可溯源率（规则）：回答中的引用编号 [n] 是否都能对应到实际检索来源
  - 引用支持度（LLM-judge，可选）：被引用来源是否确实支持陈述

可溯源率是纯规则度量，只要 RAG 可运行即可算；支持度需 --judge。
"""

import re
from ._common import SuiteResult, load_jsonl, dataset


def _extract_citation_ids(text: str):
    return set(int(m) for m in re.findall(r"\[(\d+)\]", text or ""))


def eval_citation(min_traceable=0.8, use_judge=False, config=None) -> SuiteResult:
    try:
        from config import Config
        from agents.rag_agent import MedicalRAG
    except Exception as e:
        return SuiteResult(name="L5 Citation traceability", passed=True,
                           skipped=True, skip_reason=f"deps unavailable ({e})")

    cfg = config or Config()
    try:
        rag = MedicalRAG(cfg)
    except Exception as e:
        return SuiteResult(name="L5 Citation traceability", passed=True,
                           skipped=True, skip_reason=f"cannot init RAG ({e})")

    rows = load_jsonl(dataset("golden_citation.jsonl"))
    traceable_scores = []
    judge_scores = []
    failures = []

    for r in rows:
        try:
            resp = rag.process_query(r["question"])
            answer = resp.get("response", "") if isinstance(resp, dict) else str(resp)
            sources = resp.get("sources", []) if isinstance(resp, dict) else []
        except Exception as e:
            failures.append(f"{r['id']}: query error {e}")
            traceable_scores.append(0.0)
            continue

        cited = _extract_citation_ids(answer)
        n_sources = len(sources)
        if not cited:
            # 无引用标记：若有来源却不引用，可溯源率记 0（应引用未引用）
            traceable = 0.0 if n_sources else 1.0
            failures.append(f"{r['id']}: no citation markers in answer")
        else:
            valid = sum(1 for c in cited if 1 <= c <= n_sources)
            traceable = valid / len(cited)
            if traceable < 1.0:
                failures.append(f"{r['id']}: {len(cited)-valid} citation(s) out of source range")
        traceable_scores.append(traceable)

        if use_judge:
            from judges.llm_judge import judge as _judge
            ctx = "\n".join(
                str(s.get("content") or s.get("text") or s) if isinstance(s, dict) else str(s)
                for s in sources
            )
            jr = _judge(cfg, "citation_support", r["question"], answer, ctx)
            if jr.get("score") is not None:
                judge_scores.append(jr["score"])

    avg_trace = sum(traceable_scores) / len(traceable_scores) if traceable_scores else 1.0
    metrics = {"traceability": avg_trace}
    if judge_scores:
        metrics["citation_support_1to5"] = sum(judge_scores) / len(judge_scores)

    return SuiteResult(
        name="L5 Citation traceability", samples=len(rows),
        metrics=metrics, threshold={"traceability": min_traceable},
        passed=avg_trace >= min_traceable, failures=failures,
    )


def run(min_traceable=0.8, use_judge=False):
    return [eval_citation(min_traceable, use_judge)]
