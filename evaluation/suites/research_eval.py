"""L2 Deep Research 评测（需 LLM + 知识库，离线自动 SKIP）。

评证据覆盖率（有来源结论/总结论）与收敛性。DeepResearchAgent.run 目前只返回
report 文本，覆盖率用 report 中的引用标记密度作近似（有 [n]/来源段落的比例）。
真实覆盖率应从 research_graph 内部 findings 采集——见 EVALUATION_DESIGN.md 演进项。
"""

import re
from ._common import SuiteResult, load_jsonl, dataset


def _citation_density(report: str) -> float:
    """近似证据覆盖率：带引用标记/来源提示的句子占比。"""
    if not report:
        return 0.0
    sentences = [s for s in re.split(r"[。.!?\n]+", report) if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[\d+\]|source|来源|参考", s, re.I))
    return cited / len(sentences)


def eval_research(min_coverage=0.5) -> SuiteResult:
    try:
        from config import Config
        from agents.deep_research_agent import DeepResearchAgent
    except Exception as e:
        return SuiteResult(name="L2 Deep Research coverage", passed=True,
                           skipped=True, skip_reason=f"deps unavailable ({e})")

    try:
        agent = DeepResearchAgent(Config())
    except Exception as e:
        return SuiteResult(name="L2 Deep Research coverage", passed=True,
                           skipped=True, skip_reason=f"cannot init agent ({e})")

    rows = load_jsonl(dataset("golden_research.jsonl"))
    coverages = []
    failures = []
    for r in rows:
        try:
            report = agent.run(r["query"])
            cov = _citation_density(report)
        except Exception as e:
            cov = 0.0
            failures.append(f"{r['id']}: run error {e}")
        coverages.append(cov)
        if cov < r.get("min_coverage", min_coverage):
            failures.append(f"{r['id']}: coverage={cov:.2f} < {r.get('min_coverage', min_coverage)}")

    avg_cov = sum(coverages) / len(coverages) if coverages else 0.0
    return SuiteResult(
        name="L2 Deep Research coverage", samples=len(rows),
        metrics={"avg_evidence_coverage": avg_cov},
        threshold={"avg_evidence_coverage": min_coverage},
        passed=avg_cov >= min_coverage, failures=failures,
    )


def run(min_coverage=0.5):
    return [eval_research(min_coverage)]
