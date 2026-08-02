"""L1 路由/意图准确率评测（需 LLM，离线自动 SKIP）。

调用真实 ``process_query`` 取选中的 Agent，与金标标签对比，输出 Top-1 准确率
与按 Agent 的混淆情况。
"""

from ._common import SuiteResult, load_jsonl, dataset


def eval_routing(min_accuracy=0.85) -> SuiteResult:
    try:
        from agents.agent_decision import process_query
    except Exception as e:
        return SuiteResult(name="L1 Routing accuracy", passed=True,
                           skipped=True, skip_reason=f"cannot import process_query ({e})")

    rows = load_jsonl(dataset("golden_routing.jsonl"))
    correct = 0
    failures = []
    confusion = {}
    for r in rows:
        try:
            result = process_query(r["query"], session_id=f"eval-route-{r['id']}")
            agent = result.get("agent_name") if isinstance(result, dict) else None
        except Exception as e:
            agent = f"<error: {e}>"
        exp = r["expected_agent"]
        confusion[(exp, agent)] = confusion.get((exp, agent), 0) + 1
        if agent == exp:
            correct += 1
        else:
            failures.append(f"{r['id']}: got={agent} expected={exp}")

    acc = correct / len(rows) if rows else 1.0
    return SuiteResult(
        name="L1 Routing accuracy", samples=len(rows),
        metrics={"top1_accuracy": acc}, threshold={"top1_accuracy": min_accuracy},
        passed=acc >= min_accuracy, failures=failures,
    )


def run(min_accuracy=0.85):
    return [eval_routing(min_accuracy)]
