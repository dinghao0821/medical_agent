"""L7 端到端轨迹/效率评测（需 LLM，离线自动 SKIP）。

对金标场景运行完整 ``process_query``，采集：
  - 端到端延迟（秒）
  - 是否收敛（无异常/有非空回答）
  - 应/不应触发某能力的断言（如闲聊不应进 Deep Research）

延迟/成本作为**基线记录**（不硬失败），仅收敛性与能力触发断言参与门禁。
轨迹细节（工具序列/步骤数）依赖 agent_trace，若不可用则只记延迟。
"""

import time
from ._common import SuiteResult, load_jsonl, dataset


# 轨迹断言：query -> 不应出现的 agent（负向约束）
_TRAJECTORY_RULES = [
    {"query": "Hello, how are you?", "must_not_agent": "DEEP_RESEARCH_AGENT"},
    {"query": "Thanks, that helped!", "must_not_agent": "RAG_AGENT"},
]


def eval_trajectory(max_latency_s=60.0) -> SuiteResult:
    try:
        from agents.agent_decision import process_query
    except Exception as e:
        return SuiteResult(name="L7 Trajectory & efficiency", passed=True,
                           skipped=True, skip_reason=f"cannot import process_query ({e})")

    failures = []
    latencies = []
    converged = 0
    total = 0

    # 收敛性 + 延迟（用路由金标里的场景）
    rows = load_jsonl(dataset("golden_routing.jsonl"))
    for r in rows:
        total += 1
        t0 = time.time()
        try:
            result = process_query(r["query"], session_id=f"eval-traj-{r['id']}")
            dt = time.time() - t0
            latencies.append(dt)
            answer = result.get("response") if isinstance(result, dict) else str(result)
            if answer and str(answer).strip():
                converged += 1
            else:
                failures.append(f"{r['id']}: empty answer (non-convergent)")
            if dt > max_latency_s:
                failures.append(f"{r['id']}: latency {dt:.1f}s > {max_latency_s}s")
        except Exception as e:
            failures.append(f"{r['id']}: crashed {e}")

    # 负向轨迹断言
    trajectory_ok = 0
    for rule in _TRAJECTORY_RULES:
        try:
            result = process_query(rule["query"], session_id="eval-traj-neg")
            agent = result.get("agent_name") if isinstance(result, dict) else None
            if agent != rule["must_not_agent"]:
                trajectory_ok += 1
            else:
                failures.append(f"trajectory: '{rule['query']}' wrongly used {agent}")
        except Exception as e:
            failures.append(f"trajectory rule error {e}")

    conv_rate = converged / total if total else 1.0
    traj_rate = trajectory_ok / len(_TRAJECTORY_RULES) if _TRAJECTORY_RULES else 1.0
    metrics = {"convergence_rate": conv_rate, "trajectory_rule_pass": traj_rate}
    if latencies:
        srt = sorted(latencies)
        metrics["latency_p50_s"] = srt[len(srt) // 2]
        metrics["latency_p95_s"] = srt[min(len(srt) - 1, int(len(srt) * 0.95))]

    passed = conv_rate >= 0.95 and traj_rate >= 1.0
    return SuiteResult(
        name="L7 Trajectory & efficiency", samples=total,
        metrics=metrics,
        threshold={"convergence_rate": 0.95, "trajectory_rule_pass": 1.0},
        passed=passed, failures=failures,
    )


def run(max_latency_s=60.0):
    return [eval_trajectory(max_latency_s)]
