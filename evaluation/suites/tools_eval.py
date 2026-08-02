"""L2 工具选择与参数校验评测（纯规则，CI 安全）。

  - 工具选择：关键词匹配是否选中期望工具（``_match_by_keywords``）
  - 参数/安全拦截：非法或危险参数应被工具拒绝（``run_tool`` 返回 None）

工具是确定性离线组件，本套件不需要 LLM。
"""

from ._common import SuiteResult, load_jsonl, dataset


def _ensure_registered():
    # 触发 builtin 工具注册
    import agents.tools.builtin  # noqa: F401


def eval_tools(min_select=0.8, min_param=1.0) -> SuiteResult:
    _ensure_registered()
    from agents.tools.registry import _match_by_keywords, run_tool

    rows = load_jsonl(dataset("golden_tools.jsonl"))
    select_total = select_ok = 0
    param_total = param_ok = 0
    failures = []

    for r in rows:
        expected = r.get("expected_tool")
        matched = _match_by_keywords(r["text"])

        # 工具选择正确性
        select_total += 1
        if matched == expected:
            select_ok += 1
        else:
            failures.append(f"{r['id']}: select got={matched} expected={expected}")

        # 输出/参数拦截：expect_output=False 表示应拒绝(None)
        expect_output = bool(r.get("expect_output", True))
        target = matched or expected
        if target:
            out = run_tool(target, r["text"])
            produced = out is not None
            param_total += 1
            if produced == expect_output:
                param_ok += 1
            else:
                failures.append(
                    f"{r['id']}: output={produced} expected={expect_output} (param/safety gate)"
                )

    select_acc = select_ok / select_total if select_total else 1.0
    param_acc = param_ok / param_total if param_total else 1.0
    passed = select_acc >= min_select and param_acc >= min_param
    return SuiteResult(
        name="L2 Tool selection & param gate", samples=len(rows),
        metrics={"select_acc": select_acc, "param_gate_acc": param_acc},
        threshold={"select_acc": min_select, "param_gate_acc": min_param},
        passed=passed, failures=failures,
    )


def run(min_select=0.8, min_param=1.0):
    return [eval_tools(min_select, min_param)]
