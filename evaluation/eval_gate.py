"""SAGE Agent evaluation gate (unified layered entry).

This is the new unified entry that aggregates all layered suites under
``evaluation/suites/``. The legacy ``run_eval.py`` is kept as-is for backward
compatibility. See ``evaluation/EVALUATION_DESIGN.md`` for the full design.

Rule suites (CI-safe, no LLM):   safety, tools, deliberation
Online suites (need LLM + KB):   routing, citation, research, imaging, trajectory

Usage:
    python -m evaluation.eval_gate                 # rule suites only (CI gate)
    python -m evaluation.eval_gate --full          # + online suites (needs LLM)
    python -m evaluation.eval_gate --suite routing # single suite
    python -m evaluation.eval_gate --suite citation --judge
    python -m evaluation.eval_gate --full --update-baseline

Non-zero exit means a suite fell below threshold or regressed vs baseline.
"""

import os
import sys
import json
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASELINE_PATH = os.path.join(_HERE, "baselines", "baseline.json")
REPORTS_DIR = os.path.join(_HERE, "reports")

RULE_SUITES = ["safety", "tools", "deliberation"]
ONLINE_SUITES = ["routing", "citation", "research", "imaging", "trajectory"]


def _run_suite(name, use_judge=False):
    from importlib import import_module
    mod = import_module("suites." + name + "_eval")
    if name == "citation":
        return mod.run(use_judge=use_judge)
    return mod.run()


def _load_baseline():
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_baseline(results):
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    baseline = {}
    for r in results:
        if not r.skipped:
            baseline[r.name] = r.metrics
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print("\n[baseline] updated -> " + BASELINE_PATH)


def _check_regression(results, tolerance=0.03):
    baseline = _load_baseline()
    regressed = False
    for r in results:
        base = baseline.get(r.name)
        if not base or r.skipped:
            continue
        for k, v in r.metrics.items():
            bv = base.get(k)
            if bv is None or abs(v) > 1.0 or "latency" in k:
                continue
            if v < bv - tolerance:
                print("  [REGRESSION] %s.%s: %.1f%% < baseline %.1f%%" % (r.name, k, v * 100, bv * 100))
                regressed = True
    return regressed


def _write_report(results, suites):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suites_run": suites,
        "suites": [r.to_dict() for r in results],
    }
    path = os.path.join(REPORTS_DIR, "report_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n[report] -> " + path)


def main():
    ap = argparse.ArgumentParser(description="SAGE Agent evaluation gate")
    ap.add_argument("--full", action="store_true", help="run online suites too (needs LLM + KB)")
    ap.add_argument("--suite", type=str, default="", help="run a single suite by name")
    ap.add_argument("--judge", action="store_true", help="enable LLM-as-judge for citation support")
    ap.add_argument("--update-baseline", action="store_true", help="save current metrics as baseline")
    ap.add_argument("--no-regression-gate", action="store_true", help="don't fail on regression")
    args = ap.parse_args()

    if args.suite:
        suites = [args.suite]
    else:
        suites = list(RULE_SUITES)
        if args.full:
            suites += ONLINE_SUITES

    results = []
    for name in suites:
        try:
            results.extend(_run_suite(name, use_judge=args.judge))
        except Exception as e:
            print("\n=== " + name + " ===\n  [ERROR] suite crashed: " + str(e))

    for r in results:
        r.print_report()

    _write_report(results, suites)

    if args.update_baseline:
        _save_baseline(results)

    active = [r for r in results if not r.skipped]
    gate_ok = all(r.passed for r in active)

    regressed = False
    if not args.update_baseline and not args.no_regression_gate:
        regressed = _check_regression(results)

    ok = gate_ok and not regressed
    n_skip = sum(1 for r in results if r.skipped)
    print("\nOVERALL: %s (%d active, %d skipped)" % ("PASS" if ok else "FAIL", len(active), n_skip))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
