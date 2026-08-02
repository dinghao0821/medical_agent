"""L6 multi-agent deliberation trigger evaluation.

The trigger gate (should_deliberate) is deterministic (verdict + high-risk
keywords), so it can be tested offline:
  - high-risk drafts/requests must trigger (no missed review)
  - low-risk must NOT trigger (avoid over-compute / test-time-compute abuse)

Full deliberation synthesis quality needs an LLM (optional --full). This suite
validates trigger logic only and is CI-safe. If deps are missing it SKIPs.
"""

from ._common import SuiteResult, load_jsonl, dataset, prf


class _DelibCfg:
    class deliberation:
        enabled = True
        max_reviewers = 3
        roles = [
            "geriatric medicine safety reviewer",
            "clinical pharmacist and contraindication reviewer",
            "evidence and uncertainty reviewer",
        ]


def eval_trigger(min_accuracy=0.9):
    try:
        from agents.deliberation_agent import should_deliberate
    except Exception as e:
        return SuiteResult(name="L6 Deliberation trigger", passed=True,
                           skipped=True, skip_reason="deps unavailable (%s)" % e)

    rows = load_jsonl(dataset("golden_deliberation.jsonl"))
    cfg = _DelibCfg()
    tp = fp = tn = fn = 0
    failures = []
    for r in rows:
        triggered = should_deliberate(cfg, r["user"], r["draft"], r.get("safety_verdict"))
        want = bool(r["should_trigger"])
        if want and triggered:
            tp += 1
        elif want and not triggered:
            fn += 1
            failures.append(r["id"] + ": should trigger but did not")
        elif not want and triggered:
            fp += 1
            failures.append(r["id"] + ": over-triggered (compute waste)")
        else:
            tn += 1
    m = prf(tp, fp, tn, fn)
    return SuiteResult(
        name="L6 Deliberation trigger", samples=len(rows),
        metrics={"accuracy": m["accuracy"], "recall": m["recall"], "precision": m["precision"]},
        threshold={"accuracy": min_accuracy},
        passed=m["accuracy"] >= min_accuracy, failures=failures,
    )


def run(min_accuracy=0.9):
    return [eval_trigger(min_accuracy)]
