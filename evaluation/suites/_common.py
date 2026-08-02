"""评测套件共享工具：结果结构、数据加载、路径解析、指标计算。"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(_HERE)          # evaluation/
PROJECT_ROOT = os.path.dirname(EVAL_ROOT)   # 项目根
DATASETS_DIR = os.path.join(EVAL_ROOT, "datasets")


def dataset(name: str) -> str:
    """返回 datasets/ 下数据集的绝对路径；兼容旧位置（evaluation/ 根目录）。"""
    p = os.path.join(DATASETS_DIR, name)
    if os.path.exists(p):
        return p
    legacy = os.path.join(EVAL_ROOT, name)
    return legacy if os.path.exists(legacy) else p


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class SuiteResult:
    name: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    threshold: Dict[str, float] = field(default_factory=dict)
    samples: int = 0
    failures: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def print_report(self):
        head = f"\n=== {self.name} ==="
        if self.skipped:
            print(head)
            print(f"  [SKIP] {self.skip_reason}")
            return
        print(head)
        metric_str = " ".join(
            f"{k}={v:.2%}" if abs(v) <= 1.0 else f"{k}={v:.3f}"
            for k, v in self.metrics.items()
        )
        print(f"  samples={self.samples} {metric_str}")
        for f_ in self.failures[:20]:
            print(f"  [FAIL] {f_}")
        if len(self.failures) > 20:
            print(f"  ... and {len(self.failures) - 20} more failures")
        thr = " ".join(f"{k}>={v}" for k, v in self.threshold.items())
        print(f"  threshold: {thr} -> {'PASS' if self.passed else 'FAIL'}")


def prf(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
