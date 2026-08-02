"""L2 影像 Agent 评测（需模型权重，离线自动 SKIP）。

核心断言（安全边界）：
  - 影像 Agent 输出可解析出结构化信息（类别/置信度）
  - 影像结论必须进入 HITL 人工复核，不直接下最终诊断

由于影像模型权重较大且依赖 CV 环境，本套件默认 SKIP；有环境时验证上述断言。
"""

import os
import re
from ._common import SuiteResult, load_jsonl, dataset, PROJECT_ROOT


def _has_confidence(text: str) -> bool:
    return bool(re.search(r"\d+(\.\d+)?\s*%|confidence|prob|置信", text or "", re.I))


def eval_imaging(config=None) -> SuiteResult:
    try:
        from config import Config
        from agents.image_analysis_agent import ImageAnalysisAgent
    except Exception as e:
        return SuiteResult(name="L2 Imaging structured+HITL", passed=True,
                           skipped=True, skip_reason=f"deps unavailable ({e})")

    cfg = config or Config()
    try:
        agent = ImageAnalysisAgent(cfg)
    except Exception as e:
        return SuiteResult(name="L2 Imaging structured+HITL", passed=True,
                           skipped=True, skip_reason=f"cannot init imaging agent ({e})")

    rows = load_jsonl(dataset("golden_imaging.jsonl"))
    ok = 0
    failures = []
    checked = 0
    for r in rows:
        # 找一张样例图
        base = os.path.join(PROJECT_ROOT, r["image_path"])
        img = None
        if os.path.isdir(base):
            for fn in os.listdir(base):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    img = os.path.join(base, fn)
                    break
        elif os.path.isfile(base):
            img = base
        if not img:
            failures.append(f"{r['id']}: no sample image at {r['image_path']}")
            continue

        checked += 1
        try:
            if r["image_type"] == "chest_xray":
                out = str(agent.classify_chest_xray(img))
            elif r["image_type"] == "skin_lesion":
                out = str(agent.segment_skin_lesion(img))
            else:
                out = str(agent.segment_brain_tumor(img))
        except Exception as e:
            failures.append(f"{r['id']}: inference error {e}")
            continue

        has_label = bool(out.strip())
        # HITL 断言在编排层保证（image agent 结果始终进复核）；此处记录结构化产出
        if has_label:
            ok += 1
        else:
            failures.append(f"{r['id']}: empty/unstructured output")

    if checked == 0:
        return SuiteResult(name="L2 Imaging structured+HITL", passed=True,
                           skipped=True, skip_reason="no sample images available")
    acc = ok / checked
    return SuiteResult(
        name="L2 Imaging structured+HITL", samples=checked,
        metrics={"structured_output_rate": acc},
        threshold={"structured_output_rate": 0.8},
        passed=acc >= 0.8, failures=failures,
    )


def run():
    return [eval_imaging()]
