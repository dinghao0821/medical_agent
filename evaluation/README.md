# SAGE 智能体评测体系 — 使用指南

> 完整设计见 `EVALUATION_DESIGN.md`。本文件是快速上手。

## 一、设计概览

分层评测，覆盖从单组件到端到端轨迹、从质量到安全到成本延迟的全链路：

| 层 | 套件 | 类型 | 度量 |
|----|------|------|------|
| L1 | routing | 在线(需LLM) | 路由 Top-1 准确率 |
| L2 | tools | 规则 | 工具选择 + 参数/安全拦截 |
| L2 | research | 在线(需LLM+KB) | Deep Research 证据覆盖率 |
| L2 | imaging | 在线(需模型权重) | 影像结构化输出 + HITL |
| L5 | citation | 在线(需KB,可选judge) | 引用可溯源率 + 支持度 |
| L6 | safety | 规则 | 注入/安全Critic/急诊分诊 |
| L6 | deliberation | 规则 | 多Agent审议触发逻辑 |
| L7 | trajectory | 在线(需LLM) | 端到端收敛/延迟/轨迹断言 |

## 二、目录结构

```
evaluation/
├─ EVALUATION_DESIGN.md   # 总体设计
├─ README.md              # 本文件
├─ eval_gate.py           # 统一入口（推荐用这个）
├─ run_eval.py            # 旧入口（保留，仅注入/安全/路由）
├─ suites/                # 各层评测器
│  ├─ _common.py          # 结果结构/PRF/数据加载
│  ├─ safety_eval.py      # L6 注入/安全/急诊
│  ├─ tools_eval.py       # L2 工具
│  ├─ deliberation_eval.py# L6 审议触发
│  ├─ routing_eval.py     # L1
│  ├─ citation_eval.py    # L5
│  ├─ research_eval.py    # L2 Deep Research
│  ├─ imaging_eval.py     # L2 影像
│  └─ trajectory_eval.py  # L7 轨迹
├─ judges/
│  └─ llm_judge.py        # LLM-as-judge 通用封装 + rubric
├─ datasets/              # golden 金标数据集（jsonl）
├─ baselines/baseline.json  # 各指标历史基线（回归对比）
└─ reports/               # 每次运行的结构化报告
```

## 三、运行方式

```bash
# 规则套件（CI 门禁，无需 LLM/网络）
python evaluation/eval_gate.py

# 单套件
python evaluation/eval_gate.py --suite safety

# 完整评测（需 LLM + 已建知识库 + 模型权重；缺依赖的套件自动 SKIP）
python evaluation/eval_gate.py --full

# 引用支持度（额外跑 LLM-as-judge）
python evaluation/eval_gate.py --suite citation --judge

# 建立基线（人工确认质量后）
python evaluation/eval_gate.py --full --update-baseline
```

退出码非零 = 有套件低于阈值或相对基线回归 → 可直接接入 CI。

## 四、CI 集成（示例，GitHub Actions）

```yaml
- name: Agent quality gate (rule suites)
  run: python evaluation/eval_gate.py
```

## 五、重要说明

- **诚实性**：所有阈值为工程默认值，非临床验证结论。真实医疗部署需临床专家标注与验证。
- **数据集规模**：当前为方法框架 + 种子样本。可信基线建议每层 ≥ 50 条，安全/急诊 ≥ 100 条，需持续扩充。
- **延迟/成本**：受模型供应商与硬件影响，基线仅同环境可比。
- **LLM-as-judge**：近似人工评估，不能替代；高风险维度保留人工校准子集。
- **本环境说明**：当前 boxenv 未装 pydantic，deliberation 套件自动 SKIP（其触发逻辑依赖 deliberation_agent 模块导入）。装好依赖后该套件即可离线运行。

## 六、评测驱动发现的真实问题（已修复）

运行急诊分诊评测时发现两个漏检：服药过量（overdose）和中文急症（胸痛/中风）。
已在 `agents/guardrails/emergency_triage.py` 补充 overdose 类别及中文关键词，
评测 recall 从 60% 提升到 100%。这正是分层评测的价值——安全红线缺陷被自动化捕获。
