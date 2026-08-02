# SAGE 智能体评测体系设计（Agent Evaluation Framework）

> 目标：为本项目建立一套**分层、可回归、可解释**的 Agent 评测体系，覆盖从单组件到端到端轨迹、从质量到安全到成本延迟的全链路，符合当前 Agent 评测的主流实践（组件级+轨迹级+LLM-as-judge+回归门禁）。
>
> 设计原则：
> 1. **分层解耦** —— 每一层能独立评测、独立定位问题，不是只看最终答案。
> 2. **确定性优先** —— 能用规则/精确匹配就不用 LLM 评判；LLM-as-judge 仅用于开放文本质量，且带偏差缓解。
> 3. **CI 可回归** —— 每个套件有阈值，低于基线则退出码非零，可直接作质量门禁。
> 4. **离线可跑** —— 纯规则套件不依赖 LLM/网络，CI 安全；需 LLM 的套件用开关和 `[SKIP]` 优雅降级。
> 5. **贴合真实功能** —— 评测直接调用项目真实入口（`process_query` / `deliberate_response` / `DeepResearchAgent.run` 等），不 mock 业务逻辑。

---

## 一、现状盘点

项目已有 `evaluation/` 目录，包含：

| 已有 | 覆盖 | 不足 |
|------|------|------|
| `run_eval.py` | 注入防御、安全 Critic、路由准确率 | 数据集小（8/3/3 条）、无统一报告、无基线管理 |
| `ragas_eval.py` | RAG faithfulness/relevancy/context precision | 数据集 3 条、无 recall、未纳入门禁 |
| `golden_*.jsonl` | 路由/注入/安全金标 | 样本量不足以给出可信基线 |

**核心缺口**：Deep Research 质量、引用正确性、多 Agent 审议有效性、影像 Agent、HITL 触发、端到端轨迹（步骤/成本/延迟预算）、LLM-as-judge、统一报告与基线快照。本设计补齐这些。

---

## 二、评测分层模型（7 层）

```
L7 端到端轨迹    ┌─ 步骤数/工具序列/成本/延迟是否在预算内、是否收敛
L6 安全与合规    ├─ 注入拦截、安全 Critic、急诊分诊、审议触发、HITL 触发
L5 引用与忠实    ├─ 引用可溯源率、citation precision/recall、幻觉率
L4 生成质量      ├─ faithfulness / answer relevancy / 有用性（RAGAS + LLM-judge）
L3 检索质量      ├─ context precision/recall、混合检索命中、grader 准确率
L2 工具与能力    ├─ 工具选择正确率、参数校验、Deep Research 覆盖率、影像结构化
L1 路由/意图     └─ 路由准确率、置信度校准、拒答率
```

每层对应一个评测套件（suite），可单独运行也可整体运行。

---

## 三、各层评测设计

### L1 路由 / 意图（Routing）
- **指标**：Top-1 准确率、按 Agent 的 precision/recall、混淆矩阵、低置信度拒答/澄清率。
- **数据**：`golden_routing.jsonl`（扩充到覆盖 7 类 Agent，含边界/歧义样本）。
- **方法**：调用真实 `process_query` 取 `agent_name` 对比标签；离线无 LLM 时 `[SKIP]`。
- **阈值**：Top-1 ≥ 0.85（可配）。

### L2 工具与能力（Tools / Capabilities）
- **工具选择**：`golden_tools.jsonl`，检查 ReAct 是否选对工具、参数是否通过校验（非法参数应被拒）。
- **Deep Research**：`golden_research.jsonl`，评 **证据覆盖率**（有来源结论/总结论）、是否收敛（未超最大反思轮）、子问题数合理性。
- **影像**：`golden_imaging.jsonl`，评结构化输出完整性（类别+置信度）、是否强制进入 HITL（不直接下诊断）。
- **阈值**：工具选择 ≥ 0.8；研究覆盖率 ≥ 0.6。

### L3 检索质量（Retrieval）
- **指标**：context precision / context recall（RAGAS）、Grader 二分类准确率（相关性判断 vs 人工标注）。
- **数据**：`eval_dataset.json` 扩充（question + ground_truth + relevant_doc_ids）。
- **阈值**：context precision ≥ 0.6。

### L4 生成质量（Generation）
- **指标**：RAGAS 的 faithfulness、answer relevancy；LLM-as-judge 的有用性/完整性（1-5 分）。
- **偏差缓解**：judge 采用固定维度 rubric、位置随机化、必要时人工校准子集。
- **阈值**：faithfulness ≥ 0.7。

### L5 引用与忠实（Citation / Faithfulness）
- **指标**：引用可溯源率（回答中引用编号能对应到真实来源）、citation precision（引用确实支持陈述）、citation recall（需引用的陈述都有引用）、幻觉率（无依据陈述占比）。
- **方法**：解析回答中的 `[n]` 标记与 `sources` 元数据对齐 → 规则算可溯源率；支持度用 LLM-as-judge/NLI 近似。
- **阈值**：可溯源率 ≥ 0.8；幻觉率 ≤ 0.1。

### L6 安全与合规（Safety）
- **注入防御**：`golden_injection.jsonl`（含直接/间接注入、越狱、正常医学问题），precision/recall on should_block。
- **安全 Critic**：`golden_safety.jsonl`，verdict 分类 + 必含免责话术。
- **急诊分诊**：`golden_emergency.jsonl`，急症关键词是否确定性拦截（高召回优先）。
- **审议触发**：`golden_deliberation.jsonl`，高风险是否触发多 Reviewer、低风险是否不触发（避免滥用）、单 Reviewer 失败是否 fail-open。
- **HITL 触发**：影像/高风险是否置位 `needs_human_review`。
- **阈值**：注入 accuracy ≥ 0.9；安全 accuracy ≥ 0.9；急诊 recall = 1.0。

### L7 端到端轨迹（Trajectory / Efficiency）
- **指标**：步骤数是否在预算内、工具调用序列是否合理（无死循环/无多余跳转）、单请求 LLM 调用次数、端到端延迟 P50/P95、token 成本。
- **方法**：从 `agent_trace` / `cost_tracker` 采集轨迹，对金标场景断言"应/不应触发某能力"。
- **阈值**：无死循环、步骤 ≤ 预算；延迟/成本记录基线用于回归对比（不硬失败）。

---

## 四、LLM-as-Judge 规范

用于 L4/L5 的开放文本评判，遵循：
1. **固定 rubric**：每个维度给明确评分标准（如 faithfulness：5=全部有据，1=大量臆造）。
2. **结构化输出**：judge 返回 `{score, reason, evidence}`，便于审计。
3. **偏差缓解**：多答案对比时随机化位置；避免以长度/靠前判优；关键结论保留人工校准子集（golden judge labels）。
4. **可替换模型**：judge 模型经配置层，支持与被测模型不同以减少自我偏好。
5. **降级**：judge 不可用时该指标标记 `n/a` 而非 0，不污染门禁。

---

## 五、目录与文件组织

```
evaluation/
├─ README.md                  # 本设计 + 使用说明（本文件精简版指向）
├─ EVALUATION_DESIGN.md       # 本设计文档
├─ run_eval.py                # 统一入口（已有，将扩展为聚合所有 suite）
├─ ragas_eval.py              # RAG 质量（已有）
├─ suites/                    # 各层评测器
│  ├─ routing_eval.py         # L1
│  ├─ tools_eval.py           # L2 工具选择
│  ├─ research_eval.py        # L2 Deep Research 覆盖率
│  ├─ imaging_eval.py         # L2 影像结构化 + HITL
│  ├─ citation_eval.py        # L5 引用
│  ├─ safety_eval.py          # L6 注入/安全/急诊（整合已有）
│  ├─ deliberation_eval.py    # L6 审议触发
│  └─ trajectory_eval.py      # L7 轨迹/成本/延迟
├─ judges/
│  └─ llm_judge.py            # LLM-as-judge 通用封装 + rubric
├─ datasets/                  # golden 数据集（jsonl）
│  ├─ golden_routing.jsonl
│  ├─ golden_injection.jsonl
│  ├─ golden_safety.jsonl
│  ├─ golden_emergency.jsonl
│  ├─ golden_tools.jsonl
│  ├─ golden_research.jsonl
│  ├─ golden_imaging.jsonl
│  ├─ golden_citation.jsonl
│  └─ golden_deliberation.jsonl
├─ baselines/
│  └─ baseline.json           # 各指标历史基线，回归对比用
└─ reports/
   └─ report_YYYYMMDD.json    # 每次运行的结构化报告
```

> 注：为兼容现有 CI（`python -m evaluation.run_eval`），保留原有文件位置，新套件放 `suites/`，`run_eval.py` 聚合调用。

---

## 六、运行方式与门禁

```bash
# 1) CI 安全：纯规则套件（注入/安全/急诊/审议触发逻辑/工具参数校验），无需 LLM
python -m evaluation.run_eval

# 2) 完整离线质量评测（需 LLM + 已建知识库）
python -m evaluation.run_eval --full

# 3) 单套件
python -m evaluation.run_eval --suite routing
python -m evaluation.run_eval --suite citation --judge

# 4) 更新基线（人工确认质量后）
python -m evaluation.run_eval --full --update-baseline
```

- **退出码**：任一启用套件低于阈值 → 非零，阻断合入。
- **回归对比**：与 `baselines/baseline.json` 比较，指标下降超过容差（如 3%）标 `REGRESSION`。
- **报告**：每次运行输出结构化 JSON 到 `reports/`，含每层分数、失败样本、与基线的 diff。

---

## 七、指标口径与诚实性说明

- 所有阈值为**工程默认值**，非临床验证结论；真实医疗部署需临床专家参与标注和验证。
- 数据集规模决定基线可信度：当前为**方法框架 + 种子样本**，规模化需持续扩充标注（建议每层 ≥ 50 条，安全/急诊 ≥ 100 条）。
- 延迟/成本受模型供应商和硬件影响，基线只在**同一环境**内可比。
- LLM-as-judge 是近似人工评估，不能完全替代；高风险维度保留人工校准。

---

## 八、演进路线

1. **P0（本次）**：搭建分层框架、各套件评测器、种子数据集、统一报告与基线。
2. **P1**：扩充各层数据集到可信规模；接入 CI（GitHub Actions）跑规则套件门禁。
3. **P2**：引入人工标注平台产出黄金 judge 标签，校准 LLM-judge；加 A/B（审议 on/off、模型切换）对照评测。
4. **P3**：线上评测闭环——采样真实请求（脱敏）回放、医生驳回率/用户追问率反馈进数据集。
