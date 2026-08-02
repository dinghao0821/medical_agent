# P4 实现说明与验收 —— 可观测性 & 评估 & CI/CD

> 对应路线图 P4：**Prometheus 指标 + 结构化日志 + LangSmith 追踪、RAGAS 评估、CI/CD + 自动化测试**。
> 全部 opt-in / 可降级；指标默认开启（开销极低），追踪/JSON日志默认关闭。

---

## 1. 交付内容

### 可观测性（`services/observability.py` + `config.py:ObservabilityConfig`）
| 能力 | 说明 | 开关 |
|------|------|------|
| Prometheus 指标 | `/metrics` 暴露 `http_requests_total`、`http_request_duration_seconds`、`agent_routed_total`、`llm_errors_total` | `ENABLE_METRICS`(默认 true) |
| 请求中间件 | 统计每请求方法/路由/状态/耗时（用路由模板避免高基数） | 同上 |
| 结构化日志 | 可选 JSON 日志（便于 Loki/ELK） | `ENABLE_JSON_LOGS`(默认 false)、`LOG_LEVEL` |
| LangSmith 追踪 | 配置后导出标准 LangChain 环境变量启用全链路 agent 追踪 | `LANGSMITH_TRACING`、`LANGSMITH_API_KEY`、`LANGSMITH_PROJECT` |

无 `prometheus-client` 时 `/metrics` 返回 501、中间件 no-op；不影响主流程。

### 自动化测试（`tests/`，离线）
16 条离线单测覆盖 P1–P3 基础设施的**降级/边界**（用轻量假 config，不加载 torch/langchain）：
- `test_pii.py`：邮箱/手机/ID 脱敏
- `test_rate_limiter.py`：内存回退限流、超限拦截、按客户端隔离
- `test_cache.py`：无 Redis no-op、键确定性
- `test_object_storage.py`：本地 URL 映射、S3 不可用回退
- `test_checkpointer.py`：Redis 无 URL 回退内存
- `test_auth.py`：密码哈希/校验、JWT 签发解析/防篡改

本地结果：**16 passed, 3 skipped**（skip 因本机未装 passlib/jose；CI 会全装）。

### CI/CD（`.github/workflows/ci.yml` + `requirements-dev.txt`）
- **test 作业**：装轻量 dev 依赖 → ruff（非阻塞）→ `compileall` 语法检查 → `pytest`
- **docker-build 作业**：push 到默认分支时验证镜像可构建（不推送；torch 较重，`continue-on-error`）

### RAGAS 评估（`evaluation/ragas_eval.py` + `eval_dataset.json`，可选）
将小型 QA 数据集跑过 RAG，用 RAGAS 打分（faithfulness / answer_relevancy / context_precision）。惰性依赖，缺 `ragas` 时给出提示不报错。

### 编排
`docker-compose.yml` 新增 `prometheus`(9090) 与 `grafana`(3000)；`prometheus.yml` 抓取 `app:8000/metrics`。

---

## 2. 本地验收（无需 Docker）

```
python -m pytest tests/ -q          # 期望：passed（CI 环境 19 passed）
uvicorn app:app --port 8000
curl http://localhost:8000/metrics  # 期望：Prometheus 文本指标；发几次 /chat 后 agent_routed_total 增长
```

CI 本地预演：
```
pip install -r requirements-dev.txt
ruff check .
pytest tests/ -q
```

---

## 3. 完整验收（Docker）

```
docker compose up --build -d
```
| 组件 | 地址 | 验收 |
|------|------|------|
| 应用指标 | http://localhost:8080/metrics（经 nginx）或 app 内部 | 有指标输出 |
| Prometheus | http://localhost:9090 | Targets 中 `medical-assistant` 为 UP；可查 `http_requests_total` |
| Grafana | http://localhost:3000 (admin/admin) | 添加 Prometheus 数据源(`http://prometheus:9090`)后建面板 |

LangSmith：`.env` 设 `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY=...`，请求后在 LangSmith 项目中可见 trace。

RAGAS：`pip install ragas datasets` 并确保已 ingest 数据后：
```
python -m evaluation.ragas_eval
```

---

## 4. 降级 / 不回归矩阵

| 情况 | 表现 |
|------|------|
| 无 prometheus-client | `/metrics` 501、中间件 no-op |
| `ENABLE_JSON_LOGS=false` | 标准日志（默认） |
| `LANGSMITH_TRACING=false` | 不动环境变量，无追踪（默认） |
| 无 ragas | 评估脚本提示安装，不影响应用 |
| 全默认 | 仅指标开启（轻量），其余行为同 P3 |

---

## 5. 后续（P5）
Kubernetes 编排（Deployment/Service/Ingress、HPA 自动扩缩容）、推理服务化（Triton/TorchServe + GPU 节点池）、多副本滚动发布/灰度。
