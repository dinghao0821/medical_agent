# P1 验收测试方案

> 本文档给出将「第一阶段（P1）」改造验收所需的完整测试步骤，供换机器 / 换环境后照着执行。
>
> 配套：`docs/ENTERPRISE_ARCHITECTURE.md`（P1–P5 路线图）、`smoke_test_C.cmd`（C 档一键脚本）、`docker-compose.yml`、`nginx.conf`。

---

## 0. P1 验收标准（来自路线图）

| # | 验收项 | 在哪一档验证 |
|---|--------|-------------|
| 1 | 现有功能无回归（对话/RAG/Web/影像三类/语音/guardrails/人工验证） | A + B |
| 2 | 多用户并发**不串话**（会话隔离） | B（单机双 session）/ C（跨 worker） |
| 3 | **多 worker 正常启动**（gunicorn + UvicornWorker） | C（Docker，Windows 本机不支持 gunicorn） |
| 4 | 慢推理**不阻塞**事件循环 | B / C（并发） |
| 5 | 跨副本会话共享（Redis 外部 checkpointer） | C |
| 6 | SSE 流式经网关透传（nginx `proxy_buffering off`） | C |

> 当前状态（截至编写时）：**A 档=间接覆盖；B 档=已完成；C 档=待有 Docker/Linux 环境补验。**

---

## 1. 前置依赖（换机器后先做）

1. 克隆代码，进入项目根目录，创建并激活 Python 环境（conda 示例）：
   ```
   conda create -n medical-assistant python=3.11 -y
   conda activate medical-assistant
   pip install -r requirements.txt
   ```
   > `requirements.txt` 已含 `fastembed==0.6.1`、`redis`、`langgraph-checkpoint-redis`、`sse-starlette`、`gunicorn`。
   > 若单独遇到 `ModuleNotFoundError: No module named 'fastembed'`，执行 `pip install fastembed==0.6.1`（否则 RAG 稀疏检索报错，会被动降级为 Web 搜索）。

2. 复制并填写环境变量：
   ```
   copy .env.example .env
   ```
   至少填：`OPENAI_API_KEY`（或 `DASHSCOPE_API_KEY`）、`OPENAI_BASE_URL`、`MODEL_NAME`、`TAVILY_API_KEY`。

3. 确认知识库数据：RAG 需要先 ingest 文档到 Qdrant（`python ingest_rag_data.py ...`），否则 RAG 无命中会转 Web。

---

## 2. A 档：零依赖冒烟（不需要 API Key / 不联网）

目的：确认 P1 代码结构本身没问题（可导入、图可编译、回退可用）。

```
python -m py_compile app.py config.py agents/agent_decision.py agents/session/checkpointer_factory.py agents/rag_agent/grader.py agents/deep_research_agent/__init__.py agents/deep_research_agent/planner.py agents/deep_research_agent/research_graph.py agents/deep_research_agent/report_composer.py
```
期望：无报错（返回 `COMPILE_OK` 即通过语法/导入层面）。

可选：Python 交互里验证 checkpointer 回退与工具函数
```python
from agents.session.checkpointer_factory import build_checkpointer
cp = build_checkpointer(backend="redis", redis_url=None)   # 无 URL -> 应回退 MemorySaver 并告警
print(type(cp).__name__)                                    # 期望: MemorySaver
```

**通过标准**：编译无错；无 Redis 时 `build_checkpointer` 回退 `MemorySaver`。

---

## 3. B 档：本机功能验收（需要 API Key + Qdrant，Windows 可跑）

### 启动（开发模式，单进程）
```
python app.py
```
或（推荐，便于控制）：
```
uvicorn app:app --host 0.0.0.0 --port 8000
```
> ⚠️ 不要用 `--reload`：`app.py` 有后台线程定时清理 `uploads/speech/*.mp3`，且影像会写 `uploads/`，reload 监视文件变化会不断重启/退出。

### 逐项验收（base = http://localhost:8000）

| 验收点 | 操作 | 期望 |
|--------|------|------|
| 健康检查 | `curl http://localhost:8000/health` | `{"status":"healthy"}` |
| 无回归-对话 | `/chat` 发普通问候 | 走 CONVERSATION_AGENT 正常回答 |
| 无回归-RAG | `/chat` 问已 ingest 的医学知识 | 走 RAG_AGENT，命中知识库 |
| 无回归-Web | `/chat` 问时效性问题 | 走 WEB_SEARCH_PROCESSOR_AGENT |
| 无回归-影像 | `/upload` 传脑 MRI / 胸片 / 皮肤图 | 对应影像 agent 处理，返回结果 |
| **会话隔离** | 用两个不同 session_id cookie 各说一句再互相追问 | 两会话记忆互不可见 |
| 影像唯一路径 | `/upload` 连传两张图 | 返回的 `result_image` 是不同 UUID 文件名，不互相覆盖 |
| **原生 HITL** | `/upload` 传医学影像 → 响应含 `awaiting_validation:true`；再调 `/validate`（带同一 cookie，`validation_result=yes/no`） | 图从 `interrupt` 恢复返回确认/驳回（非重跑） |
| SSE 流式 | `POST /chat/stream` | 逐段 `token` 事件 + 末尾 `done` |

### curl 示例（会话隔离）
```
curl -c s1.txt -b s1.txt -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\":\"My name is Alex.\"}"
curl -c s1.txt -b s1.txt -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\":\"What is my name?\"}"
curl -c s2.txt -b s2.txt -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\":\"What is my name?\"}"
```
期望：s1 的第二问答出 Alex；s2 答不出。

**通过标准**：以上全部符合预期。（B 档单机 MemorySaver 下会话隔离靠不同 thread_id 实现。）

---

## 4. C 档：多 worker + Redis + nginx（需要 Docker，或 Linux + Redis）

> `gunicorn` 不支持 Windows，故本档需 **Docker Desktop（Windows 用 WSL2 后端）** 或 **Linux 机器**。

### 路 A：Docker（完整，推荐）

前置：安装并启动 Docker Desktop，`docker version` 可用；`.env` 已填。

```
REM 构建并后台启动 app + redis + qdrant + nginx
docker compose up --build -d
docker compose ps

REM 一键冒烟（脚本默认打 nginx 的 8080）
smoke_test_C.cmd
```

`docker-compose.yml` 已自动注入：`CHECKPOINTER_BACKEND=redis`、`REDIS_URL=redis://redis:6379/0`、`WORKERS=2`。
nginx 暴露在 **http://localhost:8080**，转发到 app:8000。

停止：
```
docker compose down        REM 保留数据卷
docker compose down -v      REM 连 redis/qdrant 卷一起清
```

### 路 B：无 Docker（Linux/WSL + Redis，验除 nginx 外的项）

在 Linux/WSL 起 Redis：
```
sudo apt update && sudo apt install -y redis-server
redis-server --daemonize yes
```
设置环境并多 worker 启动（`__main__` 在多 worker 下不生效，必须用 uvicorn 命令）：
```
export CHECKPOINTER_BACKEND=redis
export REDIS_URL=redis://localhost:6379/0
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```
可验：多 worker 启动 / 跨 worker 会话共享 / 会话隔离 / SSE（直连）/ 慢推理不阻塞。
**验不了**：nginx SSE 透传（需网关，仅路 A）。

### C 档逐项验收

| # | 检查 | 操作 | 期望 |
|---|------|------|------|
| 1 | 多 worker 启动 | `docker compose logs app \| findstr /i worker` | 出现 2 次 `Booting worker` |
| 2 | 网关健康 | `curl http://localhost:8080/health` | 200 |
| 3 | **跨 worker 会话共享** | 同 cookie 记名字→追问 | 追问答出记住的名字 |
| 4 | 会话隔离 | 换新 cookie 追问 | 答不出 |
| 5 | Redis 有状态 | `docker compose exec redis redis-cli DBSIZE` | > 0 |
| 6 | nginx SSE 透传 | `curl -N -X POST http://localhost:8080/chat/stream -H "Content-Type: application/json" -d "{\"query\":\"what is a migraine?\"}"` | 逐段 `token` + `done`（非一次性） |
| 7 | 慢推理不阻塞 | 终端 A 发慢请求（`/upload` 或开启 Deep Research 的复杂问题），终端 B 同时连发快 `/chat` | B 不等 A 即返回 |

**通过标准**：1–6 全绿（脚本 `smoke_test_C.cmd` 自动跑 1–6，返回码 0）；7 手动确认。

---

## 5. 常见问题排查

| 现象 | 原因 | 处理 |
|------|------|------|
| `ModuleNotFoundError: fastembed` | 环境没装全 | `pip install fastembed==0.6.1`（Docker 镜像已自带） |
| `/chat` 返回 503，日志 `data_inspection_failed` | DashScope 内容审查误拦医疗内容 | 已做三层优雅降级（web/guardrails fail-open/process_query 兜底）；仍频繁可换不带审查的兼容端点 |
| 服务"自动退出"（Shutting down） | 多以 `--reload` 启动，文件写入/清理线程触发重启 | 去掉 `--reload`，或 `--reload-exclude "uploads/*" --reload-exclude "data/*"` |
| C 档第 3 项失败但第 5 项 Redis 有 key | `langgraph-checkpoint-redis` 版本 API 差异 | 调整 `agents/session/checkpointer_factory.py` 适配层 |
| `wsl --install` 报 `已禁止(403)` | 公司网络/代理封了微软下载 | 换非受限网络，或用内部远程 Linux/云研发环境 |
| RAG 总是转 Web | 未 ingest 数据 / fastembed 缺失 / 置信度低 | 先 `ingest_rag_data.py` 灌数据并装 fastembed |

---

## 6. 验收结论记录（填写）

- [ ] A 档：编译 + 回退通过
- [ ] B 档：功能无回归 + 会话隔离 + HITL + SSE 通过
- [ ] C 档-1：多 worker 启动
- [ ] C 档-2：网关健康
- [ ] C 档-3：跨 worker 会话共享（Redis）
- [ ] C 档-4：会话隔离
- [ ] C 档-5：Redis 有 checkpoint
- [ ] C 档-6：nginx SSE 透传
- [ ] C 档-7：慢推理不阻塞（手动）

全部勾选 → **P1 正式关闭，进入 P2（异步计算 + 缓存 + 对象存储）**。
