# P2 实现说明与验收 —— 异步计算与缓存

> 对应路线图 P2（见 `ENTERPRISE_ARCHITECTURE.md`）：**消息队列 + Worker、Redis 缓存 + 分布式限流、对象存储**。
> 全部特性**默认关闭 / 优雅降级**：无 Redis / 无 MinIO / 无 Celery 时应用照常运行。

---

## 1. 交付内容

### 新增服务层 `services/`
| 文件 | 作用 | 降级行为 |
|------|------|---------|
| `redis_client.py` | 共享 Redis 连接（懒加载、探活一次） | 连不上返回 `None`，依赖它的特性 no-op |
| `cache.py` | 响应/结果缓存（TTL） | 无 Redis 时每次 miss（等于无缓存） |
| `rate_limiter.py` | 固定窗口分布式限流 | 无 Redis 用进程内计数；出错 **fail-open** |
| `object_storage.py` | 对象存储（local/s3 双后端） | S3 不可用回退本地 `/uploads` |
| `celery_app.py` | Celery 应用 | 无 celery 包时 `celery_app=None` |
| `tasks.py` | 异步任务（RAG 摄取 / 影像推理） | 同步实现 `_func` 供回退 |
| `task_queue.py` | 提交门面：async/sync 自动选择 | 队列关闭时同步执行 |

### 配置（`config.py`，全部走环境变量）
- `CacheConfig`：`ENABLE_CACHE`(默认 false)、`CACHE_TTL`
- `RateLimitConfig`：`ENABLE_RATE_LIMIT`(默认 false)、`RATE_LIMIT_MAX_REQUESTS`、`RATE_LIMIT_WINDOW_SECONDS`
- `ObjectStorageConfig`：`OBJECT_STORAGE_BACKEND`(默认 local)、`S3_*`
- `TaskQueueConfig`：`ENABLE_TASK_QUEUE`(默认 false)、`CELERY_BROKER_URL`、`CELERY_RESULT_BACKEND`

### 端点变化（`app.py`）
- `/chat`、`/upload`、`/validate`、`/chat/stream`：加入**限流依赖** `rate_limit_dep`（按 session cookie / IP 计数，超限返回 429）
- `/chat`：加入**可选响应缓存**（命中返回 `cached:true`；不缓存 HITL 暂停结果）
- 新增 `POST /ingest`：提交 RAG 文档摄取（队列开启则异步返回 `task_id`，否则同步）
- 新增 `GET /tasks/{task_id}`：轮询异步任务状态/结果

### 对象存储接管
`agents/agent_decision.py` 的 `_to_public_url`：S3 后端时上传分割结果并返回远程 URL，否则/失败回退本地 `/uploads` URL。

### 依赖 & 编排
- `requirements.txt`：新增 `boto3`(S3 可选)、`celery`
- `docker-compose.yml`：新增 `minio` 与 `worker` 服务；app/worker 注入 P2 环境变量
- `.env.example`：补全 P2 全部变量

---

## 2. 本地验收（无需 Docker，验降级 + 逻辑）

限流（用进程内回退即可验，不需 Redis）：
```
set ENABLE_RATE_LIMIT=true
set RATE_LIMIT_MAX_REQUESTS=3
set RATE_LIMIT_WINDOW_SECONDS=60
uvicorn app:app --port 8000
```
连发 4 次 `/chat` → 第 4 次应返回 **429**。

同步摄取（队列关闭，验 /ingest 同步可用）：
```
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"directory\":\"./data/your_docs\"}"
```
期望返回 `{"status":"submitted","mode":"sync","result":{...}}`。

对象存储降级（默认 local）：上传影像后 `result_image` 仍是 `/uploads/...`。

---

## 3. 完整验收（Docker，含 Redis/MinIO/Worker）

```
docker compose up --build -d
```
对齐 P2 验收标准：

| 验收项 | 操作 | 期望 |
|--------|------|------|
| **重复查询命中缓存** | `ENABLE_CACHE=true`（compose 已设），对同一 query 连发两次 `/chat` | 第二次响应含 `"cached": true`，且明显更快 |
| **限流生效** | `ENABLE_RATE_LIMIT=true`，超频请求 | 返回 429；Redis 中有 `ratelimit:*` key |
| **推理任务不占用 API 连接** | `POST /ingest` 提交目录 → 立即拿到 `task_id`；`GET /tasks/{id}` 轮询 | 提交瞬时返回；worker 日志在跑摄取；状态从 PENDING→SUCCESS |
| **对象存储** | 上传影像触发分割 | `result_image` 变为 MinIO 的 URL（`S3_ENDPOINT_URL/bucket/...`）；MinIO 控制台(9001)可见对象 |

worker 独立启动（非 compose 时）：
```
celery -A services.celery_app:celery_app worker --loglevel=info
```

---

## 4. 降级矩阵（关键：不回归）

| 缺失 | 表现 |
|------|------|
| 无 Redis | 缓存 no-op、限流走进程内、Celery 若指向该 Redis 则入队失败→同步执行 |
| 无 MinIO / boto3 | 对象存储回退本地 `/uploads` |
| 无 celery 包 | `/ingest` 同步执行、`/tasks/{id}` 返回 UNAVAILABLE |
| 全部关闭（默认） | 行为与 P1 完全一致 |

---

## 5. 已知边界 / 后续

- **`/chat` 缓存忽略会话上下文**：故默认关闭，仅建议用于无状态事实类查询；更精细的做法是缓存 RAG 检索层 / Web 结果层（后续）。
- **影像推理异步化**：已提供 `segment_image_task`，但未默认改写同步 `/upload`（避免破坏 HITL 与即时返回 UX）。完整异步需配套 job 状态 API + 前端轮询（后续）。
- 下一阶段 **P3**：JWT/OAuth2 + RBAC、PII 脱敏、审计留痕、用户/历史持久化（PostgreSQL）。
