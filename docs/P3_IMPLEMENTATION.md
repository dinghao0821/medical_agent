# P3 实现说明与验收 —— 认证授权 & 合规

> 对应路线图 P3：**JWT/OAuth2 认证 + RBAC 授权、数据加密 / PII 脱敏 / 审计留痕、用户持久化**。
> 全部 **opt-in**：`ENABLE_AUTH=false`（默认）时端点不受保护、行为与 P2 完全一致；数据库默认 SQLite，无需 PostgreSQL。

---

## 1. 交付内容

### 新增服务层
| 文件 | 作用 |
|------|------|
| `services/db.py` | SQLAlchemy 引擎/会话；默认 SQLite(`data/app.db`)，`DATABASE_URL` 可切 PostgreSQL |
| `services/models.py` | `User`（用户名/邮箱/哈希密码/角色）、`AuditLog`（审计留痕） |
| `services/auth.py` | 密码哈希(bcrypt) + JWT 签发/校验（jose/passlib **延迟导入**） |
| `services/pii.py` | PII 脱敏（邮箱/手机号/身份证等 ID） |
| `services/audit.py` | 审计落库（PII 脱敏 + 可选 Fernet 加密），失败不影响主流程 |

### 配置（`config.py` → `AuthConfig`，全走环境变量）
`ENABLE_AUTH`(默认 false)、`JWT_SECRET`、`JWT_ALGORITHM`、`ACCESS_TOKEN_EXPIRE_MINUTES`、`DATABASE_URL`、`ENABLE_AUDIT`(默认 true)、`ENABLE_PII_MASKING`(默认 true)、`AUDIT_ENCRYPTION_KEY`(可选)。

### 端点（`app.py`）
- `POST /auth/register`：注册；**首个用户自动成为 admin**，其余为 patient
- `POST /auth/login`：返回 JWT `access_token`
- `GET /auth/me`：返回当前用户
- **保护**（仅当 `ENABLE_AUTH=true`）：
  - `/chat`、`/upload`、`/chat/stream` → 需要有效 JWT
  - `/validate` → **需要 doctor 或 admin 角色**（与人工审批 HITL 结合）
- **审计**：`/chat`、`/upload`、`/validate` 处理后写 `audit_logs`（记录 who/when/action/agent/session/validation_result/脱敏detail）

### RBAC 角色
`patient | doctor | admin`（`services/auth.py:VALID_ROLES`）。角色写入 JWT，`require_roles(...)` 依赖做校验。

### 依赖 & 编排
- `requirements.txt`：+`python-jose[cryptography]`、`passlib`、`bcrypt`、`psycopg2-binary`
- `docker-compose.yml`：+`postgres` 服务；app 注入 `DATABASE_URL`
- `.env.example`：补全 P3 变量

---

## 2. 本地验收（SQLite，无需 PostgreSQL）

```
set ENABLE_AUTH=true
set JWT_SECRET=my-dev-secret
uvicorn app:app --port 8000
```

注册（首个用户 = admin）与登录：
```
curl -X POST http://localhost:8000/auth/register -H "Content-Type: application/json" -d "{\"username\":\"admin\",\"password\":\"pass123\"}"
curl -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" -d "{\"username\":\"admin\",\"password\":\"pass123\"}"
```
拿到 `access_token` 后：

| 验收项 | 操作 | 期望 |
|--------|------|------|
| **未授权被拒** | 不带 token 调 `/chat` | 401 Not authenticated |
| 授权通过 | 带 `Authorization: Bearer <token>` 调 `/chat` | 正常 200 |
| **RBAC** | 用 patient 账号 token 调 `/validate` | 403 Requires role in (doctor, admin) |
| **审计可查** | 调用若干接口后查库 | `audit_logs` 表有记录，`detail` 已脱敏 |
| **PII 脱敏** | `/chat` 发含邮箱/手机号的 query | 审计 `detail` 中变为 `[EMAIL]`/`[PHONE]` |

带 token 请求示例：
```
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -H "Authorization: Bearer <token>" -d "{\"query\":\"hello\"}"
```
查看 SQLite 审计：
```
python -c "import sqlite3;print(sqlite3.connect('data/app.db').execute('select username,action,agent,detail from audit_logs').fetchall())"
```

> 关闭态验收：`ENABLE_AUTH=false`（默认）时，`/chat` 等无需 token 即可访问（不回归）；审计仍按 `ENABLE_AUDIT` 记录（匿名用户）。

---

## 3. 完整验收（Docker + PostgreSQL）

`docker compose up --build -d` 后（compose 已起 postgres 并注入 `DATABASE_URL`）。若要强制鉴权，在 `.env` 设 `ENABLE_AUTH=true` 再起。验收项同上；额外可查 PostgreSQL：
```
docker compose exec postgres psql -U medical -d medical_assistant -c "select username,action,validation_result from audit_logs order by id desc limit 10;"
```

### 字段加密（可选）
生成密钥并配置后，审计 `detail` 以 Fernet 加密存储：
```
python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"
```
写入 `.env` 的 `AUDIT_ENCRYPTION_KEY=<key>`。

---

## 4. 降级 / 不回归矩阵

| 情况 | 表现 |
|------|------|
| `ENABLE_AUTH=false`（默认） | 端点不校验；`get_current_user` 返回匿名 patient；行为同 P2 |
| 未装 jose/passlib | 只要 auth 关闭即可正常运行（延迟导入）；开启且缺包时登录/注册报错 |
| 无 `DATABASE_URL` | 自动用 SQLite；无需外部 DB |
| 审计写入失败 | 记 warning，主请求不受影响 |
| 无 `AUDIT_ENCRYPTION_KEY` | 审计 detail 存脱敏后的明文 |

---

## 5. 安全注意事项 / 后续

- 生产必须改 `JWT_SECRET`（强随机）并用 HTTPS。
- 当前注册开放（首用户 admin，其余 patient）；生产应加**管理员改角色接口**、注册审批、刷新令牌、登录失败限流。
- PII 脱敏为正则best-effort，非完整去标识化；姓名类需 NER 增强（后续）。
- 数据主体删除 / 数据留存策略、传输层 TLS 由网关侧保障（见 `ENTERPRISE_ARCHITECTURE.md` 第 4 节）。
- 下一阶段 **P4**：可观测性（Prometheus/OpenTelemetry/Grafana）+ LangSmith/RAGAS 评估 + CI/CD。
