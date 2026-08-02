# 生产部署指南

## 快速开始（本地开发）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API Key

# 3. 初始化数据库
python -c "from services.db import init_db; from config import Config; init_db(Config())"

# 4. 启动服务
python app.py
# 访问 http://localhost:8000
```

## Docker 生产部署（推荐）

```bash
# 启动完整技术栈（含 Redis + PostgreSQL + Qdrant + MinIO + Nginx + 监控）
docker compose up --build -d

# 访问入口
# 患者端:    http://localhost:8080    (Nginx -> App)
# 医生端:    http://localhost:8080/doctor
# MinIO:     http://localhost:9001    (admin/minioadmin)
# Grafana:   http://localhost:3000    (admin/admin)
# Prometheus: http://localhost:9090
```

## 生产环境配置清单

### 1. 必须修改的环境变量

在 `.env` 文件中：

```ini
# 启动认证（生产必开）
ENABLE_AUTH=true

# JWT 密钥（用 openssl rand -hex 32 生成一个 https://www.codebuddy.ai/docs/zh/ide/User-guide/Overview）
JWT_SECRET=替换为随机32位以上字符串

# 医生邀请码（定期更换）
DOCTOR_INVITE_CODE=替换为复杂邀请码

# 数据库（Docker 部署使用 PostgreSQL，本地开发用 SQLite）
DATABASE_URL=postgresql+psycopg2://medical:替换密码@postgres:5432/medical_assistant

# CORS 来源（生产环境改为实际前端域名）
CORS_ORIGINS=https://your-domain.com

# API 密钥（填入你自己的密钥）
OPENAI_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx
```

### 2. Nginx + HTTPS

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate     /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;

    location / {
        proxy_pass http://app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto https;
    }
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}
```

### 3. K8s 部署

项目已提供 `k8s/` 目录下的 YAML 配置文件，直接在集群中 apply 即可：

```bash
kubectl apply -f k8s/
```

### 4. 首次启动后的操作

1. 打开网站首页 → 注册第一个用户（自动成为 admin）
2. 打开 `/doctor` → 用邀请码注册医生账号
3. 患者上传影像 → 医生审核 → SSE 实时推送结果

## 架构概览

```
                         ┌─────────────────────────┐
                         │      Nginx :8080         │
                         │   (反向代理 + HTTPS)      │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │   FastAPI App × N        │
                         │   (Gunicorn + Uvicorn)   │
                         └──┬──────┬──────┬──────┬─┘
                            │      │      │      │
                   ┌────────▼┐ ┌──▼──┐ ┌─▼───┐ ┌▼─────┐
                   │PostgreSQL│ │Redis│ │Qdrant│ │MinIO │
                   │  (用户)  │ │(会话)│ │(RAG) │ │(存储)│
                   └─────────┘ └─────┘ └──────┘ └──────┘

                   监控: Prometheus + Grafana
```

## 安全建议

- [ ] 定期更换 JWT_SECRET
- [ ] 启用 `ENABLE_RATE_LIMIT=true` 防止暴力破解
- [ ] 使用 HTTPS（生产环境必须）
- [ ] 定期备份 PostgreSQL 数据库
- [ ] 使用强密码的 PostgreSQL 和 MinIO 凭据
- [ ] 配置防火墙，仅暴露 443/80 端口
- [ ] 使用 `docker-compose` secrets 代替环境变量传敏感信息
