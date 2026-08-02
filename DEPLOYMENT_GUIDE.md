# Multi-Agent Medical Assistant 完整部署指南

这份文档用于帮助你在本机或服务器上完整部署当前项目，并尽量把它部署成一个接近真实企业环境的可运行系统。

---

## 1. 部署目标

你要部署的不是一个“单机 demo”，而是一个尽量接近真实企业应用的完整系统，包含：

- Web 应用入口
- 用户注册/登录/鉴权
- 医疗对话
- RAG / 向量检索
- 图像分析能力
- 可选异步任务队列
- Redis / PostgreSQL / Qdrant / MinIO 等基础设施
- 监控面板

---

## 2. 部署前准备

### 2.1 机器要求

建议最低配置：

- CPU：4 核以上
- 内存：8GB 以上
- 磁盘：50GB 以上
- 操作系统：Linux / macOS / Windows 10/11

如果你想完整体验图像分析和模型推理，内存更大更好。

### 2.2 安装基础软件

必装：

- Docker
- Docker Compose
- Python 3.11
- Git
- curl

推荐：

- make
- unzip
- jq

### 2.3 检查环境

在终端执行：

```bash
docker --version
docker compose version
python3 --version
git --version
```

如果你看到类似下面的报错：

```bash
docker: unknown command: docker compose
```

说明当前机器上的 Docker CLI 版本较老，或者没有安装 `docker compose` 插件。此时请改用旧版的 Compose 命令：

```bash
docker-compose --version
```

如果 `docker-compose` 也不存在，说明需要安装新版 Docker Desktop / Docker Engine，并启用 Compose v2 插件。

在 Ubuntu / Debian 类系统上，可以先尝试安装：

```bash
sudo apt-get update
sudo apt-get install -y docker-compose-plugin
```

如果系统包仓库里没有插件，说明当前系统的仓库没有提供这个包。可以先添加 Docker 官方仓库，然后再安装：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

如果你使用的不是 Ubuntu，可以把上面的 `ubuntu` 替换为对应发行版名称。

如果系统包仓库里没有插件，或者你不想折腾仓库，也可以安装旧版 Compose：

```bash
sudo apt-get install -y docker-compose
```

安装完成后，重新检查：

```bash
docker compose version
docker-compose --version
```

另外请确保 Docker 服务已经启动：

```bash
sudo systemctl start docker
sudo systemctl enable docker
```

---

## 3. 项目代码准备

进入项目目录：

```bash
cd /data/dh/Multi-Agent-Medical-Assistant-main/Multi-Agent-Medical-Assistant-main
```

确认关键文件存在：

- Dockerfile
- docker-compose.yml
- nginx.conf
- .env.example

---

## 4. 配置环境变量

### 4.1 复制示例环境文件

```bash
cp .env.example .env
```

### 4.2 编辑 .env

至少要配置以下内容。

#### 4.2.1 LLM 配置

你需要一个兼容 OpenAI 的模型服务，项目默认示例使用 DashScope。

建议至少填：

```env
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_API_KEY=你的key
MODEL_NAME=qwen3-max
VISION_MODEL=qwen-vl-plus
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1536
```

#### 4.2.2 认证配置

为了更接近完整功能，建议开启认证：

```env
ENABLE_AUTH=true
JWT_SECRET=换成你自己的随机字符串
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

可以用下面命令生成随机字符串：

```bash
python3 - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
```

#### 4.2.3 数据库配置

建议使用 PostgreSQL：

```env
DATABASE_URL=postgresql+psycopg2://medical:medical@postgres:5432/medical_assistant
```

#### 4.2.4 Redis 配置

为了让会话和缓存更接近真实部署：

```env
CHECKPOINTER_BACKEND=redis
REDIS_URL=redis://redis:6379/0
ENABLE_CACHE=true
ENABLE_RATE_LIMIT=true
ENABLE_TASK_QUEUE=true
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/1
```

#### 4.2.5 对象存储配置

如果你想完整体验上传与对象存储：

```env
OBJECT_STORAGE_BACKEND=s3
S3_ENDPOINT_URL=http://minio:9000
S3_BUCKET=medical-assistant
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_REGION=us-east-1
```

#### 4.2.6 监控配置

```env
ENABLE_METRICS=true
ENABLE_JSON_LOGS=false
LOG_LEVEL=INFO
```

---

## 5. 安装依赖

### 5.1 推荐：使用 Docker 部署

这是最稳妥、最接近真实部署的方式。

Docker 会自动构建镜像并安装依赖。

### 5.2 可选：本地 Python 环境部署

如果你不想用 Docker，可以本地安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

但请注意：

- 本地部署对系统依赖更敏感
- 某些图像/ocr 包安装更慢
- 不是最稳定的方式

所以推荐优先使用 Docker。

---

## 6. 启动完整服务栈

### 6.1 启动所有容器

```bash
docker compose up --build -d
```

这一步会启动：

- app
- redis
- postgres
- qdrant
- minio
- nginx
- prometheus
- grafana

### 6.2 查看容器状态

```bash
docker compose ps
```

你应该看到这些服务处于 running 或 healthy 状态。

### 6.3 查看日志

如果某个服务启动失败，可以看日志：

```bash
docker compose logs -f app
docker compose logs -f redis
docker compose logs -f postgres
docker compose logs -f qdrant
docker compose logs -f nginx
```

---

## 7. 初始化数据库和基础数据

### 7.1 初始化数据库表

应用启动后会自动尝试初始化数据库，但你也可以手动触发一次：

```bash
docker compose exec app python -c "from services.db import init_db; from config import Config; init_db(Config())"
```

### 7.2 初始化 RAG 数据

如果你想让检索问答功能真正可用，需要先 ingest 文档：

```bash
docker compose exec app python ingest_rag_data.py --dir ./data
```

如果你有自己的文档目录，也可以改成：

```bash
docker compose exec app python ingest_rag_data.py --dir ./data/raw
```

---

## 8. 访问系统

部署完成后，可以访问这些地址：

### 8.1 应用入口

- 首页：http://localhost:8080
- 患者端：http://localhost:8080/app
- 医生端：http://localhost:8080/doctor
- reviewer：http://localhost:8080/reviewer

### 8.2 其他服务

- MinIO 控制台：http://localhost:9001
  - 用户名：minioadmin
  - 密码：minioadmin
- Prometheus：http://localhost:9090
- Grafana：http://localhost:3000
  - 用户名：admin
  - 密码：admin

---

## 9. 完整功能验证清单

下面这套验证很重要，能确保你部署出来的系统不是“只启动了”。

### 9.1 验证应用健康

```bash
curl http://localhost:8080/health
```

应该返回类似：

```json
{"status":"healthy"}
```

### 9.2 验证注册登录

在首页或接口中注册一个用户。

如果开启了认证，第一次注册用户会成为管理员。

建议测试：

- 注册
- 登录
- 访问受保护页面

### 9.3 验证聊天对话

进入患者端页面或调用接口，测试普通问答。

如果 LLM 配置正确，应该能正常返回回答。

### 9.4 验证 RAG 检索

把一些医学文档导入后，测试问答是否基于知识库结果进行回答。

如果没有 ingest 数据，RAG 功能可能不明显。

### 9.5 验证上传与对象存储

上传一张图片，测试是否能保存并返回结果。

### 9.6 验证异步任务

如果开启了任务队列，可以测试：

- /ingest
- /tasks/{id}

### 9.7 验证监控

访问：

- Prometheus
- Grafana

确认指标已经采集到。

---

## 10. 完整功能建议开启的配置

### 10.1 必开项

```env
ENABLE_AUTH=true
ENABLE_METRICS=true
```

### 10.2 推荐开启

```env
CHECKPOINTER_BACKEND=redis
ENABLE_CACHE=true
ENABLE_RATE_LIMIT=true
ENABLE_TASK_QUEUE=true
OBJECT_STORAGE_BACKEND=s3
```

### 10.3 如果你想要更强的体验

```env
ENABLE_STREAMING=true
ENABLE_CRAG=true
```

> 说明：有些功能更重，默认关闭更稳妥。你可以先把核心功能跑通，再逐步打开高级能力。

---

## 11. 常见问题与解决方法

### 11.1 Docker 启动失败

检查：

```bash
docker compose logs
```

常见原因：

- 端口冲突
- Docker 资源不足
- 网络问题

### 11.2 依赖安装超时

可以重试：

```bash
docker compose build --no-cache app
```

### 11.3 RAG/向量检索不可用

通常是：

- Qdrant 没启动
- 数据没 ingest
- EMBEDDING 模型配置不正确

可以先检查：

```bash
docker compose ps
docker compose logs qdrant
```

### 11.4 LLM 调用失败

通常是：

- API Key 为空
- 网络不可访问
- base URL 配置错误

检查 .env 里的：

- OPENAI_API_KEY
- OPENAI_BASE_URL

### 11.5 数据库初始化失败

检查：

```bash
docker compose logs postgres
docker compose logs app
```

如果是 PostgreSQL 初始化问题，确认：

- 端口 5432 是否被占用
- .env 中 DATABASE_URL 是否正确

---

## 12. 推荐执行顺序

### 第一阶段：先把系统跑起来

1. 配置 .env
2. docker compose up --build -d
3. 查看服务状态
4. 访问 /health

### 第二阶段：验证核心业务

1. 注册/登录
2. 发送聊天
3. 上传图片
4. 初始化 RAG 数据

### 第三阶段：增强生产化能力

1. 开启 Redis/缓存/限流
2. 开启 MinIO 对象存储
3. 配置监控
4. 进一步配置 HTTPS / 域名 / 证书

---

## 13. 如果你想更接近企业级部署

### 13.1 本地准生产环境

直接使用 Docker Compose 即可，已经足够接近真实部署。

### 13.2 Kubernetes 本地模拟

如果你想更进一步，可以在本机装 kind 或 minikube，再部署 k8s 目录下的 YAML。

这样可以模拟：

- Deployment
- Service
- Ingress
- HPA
- 自动扩缩容

---

## 14. 最关键的一句话

如果你想“完整部署好当前项目完整功能”，最稳妥的方式是：

1. 先准备好 .env
2. 用 Docker Compose 拉起整套服务
3. 确保 LLM / 数据库 / Redis / Qdrant / MinIO 都通
4. 依次验证聊天、RAG、上传、监控和认证

---

## 15. 额外建议

- 建议优先部署 Docker Compose，再逐步扩展到 K8s。
- 优先保证核心功能可用，再开启高级能力。
- 如果没有外部模型服务，系统的大部分 AI 能力将无法完整体验。
- 生产环境必须使用 HTTPS、强密钥和安全配置。
