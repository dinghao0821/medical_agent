<div align="center">

![logo](assets/logo_rounded.png)

# 👵 失能失智老年人康复与适老环境智能评估平台

### “十四五”国家重点研发计划方向 · 老年综合能力、居住环境与康复辅助器具检测评价

*An AI-powered multi-agent platform for disability/dementia screening, age-friendly environment evaluation and rehabilitation assistive technology assessment.*

<br/>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-1C3C3C?style=for-the-badge&logo=langgraph&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.13+-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)

![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📚 目录

- [项目概述](#-项目概述)
- [核心亮点](#-核心亮点)
- [系统架构](#-系统架构)
- [多智能体工作流](#-多智能体工作流)
- [技术栈](#-技术栈)
- [目录结构](#-目录结构)
- [快速开始](#-快速开始)
  - [方式一：Docker Compose（推荐，一键全栈）](#方式一docker-compose推荐一键全栈)
  - [方式二：单容器 Docker](#方式二单容器-docker)
  - [方式三：本地手动安装](#方式三本地手动安装)
- [环境变量配置](#-环境变量配置)
- [功能开关（渐进增强）](#-功能开关渐进增强)
- [数据摄取（RAG 知识库）](#-数据摄取rag-知识库)
- [API 端点](#-api-端点)
- [Kubernetes 部署](#-kubernetes-部署)
- [可观测性与评估](#-可观测性与评估)
- [测试](#-测试)
- [安全与医疗合规](#-安全与医疗合规)
- [免责声明](#-免责声明)
- [许可证](#-许可证)

---

## 📌 项目概述

本项目面向“十四五”国家重点研发计划 **“失能失智老年人居住环境及康复辅助器具检测与评价关键技术研究”** 方向，将原有通用医疗多智能体平台聚焦为 **失能失智老年人康复与适老环境智能评估平台**。系统以老年人为中心，协同完成 **日常生活活动能力（ADL）筛查、认知功能风险初筛、居住环境安全评价、康复辅助器具适配、老年健康知识问答及专业人员复核**。平台基于 **LangGraph** 构建图式编排工作流，并保留原有医学影像与企业级工程能力：

- 🤖 **大语言模型（LLM）** —— 会话、推理、路由与生成
- 🖼️ **计算机视觉模型（PyTorch）** —— 脑肿瘤、胸片、皮肤病变影像分析
- 📚 **检索增强生成（RAG）** —— 基于 Qdrant 向量库的混合检索
- 🌐 **实时联网搜索** —— Tavily / PubMed 获取最新医学研究
- 👨‍⚕️ **人工审核闭环（HITL）** —— 医生对 AI 影像诊断结果进行审批

> 本项目从「单机原型」演进为「**支撑高并发的企业级平台**」，落地了 P1–P5 五阶段架构升级：会话隔离、异步计算、认证合规、可观测性、云原生弹性伸缩。所有企业级能力均为 **可开关、可降级、零回归**，默认关闭时行为与原型完全一致。

---

## 👵 老年专项能力

- **失能与照护需求筛查**：围绕洗澡、穿衣、如厕、移位、控便、进食开展结构化 ADL 评估和依赖分层。
- **失智风险初筛**：从定向力、延迟回忆与执行功能开展风险分层，明确筛查不等于诊断，并提示听视力、教育程度、情绪、谵妄和药物等混杂因素。
- **居住环境检测评价**：覆盖出入口、卫生间、防滑、照明、跌倒障碍、走失、用火、服药与紧急呼叫风险，给出适老化改造优先级。
- **康复辅助器具适配**：综合移动、移位、上肢、认知、跌倒史、独居和使用环境，形成辅具类别与专业适配建议。
- **人机协同专业复核**：医生须上传执业资格证并经管理员审核通过，才可查看老年人评估与医学影像资料。
- **科研友好 API**：`GET /elderly/assessments/catalog` 与 `POST /elderly/assessments` 支持四类标准化评估的数据采集和系统集成。

> 所有专项评估均属于科研筛查与辅助评价，不构成疾病诊断、伤残等级认定或治疗建议。

---

## ✨ 核心亮点

### 🧠 多智能体智能
- **图式编排（LangGraph）**：结构化路由 + 智能体间置信度切换（Agent-to-Agent Handoff）
- **置信度路由**：RAG 检索置信度低于阈值时自动降级到 Web Search，抑制幻觉
- **原生人工审核（`interrupt()`）**：影像诊断结果暂停工作流，等待医生审批后恢复
- **自适应多 Agent 审议**：仅在高风险医疗回答触发时，并行调用老年医学安全、临床药学和证据不确定性三个 Reviewer，再由 Chair Agent 合成更谨慎的最终回答
- **Test-time Compute**：根据任务风险动态增加推理预算；普通问答保持单 Agent 低延迟，高风险问题升级为 Specialist Debate，并记录结构化审议元数据

### 🔍 高级 Agentic RAG
- **Docling 文档解析**：从 PDF 抽取文本、表格、图像，并生成 LLM 图像摘要后嵌入
- **语义分块**：LLM 感知结构边界的智能切分（Semantic Chunking）
- **查询扩展**：LLM 补充医学领域相关术语，提升召回
- **混合检索**：Qdrant `Dense 向量 + BM25 稀疏关键词` 融合搜索
- **交叉编码器重排**：HuggingFace Cross-Encoder 对检索结果重排序
- **Parent-Child（小-大）检索**：小块精确命中、大块提供丰富上下文
- **CRAG / Self-RAG**：文档相关性打分 + 生成后幻觉自反思，不足时回退联网
- **句级引用溯源**：响应附带来源文档与参考图片链接

### 🏥 医学影像诊断（PyTorch）
- **脑肿瘤**：语义分割（Brain Tumor Segmentation）
- **胸部 X 光**：COVID / 疾病分类
- **皮肤病变**：病灶分割 / 分类
- 诊断结果 **强制进入医生审核队列**，通过 SSE 实时推送最终结论给患者

### 🛡️ 智能体安全加固
- **Prompt 注入检测**：本地启发式过滤，拦截提示注入
- **不可信内容隔离（Spotlighting）**：将检索/联网内容作为「数据」围栏化
- **输出泄漏防护**：扫描并脱敏泄漏的系统提示词与密钥
- **急症红旗分诊**：心梗、卒中、过敏、自伤等危急信号即时预警（默认开启）
- **输入/输出护栏**：确保回答安全、无偏见、可靠

### 🏗️ 企业级工程能力
- **无状态多副本**：Redis 外部化 checkpointer，会话跨副本共享、隔离不串话
- **异步任务队列**：Celery + Redis 解耦影像推理 / RAG 摄取等重任务
- **对象存储**：MinIO / S3 接管上传与分割结果（可降级本地磁盘）
- **认证授权**：JWT/OAuth2 + RBAC（患者 / 医生 / 管理员），Refresh Token 轮换与吊销
- **合规**：审计留痕、PII 脱敏、静态加密
- **可观测性**：Prometheus 指标 + 结构化日志 + LangSmith 链路追踪
- **成本治理**：Token 计量 + 每日预算、语义缓存、对话摘要压缩
- **云原生**：Docker Compose 全栈 + Kubernetes（HPA 自动扩缩容）

### 🎙️ 交互能力
- **语音交互**：Eleven Labs 语音转文字 / 文字转语音
- **SSE 流式响应**：Token / Step 级流式输出
- **直观 Web UI**：面向患者与医生的双端界面

---

## 🏛️ 系统架构

```mermaid
flowchart TB
    U[客户端 / 前端 SPA] --> GW["API 网关 / Nginx<br/>负载均衡 · 限流 · TLS · SSE 透传"]
    GW --> APP["FastAPI 无状态应用副本 x N<br/>请求编排 · Agent 调度"]

    APP --> RD[("Redis<br/>会话 checkpointer / 缓存 / 限流 / Broker")]
    APP --> PG[("PostgreSQL<br/>用户 / 会话历史 / 审计日志")]
    APP --> QD[("Qdrant<br/>向量混合检索")]
    APP --> MQ[消息队列 Broker]

    MQ --> WK["异步 Worker 池<br/>Celery"]
    WK --> INF["PyTorch 影像推理<br/>脑肿瘤 / 胸片 / 皮肤"]
    WK --> OBJ[("对象存储<br/>S3 / MinIO")]

    APP --> EXT["外部工具<br/>Tavily / PubMed / Eleven Labs"]
    APP --> OBS["可观测性<br/>Prometheus / Grafana / LangSmith"]
```

> 详细的企业级架构蓝图与 P1–P5 演进路线见 [`docs/ENTERPRISE_ARCHITECTURE.md`](docs/ENTERPRISE_ARCHITECTURE.md)。

---

## 🔀 多智能体工作流

用户请求进入后，`analyze_input` 会依次执行安全与预处理，再进行智能体路由：

```
注入检测 → 急症分诊 → 工具调用 → 澄清追问 → 路由决策
```

| 智能体 | 职责 | 触发条件 |
|--------|------|----------|
| **CONVERSATION_AGENT** | 通用医疗对话 | 日常问答、闲聊 |
| **RAG_AGENT** | 知识库检索问答（Agentic RAG） | 命中本地医学知识库 |
| **WEB_SEARCH_AGENT** | 实时联网研究（Tavily / PubMed） | RAG 置信度不足时自动切换 |
| **DEEP_RESEARCH_AGENT** | 计划-执行-反思的深度研究报告 | 复杂研究型问题（可选） |
| **BRAIN_TUMOR_AGENT** | 脑肿瘤影像分割 | 上传脑部 MRI |
| **CHEST_XRAY_AGENT** | 胸片疾病分类 | 上传胸部 X 光 |
| **SKIN_LESION_AGENT** | 皮肤病变分析 | 上传皮肤病灶图像 |
| **TOOL_AGENT** | BMI / 单位换算 / 药物查询 | 命中工具关键词（可选） |

> 三类 **影像诊断智能体输出必须经医生人工审核**（HITL），审批结果通过 SSE 实时回推。
> Agent 层细节见 [`agents/README.md`](agents/README.md) 与 [`docs/AGENT_CAPABILITIES.md`](docs/AGENT_CAPABILITIES.md)。

---

## 🛠️ 技术栈

| 领域 | 技术选型 |
|------|----------|
| **后端框架** | FastAPI + Uvicorn / Gunicorn（多 worker） |
| **智能体编排** | LangGraph + LangChain（图式工作流、原生 HITL、checkpointer） |
| **大语言模型** | OpenAI 兼容接口（默认 DashScope Qwen：`qwen3-max` / `qwen-vl-plus`；可切换 OpenAI / Azure / 本地） |
| **向量数据库** | Qdrant（Dense + BM25 混合检索） |
| **Embedding** | `text-embedding-v4`（1536 维，可配置） |
| **重排模型** | HuggingFace Cross-Encoder `ms-marco-TinyBERT-L-6` |
| **文档解析** | Docling（文本 / 表格 / 图像抽取 + OCR） |
| **医学影像** | PyTorch 2.7 + torchvision（分割 / 分类 / 检测） |
| **语音处理** | Eleven Labs API（STT / TTS） |
| **联网搜索** | Tavily API + PubMed |
| **会话 / 缓存 / 限流 / Broker** | Redis 7 |
| **关系数据库** | PostgreSQL 16（默认降级 SQLite） |
| **对象存储** | MinIO / S3（boto3） |
| **异步任务** | Celery + Redis |
| **认证授权** | python-jose（JWT/OAuth2）+ passlib/bcrypt + RBAC |
| **可观测性** | Prometheus + Grafana + LangSmith |
| **评估** | RAGAS + 黄金测试集（routing / safety / injection） |
| **前端** | HTML / CSS / JavaScript（Jinja2 模板） |
| **部署** | Docker / Docker Compose / Kubernetes（HPA） |
| **测试** | Pytest |

---

## 📁 目录结构

```
Multi-Agent-Medical-Assistant/
├── app.py                      # FastAPI 应用入口与全部 API 端点
├── config.py                   # 集中式配置（LLM/RAG/CV/认证/可观测性等）
├── ingest_rag_data.py          # RAG 知识库文档摄取脚本
├── agents/                     # 多智能体核心
│   ├── agent_decision.py       # LangGraph 编排 / 路由 / HITL 中枢
│   ├── rag_agent/              # Agentic RAG（解析/分块/扩展/检索/重排/生成/引用/CRAG）
│   ├── image_analysis_agent/   # 脑肿瘤 / 胸片 / 皮肤病变 影像推理
│   ├── web_search_processor_agent/  # Tavily + PubMed 联网研究
│   ├── deep_research_agent/    # 计划-执行-反思 深度研究
│   ├── guardrails/             # 注入过滤/急症分诊/澄清/安全评审
│   ├── tools/                  # 结构化工具框架（BMI/换算/药物）
│   └── session/                # checkpointer 工厂（Redis / 内存）
├── services/                   # 企业级服务层
│   ├── auth.py / token_store.py        # 认证 / Refresh Token
│   ├── db.py / models.py               # 数据库与 ORM
│   ├── cache.py / semantic_cache.py    # 缓存 / 语义缓存
│   ├── rate_limiter.py                 # 分布式限流
│   ├── celery_app.py / tasks.py        # 异步任务队列
│   ├── object_storage.py               # S3 / MinIO 对象存储
│   ├── observability.py / agent_trace.py  # 指标 / 链路追踪
│   ├── audit.py / pii.py               # 审计 / PII 脱敏
│   ├── cost_tracker.py                 # 成本治理
│   ├── long_term_memory.py             # 跨会话长期记忆
│   └── review_queue.py                 # 医生审核队列（HITL）
├── evaluation/                 # RAGAS 评估 + 黄金测试集
├── tests/                      # Pytest 单元 / 集成测试
├── templates/                  # 前端页面（index / doctor / landing）
├── data/                       # 向量库 / 解析文档 / 原始资料 / SQLite
├── sample_images/              # 影像诊断示例图片
├── docs/                       # 架构与实现文档（P1–P5）
├── k8s/                        # Kubernetes 部署清单
├── docker-compose.yml          # 全栈编排（app/redis/qdrant/pg/minio/nginx/监控）
├── Dockerfile                  # 应用镜像（gunicorn + Uvicorn workers）
├── nginx.conf                  # 反向代理（SSE 友好）
├── prometheus.yml              # 指标采集配置
└── requirements.txt            # Python 依赖
```

---

## 🚀 快速开始

### 前置条件

- **Docker & Docker Compose**（推荐部署方式）或 **Python 3.11+**
- 相关服务的 **API Key**（LLM、Eleven Labs、Tavily、HuggingFace，见下文）
- 首次运行会自动下载多个模型（OCR、CV 模型、重排模型），请耐心等待

---

### 方式一：Docker Compose（推荐，一键全栈）

一键拉起完整企业级技术栈：`app + redis + qdrant + postgres + minio + nginx + prometheus + grafana`。

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd Multi-Agent-Medical-Assistant

# 2. 创建并填写环境变量（见下方「环境变量配置」）
cp .env.example .env   # 若无示例文件，请手动创建 .env

# 3. 构建并启动全栈
docker compose up --build -d

# 4. 查看服务状态
docker compose ps
```

服务访问地址：

| 服务 | 地址 |
|------|------|
| 应用（经 Nginx） | http://localhost:8080 |
| 应用（直连） | http://localhost:8000 |
| Qdrant | http://localhost:6333 |
| MinIO 控制台 | http://localhost:9001 （minioadmin/minioadmin） |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 （admin/admin） |

---

### 方式二：单容器 Docker

```bash
# 构建镜像
docker build -t medical-assistant .

# 运行容器
docker run -d --name medical-assistant-app -p 8000:8000 --env-file .env medical-assistant

# 查看日志 / 健康状态
docker logs -f medical-assistant-app
docker inspect --format='{{.State.Health.Status}}' medical-assistant-app
```

应用地址：http://localhost:8000

---

### 方式三：本地手动安装

> ⚠️ 语音功能依赖 **ffmpeg**，请先安装。

```bash
# 1. 创建并激活虚拟环境（conda 示例）
conda create -n medical python=3.11 -y
conda activate medical
conda install -c conda-forge ffmpeg -y        # Windows 亦可：winget install ffmpeg

# 或使用 venv
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 创建 .env 并填入 API Key（见下文）

# 4. 启动应用
python app.py
```

应用地址：http://localhost:8000

---

## 🔑 环境变量配置

在项目根目录创建 `.env` 文件：

```bash
# ===== LLM（OpenAI 兼容接口，默认 DashScope/Qwen）=====
# DashScope 示例：OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=
OPENAI_BASE_URL=
DASHSCOPE_API_KEY=              # 向后兼容，仅有此项也可
MODEL_NAME=qwen3-max            # 主对话/推理模型
VISION_MODEL=qwen-vl-plus       # 视觉/图像摘要模型
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1536              # 需与 Qdrant collection 维度一致

# ===== 语音（Eleven Labs，注册送免费额度）=====
ELEVEN_LABS_API_KEY=

# ===== 联网搜索（Tavily，注册送免费额度）=====
TAVILY_API_KEY=

# ===== 重排模型（HuggingFace Token）=====
HUGGINGFACE_TOKEN=

# ===== Qdrant（本地模式无需填写；使用远程 server 版才需要）=====
QDRANT_URL=
QDRANT_API_KEY=
```

> **切换模型**：
> - 使用 **OpenAI / Azure OpenAI**：调整 `MODEL_NAME`、`EMBEDDING_MODEL` 与 `OPENAI_BASE_URL`，无需改代码。
> - 使用 **本地模型**：可能需要在 `config.py` 与 `agents/` 中做相应适配。
> - 注意：环境变量名后不要有多余空格，值也不要有尾随空格。

---

## 🎚️ 功能开关（渐进增强）

所有企业级能力通过环境变量按需开启，**默认关闭时行为与原型一致、零回归**。常用开关：

```bash
# ---- 部署 / 扩展 ----
WORKERS=2                          # gunicorn worker 数
CHECKPOINTER_BACKEND=redis         # memory | redis（多副本会话共享）
REDIS_URL=redis://redis:6379/0

# ---- 性能 / 缓存 / 限流 ----
ENABLE_CACHE=true                  # Redis 响应缓存
ENABLE_SEMANTIC_CACHE=true         # 语义相似缓存
ENABLE_RATE_LIMIT=true             # 分布式限流
ENABLE_TASK_QUEUE=true             # Celery 异步任务

# ---- 对象存储 ----
OBJECT_STORAGE_BACKEND=s3          # local | s3(MinIO)

# ---- 认证 / 合规 ----
ENABLE_AUTH=true                   # JWT/OAuth2 + RBAC
ENABLE_REFRESH_TOKEN=true          # Refresh Token 轮换 + 吊销
ENABLE_AUDIT=true                  # 审计日志
ENABLE_PII_MASKING=true            # PII 脱敏
DATABASE_URL=postgresql+psycopg2://medical:medical@postgres:5432/medical_assistant
DOCTOR_INVITE_CODE=                # 医生注册邀请码

# ---- 可观测性 ----
ENABLE_METRICS=true                # Prometheus 指标（默认开）
LANGSMITH_TRACING=true             # LangSmith 链路追踪
LANGSMITH_API_KEY=

# ---- Agent 能力增强 ----
ENABLE_EMERGENCY_TRIAGE=true       # 急症红旗分诊（默认开，安全优先）
ENABLE_CRAG=true                   # CRAG/Self-RAG（默认开）
ENABLE_LONG_TERM_MEMORY=true       # 跨会话长期记忆
ENABLE_TOOLS=true                  # 结构化工具（BMI/换算/药物）
ENABLE_CLARIFICATION=true          # 主动澄清追问
ENABLE_DEEP_RESEARCH=true          # 深度研究智能体
DEEP_RESEARCH_PARALLEL=true        # 并行执行独立研究子任务
DEEP_RESEARCH_MAX_WORKERS=4        # 并行任务上限
DEEP_RESEARCH_EVIDENCE_MIN_COVERAGE=0.6 # 最低可溯源证据覆盖率
ENABLE_STREAMING=true              # SSE 流式响应（默认开）
ENABLE_COST_TRACKING=true          # 成本 / Token 预算治理
```

> 完整开关清单见 [`config.py`](config.py) 及 [`docs/`](docs/) 下各阶段实现文档。

---

## 📥 数据摄取（RAG 知识库）

将医学文档解析、分块、嵌入并写入 Qdrant 向量库：

```bash
# 本地运行：摄取单个文档
python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf

# 摄取整个目录
python ingest_rag_data.py --dir ./data/raw

# Docker 容器内摄取
docker exec medical-assistant-app python ingest_rag_data.py --dir ./data/raw
```

---

## 🌐 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` `/app` `/doctor` | 前端页面（落地页 / 患者端 / 医生端） |
| GET | `/health` | 健康检查 |
| GET | `/metrics` | Prometheus 指标 |
| POST | `/auth/register` `/auth/register/doctor` | 用户 / 医生注册 |
| POST | `/auth/login` `/auth/refresh` `/auth/logout` | 登录 / 刷新 / 登出 |
| GET | `/auth/me` | 当前用户信息 |
| POST | `/chat` | 对话（多智能体路由） |
| POST | `/chat/stream` | SSE 流式对话 |
| POST | `/upload` | 上传医学影像并诊断 |
| POST | `/ingest` | 摄取文档到知识库 |
| GET | `/tasks/{task_id}` | 查询异步任务状态 |
| POST | `/validate` | 提交人工审核结果 |
| GET | `/review/pending` `/review/{case_uid}` | 待审核队列 / 详情 |
| POST | `/review/{case_uid}` | 医生审批诊断 |
| GET | `/review/stream/{case_uid}` | SSE 推送审核结论给患者 |
| POST | `/transcribe` `/generate-speech` | 语音转文字 / 文字转语音 |
| GET/POST | `/memory` | 跨会话长期记忆读写 |

---

## ☸️ Kubernetes 部署

`k8s/` 目录提供完整的云原生部署清单（Deployment / Service / Ingress / ConfigMap / Secret / HPA / Worker / 推理服务）。

```bash
# 1. 配置 Secret（参考 secret.example.yaml）
cp k8s/secret.example.yaml k8s/secret.yaml   # 填入真实 Key

# 2. 使用 Kustomize 一键部署
kubectl apply -k k8s/

# 3. 查看状态
kubectl get pods -n medical-assistant
```

支持基于负载的 **HPA 自动扩缩容**、滚动发布、健康探针。详见 [`k8s/README.md`](k8s/README.md)。

---

## 📊 可观测性与评估

- **指标**：`/metrics` 暴露 Prometheus 指标，Grafana 面板可视化
- **链路追踪**：开启 `LANGSMITH_TRACING` 后可在 LangSmith 观测 Agent 决策链
- **结构化日志**：`ENABLE_JSON_LOGS=true` 输出 JSON 日志（便于 Loki 采集）
- **RAG 评估**：使用 RAGAS 与黄金测试集

```bash
# RAGAS RAG 质量评估
python evaluation/ragas_eval.py

# 路由 / 安全 / 注入 黄金集评估
python evaluation/run_eval.py
```

---

## 🧪 测试

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行全部测试
pytest -q

# 运行指定测试
pytest tests/test_agent_enhancements.py -v
```

测试覆盖：认证、缓存、限流、注入过滤、PII 脱敏、成本/语义缓存、工具/分诊/澄清/记忆、引用与 checkpointer 等。

---

## 🔐 安全与医疗合规

对标 HIPAA / 国内等保 / 数据安全法的关键实践：

1. **传输与存储加密**：全链路 TLS；数据库与对象存储静态加密
2. **PII 脱敏**：日志与追踪中对患者敏感信息脱敏
3. **审计留痕**：诊断类操作全程可追溯，写入不可篡改审计日志
4. **人工验证闭环**：影像诊断必须经医生 HITL 确认方可采信
5. **访问控制**：JWT/OAuth2 + RBAC 分级授权，最小权限
6. **可解释性**：保留 Agent 决策 reasoning 与检索来源，支持诊断依据回溯

---

## ⚠️ 免责声明

本项目仅用于 **研究、学习与技术演示**，**不构成任何医疗建议、诊断或治疗依据**。所有 AI 生成的医学内容必须经过持证医疗专业人员审核。请勿将本系统用于真实临床决策。如遇紧急医疗状况，请立即就医或拨打急救电话。

---

## ⚖️ 许可证

本项目基于 **Apache-2.0 License** 开源，详见 [LICENSE](LICENSE)。

---

<div align="center">

⭐ 如果这个项目对你有帮助，欢迎点个 Star！

</div>
� 如果这个项目对你有帮助，欢迎点个 Star！

</div>
