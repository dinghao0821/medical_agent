# P5 实现说明与验收 —— 云原生弹性扩展

> 对应路线图 P5：**Kubernetes 编排（Deployment/Service/Ingress、ConfigMap/Secret、HPA）、推理服务化（TorchServe/Triton + GPU）、多副本滚动发布 / 健康探针 / 灰度**。
> 交付形态为 K8s 清单（`k8s/`），需要一个集群才能运行；应用代码无需改动（无状态 + Redis 会话共享是前提，P1 已具备）。

---

## 1. 交付内容（`k8s/`）

| 文件 | 内容 |
|------|------|
| `namespace.yaml` | 独立命名空间 `medical-assistant` |
| `configmap.yaml` | 非敏感配置（连接串、各阶段开关，指向集群内服务 DNS） |
| `secret.example.yaml` | 密钥模板（API Key/JWT/加密key/对象存储凭据）→ 复制为 `secret.yaml` 填写 |
| `dependencies.yaml` | 集群内 redis / qdrant / postgres / minio（PVC+Deployment+Service，demo 级） |
| `app.yaml` | **API Deployment**（3 副本、RollingUpdate maxUnavailable=0、就绪/存活探针 `/health`、preStop 优雅退出、资源 requests/limits）+ Service |
| `worker.yaml` | Celery worker Deployment（独立伸缩、exec 就绪探针） |
| `hpa.yaml` | **HPA**：API 按 CPU70%/内存80% 在 3–10 副本自动扩缩；worker 1–6 副本 |
| `ingress.yaml` | nginx Ingress：`/chat/stream` SSE 关闭缓冲、放大 body/超时 |
| `inference-torchserve.yaml` | **推理服务化骨架**：TorchServe GPU Deployment+Service+HPA |
| `kustomization.yaml` | 聚合资源、统一命名空间与镜像 tag 覆盖 |
| `README.md` | 部署/验收/灰度操作手册 |

---

## 2. 关键设计

- **无状态水平扩展**：API 不保存本地状态，会话在 Redis checkpointer（P1），故可任意加副本、滚动更新不丢会话。
- **零停机滚动发布**：`maxSurge=1, maxUnavailable=0` + 就绪探针 + `preStop sleep` + 60s 优雅终止，保证在途慢推理请求完成。
- **自动扩缩容**：HPA 基于 CPU/内存；需集群装 `metrics-server`。
- **SSE 透传**：Ingress `nginx.ingress.kubernetes.io/proxy-buffering: "off"`，与 P1 的 nginx.conf 一致。
- **推理服务化**：把 PyTorch 影像模型剥离到 TorchServe（C++ 后端绕过 GIL、支持批处理与 GPU 复用），API 改为 HTTP 调用（见第 4 节，属后续集成）。
- **配置/密钥分离**：非敏感走 ConfigMap，敏感走 Secret；`secret.yaml` 不入库。

---

## 3. 部署与验收（对齐 P5 验收标准）

详见 `k8s/README.md`。核心：
```
cp k8s/secret.example.yaml k8s/secret.yaml   # 填写
kubectl apply -f k8s/secret.yaml
cd k8s && kustomize edit set image medical-assistant=<registry>/medical-assistant:v1
kubectl apply -k .
```

| 验收项 | 操作 | 期望 |
|--------|------|------|
| **按负载自动扩缩容** | 压测 + `kubectl get hpa -w` | 副本 3→N 随 CPU 上升，回落时缩回 |
| **滚动发布不中断** | 换镜像 tag 重新 apply，边发压测流量 | `rollout status` 平滑；无 5xx 抖动；可 `rollout undo` |
| **推理 GPU 利用率** | 部署 TorchServe 到 GPU 节点、迁移推理 | 推理批处理 + GPU 复用，API pod CPU 下降 |
| 健康探针 | `kubectl get pods` | READY 前不接流量；异常自动重启 |

---

## 4. 已知边界 / 后续

- **TorchServe 为骨架**：需先用 `torch-model-archiver` 把 brain_tumor / skin_lesion / chest_xray 打成 `.mar`，通过 PVC 或 initContainer（从对象存储下载）挂载；再把 `agents/image_analysis_agent/*` 的进程内推理改为调用 `http://inference:8080/predictions/<model>`。这是把"推理 Worker 池"完全落地的后续集成点。
- **依赖服务 demo 级**：生产用托管服务（ElastiCache / 托管 PostgreSQL / Qdrant Cloud / S3）或官方 Operator/Helm。
- **灰度**：README 给了 nginx ingress 权重灰度；更完善用 Argo Rollouts / Flagger（渐进式 + 自动回滚 + 指标门禁）。
- **GPU HPA**：CPU HPA 为起步；GPU/队列深度指标扩缩需 KEDA 或 Prometheus Adapter。

---

## 5. 路线图收尾
至此 **P1–P5 全部落地**：致命修复与现代化地基 → 异步/缓存/对象存储 → 认证授权与合规 → 可观测性/评估/CI-CD → 云原生弹性。后续可深化：GraphRAG 知识图谱、多智能体 ensemble 诊断、长期记忆服务、自建 MCP Server、RAGAS 回归数据集扩充（见 `AGENT_MODERNIZATION.md`）。
