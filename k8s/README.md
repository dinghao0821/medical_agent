# Kubernetes deployment (P5)

Cloud-native manifests for the Multi-Agent Medical Assistant: stateless API with
HPA autoscaling, a Celery worker pool, in-cluster dependencies, an SSE-friendly
Ingress, and a GPU inference-serving skeleton.

## Prerequisites
- A Kubernetes cluster (`kubectl` configured) with:
  - **metrics-server** (for HPA)
  - **ingress-nginx** controller (for the Ingress)
  - (optional) an **NVIDIA device plugin** + GPU node pool for TorchServe
- The app image built and pushed to a registry your cluster can pull:
  ```
  docker build -t <registry>/medical-assistant:v1 .
  docker push <registry>/medical-assistant:v1
  ```

## Deploy
```
# 1) Secrets (never commit the filled file)
cp k8s/secret.example.yaml k8s/secret.yaml   # edit with real keys
kubectl apply -f k8s/secret.yaml

# 2) Point manifests at your image tag
cd k8s
kustomize edit set image medical-assistant=<registry>/medical-assistant:v1

# 3) Apply everything
kubectl apply -k .

# 4) (optional) GPU inference serving
kubectl apply -f inference-torchserve.yaml

# 5) Watch rollout
kubectl -n medical-assistant rollout status deploy/app
kubectl -n medical-assistant get pods,svc,hpa,ingress
```

## Verify (maps to P5 acceptance)
- **Autoscaling**: generate load; `kubectl -n medical-assistant get hpa -w` shows replicas scaling 3→N.
- **Rolling update, zero downtime**:
  ```
  kustomize edit set image medical-assistant=<registry>/medical-assistant:v2
  kubectl apply -k .
  kubectl -n medical-assistant rollout status deploy/app   # new pods ready before old removed
  # rollback if needed:
  kubectl -n medical-assistant rollout undo deploy/app
  ```
- **SSE**: `curl -N https://<host>/chat/stream ...` streams tokens (Ingress `proxy-buffering: off`).

## Canary release (nginx ingress, weight-based)
Deploy a second `app-canary` Deployment (image = new version, label `version: canary`)
with its own Service `app-canary`, then a canary Ingress splitting traffic:
```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"   # 10% to canary
```
Increase `canary-weight` gradually; promote by pointing the stable image to the new
tag and removing the canary. For richer strategies use **Argo Rollouts** / **Flagger**.

## Notes
- The API is **stateless**: session state lives in Redis (checkpointer), so replicas
  and rolling updates never lose conversations.
- In-cluster `dependencies.yaml` (redis/qdrant/postgres/minio) is demo-grade; use
  managed services or operators/Helm charts in production.
