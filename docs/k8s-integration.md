# Hetu-DiT 集成到 Kubernetes

## 目标架构

推荐使用 `KubeRay/Ray Operator` 承载 Hetu-DiT。

- Ray head Pod：运行 Ray head 和 Hetu-DiT API sidecar。
- Ray worker Pods：运行 GPU Ray workers。
- PVC：挂载模型目录、结果目录和 profile cache 目录。
- Service：对外暴露 Hetu-DiT API。

仓库中的示例 manifest 位于 [`k8s/raycluster.yaml`](/home/youth/hetudit/Hetu-DiT/k8s/raycluster.yaml)。

## 当前实现约束

当前代码实现有两个必须保留的约束：

1. driver 进程要求有一组 Ray worker 和它在同一个节点。
2. 多机逻辑默认按 `8 GPU = 1 machine` 分组。

因此示例 `RayCluster` 默认让 head 也占用一个完整 8-GPU 节点。

## 运行时配置

容器启动 Hetu-DiT API 时，建议至少配置：

- `--host=0.0.0.0`
- `--port=8000`
- `--ray-address=127.0.0.1:6379`（API sidecar 与 Ray head 同 Pod 时）
- `--model=/models/<model-dir>`
- `--results_dir=/results`
- `--profile-cache-dir=/profile-cache`
- `--machine_nums=<RayCluster 中 8-GPU 节点总数>`

如果从单节点扩展到多节点：

- 将 `workerGroupSpecs[].replicas` 提升到对应的 GPU 节点数。
- 同步把 `--machine_nums` 改成 `1 + worker replicas`。
- 确保所有 Ray 节点都挂载同一份模型目录和结果目录，或改造成节点本地缓存分发。

## 存储建议

当前仓库默认使用 PVC 方案：

- `/models`：模型权重。
- `/results`：推理输出图片或视频。
- `/profile-cache`：`ModelProfiler` 缓存。

多节点下三者都更适合 `ReadWriteMany` 存储类；如果集群只支持 `ReadWriteOnce`，需要额外引入对象存储或节点本地缓存同步方案。

示例 PVC 在 [`k8s/pvc.yaml`](/home/youth/hetudit/Hetu-DiT/k8s/pvc.yaml) 中显式写了 `storageClassName: shared-rwx`。这是占位值，部署前需要替换成集群里实际可用的 RWX StorageClass。

## 健康检查与接口

服务现在提供：

- `/health`：进程级 liveness。
- `/readyz`：检查 engine 和 Ray worker 是否已就绪。
- `/status/{task_id}`：查询任务状态。

这意味着 K8s 可以直接把：

- livenessProbe 指向 `/health`
- readinessProbe 指向 `/readyz`

## 部署步骤

1. 构建镜像：

```bash
docker build -t hetudit:latest .
```

2. 安装 KubeRay Operator。

3. 准备支持 `ReadWriteMany` 的存储类，或修改 [`k8s/pvc.yaml`](/home/youth/hetudit/Hetu-DiT/k8s/pvc.yaml)。

4. 将模型放到 `hetudit-models` PVC 对应目录。

5. 应用 manifests：

```bash
kubectl apply -k k8s
```

6. 通过 Service 访问 API，或挂接 Ingress。

Ingress 示例位于 [`k8s/ingress.example.yaml`](/home/youth/hetudit/Hetu-DiT/k8s/ingress.example.yaml)。

## 可用性与安全

- [`k8s/pdb.yaml`](/home/youth/hetudit/Hetu-DiT/k8s/pdb.yaml) 为 head pod 提供最小可用保护，避免节点维护时被直接驱逐。
- 这个 PDB 对单副本 head 的实际效果是“阻止自愿驱逐”；节点维护前需要先移除 PDB，或先手动迁移/停机。
- [`k8s/networkpolicy.yaml`](/home/youth/hetudit/Hetu-DiT/k8s/networkpolicy.yaml) 将 worker 通信端口、API 访问端口和 dashboard 管理端口拆分开控制。
- API 入口默认只放行两类来源：
  - 带 `hetudit-access: "true"` 标签的同 namespace Pod
  - `ingress-nginx` namespace 中带 `app.kubernetes.io/name=ingress-nginx` 的 Pod
- dashboard `8265` 默认只放行带 `hetudit-admin: "true"` 标签的同 namespace Pod。
- NetworkPolicy 还显式允许到 `hetudit` namespace 的集群内通信，以及到 `kube-dns` 的 DNS egress；如果集群 DNS 标签不同，需要按实际环境修改。

## `machine_nums` 配置

示例 `RayCluster` 不再把 `--machine_nums` 写死在命令行，而是通过 `HETUDIT_MACHINE_NUMS` 环境变量传入。

- 单节点 8-GPU 部署：`HETUDIT_MACHINE_NUMS=1`
- 1 个 head + 2 个 8-GPU worker 节点：`HETUDIT_MACHINE_NUMS=3`

扩缩容 `workerGroupSpecs[].replicas` 时，要同步更新这个值。