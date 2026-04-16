# Hetu-DiT 模块说明

## 整体结构

当前开源版 Hetu-DiT 是一个基于 `FastAPI + Ray + 多 GPU worker` 的 DiT 推理服务。系统不是传统的单进程模型服务，而是由入口层接收请求、调度层决定并行策略、执行层组织 Ray worker、worker 层在 GPU 上实际跑模型，最终把结果写到输出目录。

核心目录：

- `hetu_dit/entrypoint`：服务入口和 HTTP API。
- `hetu_dit/config`：CLI 参数、运行时配置、并行配置。
- `hetu_dit/engine`：Ray 集群接入、executor 生命周期、异步调度。
- `hetu_dit/core/request_manager`：请求排队和 ILP 调度逻辑。
- `hetu_dit/executor`：将多个 Ray worker 组织成一个可执行实例。
- `hetu_dit/worker`：单 GPU worker，负责模型初始化和执行。
- `hetu_dit/model_executor`：模型包装、并行执行组件、缓存和 pipeline 适配。
- `hetu_dit/model_profiler.py`：profile cache 和性能估计。

## 启动链路

入口是 [`hetu_dit/entrypoint/api_server.py`](/home/youth/hetudit/Hetu-DiT/hetu_dit/entrypoint/api_server.py)。

启动时序：

1. 解析 CLI 参数，构造 `hetuDiTArgs`。
2. 通过 `create_config()` 生成 `EngineConfig` 和 `InputConfig`。
3. 根据 `--model-class` 选择对应的 Diffusers pipeline 包装。
4. 调用 `AsyncServingEngine.from_engine_args()` 初始化 Ray 连接和 worker 池。
5. `startup()` 中完成 executor 初始化、monitor 初始化，以及可选的 profiler 预热。
6. 后台启动 `process_queue()`，持续从 scheduler 拉取请求并执行。

## 模块职责

### 1. 入口层

[`hetu_dit/entrypoint/api_server.py`](/home/youth/hetudit/Hetu-DiT/hetu_dit/entrypoint/api_server.py)

- 提供 `/generate`、`/generate_with_workers`、`/health`、`/readyz`、`/status/{task_id}`。
- 把 HTTP 请求转成 `InputConfig` 和局部 `EngineConfig`。
- 调用并行度选择逻辑，决定当前请求使用多少 worker。
- 将请求放入 scheduler，并由后台队列消费。

### 2. 配置层

[`hetu_dit/config/args.py`](/home/youth/hetudit/Hetu-DiT/hetu_dit/config/args.py)

- 定义 CLI 参数。
- 支持模型路径、结果目录等运行时参数。
- 把 CLI 映射为 `ModelConfig`、`RuntimeConfig`、`ParallelConfig`、`EngineConfig` 和 `InputConfig`。

[`hetu_dit/config/config.py`](/home/youth/hetudit/Hetu-DiT/hetu_dit/config/config.py)

- 定义系统内的主要 dataclass。
- `RuntimeConfig` 包含推理策略和输出目录。
- `ParallelConfig` 抽象 DP/SP/TP/PP 组合。

### 3. 引擎层

[`hetu_dit/engine/async_serving_engine.py`](/home/youth/hetudit/Hetu-DiT/hetu_dit/engine/async_serving_engine.py)

- 是实际运行时中心。
- 管理所有 Ray workers 和 executor 池。
- 负责 worker 复用、重配置、stage-level 调度和多机分配。

[`hetu_dit/engine/ray_utils.py`](/home/youth/hetudit/Hetu-DiT/hetu_dit/engine/ray_utils.py)

- 封装 `ray.init()` 和 placement group 初始化。
- 定义 `RayWorkerHetudit`，作为真实 `Worker` 的 Ray actor 包装层。

### 4. 调度层

[`hetu_dit/core/request_manager/scheduler.py`](/home/youth/hetudit/Hetu-DiT/hetu_dit/core/request_manager/scheduler.py)

- 负责请求入队、出队和优先级管理。

[`hetu_dit/core/request_manager/efficient_ilp.py`](/home/youth/hetudit/Hetu-DiT/hetu_dit/core/request_manager/efficient_ilp.py)

- 提供 ILP 搜索逻辑，用于提升多请求下的资源分配效率。

### 5. 执行层

[`hetu_dit/executor/gpu_executor.py`](/home/youth/hetudit/Hetu-DiT/hetu_dit/executor/gpu_executor.py)

- `RayGPUExecutor` / `RayGPUExecutorAsync` 将一组 worker 组合成一个执行实例。
- 在执行前后管理 worker 的 `idle/ready/busy` 状态。

### 6. Worker 层

[`hetu_dit/worker/worker.py`](/home/youth/hetudit/Hetu-DiT/hetu_dit/worker/worker.py)

- 每个 worker 绑定一个 GPU。
- 初始化分布式环境、模型实例和并行 group。
- 执行 encode/diffusion/decode 或整条 pipeline。
- 将图片或视频写到 `RuntimeConfig.results_dir`。

### 7. 模型执行层

[`hetu_dit/model_executor`](/home/youth/hetudit/Hetu-DiT/hetu_dit/model_executor)

- 保存对不同模型族的 pipeline 包装。
- 提供 attention、cache、text encoder、VAE 等模块的并行执行适配。
- 是 Hetu-DiT 对原始 Diffusers 模型进行并行化改造的主体。

## 请求数据流

1. 客户端调用 `/generate`。
2. API 读取 prompt、高宽、帧数、steps 等参数。
3. 根据输入规模推断并行度。
4. 请求进入 scheduler。
5. `process_queue()` 取出任务并调用 `generate_image()`。
6. `AsyncServingEngine` 选择现有 executor 或重建 executor。
7. `RayGPUExecutorAsync` 把任务分发到一组 worker。
8. worker 执行模型并把结果写入输出目录。
9. API 把状态写入内存 `results_store`，客户端可通过 `/status/{task_id}` 查询。

## 与 K8s 集成直接相关的点

- API 现在支持 `--host`，容器内可监听 `0.0.0.0`。
- Ray 地址支持 `--ray-address` 或 `RAY_ADDRESS` 注入，适配集群内连接。
- 输出目录支持 `--results_dir` 或 `HETUDIT_RESULTS_DIR` 注入，适配 PVC。
- profile cache 支持 `--profile-cache-dir`，适配共享卷。
- 现有 machine 划分逻辑默认按 8 GPU 一组推导机器编号，K8s 资源规划需要保持这一假设。
- 当前 `results_store` 是进程内状态，Pod 重启后不会保留；如果后续需要可靠任务状态，需要外部存储。
