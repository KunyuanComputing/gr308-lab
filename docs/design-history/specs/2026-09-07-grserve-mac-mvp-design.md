# GRServe Mac MVP 设计规格

版本：2026-09-07  
状态：已确认，可进入实施计划  
范围：仅限 Mac 本地控制面、接口契约和 CPU/模拟执行后端

## 1. 目标

第一阶段交付一个可在 Apple Silicon Mac 上编译、运行和自动测试的 GRServe 最小系统，用来提前验证未来 GR308 推理框架的接口边界、请求生命周期、跨 rank 事务语义和故障处理。

该阶段不以模型吞吐为目标，也不对 GR308 驱动、算子、通信、Graph 或性能作任何结论。服务器继续独立执行冻结的 baseline；本地开发过程不登录服务器、不修改服务器文件、不启动或停止服务器服务。

## 2. 成功标准

Mac MVP 完成时必须同时满足：

- C++20 Engine Core 通过稳定的纯 C HAL ABI 调用执行后端，核心代码不依赖 Fantasy、CUDA、PyTorch 或 Python 运行时。
- Python API 适配层提供最小 OpenAI-compatible `/health`、`/v1/models` 和流式 `/v1/completions` 接口。
- 同一请求可以在 CPU Reference HAL 与 MockAsync HAL 上运行，并产生一致的协议级结果。
- 两 rank 模拟执行严格遵守 `RESERVED → PREPARED → EXECUTED → COMMIT_DECIDED → LOCAL_COMMITTED → PUBLISHED` 状态迁移。
- 只有所有 rank 完成本地提交后，token 才能对客户端可见；提交决定不确定时，整个 replica 进入 fatal 状态并停止接收请求。
- FaultInjection HAL 能稳定复现 prepare、execute、commit、timeout 和 rank-loss 故障，测试证明不会发布未提交 token。
- ABI 版本、能力位和模型 manifest 不兼容时，在任何执行发生前返回结构化错误。
- macOS arm64 本机构建、单元测试、接口测试和 sanitizer 测试可由一条项目命令重复执行。
- Linux amd64 至少完成交叉构建/容器构建验证；该结果只证明可移植性，不作为服务器运行资格。

## 3. 明确不做的内容

首个增量不包含：

- Qwen3.8-27B 权重加载或真实 Transformer 推理。
- GR308 Fant HAL、FHCCL、设备内存分配、真实 stream/event、Graph capture 或 GPU kernel。
- vLLM/SGLang 的完整插件实现。
- continuous batching、PagedAttention、prefix cache、speculative decoding、量化、LoRA 或多模型共驻。
- tokenizer 完整兼容、Chat Completions、Embeddings、Responses API、鉴权、TLS、计费或生产网关。
- 吞吐、TTFT、TPOT、显存占用或功耗优化结论。

这些功能只有在 Mac MVP 契约通过后，按独立规格逐项进入后续阶段。

## 4. 技术选型

| 层 | 选型 | 原因 |
|---|---|---|
| Engine Core | C++20 | 与未来 native runtime 方向一致，便于明确所有权、生命周期与并发约束 |
| HAL 边界 | C11 ABI、opaque handle | 避免 C++ ABI 和编译器绑定，允许未来独立装载 Fant 后端 |
| 构建 | CMake + Ninja | 同时支持 macOS arm64 与 Linux amd64，适合 C/C++ 测试和 sanitizer |
| C++ 测试 | GoogleTest | 覆盖状态机、HAL 契约、资源所有权与故障注入 |
| Engine RPC | protobuf + Unix Domain Socket | 保持版本化消息边界；本地无需引入外部网络依赖 |
| Python API | Python 3.11+、FastAPI、uvicorn | 快速验证 OpenAI-compatible HTTP/SSE 行为，且不进入核心状态所有权 |
| Python 测试 | pytest + httpx | 进行黑盒 API、SSE 和错误映射测试 |
| 动态检查 | ASan、UBSan；可用平台再启用 TSan | 提前发现越界、未定义行为与并发错误 |

Python 最低版本固定为 3.11。当前系统自带 Python 3.9 不进入项目环境，项目使用独立虚拟环境，避免污染系统 Python。

## 5. 总体架构

```text
HTTP Client
    │ OpenAI-compatible HTTP/SSE
    ▼
Python API Adapter
    │ versioned Engine RPC over UDS
    ▼
Engine Controller ─────────────── Replica Health/Fatal Gate
    │ owns request, schedule, transaction and publish decisions
    ├───────────────┐
    ▼               ▼
Rank Worker 0    Rank Worker 1
    │ C HAL ABI      │ C HAL ABI
    ▼                ▼
CPU / MockAsync / FaultInjection HAL
```

API Adapter 只做协议解析、RPC 调用、SSE 编码和错误映射，不保存调度、KV、GDN、RNG 或提交真相。Engine Controller 是请求和事务的唯一逻辑所有者。Rank Worker 只拥有本 rank 的本地上下文与 buffer，不能自行决定提交或发布。

## 6. 组件边界

### 6.1 公共类型与错误模型

公共类型定义 request ID、replica ID、rank ID、transaction ID、token、状态枚举、能力位和错误码。ID 使用强类型包装，禁止在 rank、request 和 transaction 之间隐式转换。

错误分为：

- `INVALID_ARGUMENT`：请求字段或 manifest 非法。
- `INCOMPATIBLE_VERSION`：RPC、HAL ABI 或模型 manifest 不兼容。
- `UNSUPPORTED_CAPABILITY`：后端缺少请求所需能力。
- `RESOURCE_EXHAUSTED`：逻辑资源预留失败。
- `DEADLINE_EXCEEDED`：阶段超时。
- `RANK_FAILURE`：rank 进程或本地执行失败。
- `REPLICA_FATAL`：提交决定不确定或出现不可恢复的一致性错误。
- `INTERNAL`：违反不变量或未分类内部错误。

所有错误都携带稳定错误码、可读消息、request/transaction 关联信息和 `retryable` 标志，不把堆栈或后端私有信息直接返回 HTTP 客户端。

### 6.2 C HAL ABI

HAL 暴露：

- ABI 主/次版本查询。
- 后端名称、构建标识和 capability bitset 查询。
- runtime、device、stream、event、buffer、executable 的 opaque handle 生命周期。
- host/device buffer 创建、销毁和显式 copy。
- stream submit、event record/wait、同步与状态查询。
- executable load、shape 校验和异步 launch。
- 最小 collective 接口占位能力查询；Mac MVP 不实现真实 collective。
- 线程本地的稳定错误对象读取。

ABI 规则：调用方分配的结构体必须带 `struct_size`；新增字段只能追加；主版本不匹配直接拒绝装载；后端不能把异常跨过 C 边界；所有成功创建的 handle 都必须有唯一且幂等的销毁路径。

### 6.3 三种本地 HAL

`CPU Reference HAL` 使用确定性 CPU 算子执行微型模型，是协议与数值的参考实现。

`MockAsync HAL` 不做真实矩阵计算，通过可控队列模拟异步 submit、event 完成、乱序完成和虚拟时钟，用于快速覆盖并发状态空间。

`FaultInjection HAL` 包装 MockAsync HAL，按声明式 fault plan 在指定 rank、阶段和调用序号注入失败、阻塞、超时或进程丢失。故障计划必须记录到测试输出，使失败可重复。

### 6.4 Engine Controller

Controller 负责：

- 请求校验、ID 分配和 deadline 固化。
- admission 和逻辑资源预留。
- 为所有 rank 创建同一个 transaction epoch。
- 驱动 prepare、execute、commit 和 publish。
- 维护 replica 健康状态。
- 将 token 以单调 sequence number 发布给 API Adapter。

Controller 不允许根据某个 rank 的先到结果提前发布。取消请求在 `COMMIT_DECIDED` 前可终止并回滚；达到 `COMMIT_DECIDED` 后必须完成一致提交，客户端断开只影响传输，不改变内部提交决定。

### 6.5 Rank Worker

每个 Rank Worker 管理本 rank 的本地资源句柄和 transaction shadow state。它接受 Controller 命令并返回带 epoch 的确认：

- `Prepare`：验证资源和输入，建立未公开的 next state。
- `Execute`：通过 HAL 产生候选 token 与 next state。
- `Commit`：原子切换本 rank 的 canonical state。
- `Abort`：仅在尚未决定提交时释放 shadow state。

迟到、重复或 epoch 不匹配的消息不得修改 canonical state。Prepare、Execute、Commit 和 Abort 都必须具备幂等定义。

### 6.6 MiniQwen38HybridModel

本地微型模型保留 Qwen3.8 的结构特征，而不复刻模型规模：

- 4 个 decoder layer，排列为 3 个 GDN/linear-attention 风格层加 1 个 full-attention 层。
- 小词表、小 hidden size、固定随机种子和随仓库冻结的小权重。
- 明确拆分 FullKV、GDN recurrent state、RNG state 与候选输出。
- greedy decode，固定输入产生稳定 golden token 序列和 state digest。

该模型只用于验证状态组织、事务切换和接口一致性。它的输出不宣称与真实 Qwen3.8-27B 数值等价。

### 6.7 Engine RPC 与 Python API

Engine RPC 消息包含协议版本、request ID、模型 ID、输入 token IDs、生成上限、deadline 和流式 token event。首个增量只接受预分词 token IDs，避免 tokenizer 差异污染核心测试。

HTTP 层提供：

- `GET /health`：返回 API、Engine 和 replica 状态。
- `GET /v1/models`：返回唯一的 `mini-qwen38-hybrid`。
- `POST /v1/completions`：接受 `prompt_token_ids`、`max_tokens`、`stream=true` 和 greedy 参数。

流式返回遵守 SSE 帧边界，最后输出 usage 和 `[DONE]`。HTTP 客户端断开会发送取消；若事务已经决定提交，Engine 完成内部提交但丢弃无法送达的网络帧。

## 7. 单 token 数据流

1. API Adapter 校验 HTTP 形状并向 Engine 发起版本化 RPC。
2. Controller 校验 manifest、replica 状态和 deadline，创建 transaction。
3. 所有 rank 完成逻辑资源 `RESERVED`。
4. 所有 rank 建立 shadow state 并返回 `PREPARED`。
5. Controller 命令执行；所有 rank 返回候选 token、state digest 和 `EXECUTED`。
6. Controller 校验 rank 结果一致性，然后持久化内存内 commit decision，进入 `COMMIT_DECIDED`。
7. 所有 rank 原子切换 canonical state，返回 `LOCAL_COMMITTED`。
8. Controller 将唯一 token event 标为 `PUBLISHED` 并交给 RPC/API 层。
9. 下一 token 使用新的 transaction ID 和递增 sequence number 重复流程。

任一 rank 在步骤 2–5 失败时，Controller 对已准备 rank 发出 Abort，token 不可见。步骤 6 后任一 rank 无法确认提交时，replica 转为 fatal，token仍不可见，且拒绝后续请求，直到显式重建 replica。

## 8. 并发与资源模型

首个增量使用一个 Controller 线程和每 rank 一个 Worker 线程，HAL completion 通过事件队列回到 Controller。所有状态迁移只在 Controller 线程发生，从设计上避免多写者竞争。

MVP admission 只维护确定性的逻辑配额：最大并发请求、每请求最大 token 和每 rank 最大 state slot。它不模拟真实显存字节，也不据此推断 GR308 容量。资源预留是全 rank 原子语义：任一 rank 无容量则所有 rank 都不保留资源。

## 9. Manifest 与兼容性门禁

启动时加载不可变 manifest，至少包含：

- schema version、model ID、model format version。
- layer topology `GDN,GDN,GDN,FULL`。
- dtype、vocab size、hidden size、最大上下文。
- 权重文件 SHA256 和 golden vector version。
- 所需 HAL ABI 范围与 capability bitset。

Engine 在创建 replica 前完成全部门禁。任何文件缺失、哈希变化、未知必需字段、ABI 主版本差异或能力缺失都必须终止启动，禁止自动降级和静默 fallback。

## 10. 测试设计

实现严格执行 RED → GREEN → REFACTOR，每个生产行为先观察到对应测试因缺少实现而失败。

### 10.1 C++ 单元与属性测试

- 合法状态迁移逐步通过，非法跳转被拒绝。
- Prepare/Execute/Commit/Abort 重复消息保持幂等。
- epoch 不匹配和迟到响应不能修改 canonical state。
- 任一 rank 在 commit decision 前失败，发布计数保持为零。
- commit decision 后缺失确认会使 replica fatal，发布计数保持为零。
- 两种完成顺序得到相同 token sequence 和 state digest。
- handle 创建/销毁计数在成功、异常和取消路径均回到零。
- ABI 结构体截短、扩展字段、主版本不匹配和能力缺失行为固定。

### 10.2 数值与状态 golden

- CPU Reference HAL 对固定输入产生固定 token 序列。
- 每个 token 后的 FullKV、GDN state、RNG state 和整体 digest 与 golden 匹配。
- 故障后重试只能从最后一次 canonical commit 继续，不得从 shadow state 继续。

### 10.3 Python 黑盒接口测试

- health 和 model discovery 返回固定 schema。
- completions SSE 帧可增量解析，token sequence 单调且只出现一次。
- 最终 usage 精确匹配输入/输出 token 数，结束帧为 `[DONE]`。
- 非法参数、超时、资源耗尽、后端失败和 replica fatal 映射到稳定 HTTP/JSON 错误。
- 客户端断开前后的取消语义分别符合 commit boundary。

### 10.4 动态与可移植性测试

- macOS arm64 Debug、Release、ASan/UBSan 测试。
- 若当前 Apple Clang 支持且运行稳定，单独执行 TSan；不把平台不支持误报成代码失败。
- Linux amd64 容器内完成配置、编译和测试。
- 多次重复 fault plan，验证结果、事件序列和日志摘要完全一致。

## 11. 可观测性与证据

每个请求输出结构化 JSONL 事件，字段包括 monotonic timestamp、request ID、transaction ID、rank、epoch、state、error code 和 sequence number。日志不记录 prompt 原文，只记录 token 数与可选摘要。

测试运行产出：构建身份、编译器版本、依赖锁定摘要、测试结果、sanitizer 结果、manifest SHA256、golden SHA256 和 fault-plan seed。证据目录默认位于本地 `artifacts/mac-mvp/<run-id>/`，不提交大型临时产物。

## 12. 目录规划

```text
GR308/
├── CMakeLists.txt
├── cmake/
├── include/grserve/
│   ├── hal.h
│   ├── error.h
│   └── version.h
├── proto/
│   └── engine.proto
├── src/
│   ├── engine/
│   ├── hal_cpu/
│   ├── hal_mock/
│   ├── model/
│   └── rpc/
├── python/grserve_api/
├── models/mini-qwen38-hybrid/
├── tests/
│   ├── cpp/
│   └── python/
├── tools/
├── docs/superpowers/specs/
├── docs/superpowers/plans/
└── artifacts/                 # ignored local evidence
```

每个源文件只承担一个主要职责；Engine、HAL、模型与 API 通过公开接口通信，测试不得依赖私有成员布局。

## 13. 阶段拆分

本规格对应一个可独立验收的实现计划，按以下顺序交付：

1. 仓库与可复现工具链基础。
2. 公共错误、版本和强类型 ID。
3. HAL ABI 与装载兼容性门禁。
4. MockAsync/FaultInjection HAL。
5. 事务状态机与两 rank Controller。
6. CPU Reference HAL 与 MiniQwen38HybridModel。
7. Engine RPC。
8. Python OpenAI-compatible API。
9. 故障矩阵、sanitizer 和 Linux amd64 可移植性门禁。
10. Mac MVP 验收报告。

首个计划完成后再分别为 Fant HAL、GR308 资格探针、vLLM/SGLang adapter 和真实模型执行写独立规格，避免把硬件未知项带入本地核心设计。

## 14. 服务器上线门槛

Mac MVP 通过不等于可上线。进入服务器必须满足：

- baseline 已完成或用户明确分配了独占维护窗口。
- 服务器驱动、runtime、模型和 baseline 证据保持冻结。
- Fant HAL 通过独立 P0 资格：设备身份、内存、stream/event、kernel launch、copy、错误传播、资源回收和 FHCCL 冷启动。
- 全部新增依赖使用隔离目录、虚拟环境或容器，不覆盖系统与 baseline 环境。
- 先跑正确性和资源门，再跑性能；性能使用同模型、同请求、同并发和 A-B-A 配对校准。

## 15. 验收定义

本地阶段的完成定义是：从干净环境执行项目规定的一条验证命令，能够成功构建 C++ 核心和三个本地 HAL，运行全部 C++/Python/协议/golden/故障测试，无 sanitizer 错误，并生成完整本地证据清单。

任何测试被跳过都必须带机器可读原因，并在验收摘要中单列；关键事务、发布安全、ABI 兼容和 CPU golden 测试不允许跳过。
