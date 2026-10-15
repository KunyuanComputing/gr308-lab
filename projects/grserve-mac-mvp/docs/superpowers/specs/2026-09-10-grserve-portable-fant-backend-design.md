# GRServe 可移植核心与 Fant 首个后端设计规格

版本：2026-09-10

状态：设计方向已确认，待用户书面复核

首个硬件后端：Fantasy GR308

首个真实模型：`/data/models/Qwen3.8-27B`，BF16，TP4

## 1. 决策背景

项目不再把“短期提高当前 GR308 的吞吐数字”作为唯一目标。GR308 仍是当前可用、已有完整
baseline 和厂商编程资料的设备，因此继续作为 GRServe 的第一个真实硬件后端；但新增核心
代码必须能够在后续更换国产加速卡时复用。

本设计取代“先实现 GRServe 到 vLLM 的 HTTP/SSE 桥接，再验证上层控制逻辑”的近期执行
方向。已有桥接设计和计划保留为历史资料，不删除、不伪装为原生执行路径，也不继续实施。
既有 vLLM、SGLang 和 FANT Graph 结果只作为只读 baseline。

核心决策是：

> 建设厂商无关的 GRServe Portable Core、版本化 Backend Contract 和推理执行计划；以
> GR308 同时实现 `fant-torch` 兼容路径与 `fant-native` 性能路径。前者用于尽快获得真实
> 模型正确性，后者使用 HIP、FTUCA、FHCCL 和 FANT Graph 渐进替换热点。下一种国产卡
> 复用同一核心、模型计划、测试和证据体系，只新增后端实现与专用算子。

## 2. 目标

### 2.1 产品目标

- GRServe Core 不直接依赖 Fantasy、CUDA、CANN、CNRT、ROCm、PyTorch 或某一厂商通信库。
- 同一模型计划能在 CPU Reference、Mock/Fault、`fant-torch` 和 `fant-native` 后端执行。
- 后端通过能力协商暴露真实限制，核心不依赖厂商名称分支。
- GR308 完成 Qwen3.8-27B 的 Eager、TP4、C1-C64 和 FANT Graph 路径后，能与冻结的
  vLLM/SGLang baseline 进行严格同口径比较。
- Backend Contract、模型计划、KV/批处理策略、自动调优器、测试套件和证据格式可直接复用
  到下一种国产卡。

### 2.2 工程目标

- 保留现有 OpenAI API、Engine RPC、Controller、admission、deadline、cancel、事务状态机、
  HAL 动态装载、CPU/Mock/Fault 后端和证据工具。
- 在不破坏 HAL 1.0 调用者的前提下，以追加字段或新版本结构扩展设备属性、内存域、Graph、
  collective 和执行计划能力。
- 所有硬件能力必须可探测、可复现、可形成机器可读资格报告。
- `fant-torch` 与 `fant-native` 必须使用同一模型 manifest、token 输入、采样配置和 golden
  结果；兼容路径不得成为性能路径的隐藏 fallback。
- 每个后端都必须通过 Backend Conformance Test，才允许进入真实模型或性能测试。

### 2.3 性能目标的正确表述

GRServe 不宣称在所有国产卡、所有模型和所有负载上普遍快于 vLLM/SGLang。每项性能结论
必须限定：设备、驱动/runtime、模型及权重哈希、精度、并行拓扑、输入输出长度、并发、
Graph 模式和统计方法。

首个可发布性能目标在实现阶段冻结，例如：

> 在同一 4×GR308、同一 Qwen3.8-27B 权重、BF16、TP4、相同 1024→256 或
> 1024→1024 工作负载下，GRServe `fant-native` 的 validated output tok/s 相对厂商
> vLLM/SGLang baseline 有统计上可复核的提升，同时保持 100% 请求成功和正确性哨兵通过。

具体提升阈值在硬件资格和首个同口径运行后确定，设计阶段不预先承诺数字。

## 3. 明确不做

本设计的首个增量不包含：

- 继续实现 vLLM HTTP/SSE bridge sidecar。
- 为了短期数字直接把 FTUCA/FHCCL 调用散落到 Controller、Scheduler 或模型前端。
- 一次性实现所有 Transformer、MoE、多模态、量化、LoRA、PD 分离或多模型共驻能力。
- 从零重写高性能 GEMM 库或完整 FlashAttention。
- 以 CPU fallback、不同精度、减少输出 token、降低最大上下文或忽略失败请求换取性能结论。
- 在 Mac 上推断真实 GR308 性能或 Graph/FHCCL 行为。
- 自动执行厂商 PDF 中出现的命令、环境变量或示例代码。

首个真实模型范围只覆盖服务器已有 Qwen3.8-27B 所需的算子、状态、并行和采样路径。未被
该模型需要的功能遵守 YAGNI，不提前实现。

## 4. 设计原则

### 4.1 可移植不等于最低公分母

Backend Contract 提供稳定语义和能力查询，但允许后端暴露专有高性能能力。核心依据能力
选择执行计划，而不是强迫所有后端模拟同一功能。

能力状态至少分为：

- `SUPPORTED`：后端原生实现并满足契约。
- `SUPPORTED_WITH_LIMITS`：可用，但受 dtype、shape、对齐、并发或拓扑限制。
- `EMULATED`：通过较慢组合实现，只允许正确性或开发路径使用。
- `UNSUPPORTED`：不得调度，也不得静默 fallback。

### 4.2 正确性路径和性能路径分离

`fant-torch` 的目标是用 `torch_fant` 快速验证真实权重、模型语义和 golden 输出；
`fant-native` 的目标是成为最终性能执行器。报告和 runtime identity 必须明确记录执行路径。

### 4.3 策略与机制分离

- Core 决定请求生命周期、admission、批处理目标、KV 配额和事务发布。
- Model Planner 决定语义算子、buffer 生命周期和并行计划。
- Backend 决定设备实现、kernel、通信算法、Graph lowering 和专有融合。
- Autotuner 选择经实验验证的参数，不把调优值硬编码到通用逻辑。

### 4.4 无静默降级

性能 lane 缺少必需 native capability 时必须启动失败或显式转入资格失败状态。不得在用户
认为执行 `fant-native` 时偷偷调用 `fant-torch`、CPU 或 vLLM。

## 5. 总体架构

```text
OpenAI-compatible Client
          │ HTTP/SSE
          ▼
Python API Adapter
          │ versioned UDS/protobuf
          ▼
GRServe Portable Core
  ├── Request Controller / Lifecycle / Admission
  ├── Continuous Batch Scheduler
  ├── KV and Recurrent-State Manager
  ├── Model Planner and Inference Plan IR
  ├── Parallel/Graph Policy
  ├── Backend Loader and Capability Resolver
  └── Metrics / Evidence / Autotuning
          │ Execution Backend Protocol
          ├───────────────────────────────┐
          ▼                               ▼
fant-torch Worker                   fant-native Rank Workers
  ├── torch 2.7                         ├── HAL Runtime ABI
  ├── torch_fant                        ├── HIP Runtime
  └── correctness runner                ├── FTUCA kernels
                                        ├── FHCCL collectives
                                        └── FANT Graph
```

CPU Reference、MockAsync 和 FaultInjection 后端继续存在，与两个 Fant 后端共享契约测试。

## 6. 分层边界

### 6.1 Portable Serving Core

Core 是请求真相的唯一所有者，负责：

- request/transaction ID、deadline、cancel 和 publish sequence。
- admission、请求队列、batch 形成和公平性策略。
- KV/state 的逻辑配额、页所有权和回收决定。
- 对多 rank 执行的 prepare/execute/commit/abort 协调。
- 选择满足模型计划所需能力的后端和执行 profile。
- 输出与后端身份绑定的 metrics 和 evidence。

Core 不直接持有厂商原生指针，不调用 `hip*`、`nccl*`、`torch_fant` 或 FTUCA kernel。

### 6.2 Model Frontend

Model Frontend 读取 Hugging Face config、safetensors index 和 GRServe manifest，生成厂商无关
模型描述。Qwen3.8-27B 首版至少描述：

- embedding、输出 head 和 layer topology。
- hidden/intermediate/head 维度、dtype 和 tensor parallel shard 规则。
- full-attention 与线性/递归状态层的排列。
- RoPE、normalization、MLP/gating、attention/state 更新语义。
- KV 与 recurrent state 的 shape、生命周期和一致性要求。
- MTP 是否存在以及所需 checkpoint/能力；未知或初始化失败时显式关闭并记录原因。

模型前端不选择 FTUCA block size、FHCCL channel 或 Graph bucket。

### 6.3 Inference Plan IR

首版 IR 是小型、版本化的推理执行计划，不建设通用编译器。它包含：

- tensor ID、shape、stride、dtype、layout、memory domain 和 alignment。
- semantic op ID、输入输出、attributes 和副作用。
- buffer alias、原地写和生命周期区间。
- rank-local、collective、host-control 与 barrier 节点。
- Prefill、Decode、Sampling 和 Commit 子图。
- 静态维度、动态维度上界和 Graph bucket key。
- 可融合区域及其数值容差，不在 IR 中写死厂商 kernel 名称。

后端可以逐算子解释执行，也可以 pattern-match 后融合或编译成 opaque executable。Plan IR
带 schema version、模型 manifest hash 和 planner build ID；任何不兼容必须在执行前失败。

### 6.4 HAL Runtime ABI

现有 `include/grserve/hal.h` 的 C ABI 继续作为低层 runtime 边界。扩展遵守：

- HAL 1.0 已有结构语义不变；兼容字段只追加在 API table 尾部。
- 新结构均含 `struct_size` 和版本字段。
- 改变所有权、线程安全或错误语义时提升 ABI major，不复用旧字段。
- C++ 异常不得跨 ABI；所有错误通过稳定状态码和 thread-local detail 返回。

新增能力族包括：

- 设备属性与限制：内存、alignment、最大 grid/block、并发 kernel、P2P 拓扑。
- memory domain：host、pinned host、device、peer-visible、graph-static。
- stream/event：优先级、non-blocking、跨进程限制、计时支持。
- Graph：capture、instantiate、upload、launch、parameter update 和 debug dump。
- collective：communicator、AllReduce、AllGather、ReduceScatter、Broadcast、Send/Recv、group。
- executable/plan：validate、prepare、bind、launch 和销毁。

64 位 capability bitset 表示能力族；shape、dtype、对齐、拓扑等结构化限制通过追加的属性查询
返回，不为每个细节消耗 bit。

### 6.5 Semantic Operator Contract

跨卡复用以算子语义为中心，而不是以 FTUCA/HIP 函数名为中心。首个 Qwen 所需 operator
contract 至少包括：

- embedding/gather。
- GEMM/linear 及 TP shard 语义。
- RMSNorm、residual-add、RoPE。
- attention 或模型要求的线性/递归状态更新。
- SiLU/gated MLP。
- KV/state read/write。
- logits、argmax/top-k/sampling。
- collective 节点。

后端可注册单算子实现或融合 pattern，例如 residual-add + RMSNorm。融合实现必须声明支持
的 dtype、shape、alignment、数值容差、Graph 安全性和是否修改输入。

### 6.6 KV 与 Recurrent-State Manager

Core 管理逻辑 page、request ownership、引用计数和事务边界；Backend 管理物理 buffer、
layout 和 copy/kernel。接口需要同时表达 full-attention KV 与 Qwen 混合层 recurrent state，
不能假设所有 layer 都具有相同 KV 形态。

首版支持：

- 固定 page pool 与显式预算。
- 每请求逻辑 page table。
- prepare 阶段保留、commit 后切换、abort 后回收。
- backend 建议的 page size、alignment 和 layout profile。
- Graph-static address 模式。

Prefix cache、offload、跨节点 KV 和 PD 分离不进入首个真实模型增量。

### 6.7 Parallel Plan

首个拓扑固定为物理 GR308 对应的逻辑设备 `0,2,4,6`，TP4。Parallel Plan 描述 shard 和
collective 语义，Fant 后端将其 lowering 到 FHCCL。

通信资格必须验证：

- rank 到物理设备映射和唯一性。
- P2P 是否直接可用，禁止把 host staging 误称为 P2P。
- FP16/BF16 collective 正确性。
- AllReduce/AllGather/ReduceScatter 的 size sweep。
- group 调用的 host 阻塞和 stream 顺序。
- communicator abort、异步错误和重建。

`NCCL_MIN_NCHANNELS`、`NCCL_MAX_NCHANNELS`、`NCCL_BUFFSIZE`、protocol 和 algorithm
属于 Autotuner profile，不能成为通用默认值。

### 6.8 Graph Plan

Graph policy 根据后端能力生成静态 bucket。Fant 首版以 HIP/FANT Graph 的 capture、
instantiate、upload 和 launch 为基础，默认不假设支持任意参数更新。

每个 Graph executable 绑定：

- model manifest hash。
- backend build/runtime/driver ID。
- rank、dtype、batch bucket、sequence bucket。
- buffer layout/profile hash。
- capture log 和 node count。

bucket 外请求必须走明确标记的 eager/fallback 计划，不能在报告中声称使用了相应 bucket。

## 7. Fant 双路径

### 7.1 `fant-torch` 兼容路径

`fant-torch` 是独立 Python worker，直接导入 `torch` 和 `torch_fant`，不启动 vLLM 或
SGLang。它通过版本化 Execution Backend RPC 接收预分词输入、plan/profile ID 和 buffer
绑定，返回 token、状态 digest、计时和错误。

职责：

- 验证真实模型 config、权重加载和 tensor shape。
- 建立 CPU/torch_fant 的算子级和 token 级 golden。
- 识别 unsupported op、dtype 或 shape。
- 为 `fant-native` 提供逐阶段正确性 oracle。

限制：

- 不作为最终吞吐领先证明。
- 不拥有 admission、deadline 或 publish 决策。
- 不允许调用 vLLM/SGLang executor。
- Python/PrivateUse1 开销必须在 evidence identity 中记录。

### 7.2 `fant-native` 性能路径

`fant-native` 由 C++ rank worker 加载 Fant HAL，使用 HIP Runtime 管理设备、stream、event、
内存和 Graph；使用 FTUCA/厂商库执行算子；使用 FHCCL 完成 TP4。

实现顺序是渐进替换：

1. 设备和内存资格。
2. 单算子 microbenchmark 与 golden。
3. 单层 Prefill/Decode。
4. 单卡真实模型。
5. TP4 Eager C1。
6. 连续批处理和 C1-C64。
7. Graph bucket。
8. profiling 后只替换最高收益热点。

原生路径缺失算子时允许测试进程返回 `UNSUPPORTED`，不允许把单个生产请求跨进程混合执行
在 `fant-torch` 上。

## 8. Capability Model

Backend identity 至少包含：

- backend name、semantic version、build ID、source revision。
- driver、runtime、compiler、PyTorch/plugin 和 communication library 版本。
- device UUID/ordinal、物理映射、NUMA 和 P2P matrix。
- 支持的 dtype、memory domain、Graph、collective 和 operator set。
- 已通过的 conformance profile 与时间。

核心使用 required/preferred capability：required 缺失则拒绝；preferred 缺失时可选择另一个
已声明 plan，但必须记录 plan ID，不能静默改变语义。

## 9. Autotuning

Autotuner 不修改通用代码，而是为固定 backend/model/workload 生成可签名 profile。流程：

1. 设备和拓扑探测。
2. memory/copy/stream/event/Graph microbenchmark。
3. operator shape corpus benchmark。
4. collective size sweep。
5. batch、KV page、Graph bucket 和 parallel plan 候选。
6. 快速正确性门。
7. 交错 A/B 运行。
8. 输出 profile、原始结果、版本和 hash。

Profile 不跨 driver/runtime/backend build 自动复用；identity 不匹配时重新资格或回到保守
profile。

## 10. 错误与恢复

- Backend 初始化失败：Engine 不 ready，报告缺失 capability 和原始错误。
- 单算子/plan 不支持：请求执行前失败，不部分执行。
- OOM：回滚未提交预留；若 allocator 状态不确定则 rank/replica fatal。
- FHCCL 异步错误：停止新 admission、abort communicator、重建整个 replica。
- Graph capture/instantiate 失败：只有存在预注册 eager plan 时才能显式降级，并记录原因。
- rank loss 或 commit decision 不确定：沿用现有事务不变量，token 不发布，replica fatal。
- `fant-torch` worker 断开：未提交请求失败；不得自动转发到 vLLM。

## 11. 测试体系

### 11.1 Backend Conformance Test

所有后端共享：

- ABI 截短/扩展、版本和 capability 测试。
- handle 所有权、重复销毁、异常路径无泄漏。
- device/thread-local 语义。
- buffer alignment、越界 copy、异步顺序。
- stream/event wait/query/timeout。
- Graph capture/replay/重复 launch 与不支持路径。
- collective 正确性、不同完成顺序和 rank failure。
- unsupported capability 无静默 fallback。

CPU/Mock/Fault 在 Mac CI 运行；Fant conformance 在 GR308 容器中运行。

### 11.2 数值测试

- CPU float32 reference。
- `fant-torch` BF16/FP16 oracle。
- `fant-native` 输出与约定容差。
- 单算子、单层、Prefill、Decode、logits 和 token 序列逐级比较。
- 固定 seed、greedy 路径和完整 state digest。

### 11.3 真实模型资格门

按顺序通过：

1. manifest 和权重 hash。
2. 单卡加载且无 host fallback。
3. Eager C1 正确 token。
4. TP4 C1 与单卡/torch oracle 一致。
5. C1-C64 冒烟，无死锁、泄漏和重启。
6. Graph capture/replay/fallback identity 正确。
7. 长稳和性能测试。

前一门失败时不得跳到后一门。

### 11.4 性能实验

既有只读 baseline：

- `/data/infra/benchmarks/gr308-vllm-baseline-20260907T095637Z/`
- `/data/infra/benchmarks/gr308-sglang-baseline-20260908T053931Z/`
- `/data/infra/benchmarks/gr308-vllm-fant-graph-20260909T141559Z/`

新实验写入唯一 `grserve-native-<UTC>` 目录，不覆盖 baseline。比较必须使用相同模型 hash、
精度、TP、设备、输入、输出、采样和并发；至少三次 engine-cold block。主指标是 validated
output tok/s，同时报告 TTFT、TPOT、request/s、错误率、显存、功耗和重启次数。

## 12. 服务器安全与可复现性

- baseline 目录只读。
- 模型目录只读挂载。
- 新容器、socket、PID、端口和输出目录使用唯一 run ID。
- 不执行全局 kill、GPU reset、driver/runtime 升级或删除历史结果。
- 只停止本轮有明确身份记录的进程和容器。
- source bundle、镜像 digest、共享库 hash、编译器和环境变量进入 manifest。
- 每轮结束恢复约定的全停止状态。

## 13. 实施阶段与退出条件

### Phase 1：Portable Contract，预计 1-2 周

- HAL 兼容扩展设计和 capability schema。
- Execution Plan IR v1。
- Backend conformance runner。
- CPU/Mock/Fault 全通过。

退出条件：不依赖 Fant 的测试证明契约、所有权和 unsupported 行为固定。

### Phase 2：Fant Qualification，预计 1 周

- HIP、FTUCA、FHCCL、P2P、Graph 和 torch_fant 探针。
- 真实设备映射和结构化限制。
- 首个签名 backend profile。

退出条件：每项能力为 supported/limited/unsupported，有原始证据且无性能推断。

### Phase 3：`fant-torch` 真实模型，预计 2-4 周

- Qwen manifest、权重、tokenizer/预分词接口。
- Eager C1 golden。
- 算子和 layer 级 oracle。

退出条件：固定输入生成正确 token/state，执行身份明确且不经过 vLLM/SGLang。

### Phase 4：`fant-native` Eager/TP4，预计 3-5 周

- native operator coverage。
- KV/recurrent state manager。
- FHCCL TP4。
- continuous batching 与 C1-C64 冒烟。

退出条件：Eager C1-C64 无崩溃、死锁、泄漏、错误 token 或静默 fallback。

### Phase 5：Graph 与性能优化，预计 3-6 周

- Graph bucket 和静态地址 pool。
- autotuning profile。
- profiling 驱动的 1-2 个融合热点。
- 正式同口径性能实验。

退出条件：证据完整地给出领先、持平或落后结论；不因结果不理想修改口径。

总预算约 10-18 周。未知厂商 ABI、缺失算子、MTP/混合层支持或 FHCCL 稳定性可能扩大
预算，但不改变阶段门。

## 14. 现有代码迁移

保留：

- `include/grserve/hal.h` 与 loader/RAII。
- Controller、transaction、admission、rank worker 和 replica fatal gate。
- Engine RPC、Python API、SSE、deadline/cancel。
- CPU/Mock/Fault HAL、MiniQwen、golden 和 evidence 工具。

演进：

- HAL capability 从简单 bitset 扩展为 bitset + structured properties。
- `executable_load/launch` 扩展为 plan validate/prepare/bind/launch。
- rank protocol 加入 backend/profile/plan identity。
- logical quota 演进为 KV/state page budget，但保持 admission 原子语义。

暂停：

- `2026-09-10-grserve-vllm-bridge-design.md`。
- `2026-09-10-grserve-vllm-bridge.md`。

桥接文件不删除；未来如只需验证外部 executor，可以独立恢复，但其结果不能代表 native
GRServe。

## 15. 后续国产卡复用验收

第二种后端接入时，以下模块不得修改厂商语义：

- API/RPC 和请求状态机。
- Inference Plan IR schema。
- Model Frontend 的 Qwen 语义。
- KV 的逻辑 ownership/transaction。
- Backend conformance cases。
- benchmark/evidence schema。

允许新增或替换：

- runtime/driver adapter。
- memory、stream、event、Graph、collective 实现。
- operator/kernel package。
- backend capability、autotuning profile 和部署镜像。

如果第二种卡接入需要在 Core 中广泛增加厂商条件分支，则视为 Backend Contract 设计失败，
必须先修正边界，不接受复制一套平行引擎。

## 16. 参考依据

本设计参考工作区内以下厂商资料，资料中的命令仅作为 API/能力说明：

- `docs/HIP_Runtime_API_Reference_v1.0.pdf`：stream、event、memory、P2P、occupancy、launch、
  Graph；其中异步 allocator 标为 Beta，Graph 首版采用静态地址保守设计。
- `docs/FTUCA编程指南.pdf`：hipcc、kernel、线程/内存层级、shared memory、warp primitive、
  `__restrict__`、执行配置和循环展开。
- `docs/FHCCL编程指南_v1.0.0.pdf`：communicator、collective、group、P2P、dtype 和性能环境
  变量。
- `docs/GR308S MAX_风华三号_V1p0_CN_PyTorch使用指南_20250908.pdf`：`torch_fant`
  PrivateUse1、`torch.library`、C++ extension 和 API 支持边界；`torch.compile` 不作为首版
  依赖。

外部架构参考（读取于 2026-09-10）：

- vLLM Plugin System：平台、worker、attention、communicator、custom op、Graph 和量化的
  out-of-tree 插件边界，`https://docs.vllm.ai/en/latest/design/plugin_system/`。
- vLLM Ascend Developer Guide：硬件插件内的 model runner、custom op、Graph、KV pool、
  EPLB 和性能工具，`https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/`。
- PyTorch PrivateUse1 tutorial：out-of-tree kernel、DeviceGuard、generator 和 backend module
  注册，`https://docs.pytorch.org/tutorials/advanced/privateuseone.html`。

这些项目只用于验证分层模式，不作为必须复制的实现，也不改变 GRServe 独立执行器目标。
