# GRServe vLLM Bridge 可运行性验证设计

**状态：APPROVED DESIGN / IMPLEMENTATION NOT STARTED**

**日期：2026-09-10**

## 1. 目标

在不把 vLLM/SGLang 误称为 GRServe 原生执行后端的前提下，让真实
Qwen3.8-27B 请求从 GRServe 的公开接口进入，并经过 GRServe 的请求准入、
生命周期、超时、取消、流式顺序和证据链，再由厂商 vLLM 完成 GR308 上的
模型执行。

首轮仅验证以下矩阵能否跑通：

- 部署模式：厂商 vLLM eager、厂商 vLLM FANT Graph；
- 并发：C1、C2、C4、C8、C16、C32、C64；
- 模型：`/data/models/Qwen3.8-27B`，BF16，TP=4；
- 物理卡：逻辑设备 `0,2,4,6`，对应四张物理 GR308；
- 结论范围：可运行性烟测，不形成新的正式 baseline，不认证 GRServe 原生
  Fant HAL、kernel、KV/GDN、FHCCL 或 Graph Manager。

## 2. 非目标

本阶段不实现或宣称：

- GRServe 直接加载 Qwen3.8-27B 权重；
- GRServe 原生 Fant HAL、设备内存、stream/event 或 kernel launch；
- GRServe 原生 Tensor Parallel、FHCCL、KV/GDN Cache；
- GRServe 原生 eager executor 或 Graph capture/replay；
- 相对 vLLM/SGLang 的性能提升；
- 生产 SLA、业务质量或上线资格。

桥接结果必须使用“GRServe control-plane + vendor vLLM executor”命名，禁止使用
“GRServe 原生 GR308 推理”或等价表述。

## 3. 方案选择

### 3.1 采用方案

采用 **C++ Bridge Controller + Python vLLM Sidecar**：

```text
OpenAI-compatible client
  -> grserve-api
  -> versioned UDS/protobuf RPC
  -> GRServe Bridge Controller
       -> admission / request table / deadline / cancellation
       -> token sequence validation / terminal-state ownership
  -> versioned local Bridge RPC
  -> Python vLLM Bridge Sidecar
  -> vendor vLLM HTTP/SSE endpoint
  -> Qwen3.8-27B on 4 x GR308
```

vLLM Sidecar 是无权威状态的执行适配器。请求是否被接受、是否超时、是否取消、
哪些 token 可以向客户端发布，以及请求何时进入终态，都由 GRServe Bridge
Controller 决定。

### 3.2 不采用方案

- **纯 Python 反向代理**：实现快，但绕过 C++ 控制面，不能验证本阶段目标。
- **vLLM-as-HAL**：HAL 是 rank-local 设备边界，而 vLLM 已拥有完整 TP replica；
  将它塞入每个 Rank 会重复执行请求，并破坏取消与错误归因。
- **影子控制面**：若 vLLM 输出可绕过 Controller 直接发布，则 GRServe 状态不是
  权威状态，验证结果没有意义。

## 4. 组件边界

### 4.1 `BridgeController`

`BridgeController` 是桥接模式的唯一请求状态所有者。它不复用当前只支持单个
`active_` 请求的 MiniQwen `Controller` 执行循环，而是复用稳定的错误模型、
ID 类型、`AdmissionController`、`ReplicaGate`、`RequestEventSink` 与 RPC 发布
约束，维护独立的并发请求表。

每个请求至少记录：

- GRServe request ID 与不可复用 generation；
- 模型 ID、prompt token IDs、max tokens；
- admission lease；
- monotonic deadline；
- bridge execution ID；
- 下一个允许发布的 sequence number；
- accepted/running/cancelling/finished/failed 终态；
- transport 是否仍连接；
- 已发布 token 数和唯一 terminal event。

任意请求至多发布一次 Accepted、严格递增的 TokenEvent，以及一个 Finished 或
Error。Controller 销毁、sidecar 断开和客户端断线均必须释放 lease。

### 4.2 `BridgeExecutorClient`

C++ 侧的异步客户端，每个请求对应一个逻辑执行流。它负责：

- 向 Sidecar 发送 start/cancel；
- 将 sidecar token/finished/error 转为值对象并送入线程安全 completion queue；
- 断线时产生稳定 `RANK_FAILURE`，不得静默重试已经发布 token 的请求；
- 设定有界入站帧、出站队列和关闭期限。

它不解释 HTTP，不加载 tokenizer，不接触模型权重或设备。

### 4.3 `grserve-vllm-sidecar`

Python asyncio 进程，连接本机 `127.0.0.1` 的厂商 vLLM endpoint。它负责：

- 接收已经 tokenized 的 prompt token IDs；
- 使用固定、可审计参数请求 `/v1/completions`；
- 解析 SSE，并回传单 token 事件、usage、finish reason 或净化后的错误；
- 收到 cancel 后关闭对应 HTTP stream；
- 限制每请求缓冲，遵守 Bridge RPC 背压；
- 暴露只读 health，报告目标 deployment ID 与执行模式。

Sidecar 不持有 admission、deadline 真值或客户端连接状态。Sidecar 重启后不得
恢复旧请求。

### 4.4 厂商 vLLM deployment

两种模式使用相同镜像、模型、TP、dtype、设备、上下文和显存参数，仅改变执行
模式：

- eager：`--enforce-eager`；
- FANT Graph：禁用 `--enforce-eager`，固定
  `FULL_DECODE_ONLY` 与 capture sizes `[1,2,4,8,16,32]`。

C64 是捕获桶外压力路径，必须单独标注，不能声称 C64 使用单一 64 bucket Graph。

## 5. 协议与数据流

Bridge RPC 使用独立版本 `1.0`，不把厂商 HTTP 细节加入现有 Engine RPC。

最小消息集合：

- `BridgeStart`：execution ID、model、prompt token IDs、max tokens、冻结的 greedy
  解码参数、deadline budget、deployment ID；首版不扩大现有公开 API 的采样能力；
- `BridgeAccepted`：execution ID；
- `BridgeToken`：execution ID、从 1 开始的 sequence、token ID；
- `BridgeFinished`：execution ID、finish reason、usage；
- `BridgeCancel` / `BridgeCancelAck`；
- `BridgeError`：稳定错误码、retryable、净化消息；
- `BridgeHealth`：sidecar build ID、target deployment ID、mode、accepting。

正常数据流：

1. API 完成严格 schema 校验并向 Engine 提交请求。
2. BridgeController 原子取得 admission lease，发布 Accepted。
3. BridgeExecutorClient 向 Sidecar 发送 BridgeStart。
4. Sidecar 发起 vLLM streaming 请求。
5. 每个 token 先回到 Controller；Controller 校验 execution ID 和 sequence 后再发布。
6. Finished 通过后 Controller 释放 lease，并发布唯一 terminal event。

Controller 在 Accepted 之后失败时保持 HTTP 200 + SSE Error + `[DONE]` 的冻结公共
协议；Accepted 之前的准入失败仍返回现有 JSON 状态码。

## 6. 并发、背压与取消

- `--max-concurrent-requests` 是 BridgeController 的硬上限，首轮覆盖到 64。
- 每个已接受请求占用一个 lease，直到 terminal state 或有界清理完成。
- 请求表可以同时存在 64 个 running 请求；不得退化为单 `active_` 串行队列。
- completion queue 与每请求 token queue 均有界；满时停止从 Sidecar 读取，通过
  socket 背压传递到 HTTP reader。
- 客户端断线触发一次 BridgeCancel；Controller 立即停止网络发布，但必须等待
  CancelAck、Finished、Error 或清理期限后才能释放资源。
- deadline 使用 Controller 的 monotonic clock。Sidecar 超时只作为执行错误信号，
  不能替代 Controller 的权威 deadline。
- 已发布 token 的请求不得自动重试。启动前或零 token 断线也不在首版自动重试。

## 7. 故障与安全边界

- Sidecar 不可达：请求在 Accepted 后得到稳定 `RANK_FAILURE`，lease 被释放。
- vLLM 返回 4xx：映射为 `INVALID_ARGUMENT` 或 `UNSUPPORTED_CAPABILITY`。
- vLLM 429：映射为 `RESOURCE_EXHAUSTED`。
- vLLM timeout：映射为 `DEADLINE_EXCEEDED`。
- vLLM 5xx、断流、非法 SSE、token sequence 异常：映射为 `RANK_FAILURE`；不得发布
  伪造完成事件。
- Sidecar 返回未知 execution ID、重复 terminal 或倒退 sequence：视为协议错误，
  仅失败相关请求；若相关性不可证明，则将 bridge replica 置为 fatal。
- 所有监听仅绑定 loopback 或私有 Unix socket；模型 endpoint 不暴露到外网。
- 日志不得写入 prompt 文本、Authorization header 或完整请求体。

## 8. 配置与可审计身份

桥接部署 profile 必须冻结并记录：

- GRServe Git SHA、Bridge schema hash、Sidecar source hash；
- vLLM image ID/RepoDigest、vLLM/vllm_fant/torch_fant 版本；
- 模型 config/tokenizer/weight index hash；
- eager 或 graph 模式及完整启动参数；
- 逻辑设备 `0,2,4,6`、PCI ID、序列号、驱动和 firmware；
- 请求矩阵、输入/输出 token 数、timeout、deployment ID；
- 所有进程 PID、退出码、restart count 与测试后资源状态。

证据放在 `/data/infra/benchmarks/grserve-vllm-bridge-smoke-<UTC>/`。任何既有 vLLM、
SGLang baseline 目录保持只读，不作为新结果目录，也不得被覆盖。

## 9. 测试策略

### 9.1 本地自动化

- Bridge protocol 编解码、帧大小和版本门禁；
- 64 个并发请求的 admission 与 lease 回收；
- token sequence、重复/乱序/未知 execution ID；
- cancel、deadline、客户端断线和 sidecar crash；
- bounded queue 背压；
- HTTP/SSE 公共协议回归；
- Fake vLLM Sidecar 的确定性 E2E；
- ASan/UBSan、Python pytest、ruff、mypy。

### 9.2 服务器烟测

每个模式只进行一次 engine-cold 启动，不做 formal repeat 或长稳定性测试：

1. 确认无其他 GPU 进程、无运行容器、8000 和 GRServe 端口空闲。
2. 启动厂商 vLLM，等待 health，并记录 deployment identity。
3. 启动 Sidecar、GRServe Engine 与 grserve-api。
4. 执行 C1/C2/C4/C8/C16/C32/C64；每档请求数至少等于并发数。
5. 每档校验全部请求 HTTP 200、唯一 `[DONE]`、严格 sequence、预期 token 数、
   零错误和零 lease 泄漏。
6. 在 C8 额外执行取消、deadline 和客户端断线各一组。
7. 矩阵后重新检查 health、进程退出码、容器 restart、GPU 错误和资源回落。
8. 先停止 GRServe/Sidecar，再停止本轮精确命名的 vLLM 容器。

## 10. 验收条件

桥接可运行性结论为 PASS 必须同时满足：

- eager 与 FANT Graph 均成功冷启动；
- 两种模式的七档并发全部成功；
- 每档成功请求数等于请求总数；
- token sequence、完成标记和 usage 校验全部通过；
- cancel、deadline、disconnect 的外部结果与内部终态一致；
- Controller 请求表、admission lease 和 Sidecar execution table 最终均为零；
- 无容器重启、OOM、FHCCL/NCCL 错误、驱动 reset 或遗留服务；
- 机器可读 summary 明确包含
  `executor=vendor-vllm`、`native_fant_hal=false`、`formal_baseline=false`。

任一条件失败时保留证据并报告 BLOCKED/FAIL，不通过重跑覆盖失败目录。

## 11. 后续演进

桥接验证通过后，保持 GRServe API 与 BridgeController 的请求语义不变，按以下顺序
替换执行层：

1. Fant P0：设备身份、内存、stream/event、copy、kernel、错误传播、资源回收；
2. 单卡原生 eager Qwen Model Runner；
3. TP=4 与 FHCCL；
4. 原生 KV/GDN Cache 与 continuous batching；
5. FANT Graph capture/replay/fallback；
6. 使用相同矩阵与三个冻结 baseline 做正式对比。

只有完成上述原生执行路径后，才能评估 GRServe 是否可直接替代 vLLM/SGLang。
