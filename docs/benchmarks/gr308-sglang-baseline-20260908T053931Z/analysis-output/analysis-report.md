# Analysis Report

## Analysis question

在同一 Qwen3.8-27B、BF16、TP4 和 Formal 工作负载下，SGLang 的吞吐扩展、延迟和长稳表现如何；与既有 vLLM baseline 的系统级差异有多大。

## Key findings

- SGLang Formal C64 mean 为 6.3494 output tok/s，三次 CV 为 0.099%。
- 同口径 C64 稳定性为 14.3865 output tok/s，1024/1024 成功，运行 75.9 分钟。
- 并发增大后 SGLang 吞吐在 C16–C64 附近明显趋于平台；这支持后续从算子、设备利用率和批处理执行路径排查瓶颈。
- 与 vLLM 的差异仅能描述为当前系统栈差异，不能因果归因到框架调度器。

## Claim Candidates

- Claim:
  - Source evidence: Formal 三个 engine-cold block 及同口径稳定性主测。
  - Allowed wording: 当前 GR308 兼容栈下，SGLang 在 C16 以上吞吐扩展明显趋缓。
  - Forbidden stronger wording: SGLang 调度器必然比 vLLM 差。
  - Uncertainty: 框架兼容层和底层算子不同，运行未随机交错。
  - Next check: 对齐 attention/MLP kernel、批处理 shape，并做随机交错 A/B blocks。
  - Decision: keep
