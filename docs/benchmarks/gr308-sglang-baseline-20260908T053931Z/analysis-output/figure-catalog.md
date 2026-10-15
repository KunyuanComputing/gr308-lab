# Figure Catalog

## figure-01-throughput-scaling.svg

- Purpose: 展示吞吐随并发扩展，以及当前 SGLang 与既有 vLLM 系统级曲线的差异。
- Data: 两框架 Formal 三 block mean；SGLang 误差条为描述性 95% t-CI。
- Observation: SGLang 在 C16 后接近平台，vLLM 继续随并发增长。
- Interpretation: 下一步优先调查高并发执行路径；不能仅据此归因调度器。

## figure-02-latency.svg

- Purpose: 展示 SGLang Formal 的 TTFT/E2E p50 与 p95 随并发变化。
- Data: 每个并发三个 block 的成功请求合并；纵轴为对数尺度。
- Observation: 高并发下端到端延迟增长显著。
- Interpretation: 吞吐平台与排队/执行时延增长共同指向高并发瓶颈。
