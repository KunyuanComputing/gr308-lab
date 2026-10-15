# 图表目录

## figure-01-formal-throughput.pdf
- Purpose: 比较正式长上下文负载下的吞吐扩展。
- Data source: 三份 baseline-summary.json 的 Formal 三 block 均值与描述性 95% t 区间。
- Key observation: Graph32 在 C1–C32 领先，C64 回落并接近 Eager。
- Caveat: n=3；区间只作描述；未随机交错。

## figure-02-rapid-throughput.pdf
- Purpose: 比较短输出快速筛查的扩展形状。
- Data source: 三份 summary 的 Rapid 单轮值。
- Key observation: Graph32 在 C32 达峰，C64 与 Eager 收敛。
- Caveat: 单轮，无跨 block 误差估计。

## figure-03-stability-throughput.pdf
- Purpose: 比较同口径 C64 稳定性主段。
- Data source: 各方案 128→64、C64、1024 请求正式稳定性主段。
- Key observation: Eager 与 Graph32 接近，SGLang 明显较低；三者均零请求失败。
- Caveat: 单个长运行，不是独立重复。

## figure-04-stability-latency.pdf
- Purpose: 对比同口径稳定性请求的 TTFT/E2E P50。
- Data source: Eager/Graph summary 与 SGLang 1024 条请求级记录。
- Key observation: Graph32 与 Eager 接近，SGLang E2E P50 更高。
- Caveat: 闭环并发延迟受排队与服务共同影响。
