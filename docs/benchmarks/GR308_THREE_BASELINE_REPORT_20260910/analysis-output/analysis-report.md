# 三 Baseline 严格分析摘要

## 分析问题

在同一 GR308 服务器、同一 Qwen3.8-27B BF16 模型、TP=4 和冻结合成负载下，比较 FANT vLLM Eager、SGLang + GR308 兼容层、FANT vLLM Graph32 的吞吐扩展、同口径稳定性与请求延迟。

## 关键发现

- Graph32 Formal 在 C1/C8/C16/C32 相对 Eager 分别为 3.314×、2.530×、1.826×、1.419×。
- Graph32 C64 为 32.4532 tok/s，Eager 为 32.0792 tok/s，仅相差 +1.2%。
- SGLang C1 相对 Eager为 +32.1%，但 C64 为 -80.2%。
- 同口径稳定性主段三者均 1024/1024 成功；吞吐依次为 44.0619、14.3865、44.2578 tok/s。

## Claim Candidates

- Claim: Graph32 在本轮冻结负载的 C1–C32 中提供最高描述性吞吐。
  - Source evidence: 三方案 Formal 各并发 3 个 engine-cold block。
  - Allowed wording: 在本次系统级对比中，Graph32 的 C1–C32 均值最高。
  - Forbidden stronger wording: Graph 调度器在所有业务负载上显著优于其他框架。
  - Uncertainty: 未随机交错；框架与底层执行路径同时变化；n=3。
  - Next check: 随机交错 5–10 个 block，并做组件级消融。
  - Decision: keep

- Claim: Graph32 的 C64 饱和吞吐没有形成相对 Eager 的明显优势。
  - Source evidence: Formal 32.4532 vs 32.0792 tok/s；稳定性 44.2578 vs 44.0619 tok/s。
  - Allowed wording: C64 的描述性差异约为 1.2% / 0.4%。
  - Forbidden stronger wording: Graph 对高并发完全无效。
  - Uncertainty: C64 在冻结 capture bucket 之外。
  - Next check: 增加 64 capture bucket 后重跑。
  - Decision: keep
