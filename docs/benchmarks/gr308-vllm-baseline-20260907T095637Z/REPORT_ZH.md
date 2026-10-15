# GR308 vLLM Baseline 实验报告

结论：通过。本报告仅覆盖 vLLM Lane E；SGLang 未纳入本轮 baseline。

## 冻结配置

- 模型：`/data/models/Qwen3.8-27B`（served name `qwen3.8-27b`）
- BF16，TP=4，设备 0/2/4/6，eager execution
- `max-model-len=32768`，`gpu-memory-utilization=0.40`，`block-size=800`
- 接口仅绑定 `127.0.0.1:8000`

## Rapid（128→64）

| 并发 | validated output tok/s |
|---:|---:|
| 1 | 1.6748 |
| 8 | 11.1785 |
| 16 | 20.3973 |
| 32 | 32.7470 |
| 64 | 47.0706 |

## Formal（1024→256，3 个 block）

| 并发 | 原始三点 tok/s | mean | median | CV | 描述性 95% t-CI |
|---:|---|---:|---:|---:|---:|
| 1 | 1.6391, 1.6443, 1.6450 | 1.6428 | 1.6443 | 0.195% | [1.6349, 1.6507] |
| 8 | 10.2714, 10.3002, 10.2429 | 10.2715 | 10.2714 | 0.279% | [10.2003, 10.3427] |
| 16 | 17.7820, 17.9111, 17.8602 | 17.8511 | 17.8602 | 0.364% | [17.6895, 18.0127] |
| 32 | 26.4181, 26.4354, 26.4541 | 26.4358 | 26.4354 | 0.068% | [26.3911, 26.4806] |
| 64 | 32.2150, 31.9827, 32.0398 | 32.0792 | 32.0398 | 0.377% | [31.7784, 32.3799] |

## 稳定性与正确性

- 有效替代段：`stability-retry02`；主段 1024/1024，44.0619 output tok/s，1487.4 秒。
- 延长段 512/512，44.3500 output tok/s，738.9 秒；合计 2226.2 秒。
- 性能后正确性哨兵：12/12 通过。

## 排除与故障记录

- Rapid 原始 C1 因 SSH 中断仅形成部分证据，使用 `c1-rerun-01`。
- Formal 原始 R1 在 FHCCL/NCCL 拓扑初始化时失败，使用 `r1-retry01`。
- 首个 stability 因系统缺少 `jq` 错误跳过延长段，保留但排除；`stability-retry01` 又在 FHCCL/NCCL 初始化失败，最终使用 `stability-retry02`。

## 解释边界

这些数字来自固定 token-ID、`ignore_eos=true` 的合成吞吐诊断，不代表业务质量或生产 SLA。所谓 engine-cold 仅表示新容器与新 engine 进程；宿主 page cache 和驱动状态可能仍为热态。Formal 只有 3 个 block，置信区间仅作描述。

机器可读明细见 `analysis/baseline-summary.json`、`analysis/validation.json`、`analysis/run-level.csv`、`analysis/request-level.csv` 和 `analysis/telemetry-physical.csv`。
