# Qwen3.6-27B × GR308：vLLM 吞吐调优结果（KV 152）

日期：2026-09-10  
状态：单轮目标负载验证完成  
优化目标：只评价吞吐；不评价语义质量、一致性或生产 SLA

## 1. 结论

在 `Qwen3.6-27B / BF16 / 3072 input / 1024 output / 4×TP2 / 每副本 C16 / 全机 C64 / FANT Graph` 的相同负载下，将每个 vLLM 副本的 KV block override 从 144 提高到 152，消除了 KV 抢占，并显著改善 Decode：

| 指标 | KV 144 | KV 152 | 变化 |
|---|---:|---:|---:|
| 成功请求 | 64/64 | 64/64 | — |
| KV 抢占 | 144 次 | **0 次** | **-100%** |
| 全机实际输出吞吐 | 46.455 tok/s | **72.781 tok/s** | **+56.67%** |
| 对方公式口径 | 54.675 tok/s | **92.402 tok/s** | **+69.00%** |
| Median TPOT | 1177.17 ms | **705.18 ms** | **-40.09%** |
| Median E2E | 1376.003 s | **893.306 s** | **-35.08%** |
| Median TTFT | 172.363 s | 172.799 s | +0.25% |

因此，KV 152 是目前在该精确负载下验证通过的配置。它解决了 KV 144 的重算/抢占问题，但没有改善长输入 Prefill；下一阶段仍应把主要精力放在 Prefill 和低效率 Decode kernel/TP2 通信上。

## 2. 最终测试配置

| 项目 | 配置 |
|---|---|
| 模型 | `/data/models/Qwen3.6-27B` |
| 精度 | BF16 |
| 输入 / 输出 | 3072 / 1024 token |
| 并行 | 4×TP2，逻辑设备 `(0,1)`、`(2,3)`、`(4,5)`、`(6,7)` |
| 并发 | 每副本 C16，全机 C64 |
| Graph | `FULL_DECODE_ONLY`，capture sizes `[1,2,4,8,16,32]` |
| KV | `block_size=800`、`num_gpu_blocks_override=152` |
| Prefill | chunked prefill，`max_num_batched_tokens=2048`（厂商默认） |
| 其他 | `max_model_len=4096`、`gpu_memory_utilization=0.70`、关闭 custom all-reduce |
| 计数 | streaming completions、`ignore_eos`、每请求精确输出 1024 token |

正式轮之前执行了独立的 `C64 / 3072→8` Graph 预热：64/64 成功、0 抢占、共同墙钟 314.111 s。预热请求不计入正式结果。

## 3. 正式轮性能

### 3.1 全机

| 指标 | KV 152 结果 |
|---|---:|
| 成功请求 | 64/64 |
| 输出 token | 65,536 |
| 共同请求时间窗 | 900.451 s |
| 实际完成输出吞吐 | **72.781 tok/s** |
| Median TTFT | 172,799.50 ms |
| P95 TTFT | 314,876.13 ms |
| Median TPOT | 705.18 ms |
| P95 TPOT | 838.48 ms |
| 平均单请求 Decode TPS | **1.443787 tok/s** |
| Median 单请求 Decode TPS | 1.418127 tok/s |
| Median E2E | 893.306 s |
| P95 E2E | 898.651 s |

四个 vLLM generation counter 在 `16 running / 0 waiting` 区间的合计斜率为 **109.583 tok/s**。这是引擎稳态 Decode 区间的计数器速率，不包含完整 Prefill 和尾部，因此不能替代 72.781 tok/s 的端到端完成吞吐。

### 3.2 分副本

| 副本 | 设备 | 墙钟 | 实际输出吞吐 | 平均单请求 Decode TPS | 对方公式小计 |
|---|---|---:|---:|---:|---:|
| board01 | 0,1 | 897.422 s | 18.257 tok/s | 1.446297 | 23.141 |
| board23 | 2,3 | 896.931 s | 18.267 tok/s | 1.444838 | 23.117 |
| board45 | 4,5 | 900.450 s | 18.195 tok/s | 1.438159 | 23.011 |
| board67 | 6,7 | 897.846 s | 18.248 tok/s | 1.445856 | 23.134 |

四副本墙钟吞吐极差约 0.39%，没有明显的单板掉队。

## 4. 与对方 916.48 的同口径计算

外部表中 `916.48 / 8 / 16 = 7.16`，与另一份外部表的 `Decode Avg TPS=7.16` 完全一致，因此对方的吞吐高度疑似按下式得到：

```text
平均单请求 Decode TPS × 每副本并发 × TP2 副本数
```

本轮按相同公式：

```text
1.443787 × 16 × 4 = 92.402 tok/s
```

| 对比口径 | 本机 KV 152 | 对方 | 本机 / 对方 |
|---|---:|---:|---:|
| 单请求 Decode TPS | 1.443787 | 7.16 | 20.16% |
| 单 TP2、C16 推导值 | 23.101 | 114.56 | 20.16% |
| 4×TP2 公式实测值 | **92.402** | — | — |
| 8×TP2 理想线性投影 | **184.805** | 916.48 | **20.16%** |
| 8×TP2 墙钟吞吐线性投影 | 145.563 | 916.48 | 15.88% |

KV 调优后，同规模公式投影与对方的差距从 8.38× 缩小到约 **4.96×**。`8×TP2` 仍只是将本机 4×TP2 结果乘 2 的理想投影，不是本机实测；外部也没有公开原始请求与测量脚本，因此不能作完全同条件结论。

## 5. GPU、显存和功耗遥测

正式请求时间窗内使用 `fantgpu-smi` 约 1 秒采样，每逻辑芯片 895 个样本：

| 逻辑芯片 | GPU busy 均值 | GPU busy P95 | 显存占用 | Memory busy | 温度峰值 | 整卡功耗均值 / 峰值 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 94.35% | 97% | 33,771 MiB | 58% | 53°C | 298.88 / 338 W |
| 1 | 94.38% | 97% | 33,779 MiB | 58% | 53°C | — |
| 2 | 94.88% | 98% | 33,771 MiB | 58% | 56°C | 311.66 / 351 W |
| 3 | 94.28% | 97% | 33,779 MiB | 58% | 58°C | — |
| 4 | 94.68% | 97% | 33,771 MiB | 58% | 56°C | 312.61 / 352 W |
| 5 | 94.46% | 97% | 33,779 MiB | 58% | 56°C | — |
| 6 | 94.86% | 98% | 33,771 MiB | 58% | 56°C | 305.01 / 344 W |
| 7 | 94.41% | 97% | 33,779 MiB | 58% | 56°C | — |

全机 GPU busy 均值 94.54%、Median 95%、P95 97%。每逻辑芯片显存占用约 33,775/57,344 MiB，即 58.90%；四张物理卡平均总功耗约 1228.16 W。

厂商 `utilization.memory` 全程固定为 58%，其语义是采样周期内存在显存读写的时间比例，不是实际传输字节数占 1008 GB/s 峰值的比例。当前工具没有 DRAM byte counter，因此不能把 58% 直接写成 584.64 GB/s；报告中的带宽利用率只能给出下述模型估算。

## 6. AI、带宽与 TFLOPS

模型权重索引总数据量为 55,562,855,904 B，BF16 等效约 27.781B 参数。按 Decode 常用 dense-equivalent 下界：

```text
FLOPs/token ≈ 2P ≈ 55.563 GFLOP/token
```

| 指标 | KV 152 结果 | 属性 |
|---|---:|---|
| Nominal weight-only AI（C16） | ≈16 FLOP/B | 模型估算 |
| 公式口径 Decode dense-equivalent | **5.134 TFLOPS/全机** | 模型估算 |
| 每逻辑芯片 Decode dense-equivalent | **0.642 TFLOPS** | 模型估算 |
| 墙钟输出 dense-equivalent | 4.044 TFLOPS/全机 | 模型估算 |
| 16/0 稳态 counter dense-equivalent | 6.089 TFLOPS/全机 | counter + 模型估算 |
| 端到端 parameter-equivalent | 16.176 TFLOPS/全机 | 粗略估算，计 3072+1024 token |
| 每逻辑芯片权重流有效带宽 | **40.11 GB/s** | 模型估算 |
| 权重流带宽利用率 | **3.98%** | 40.11 / 1008 GB/s |
| 四副本权重流有效带宽 | 320.88 GB/s | 模型估算 |

上述 AI 只计 BF16 权重流；加入 KV、激活、linear attention state 和 TP2 集合通信后，真实 AI 会下降。TFLOPS 也不是硬件 FLOP counter，而是根据模型参数量和 token 速率得到的 dense-equivalent 值。

与 KV 144 相比，GPU busy 仍在约 94%，但权重流带宽估算从 2.35% 提高到 3.98%，Decode dense-equivalent 从 3.038 提高到 5.134 TFLOPS。说明消除抢占确实把更多设备时间转化为有效 Decode，但绝对计算和带宽效率仍很低。

## 7. 本轮参数筛选记录

| 变量 | 结果 | 决策 |
|---|---|---|
| KV blocks 144 | C64 正式轮每副本 36 次抢占 | 淘汰 |
| KV blocks 160 | C16 3072→8 首轮触发 HIP OOM | 淘汰 |
| **KV blocks 152** | C16 与 C64 均成功，0 抢占 | **采用** |
| `max_num_batched_tokens=8192` | 首次 C16 Prefill HIP OOM | 淘汰 |
| `max_num_batched_tokens=4096` | 首次 C16 Prefill HIP OOM | 淘汰 |
| `max_num_batched_tokens=3072` | 单轮预热墙钟快 1.44%，但 Median TTFT/E2E 更差且 OOM 余量更小 | 不采用 |
| `max_num_batched_tokens=2048` | 厂商默认，稳定 | 保留 |
| concurrent partial prefill=2 | 厂商后端明确报 `NotImplementedError` | 淘汰 |

## 8. 当前瓶颈与下一轮建议

1. **Prefill 未改善。** Median TTFT 与 KV 144 基本相同，3072-token Prefill 仍需约 5 分钟才能让所有 C16 请求进入运行。应针对 Qwen3.5/3.6 Prefill kernel、batch 形状和 Graph 覆盖继续定位，而不是扩大 `max_num_batched_tokens`。
2. **Decode 有效效率仍低。** GPU busy 约 94.5%，但每逻辑芯片只有约 0.642 dense-equivalent TFLOPS、3.98% 权重流带宽估算，优先检查 linear attention/state update、full attention、小算子间隙及 TP2 同步。
3. **KV 余量有限。** 峰值 KV usage 为 95.36%；152 blocks 可满足当前精确负载，但不应外推到更长输出或更高并发。
4. **TP2 通信仍待测。** custom all-reduce 由于厂商路径曾触发崩溃而关闭。需要独立 event timing 后再决定修复或融合，不能直接在正式配置中打开。

## 9. 证据位置

服务器调优根目录：

```text
/data/infra/benchmarks/gr308-qwen36-tuning-20260910T132900Z/
```

KV 152 正式轮：

```text
rollout-kv152/formal-c64-3072x1024-r1/
├── board01-responses.jsonl / board01-summary.json
├── board23-responses.jsonl / board23-summary.json
├── board45-responses.jsonl / board45-summary.json
├── board67-responses.jsonl / board67-summary.json
├── metrics-before-*.prom / metrics-after-*.prom
├── metrics-timeseries.prom
├── gpu-timeseries.csv
└── status
```

预热轮：

```text
rollout-kv152/warmup-c64-3072x8-r1/
```

正式轮结束后四个容器均为 running、restart count 0、OOM=false，端口 18000–18003 的 `/health` 均为 HTTP 200。

## 10. 结论边界

本报告是单轮描述性结果，不能给出置信区间或统计显著性。测试使用合成 prompt 和 `ignore_eos`，只用于吞吐诊断。质量、一致性、真实业务请求分布和生产 SLA 均不在本轮范围内。
