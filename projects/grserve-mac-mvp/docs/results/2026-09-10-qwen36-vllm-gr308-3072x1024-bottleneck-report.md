# Qwen3.6-27B × GR308：3072/1024、4×TP2 C64 性能与瓶颈报告

日期：2026-09-10  
状态：单轮对齐测试完成  
目标：优先评价吞吐，不评价语义质量、一致性或生产 SLA

## 1. 结论摘要

本轮按目标配置完成 64/64 个请求，未发生请求错误或容器重启。四个 TP2 副本的共同测量墙钟为 1410.741 s，全机实际完成输出吞吐为 **46.455 output tok/s**。

按外部结果 `916.48 = 7.16 × 16 × 8` 所暗示的计算方法，本机结果为：

```text
平均单请求 Decode TPS × 每副本并发 × TP2 副本数
= 0.854292 × 16 × 4
= 54.6747
```

若只做硬件数量的理想线性外推，本机 8×TP2 对应 **109.3494**，为对方 916.48 的 **11.93%**；换算为同等硬件规模后仍相差 **8.38×**。该外推不是实测值。

当前最明确的瓶颈证据不是显存容量不足或 GPU 空闲，而是：

1. 四个副本的 KV cache 均达到 100%，每副本新增 36 次 preemption，总计 144 次；
2. Median TTFT 为 172.363 s，长输入 Prefill 与调度开销非常高；
3. GPU busy 均值达到 93.80%，但模型权重流量反推的有效带宽仅约为每逻辑芯片 23.73 GB/s，即服务器报告理论带宽的 2.35%；
4. Decode dense-equivalent 算力仅约 3.038 TFLOPS/全机、0.380 TFLOPS/逻辑芯片，说明“GPU 很忙”并没有转化为高有效矩阵计算吞吐。

因此下一轮优化应优先解决 KV 抢占与长输入 Prefill，其次定位 Qwen3.5/3.6 混合线性注意力算子、TP2 集合通信和 Graph 覆盖率，而不是仅增加请求并发。

## 2. 测试配置

| 项目 | 配置 |
|---|---|
| 模型 | `/data/models/Qwen3.6-27B` |
| 模型架构 | `Qwen3_5ForConditionalGeneration`，64 层，3:1 linear/full attention |
| 精度 | BF16 |
| 输入 / 输出 | 3072 / 1024 token |
| 执行拓扑 | 4×TP2，逻辑设备 `(0,1)`、`(2,3)`、`(4,5)`、`(6,7)` |
| 并发 | 每副本 C16，全机 C64 |
| 请求数 | 每副本 16，全机 64 |
| Graph | `FULL_DECODE_ONLY`，capture sizes `[1,2,4,8,16,32]` |
| KV 配置 | `max_model_len=4096`、`block_size=800`、`num_gpu_blocks_override=144` |
| 其他 | `gpu_memory_utilization=0.70`、`max_num_seqs=64`、关闭 custom all-reduce |
| 计数方式 | streaming completions、`ignore_eos`、精确输出 1024 token |

正式轮之前单独执行了 `3072 input / 8 output / C64` 预热；预热请求不计入正式结果。正式轮四个副本同时启动，遥测间隔约 1 s。

## 3. 请求性能结果

### 3.1 全机结果

| 指标 | 结果 |
|---|---:|
| 成功请求 | 64/64 |
| 成功输出 token | 65,536 |
| 共同测量墙钟 | 1410.741 s |
| 编排器总墙钟（含前后遥测余量） | 1414.804 s |
| 实际完成输出吞吐 | **46.455 tok/s** |
| Median TTFT | 172,362.84 ms |
| P95 TTFT | 312,999.20 ms |
| Median TPOT | 1,177.17 ms |
| P95 TPOT | 1,299.10 ms |
| 平均单请求 Decode TPS | **0.854292 tok/s** |
| Median 单请求 Decode TPS | 0.849492 tok/s |
| Median E2E | 1,376.003 s |
| P95 E2E | 1,392.283 s |

按请求 TPOT 汇总得到的 54.675 tok/s 比真实墙钟吞吐高 17.69%。差异主要来自 TTFT、动态调度、KV 抢占和四副本尾部时间，因此对外应同时报告两种口径，不能把 54.675 当作实际服务器完成吞吐。

### 3.2 分副本结果

| 副本 | 逻辑设备 | 成功请求 | 墙钟（s） | 实际输出吞吐（tok/s） | 平均单请求 Decode TPS | 外部公式小计 |
|---|---|---:|---:|---:|---:|---:|
| board01 | 0,1 | 16/16 | 1392.714 | 11.764 | 0.859322 | 13.749 |
| board23 | 2,3 | 16/16 | 1404.871 | 11.662 | 0.852642 | 13.642 |
| board45 | 4,5 | 16/16 | 1410.741 | 11.614 | 0.848953 | 13.583 |
| board67 | 6,7 | 16/16 | 1397.577 | 11.723 | 0.856251 | 13.700 |

四副本实际输出吞吐极差约 1.29%，没有明显的单板异常；当前瓶颈是全局共有路径。

## 4. 与 916.48 的同口径比较

外部表格写的是 `8×TP2、并发 16、吞吐 916.48`。由于 `916.48 / 8 / 16 = 7.16`，它与另一份外部数据中的 `Decode Avg TPS = 7.16` 完全吻合，因此高度疑似采用：

```text
吞吐 = 平均单请求 Decode TPS × 每副本并发 × TP2 副本数
```

| 比较项 | 本机 | 外部 | 本机 / 外部 |
|---|---:|---:|---:|
| 单请求 Decode TPS | 0.854292 | 7.16 | 11.93% |
| 单 TP2、C16 推导吞吐 | 13.6687 | 114.56 | 11.93% |
| 4×TP2 实测公式值 | 54.6747 | — | — |
| 8×TP2 理想线性投影 | 109.3494 | 916.48 | 11.93% |
| 8×TP2 墙钟吞吐线性投影 | 92.9100 | 916.48 | 10.14% |

此处只有本机单轮原始请求记录；外部仅有汇总表，没有客户端、prompt 构造、token 计数、Graph 覆盖率和重复轮次。因此该比较可以说明数量级差距，但不能作统计显著性或完全同条件结论。

## 5. 硬件遥测（实测）

以下数据来自正式测量窗口内的 `fantgpu-smi`，每个逻辑芯片约 1406–1407 个样本。

| 物理卡 / 逻辑芯片 | GPU busy 均值 | GPU busy P95 | Memory busy | 显存占用（每逻辑芯片） | 温度峰值 | 整卡功耗均值 / 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| board01 / 0,1 | 93.29% / 93.08% | 98% / 98% | 58% / 58% | 33695 / 33703 MiB | 53 / 53 °C | 301.57 / 340 W |
| board23 / 2,3 | 94.27% / 94.03% | 98% / 98% | 58% / 58% | 33695 / 33703 MiB | 56 / 58 °C | 315.74 / 351 W |
| board45 / 4,5 | 94.39% / 94.09% | 98% / 98% | 58% / 58% | 33695 / 33703 MiB | 56 / 56 °C | 316.89 / 353 W |
| board67 / 6,7 | 93.70% / 93.53% | 98% / 98% | 58% / 58% | 33695 / 33703 MiB | 56 / 56 °C | 309.62 / 348 W |

全机汇总：

- GPU busy 均值 93.80%，Median 94%，P95 98%；
- 每逻辑芯片显存占用约 33,699 MiB，即报告总量 57,344 MiB 的 58.77%；
- Memory busy 始终为 58%；
- 显存时钟始终为 9000 MHz；
- 四张物理卡平均总功耗约 1243.83 W；
- 温度范围 49–58 °C；
- 服务器对每个逻辑芯片报告理论显存带宽 1008 GB/s。

`utilization.memory` 的厂商定义是“采样周期内全局显存被读写的时间百分比”，不是传输字节数占理论峰值的比例。因此这里将 58% 记为 **Memory busy**，不能直接写成实际带宽利用率或 `584.64 GB/s`。该字段在全程固定为 58%，分辨率/可信度也不足以单独判断带宽瓶颈。

## 6. AI、带宽与 TFLOPS（模型估算）

当前已安装工具没有可用的 DRAM byte counter 或硬件 FLOP counter；PyTorch 指南也明确不支持 `torch.autograd.profiler.profile`。所以下表严格区分模型估算与硬件实测。

模型权重索引记录总数据量 55,562,855,904 B。按 BF16 等效换算约 27.781B 参数，采用 Decode 的常用 dense-equivalent 下界 `2P`：

```text
dense-equivalent FLOPs/token ≈ 55.563 GFLOP/token
```

| 派生指标 | 结果 | 解释 |
|---|---:|---|
| Nominal weight-only AI（C16） | ≈16 FLOP/B | `2P×B / BF16权重字节`，仅计权重流量 |
| 稳态 Decode AI 范围 | ≈15–16 FLOP/B | 运行状态主要为 16 running 或 15 running + 1 waiting |
| 全机 Decode dense-equivalent | **3.038 TFLOPS** | 使用 54.6747 的 TPOT 汇总口径 |
| 每逻辑芯片 Decode dense-equivalent | **0.380 TFLOPS** | 全机 8 个逻辑芯片均分 |
| 墙钟输出 dense-equivalent | 2.581 TFLOPS | 使用 46.455 实际完成吞吐 |
| 端到端 parameter-equivalent | 10.325 TFLOPS | 粗略计入 3072+1024 token；不含注意力额外项 |
| 每 TP2 权重流有效带宽 | 47.47 GB/s | 权重大小 × 平均单请求 Decode TPS |
| 每逻辑芯片权重流有效带宽 | 23.73 GB/s | TP2 权重分片二分 |
| 权重流带宽利用率 | **2.35%** | 23.73 / 1008 GB/s |
| 四副本权重流有效带宽 | 189.87 GB/s | 4×TP2 合计估算 |

这些数值的边界：

- AI 只计算 BF16 权重流；加入 KV、激活、linear attention state 和集合通信后，真实 AI 会下降；
- TFLOPS 是 dense-equivalent 模型计算量，不是芯片性能计数器读数；
- 权重带宽是假设每个 Decode step 从显存流过一次 TP 分片权重的估算，不包含其他显存流量；
- Memory busy 58% 与权重流带宽利用率 2.35% 并不矛盾：前者只表示有读写活动的时间比例，后者才是按字节量与理论峰值比较的模型估算。

服务器实测带宽属性为 1008 GB/s/逻辑芯片；厂商公开 GR308S 页面写的是“最高 1120 GB/s”。本报告所有利用率均采用本机驱动返回的 1008 GB/s，避免使用更乐观的纸面峰值。

## 7. 调度与 KV 证据

| 指标 | board01 | board23 | board45 | board67 |
|---|---:|---:|---:|---:|
| KV cache 峰值 | 100% | 100% | 100% | 100% |
| 本轮 preemption 增量 | 36 | 36 | 36 | 36 |
| prompt recompute | 0 | 0 | 0 | 0 |
| `15 running / 1 waiting` 持续时间 | 503.6 s | 508.9 s | 512.0 s | 505.7 s |
| 存在 waiting 的活跃时间占比 | 55.29% | 55.51% | 55.56% | 55.35% |

正式轮中每副本都经历了从 Prefill 阶段的 waiting 队列，到 `16/0`，再到长时间 `15/1`。最终 preemption 计数证明 KV=144 对 `3072+1024、C16` 并不足以避免抢占。`prompt_tokens_recomputed_total=0`，但现有指标无法确认恢复过程采用了何种缓存保留或换出机制。

## 8. 瓶颈判断与调优顺序

### P0：消除 KV 抢占

显存遥测只占约 58.77%，而 vLLM 被人为限制为 `gpu_memory_utilization=0.70`、144 blocks。下一轮应在单副本上快速筛选更高 KV 配置，例如 160/176/192 blocks 或提高 memory utilization，然后只将无 OOM、preemption=0 的配置扩展到四副本。目标是保持 C16 全程 `16 running / 0 waiting`。

### P0：优化 3072-token Prefill

Median TTFT 172.4 s、P95 313.0 s，且首 token 到达呈明显阶梯状，说明 Prefill 没有形成高效批处理。应优先筛选 `max_num_batched_tokens`、chunked prefill、调度策略和 Prefill batch 形状；当前 Graph 只覆盖 Decode，不能改善 Prefill。

### P1：定位低有效 Decode 吞吐

GPU busy 接近 94%，但每逻辑芯片只有约 0.380 dense-equivalent TFLOPS 和 2.35% 权重流带宽利用率。这表明大量时间可能消耗在低效率 kernel、同步、算子间空隙或通信，而不是高吞吐 GEMM。应使用 HIP event 对下列路径逐层计时：

1. Qwen3.5/3.6 linear attention、conv/state update；
2. full attention；
3. RMSNorm、RoPE、SwiGLU、采样等小算子；
4. TP2 all-reduce / all-gather；
5. Graph 内与 Graph 外的 Decode op 覆盖率。

### P1：恢复或替代高效 TP2 通信

当前关闭 custom all-reduce，是因为厂商路径此前触发 iFormer/FHCCL 段错误。标准 FHCCL/RCCL 路径虽然稳定，但可能是 TP2 Decode 的同步瓶颈。需要对每层集合通信做 event timing，再决定修复 custom all-reduce、融合通信，还是调整张量切分。

## 9. 证据与复现位置

服务器证据根目录：

```text
/data/infra/benchmarks/gr308-qwen36-target-3072x1024-20260910T125300Z/
```

正式轮：

```text
formal-c64-3072x1024-r1/
├── board01-responses.jsonl / board01-summary.json
├── board23-responses.jsonl / board23-summary.json
├── board45-responses.jsonl / board45-summary.json
├── board67-responses.jsonl / board67-summary.json
├── fantgpu-telemetry.csv
├── vllm-metrics.prom
└── run-status.txt
```

预热轮：

```text
warmup-c64-3072x8-r1/
```

正式轮结束后，四个容器均保持 running、restart count 0、OOM=false，端口 18000–18003 的 `/health` 均返回 HTTP 200。

## 10. 结论边界

这是单轮描述性结果，不能给出置信区间。外部 916.48 没有原始请求记录和遥测，`8×TP2` 也对应本机两倍的 TP2 副本数。当前最可信的比较是硬件归一化后的单 TP2 C16：本机 13.669，对方推导 114.56，本机约为 11.93%。下一轮只有先把 preemption 清零并显著降低 TTFT，才有必要继续比较更高并发。
