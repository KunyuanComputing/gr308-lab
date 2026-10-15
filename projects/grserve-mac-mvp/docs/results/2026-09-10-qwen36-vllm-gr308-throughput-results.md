# Qwen3.6-27B × GR308 厂商 vLLM 吞吐调优阶段报告

日期：2026-09-10  
状态：短口径（input 1024 / output 1024）C8–C64 调优完成  
目标：只优化吞吐；本阶段不评价语义正确性、一致性或生产 SLA

## 1. 硬件与执行拓扑

- 服务器包含 4 张双芯 GR308 GPU 卡，HIP 枚举为 8 个逻辑设备。
- PCIe 设备为 `2e:00.0/.2`、`34:00.0/.2`、`37:00.0/.2`、`38:00.0/.2`。
- 最终执行拓扑为 4 个 TP2 vLLM 副本，设备对分别为 `(0,1)`、`(2,3)`、`(4,5)`、`(6,7)`。
- 容器镜像：`sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc`。
- 软件：厂商 vLLM 0.18.0、vllm_fant 1.1.0.20、BF16、FANT Graph。
- 模型：`/data/models/Qwen3.6-27B`，served model name 为 `qwen3.6-27b`。

所有设备均挂在 NUMA0。将每个副本硬绑定到 NUMA0 的 14 个物理核及其超线程后，吞吐下降 0.93%，因此最终配置恢复为全 CPU/NUMA 可用。

## 2. 测试口径

- 合成 exact-token streaming completions，`ignore_eos`。
- 输入 1024 token，输出固定 1024 token。
- 每个并发点先做同并发、输出 8 token 的显式预热；预热不计入正式结果。
- C64 表示四个 TP2 副本各承载 C16，共 64 个并发请求。
- 聚合吞吐按 `全部成功输出 token / 四副本共同墙钟时间` 计算。
- Graph 使用 `FULL_DECODE_ONLY`，capture sizes 为 `1,2,4,8,16,32`。
- 自定义 all-reduce 在板内 TP2 上触发 iFormer/FHCCL 路径段错误，故使用 `--disable-custom-all-reduce`，回退到稳定的标准 FHCCL/RCCL 集合通信。

## 3. Baseline

Baseline 关键参数：`max_model_len=32768`、`gpu_memory_utilization=0.70`、`num_gpu_blocks_override=64`、`max_num_seqs=64`。

| 全机并发 | 成功请求 | 墙钟（s） | 聚合输出吞吐（tok/s） |
|---:|---:|---:|---:|
| C8 | 8/8 | 359.665 | 22.777 |
| C16 | 16/16 | 378.954 | 43.235 |
| C32 | 32/32 | 473.035 | 69.272 |
| C64 | 64/64 | 1147.070 | 57.133 |

Baseline 最佳并发点为 C32。C64 时每个副本只能保持约 `12 running / 4 waiting`，KV 使用率约 95.2%，吞吐较 C32 下降约 17.5%。

## 4. 调优实验

### 4.1 缩短运行上下文并扩大 KV：88 blocks

参数改为 `max_model_len=4096`、`num_gpu_blocks_override=88`，其余保持不变。

- 单副本 C16、output 256：18.199 tok/s；相对 64-block 对照 12.214 tok/s 提升 49.0%。
- 全机 C64、output 1024：76.983 tok/s，墙钟 851.305 s；相对 baseline C64 提升 34.74%。
- 88 blocks 在前半程可达到每副本 `16 running / 0 waiting`，但序列增长后退化为约 `14 running / 2 waiting`，仍有 KV 容量瓶颈。

### 4.2 扩大 KV：144 blocks

在 `max_model_len=4096` 下把 `num_gpu_blocks_override` 提高到 144。

- 单副本 C16、output 256：18.367 tok/s，仅比 88 blocks 高 0.92%。
- 单副本 C16、output 1024：27.833 tok/s，比同板 88-block 的 19.247 tok/s 提升 44.61%。
- 全机 C64、output 1024：96.622 tok/s，墙钟 678.272 s。
- 相对 baseline C64 提升 69.12%；相对 88-block 调优版再提升 25.51%。
- 四副本均能持续保持 `16 running / 0 waiting`，KV 使用率从约 55.9% 阶梯增长到约 67.1%，未发生容量换出。

全机正式轮分板结果：

| 副本 | 设备 | 成功请求 | 时长（s） | 输出吞吐（tok/s） |
|---|---|---:|---:|---:|
| board01 | 0,1 | 16/16 | 678.217 | 24.157 |
| board23 | 2,3 | 16/16 | 587.017 | 27.911 |
| board45 | 4,5 | 16/16 | 593.204 | 27.620 |
| board67 | 6,7 | 16/16 | 586.393 | 27.940 |

64 条响应全部成功、全部精确输出 1024 token。整体 median TTFT 为 64943.1 ms，median TPOT 为 528.9 ms，median E2E 为 588446.1 ms。board01 在四副本同时运行时 TPOT 为 600.1 ms，单独运行时为 511.6 ms，表明存在并发资源或热态相关的板间拖尾。

### 4.3 CPU/NUMA 硬绑定（否决）

四个副本分别绑定到 NUMA0 的独立 CPU 集：

- board01：`0-13,112-125`
- board23：`14-27,126-139`
- board45：`28-41,140-153`
- board67：`42-55,154-167`

结果为 95.721 tok/s、墙钟 684.657 s，比不绑定的 96.622 tok/s 下降 0.93%。该方案使四板更均衡，但限制了最快三板的 host 侧吞吐，已回退到 `cpuset-cpus=0-223`、`cpuset-mems=0-1`。

## 5. 当前推荐配置

```text
4 × TP2 replicas
CUDA_VISIBLE_DEVICES: 0,1 | 2,3 | 4,5 | 6,7
dtype: bfloat16
max_model_len: 4096
gpu_memory_utilization: 0.70
block_size: 800
num_gpu_blocks_override: 144
max_num_seqs: 64
language_model_only: true
prefix_caching: false
async_scheduling: false
disable_custom_all_reduce: true
FANT Graph: FULL_DECODE_ONLY
capture_sizes: [1,2,4,8,16,32]
CPU affinity: unrestricted
```

四个最终容器均为 running、restart count 0，端口 `18000-18003` 的 `/health` 均返回 200。

## 6. 证据目录

- 根目录：`/data/infra/benchmarks/gr308-qwen36-direct-20260910T073108Z/`
- 8 芯 baseline：`topology-correction-20260910T1629/four-tp2-c{8,16,32,64}-r1`
- 88-block C64：`topology-correction-20260910T1629/tune-ctx4096-kv88/four-tp2-c64-out1024-r2`
- 144-block 单板筛选：`topology-correction-20260910T1629/tune-ctx4096-kv144-screen/`
- 144-block 全机正式轮：`topology-correction-20260910T1629/tune-ctx4096-kv144-rollout/four-tp2-c64-out1024-r1`
- CPU/NUMA 否决实验：`topology-correction-20260910T1629/tune-ctx4096-kv144-cpupin/four-tp2-c64-out1024-r1`

## 7. 边界与下一步

本报告的 96.622 tok/s 是 `1024 input / 1024 output` 的合成吞吐诊断，不能直接与外部团队的 `3072 input / 1024 output` 表格比较，也不代表质量或生产 SLA。对齐外部口径前，需要：

1. 保持同一模型、TP2、BF16、Graph 与 exact-token 计数方式；
2. 使用 3072-token 输入和 1024-token 输出；
3. 按目标是“单 TP2 的 Batch16”还是“全机四副本总 C16/C64”分别报告；
4. 如果单个 TP2 副本承载 C16×4096 token，应继续筛选更大的 KV block 配置，144 blocks 不足以保证全程驻留；
5. 单独调查 board01 在四副本满载下的 TPOT 拖尾，并评估动态路由能否减少慢板造成的全局尾部空闲。
