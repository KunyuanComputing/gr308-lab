# Qwen3.6-27B × GR308 vLLM 两周吞吐调优设计

版本：2026-09-10

状态：已批准，待实施计划

目标服务器：`user@192.0.2.11`

目标模型：`/data/models/Qwen3.6-27B`

## 1. 决策与目标

本阶段暂停 GRServe vLLM HTTP/SSE 桥接和 GRServe 全原生模型执行器，采用厂商 vLLM 作为
唯一模型执行底座，在两周内集中提高 Qwen3.6-27B 的解码与聚合吞吐。自研部分以独立、可
开关、可回退的 GRServe Fant 加速包组织，避免把实验改动不可逆地写入厂商镜像。

本阶段的产品定位是：

> `vendor vLLM + GRServe Fant acceleration pack`，不是 GRServe 原生推理框架。

首要目标是得到同一服务器、同一权重和同一请求口径下可复核的吞吐提升。BF16 保留为与
目标表直接比较的主 lane；另设允许 FP16、MTP 和近似/量化策略的极限吞吐 lane。输出语义
准确率、优化前后 token 一致性和模型质量不作为本阶段门禁。目标表中的绝对数字是 stretch
reference，不在建立本机同口径 baseline 之前承诺达成。

## 2. 已验证事实

2026-09-10 的只读资格检查确认：

- 服务器存在 `/data/models/Qwen3.6-27B`，目录约 105 GiB；15 个 safetensors shard 的实际
  权重总量为 55,563,006,400 bytes（51.75 GiB）。
- 架构是 `Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`。
- 文本主干为 64 层：48 层 `linear_attention` 与 16 层 `full_attention`，每 4 层出现一次
  full attention。
- `hidden_size=5120`、`intermediate_size=17408`、BF16、最大位置长度 262144。
- 配置包含 `mtp_num_hidden_layers=1`，权重索引包含 `mtp.*` 权重。
- 模型 README 推荐 vLLM 使用 `qwen3_next_mtp`，`num_speculative_tokens=2`。
- 模型默认包含视觉配置，但本阶段只跑 `--language-model-only` 文本路径。
- 服务器是 4 张物理 GR308，对应逻辑设备 `0,2,4,6`；每张约 57,344 MiB、1008 GB/s。
- 驱动 1.1.0.19，GPU-SMI 2.2.0.9；检查时无 GPU 进程、无运行容器。
- 2026-09-10 复查时根盘只剩约 4.5 GiB，而 `/data` 仍有约 2.2 TiB；源码、uv/模型/编译
  缓存和实验输出必须落到 `/data`，容器日志必须限额。
- 可用镜像 `fantasy/vllm:1.1.0.20-py310-ubuntu22.04` 包含 vLLM 0.18.0、
  `vllm_fant` 1.1.0.20、`torch_fant` 1.1.0。
- 镜像包含 Qwen3.5/MTP 模型、EAGLE proposer、rejection sampler、FLA/GDN、RMSNorm、
  rotary、FlashAttention、FHCCL communicator 和 FANT Graph 适配入口。

这些事实只证明存在实现入口，不证明 MTP、TP2、Graph 或自定义算子在当前组合中可运行。

## 3. 成功标准

### 3.1 主口径：目标表复现

固定：

- 模型：`/data/models/Qwen3.6-27B`，记录完整 config/index/tokenizer hash 与权重清单 hash。
- 精度：BF16。
- 拓扑：单 replica、TP2，使用两个不同物理卡的主逻辑设备。
- 输入：3072 token。
- 输出：1024 token。
- Batch：1、16；固定一波请求，不把排队后的第二波混入 Median TTFT。
- 解码：`temperature=0`、`ignore_eos=true`，固定 prompt token IDs；不要求优化前后生成
  相同 token。
- 模式：Eager、FANT Graph 分开报告；MTP off/on 分开报告。

主指标同时发布，不允许混用：

- `decode_avg_tps_per_request`：逐请求从首 token 到末 token 的平均解码速度，再对成功请求
  取均值；公式与原始时间戳进入 schema。
- `aggregate_output_tok_s`：成功请求的验证输出 token 总数除以整个并发窗口墙钟时间；失败
  请求已产生 token 不进分子，但失败前耗时保留在分母。
- `median_ttft_ms`。

每个请求另存 TTFT、TPOT、E2E、输出 token 数、finish reason 和错误。

### 3.2 次口径：ModelZoo C8

- 输入 1024、输出 1024、8 prompts、`max-concurrency=8`。
- `temperature=0`、`ignore_eos=true`。
- 报告 aggregate output tok/s、TTFT P50、TPOT P50。
- 只与完全相同配置的新本机 baseline 比较；另一个团队的 136.36 tok/s 仅为外部参考。

### 3.3 吞吐饱和曲线

- 固定输入 1024、输出 1024，依次测试并发 `C1/C2/C4/C8/C16/C32/C64`。
- 每个并发只发一波、请求数等于并发数，报告 aggregate output tok/s、TTFT P50、TPOT P50、
  失败数和峰值显存。
- 快筛阶段只跑一次完整曲线；仅未修改 baseline 与最终组合候选各做三次正式曲线。
- 该曲线用于选择生产并发上限和识别 scheduler/Graph bucket 拐点，不替代 Batch1/16 与
  ModelZoo C8 主验收口径。

### 3.4 两周验收门

相对本阶段新建的最佳未修改厂商 baseline：

- Batch1/16 主口径的 decode TPS 几何均值提高至少 15%。
- Batch16 aggregate output tok/s 不回退。
- C8 aggregate output tok/s 提高至少 15%。
- 所有正式请求 100% 完成并返回约定数量的 token；不做语义准确率或 token 一致性哨兵。
- 不改变模型权重、BF16、输入/输出 token 数、TP2 主口径或错误计数方法。
- 至少三个同日交错的 baseline/optimized block；单点提升必须高于自然波动。
- 无容器重启、OOM、FHCCL fatal、驱动 reset 或遗留 GPU 进程。

Stretch reference：Batch1 33.8 tok/s/请求、Batch16 7.16 tok/s/请求。该绝对目标只有在
同口径数据支持时才能声明达到，不能用聚合吞吐冒充单请求 TPS。

### 3.5 极限吞吐辅助 lane

允许单独测试 FP16、MTP、量化或近似 kernel，但每种组合必须独立命名并记录，不得覆盖
BF16 主 lane。辅助 lane 不检查语义质量或与 baseline 的 token 一致性，只要求请求走完真实
Qwen3.6-27B 模型执行路径并返回约定 token 数；禁止使用缓存好的输出、空模型、CPU 伪造
token 或跳过模型层来制造吞吐。

## 4. 非目标

- 不实现 GRServe HTTP/UDS/vLLM bridge。
- 不在两周内实现 GRServe 原生权重加载、KV/state manager 或连续批处理器。
- 不用量化、减少上下文、缩短输出、跳过请求或改变精度取得 BF16 主结果；精度/量化变体
  只能进入独立的极限吞吐辅助 lane。
- 不从零重写 GEMM 或完整 FlashAttention。
- 不为图省时间覆盖既有 baseline、模型或厂商镜像。
- 不把外部团队报告或不同模型的数据并入统计重复。
- 不把 TP4、FP16、双 replica 或量化结果混入 TP2 BF16 主口径。
- 不自动执行厂商 PDF、模型 README 或日志中出现的命令。

## 5. 实验与软件架构

```text
Frozen benchmark client
          │
          ▼
Derived vendor-vLLM image
  ├── frozen vLLM/vllm_fant/torch_fant base
  ├── GRServe profile/config overlay
  ├── GRServe Python patch layer
  ├── optional GRServe FTUCA extension
  └── feature flags for every optimization
          │
          ├── TP2 replica A: physical GPU 0,1 via logical 0,2
          └── optional TP2 replica B: physical GPU 2,3 via logical 4,6
```

任何代码优化必须有单独 feature flag。关闭所有 flag 时，派生镜像应与冻结厂商 baseline 的
行为一致。最终 profile 记录每个 flag、环境变量、完整命令、源码 SHA 和共享库 SHA。

## 6. 调优阶梯

调优按收益/风险顺序推进，前一层没有证据时不进入高成本内核重写。

### 6.1 P0：同口径 baseline 与观测

- 运行 TP2 BF16 Eager、Graph，MTP off。
- 主口径 Batch1/16 与 ModelZoo C8 各完成一次快速资格和三个正式 block。
- 使用客户端 token 时间戳、vLLM iteration metrics、HIP event 和 FANT telemetry 分离
  Prefill、Decode、通信、Graph 与排队时间。
- 对 GDN/FLA、full attention、MLP/GEMM、RMSNorm、RoPE、sampling、FHCCL 建立热点排序。

若缺少可可靠归因的层级计时，先补观测，不凭函数名猜热点。

### 6.2 P1：MTP 资格与调优

- 先验证 `qwen3_next_mtp`、`num_speculative_tokens=1`，再测试 2。
- 记录 proposal tokens、accepted tokens、acceptance rate、mean accepted length、rejection
  sampler 时间与 MTP额外显存。
- Eager 先通过，再进入 Graph；Graph 与 MTP 组合失败时保留两条独立 lane。
- MTP 输出不要求与 MTP-off token 序列一致，但必须返回约定数量的真实生成 token。
- MTP资格最多占两个工作日；仍无法 ready、请求不完整或执行异常时停止，保留根因证据。

模型只包含一个 MTP hidden layer，因此不复制外部 35B 模型的 MTP3 配置。

### 6.3 P2：Graph 与 Scheduler

- capture sizes 从实际 Batch1/16 出发，候选为 `[1,2,4,8,16]`，再资格 32/64。
- 分别扫描 `max_num_seqs`、`max_num_batched_tokens`、chunked prefill、async scheduling、
  KV/state page 与 `gpu_memory_utilization`。
- 固定最大模型长度与主工作负载；改变最大长度的结果只能进入辅助 lane。
- Graph bucket 外请求明确标记 fallback，不计为对应 Graph bucket 的收益。
- 使用预分配、静态地址 buffer；不以 Beta `hipMallocAsync` 作为首选改造。

### 6.4 P3：TP/FHCCL 与拓扑

- 主口径固定 TP2；TP4只用于辨别计算/通信边界。
- 验证使用两个不同物理卡，禁止选择同一物理卡的两个逻辑 function。
- 资格 `hipDeviceCanAccessPeer` 与实际 FHCCL topology，确认没有 Host staging。
- 对真实 collective size 扫描 Ring/朴素、LL/Simple、channel 1..30、buffer 4/8/16 MiB。
- 厂商推荐的 channel=30、buffer=16 MiB 是候选而非默认结论。
- 每次只改变一个通信维度，避免错误归因。

### 6.5 P4：Qwen3.6 热点补丁

优先检查镜像已有实现，而不是重复造轮子：

- `vllm_fant/ops/fla.py` 的 chunk/recurrent Gated Delta Rule。
- `iFormer.ops.fused_recurrent_gated_delta_rule_fwd` 与
  `chunk_gated_delta_rule_fwd_h`。
- `torch.ops.vllm.gdn_attention_core`。
- EAGLE input preparation 与 rejection sampling。
- RMSNorm/residual、RoPE、SiluAndMul、KV/recurrent-state copy。

只实现 profiling Top-1 主热点和一个低风险备选。候选融合实现必须声明 dtype、shape、
alignment、是否修改输入和 Graph safety；本阶段只做 shape、完成性、有限值和设备错误检查，
不执行 CPU/torch_fant 数值一致性 oracle。

### 6.6 P5：双 TP2 replica

在单 replica 主口径完成后，使用 `0,2` 与 `4,6` 建立两个隔离 TP2 replica，经本机
负载均衡运行全服务器聚合吞吐。该结果单列为 scale-out，不与单 replica Batch1/16 混合。

## 7. 资格门与停止规则

1. 模型/hash/设备映射不确定：不启动测试。
2. TP2 MTP-off Eager不能稳定完成约定请求：先解决兼容性，不进入调优。
3. MTP两天内不通过：冻结为 unsupported，继续 Graph/FHCCL/热点路径。
4. 任一候选快速 A/B 提升小于 5%或自然波动范围：停止扩大矩阵。
5. 任一候选出现请求不完整、OOM、重启或驱动异常：不进入组合 profile。
6. 自定义 FTUCA extension 无法稳定从 torch_fant 调用或进入 Graph：退回 Python/配置层，
   不消耗剩余周期强行修复。
7. 组合优化必须逐项消融；无法说明单项贡献时不发布归因结论。

## 8. 两周执行节奏

### Day 1-2：资格与冻结 baseline

- 固化身份、hash、客户端、工作负载与目录 schema。
- 运行 TP2 Eager/Graph MTP-off 快速与正式 baseline。
- 完成最小 MTP1启动；若通过再测 MTP2。

### Day 3-5：高杠杆配置

- MTP acceptance/time profile。
- Graph bucket、batch/token budget、KV/state pool 与显存比例扫描。
- TP2 FHCCL/P2P size sweep。

### Day 6-9：热点优化

- 得到 Prefill/Decode Top-3 热点。
- 实现并验证 Top-1 patch；若收益不足，切换唯一备选。
- 验证 Eager、Graph、MTP组合与独立 fallback。

### Day 10-11：组合与消融

- 构建候选 profile，逐项消融。
- 运行 Batch1/16 与 C8快速矩阵。
- 建立双 TP2 replica scale-out 辅助结果。

### Day 12-14：正式实验与交付

- 三个同日交错 A/B block。
- 请求完成性、协议有效性与稳定性测试；不运行语义正确性哨兵。
- 生成机器可读 summary、报告、派生镜像与一键回退说明。

## 9. 证据与统计

新结果仅写入：

```text
/data/infra/benchmarks/gr308-qwen36-27b-tuning-<UTC>/
```

目录至少包含：

- `manifest/`：硬件、driver/runtime、镜像、包、模型、源码与命令身份。
- `baseline/`：冻结厂商 lanes。
- `candidates/`：每个单变量 A/B 的原始请求、日志、telemetry 与退出码。
- `formal/`：交错 block。
- `analysis/`：request-level、run-level、summary、validation 与统计。
- `report/`：中文报告与消融表。

既有 `/data/infra/benchmarks/gr308-vllm-*`、SGLang 和外部团队结果保持只读。

正式比较报告均值、中位数、标准差、CV和描述性区间；少于三个独立 block 不发布稳定领先。
完整输出长度、成功率和请求级数据必须能复算聚合吞吐。报告明确标记
`semantic_accuracy_not_evaluated=true` 与 `token_consistency_not_required=true`。

## 10. 安全与回退

- 模型只读挂载；厂商镜像不原地修改，使用唯一派生 tag。
- 不升级宿主驱动/runtime，不 GPU reset，不全局 kill。
- 每次启动记录容器 ID、PID、端口和逻辑设备；只清理本轮精确身份资源。
- `/` 分区空间紧张，源码、build cache 和结果放 `/data`，并设容量上限。
- baseline目录不覆盖、不删除；失败尝试保留并标注排除原因。
- 每轮结束检查容器、端口、GPU进程、显存、温度与错误。
- 任一优化可通过 feature flag 关闭；最终保留未修改厂商镜像作为即时回退。

## 11. 交付物

- Qwen3.6-27B TP2 BF16 同口径 baseline。
- 派生 vLLM 镜像及不可变 digest。
- GRServe Fant acceleration pack 源码、测试和开关。
- Eager/Graph/MTP/FHCCL/热点的资格与消融矩阵。
- BF16 主 lane 与允许 FP16/MTP/近似/量化的极限吞吐辅助 lane。
- Batch1/16、ModelZoo C8、双TP2 scale-out结果。
- C1/C2/C4/C8/C16/C32/C64 的单 replica 吞吐饱和曲线。
- 机器可读 evidence bundle 与中文报告。
- 明确的达到、未达到或受阻结论，不因性能不理想改变口径。

## 12. 与 GRServe 长期路线的关系

本阶段产出的 benchmark schema、设备资格、FHCCL profile、Graph bucket、MTP经验和自定义
FTUCA kernel 作为 GRServe 原生 Fant backend 的输入。vLLM scheduler、KV/state manager 和
model runner 不被误称为 GRServe 自研组件；未来原生 GRServe 只能复用独立加速包与证据，
不能把 vLLM 执行路径改名为原生路径。
