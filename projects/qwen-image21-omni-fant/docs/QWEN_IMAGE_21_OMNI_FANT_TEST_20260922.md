# Qwen-Image-2.1 vLLM-Omni 思路移植到 GR308 的测试记录

## 移植范围

本次保留已经在 GR308 上跑通的 Diffusers/FANT 模型实现，把 vLLM-Omni
的四阶段接口拆分为 `prepare_encode`、`denoise_step`、`step_scheduler` 和
`post_decode`。并行部分使用 FHCCL，在两颗逻辑芯片之间分配 VAE 重叠瓦片，
随后用 all-reduce 汇总，并按 Qwen-Image-2.1 原生 overlap/blend 规则拼接。

未直接引入 vLLM-Omni 的 CUDA runtime、CUDA Graph、FlashAttention 或 FP8。

## FHCCL 实测

测试设备为同一张双芯卡的逻辑设备 `fant:0` 和 `fant:1`。

| 操作 | 结果 | 64 MiB 结果 |
|---|---:|---:|
| broadcast | 正确 | — |
| all-reduce BF16 | 正确 | 约 2.68 ms，约 25.0 GB/s 总线带宽 |
| all-gather | 正确 | 约 3.91 ms，约 34.3 GB/s 汇总吞吐 |
| all-to-all | 不支持 | `Backend fhccl does not support alltoall_base` |

因此，原版 Ulysses 序列并行不能直接运行。当前实际启用了分布式 VAE
瓦片解码，以及不依赖 all-to-all 的 DiT TP2。

## 512×512、2 步短测

| 阶段 | 结果 |
|---|---:|
| DiT 去噪 | 9.28 秒 |
| DiT 峰值显存 | 33,106,365,440 bytes |
| VAE 瓦片数 | 9（rank 0: 4，rank 1: 5） |
| 两芯 VAE 阶段 | 约 179 秒 |
| 单芯 VAE 峰值显存 | 约 5.1–5.4 GiB |
| 输出 | 512×512 RGBA |
| SHA256 | `f039e62b963108199c4de834614c3a83b013c37822dd4b997b10b9b15e5a53ae` |

输出路径：

`/data/infra/qwen-image-2.1-omni-fant-20260922/output/qwen-image-2.1-bfloat16-512x512-2steps-omni-fant-tp2.png`

两步采样只用于执行链验证，图像细节和文字质量不代表正常 40 步质量。

## 1024×1024、4 步验证

该次复用此前由相同 BF16 DiT 路径生成的 4 步 latent，只重新执行两芯 VAE
阶段，用来验证目标分辨率的瓦片分配、通信和拼接。

| 指标 | 结果 |
|---|---:|
| 瓦片数 | 36（每个 rank 18） |
| 两芯 VAE 总阶段 | 143.50 秒 |
| rank 0 本地解码 | 140.37 秒 |
| rank 1 本地解码 | 113.02 秒 |
| rank 0 峰值显存 | 18,699,708,928 bytes |
| rank 1 峰值显存 | 24,701,789,696 bytes |
| 输出 | 1024×1024 RGBA |
| SHA256 | `08e2c4a2acad43b4b2960bf4e0de9b343ed804830705f4492e66dd432e2f0b23` |

作为背景，原单芯整图 VAE 解码实测为 1241.47 秒、峰值
32,374,563,328 bytes。两条路径的算法不同（整图与瓦片），因此该时间对比
说明当前可用部署路径的收益，不等同于只改变芯片数量的严格 A/B。

同一 latent 的单芯串行瓦片对照耗时 154.70 秒，峰值
42,568,008,704 bytes。两芯版本耗时缩短约 7.2%，同时把每颗芯片的峰值压到
约 17.4–23.0 GiB。两芯和单芯瓦片输出的通道 MAE 为 0.0435/255，
99.9815% 的通道差不超过 1，最大差为 11；这是不同逻辑芯片上的 BF16 舍入
差异，瓦片分配、汇总和拼接通过数值对照。

## 当前限制

1. `_upsample_nearest_exact2d.out` 仍回退到兼容实现，是 VAE 的主要耗时点。
2. FHCCL 0.18.0 的多进程退出会触发 allocator 析构断言；短命容器在日志和
   barrier 完成后用 `os._exit(0)` 绕开退出期问题，不影响计算结果。
3. DiT 序列并行需要 FHCCL 增加 all-to-all，或实现并验证基于 all-gather 的
   注意力重排，当前未宣称支持。

## DiT 双芯 16+16 层模型并行

32 个 Transformer block 按前 16 层和后 16 层分到 `fant:0`、`fant:1`。
FANT PAL 0.18.0 在同一进程中直接执行 `fant:0 -> fant:1` 张量复制会触发
`palRbTreeGetFirst` 断言，因此阶段边界暂时通过 CPU 中转。每一步只在第 16
层边界和最终输出处传输一次。

### 512×512、2 步

| 指标 | 单芯 | 双芯 16+16 |
|---|---:|---:|
| DiT 时间 | 9.28 秒 | 9.73 秒 |
| 最大单芯峰值 | 33,106,365,440 bytes | 25,401,270,272 bytes |
| 第二芯峰值 | — | 7,640,736,768 bytes |
| latent 对照 | 基线 | MAE 0，最大误差 0 |

### 1024×1024、4 步

| 指标 | 单芯 | 双芯 16+16 |
|---|---:|---:|
| DiT 时间 | 163.33 秒 | 164.07 秒 |
| 最大单芯峰值 | 40,595,459,072 bytes | 32,889,156,608 bytes |
| 第二芯峰值 | — | 15,103,029,248 bytes |
| latent 对照 | 基线 | MAE 0，最大误差 0 |

双芯最高单芯显存下降约 19%，但单图延迟增加约 0.45%。这是模型并行的预期：
前后两段对同一张图仍串行执行。它适合解决单芯容量问题；若要提高单图速度，
仍需要序列并行或张量并行。当前 FHCCL 缺少 all-to-all，无法直接使用 Omni
的 Ulysses 路径。

## DiT TP2 张量并行

TP2 以两颗逻辑芯片各持有 16 个注意力头的方式切分所有 32 个 Transformer
block，同时按中间通道切分 SwiGLU。Q/K/V、gate 和 proj 使用列并行；注意力
输出和 MLP 输出使用行并行，并各执行一次 FHCCL all-reduce。各 rank 的
Transformer 参数量由 14,230,249,472 bytes 降至 7,250,927,616 bytes。

单 block 的实测显示序列越长，计算相对通信的比例越高：序列长度 1024 时为
1.64 倍加速，序列长度 4096 时为 1.85 倍加速。全模型短测结果如下。

| 分辨率 / 步数 | 单芯 DiT | TP2 DiT | 加速比 | TP2 每芯峰值 |
|---|---:|---:|---:|---:|
| 512×512 / 2 | 9.28 秒 | 5.51 秒 | 1.68× | 25,696,247,808 bytes |
| 1024×1024 / 4 | 163.33 秒 | 89.60 秒 | 1.82× | 29,495,120,896 bytes |

两颗芯片在两次测试中的最终 latent 完全一致。相对单芯基线，512 测试的
latent MAE 为 0.00921、最大误差为 0.203125；1024 测试的 MAE 为
0.00575、最大误差为 0.09375。所有值均为有限值，差异来自 BF16 分片计算和
all-reduce 的求和顺序。

按 1024×1024、4 步结果线性外推，40 步 TP2 DiT 约为 896 秒。加上已经实测
的双芯 VAE 143.50 秒和约 1 秒文本编码，完整出图约 1040 秒，即约 17.3 分钟。
这是短测外推，正式结论还需要一次 40 步完整成图。

## FANT Graph 状态

当前 FANT PyTorch 运行时具有 Graph 能力，已确认暴露 `HIPGraph`、
`torch.fant.graph`、`graph_pool_handle` 和 `make_graphed_callables`。文本模型
可以使用 Graph，是因为 decode 的形状固定，并且 KV cache 等缓冲区已经预分配。

Qwen-Image-2.1 当前 Diffusers 链仍走 Eager。它在第一步 prefill 和后续 cached
decode 间有不同执行路径；RoPE、mask 和 segment 仍含 CPU/Python 构造；KV cache
对象会动态修改。双芯 16+16 路径还经过 CPU 中转。这些状态不能直接套用文本
vLLM 的 Graph runner。

可行的移植路径是分别捕获固定形状的 prefill Graph 和 cached-decode Graph，
把 KV cache、RoPE、mask 和 segment 改成预分配的设备缓冲区，并只捕获 TP2 的
纯 DiT 区域。VAE 的 `_upsample_nearest_exact2d.out` 兼容回退应单独优化，不放入
首轮 Graph 捕获范围。

## 1024 DiT 单步瓶颈剖析

使用序列长度 4096 的真实 checkpoint 第 0 个 Block，在 TP2 两个 rank 上逐算子
同步计时。单 Block 为约 0.604 秒，与完整模型每步 22.4 秒除以 32 层得到的
0.700 秒量级一致。逐 Block 结果外推到完整一步：

| 部分 | 单 Block | 32 Block 外推 | 约占完整 22.4 秒 |
|---|---:|---:|---:|
| SDPA | 0.485–0.489 秒 | 15.5–15.7 秒 | 69%–70% |
| Q/K/V、Attention out 和 MLP GEMM | 约 0.0195 秒 | 约 0.62 秒 | 约 2.8% |
| LayerNorm、RMSNorm、残差及重排等 | 约 0.085–0.088 秒 | 约 2.7–2.8 秒 | 约 12%–13% |
| 两次 all-reduce 调用（含 rank 等待） | 0.0066–0.0147 秒 | 0.21–0.47 秒 | 0.9%–2.1% |
| RoPE、cache、segment、调度及剖析差额 | — | 约 3.0–3.4 秒 | 约 14%–15% |

独立 FHCCL 基准中，和 TP2 行并行输出相同大小的 32 MiB BF16 all-reduce 为
1.57–1.58 ms，约 21.2–21.3 GB/s 总线带宽。每层两次、32 层的纯传输时间约
0.10 秒/步，仅约 0.45%。因此 all-reduce 调用里的其余时间主要是两个 rank 到达
collective 的等待，芯片间通信不是当前主瓶颈。

当前容器没有 `flash_attn` 模块，`torch_fant` 和 `torch.ops.fant` 也未暴露通用
FlashAttention/SDPA 算子。Qwen-Image-2.1 的默认 processor 通过
`dispatch_attention_fn(..., backend=None)` 执行普通 SDPA；prefill 还会按 segment
拆成多次 attention 调用。证据指向注意力算子/后端路径不足，而不是 TP2 通信带宽
不足。没有可靠的硬件 DRAM 流量 counter，因此尚不能把 SDPA 内部进一步严谨拆成
计算单元瓶颈或显存带宽瓶颈。

## 算子优化短测

本轮只测试序列长度 4096 的真实 Block 和对应单算子，没有运行端到端图片生成。
由于测试期间设备 0、2、4 被另一容器的 vLLM 服务占用，算子测试显式绑定同一张
空闲物理卡的逻辑设备 6、7，没有停止或修改既有服务。

### Attention 候选

| 候选 | 结果 |
|---|---:|
| 普通 SDPA | 约 484.1–484.3 ms |
| 强制 `FLASH_ATTENTION` | 约 484.0–484.2 ms，与 eager 完全相同 |
| 强制 `EFFICIENT_ATTENTION` | 约 484.3–484.5 ms，与 eager 完全相同 |
| 显式 matmul-softmax-matmul | 约 569.0–569.5 ms，慢约 17.6% |
| eager/compiled flex attention | 拒绝 `fant` 设备，只支持 CUDA/CPU |
| 在线 Softmax，chunk 512 | 1024 序列约 76–85 ms，eager 约 34.7 ms |

所有 `SDPBackend` 选择在 FANT 上落到同一实现，没有获得融合 FlashAttention。
分块注意力数值通过，但 1024 序列已慢 2.2–2.4 倍，因此没有继续测试 4096。

### 厂商 iFormer 算子

`vllm_fant._custom_ops` 实际把厂商算子注册到 `torch.ops.iFormer`。符合
Qwen-Image 张量布局的两个算子通过测试：

| 算子 | eager | iFormer | 加速 | 数值差异 |
|---|---:|---:|---:|---:|
| Q/K RMSNorm（单个） | 约 11.69 ms | 约 0.615 ms | 约 19.0× | MAE 2.33e-8，max 0.015625 |
| Q/K RoPE（成对） | 约 24.69 ms | 约 0.62–0.64 ms | 约 38.5× | MAE 约 0.00115，max 0.03125 |
| SiLU×gate | 约 8.83 ms | 约 0.679 ms | 约 13.0× | 逐元素一致 |

SiLU×gate 虽然单算子明显加速，但它要求 gate/proj 拼接布局。把两个投影合并为一个
GEMM 后，完整 MLP 前半段仅从约 9.35 ms 降到约 9.21 ms，不到 2%，因此当前不改
MLP 权重布局。`torch.compile` 对 RMSNorm、LayerNorm+modulation 和 gated residual
均没有达到 5% 收益。

### 真实 TP2 Block 组合验证

将 iFormer RMSNorm 和 RoPE 同时代入真实 checkpoint Block 0：

| 配置 | 单 Block 时间 | 相对收益 |
|---|---:|---:|
| eager RMSNorm + eager RoPE | 0.67934 秒 | 基线 |
| iFormer RMSNorm + iFormer RoPE | 0.63190 秒 | 1.075×，耗时下降 6.98% |

优化 TP2 Block 相对 eager TP2 Block 的输出 MAE 为 0.00360、最大误差为 2，所有
值均有限。相对未分片参考，eager TP2 的最终 MAE 为 0.00394，优化 TP2 为
0.00412，最大误差均为 2。算子组合达到 5% 性能门，已接入 `tp2_full_dit.py`；
适配器在逻辑设备 6 上另行通过序列长度 256 冒烟测试：RMSNorm 与参考完全一致，
RoPE MAE 为 0.001149、最大误差为 0.015625，缓存重复调用逐元素一致。此次按约定
没有进行端到端验证，因此部署脚本默认不启用，需显式设置
`ENABLE_FANT_OPERATORS=1`。
