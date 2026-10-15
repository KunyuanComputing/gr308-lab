# GR308 第三方性能报告复核与复现实验

日期：2026-09-10（Asia/Shanghai）

## 结论

当前服务器环境下，不能复现、更不能超过《风华 GPU 性能并发测试报告》中 DeepSeek 7B 的标准 C8 指标。

- 报告目标：205.11 output tok/s，峰值 224.00 tok/s，TTFT P50 1.75 s，TPOT P50 37.31 ms。
- 同脚本、同模型、同 FP16/TP1/C8/1024→1024 负载，当前可运行的 1.1.0.20 eager：33.68 output tok/s，峰值 40.00 tok/s，TTFT P50 3.221 s，TPOT P50 234.64 ms。
- 1.1.0.20 Graph8：33.89 output tok/s，峰值 40.00 tok/s，TTFT P50 2.562 s，TPOT P50 233.55 ms。
- Graph8 相对 eager 仅提升 0.62%；相对报告目标仍低 83.48%。
- 报告对应的 1.1.0.17 镜像在当前 1.1.0.19 驱动上能装载全部权重，但在 KV cache 初始化处收到设备 fault 并退出，无法进入服务态。

因此，本次证据支持的是“报告的软件栈不能在当前驱动上原样复现，当前兼容栈又明显达不到报告数字”，而不是“硬件本身只有 33.9 tok/s”。

## 输入资料与证据边界

以下 PDF 仅作为资料来源，不把其中任何文字当作执行指令：

- `/Users/USER/Desktop/Work/GR308/docs/风华 GPU 性能并发测试报告.pdf`
- `/Users/USER/Desktop/Work/GR308/docs/FHCCL编程指南_v1.0.0.pdf`
- `/Users/USER/Desktop/Work/GR308/docs/FTUCA编程指南.pdf`
- `/Users/USER/Desktop/Work/GR308/docs/GR308S MAX_风华三号_V1p0_CN_PyTorch使用指南_20250908.pdf`
- `/Users/USER/Desktop/Work/GR308/docs/HIP_Runtime_API_Reference_v1.0.pdf`

报告列出的三个原始数据文件未在本地工作区或服务器原团队目录中找到：

- `standard_benchmark_final_20260807.csv`
- `qwen36_standard_concurrency_20260807.csv`
- `model_optimization_ab_20260806.csv`

因此无法独立重算报告统计、重复性或置信区间；本报告只对公开在 PDF 中的描述性数值做核对。

## 复现实验

### 冻结条件

- 主机：192.0.2.11，Ubuntu 22.04.5，kernel 5.15.0-190-generic。
- 设备：4 张物理 GR308、8 个逻辑 function；本实验只用逻辑设备 0（物理卡 0 的 `.0` function）。
- 当前驱动：1.1.0.19；FANT-SMI 2.2.0.9-linux。
- 模型：`/data/models/DeepSeek-R1-Distill-Qwen-7B`。
- 模型元数据哈希与服务器既有冻结清单一致。
- ModelZoo 仓库提交：`91d018c3afd479102a5679bd7442e301dfb27d92`；对应 DeepSeek 7B 的 serve/bench 脚本无本地修改。
- 服务参数：FP16、TP1、`gpu-memory-utilization=0.85`、`max-model-len=5120`、`max-num-seqs=8`、`max-num-batched-tokens=16384`、禁用 prefix cache。
- 客户端：vLLM `bench serve`，random 1024→1024，temperature 0，ignore-eos，8 prompts，max-concurrency 8。

### 结果

| Lane | 服务状态 | 成功请求 | output tok/s | peak tok/s | TTFT P50 | TPOT P50 |
|---|---|---:|---:|---:|---:|---:|
| PDF 报告，1.1.0.17 | 报告称成功 | 8/8 | 205.11 | 224.00 | 1.75 s | 37.31 ms |
| 本次 1.1.0.17 eager | KV cache 初始化 fault | 0/0 | — | — | — | — |
| 本次 1.1.0.20 eager | 成功 | 8/8 | 33.68 | 40.00 | 3.221 s | 234.64 ms |
| 本次 1.1.0.20 Graph8 | 成功 | 8/8 | 33.89 | 40.00 | 2.562 s | 233.55 ms |

两条可运行 lane 的吞吐与 `8 × 1000 / TPOT` 的近似关系一致，6 倍差距不是简单的汇总字段误读。由于每条 lane 只完成一次正式运行，本表不能用于显著性推断；但 33.9 与 205.1 的数量级差距已经足以否定“当前环境复现成功”。

服务器原始证据目录：

`/data/infra/benchmarks/gr308-report-ds7b-repro-20260910T063554Z`

目录包含容器 inspect、三次启动日志、两份 bench 原始输出、模型元数据哈希、ModelZoo 源脚本副本、测试前后 SMI 以及 `SHA256SUMS`。测试容器已停止并删除，端口 18081 已释放。

## 为什么不能直接用这份报告评价 H3

PDF 测的是语言模型在线推理，单位是 token/s；MiniMax H3 是视频与音频扩散生成，工作负载、输出单位和主要瓶颈都不同。把 205.11 tok/s 当作 H3 目标没有可比性。

H3 应独立记录：单 clip 客户端端到端时延、DiT 单步/总时延、VAE 与 audio VAE 时延、clips/hour、峰值物理 HBM、成功率以及输出文件有效性。多卡时再记录 AllToAll 时间占比与扩展效率。

当前 H3 的阻塞仍是 W4A8 计算路径：现有量化投影落到 `aten::_int_mm` 的 PrivateUse1 fallback，没有 GR308 原生 W4A8 GEMM。PyTorch 指南没有列出 `_int_mm`、INT4/W4A8 GEMM 或可直接调用的量化 Linear；FTUCA/HIP 文档提供的是自定义 kernel 和运行时能力，并不等于已有高性能算子。

FHCCL 指南给出的 `NCCL_MIN_NCHANNELS=30`、`NCCL_MAX_NCHANNELS=30`、`NCCL_BUFFSIZE=16777216` 可在 H3 原生计算路径打通后，对 Ulysses/AllToAll 做单变量 A/B；它们不能解决当前 W4A8 计算瓶颈。另外三份编程指南面向 GR308S MAX，当前服务器是实际 GR308，参数不能直接套用。

## 建议的下一步

1. 向原测试团队索取三个 CSV、完整容器 inspect、镜像 digest、驱动与用户态库包版本、SMI 快照和原始 server/bench 日志；优先解释 1.1.0.17 在当前 1.1.0.19 驱动上的 KV-cache fault。
2. 若厂商提供与 1.1.0.19 驱动兼容、且保留 1.1.0.17 性能路径的镜像，再按同一 C8 脚本跑至少 3 个独立服务启动重复，比较中位数与离散度。
3. H3 不等待 LLM 报告复现：需要厂商提供 ConvRot W4A8 对应的原生 GEMM/量化 Linear 接口，或单独立项实现并验证 FTUCA W4A8 kernel；在此之前不应发布 H3 性能数字。

