# 风华 GR308 Wan2.2 视频生成性能测试与调优报告

日期：2026-09-18  
模型：Wan2.2-TI2V-5B-Diffusers  
推理框架：SGLang Diffusers  
硬件：8 张 GR308 物理卡（16 个逻辑设备，单物理卡 56 GiB）  
驱动：1.1.0.21

## 一、结论摘要

1. 当前短测最优可用方案为 **S2 混合精度**：前三个 VAE up block 和最终 up block 的最后两个 residual unit 使用 FP16，其余高风险解码路径、归一化、输出头保持 FP32。相对 FP32-effective 基线，端到端耗时从 1190.20 秒降至 672.33 秒，降低 43.5%；VAE decode 从 1144.41 秒降至 630.31 秒，降低 44.9%，对应 1.82 倍解码加速。
2. S2 在固定 prompt、seed 和首帧下保持较高参考一致性：PSNR 39.36 dB、luma SSIM 0.9743、RGB SSIM 0.9617、MAE 1.66/255。该结果支持继续验证，但样本量仅为一帧，不能证明视频级质量等价。
3. 全局 FP16 虽将端到端耗时降低约 68.3%，但产生严重条带、块状断裂、颜色漂移和结构损坏；PSNR 仅 14.25 dB，必须拒绝。
4. 当前性能瓶颈高度集中在 VAE decode。FP32 基线中 VAE decode 占端到端耗时约 96.2%；文本编码和 3 步去噪合计不足 4%。
5. S2 尚未通过 1280×704、121 帧、20 步完整测试。TP2 路径出现 DMA/固件 semaphore 超时；SP2/SP4 在 denoise step 0 出现“报告仍有约 23 GiB 空闲却无法申请 160/80 MiB”的 HIP OOM；SP8 ring 在加载阶段遇到 SGLang native fallback 与 Fantasy attention backend 能力限制。
6. 因此当前结论是：**S2 是短测候选，而非生产验证完成的部署方案。**在完整负载打通前，不应将短测倍率外推为 121 帧视频性能。

## 二、测试口径与证据等级

### 2.1 短测口径

- 分辨率：1280×704；
- CLI 请求帧数：1；SGLang 在双卡条件下自动调整为 5 帧；
- denoise steps：3；
- scheduler：UniPC；
- guidance scale：5.0；
- seed：42；
- 拓扑：TP2，2 张 GR308 物理卡；
- 固定 checkpoint、prompt、negative prompt 和输出路径；
- 主要指标：完整 pipeline wall time、各阶段 wall time、峰值显存与首帧质量。

短测用于参数筛选和定位 VAE 热点，不代表目标生产视频规格。

### 2.2 完整目标口径

- 模型：纯文本 Wan2.2-TI2V-5B；
- 分辨率：1280×704；
- 帧数：121；
- denoise steps：20；
- shift=5.0，guidance=5.0，UniPC；
- 固定 prompt 与 seed。

截至本报告生成时，该口径没有成功的 GR308 E2E 样本，因此报告不提供 GR308 与 RTX 5090 的速度倍率。

### 2.3 证据等级

- **有效性能结果**：生成成功、存在 `perf.json`，各阶段耗时完整；
- **质量候选**：有效性能结果，并完成固定首帧的全参考指标和人工检查；
- **失败诊断**：运行未形成 `perf.json`，只用于识别软件栈/显存/固件边界；
- **外部参考**：用于定义对照方向，不与不同帧数、步数或拓扑的结果直接相除。

## 三、FP32 基线与瓶颈

FP32-effective 基线的总耗时为 1190.20 秒，其中：文本编码 20.64 秒、3 步去噪 21.90 秒、VAE decode 1144.41 秒。VAE decode 占总耗时约 96.2%，是绝对主瓶颈。

四次重复 TP2 基线总耗时均值为 1192.43 秒，标准差约 2.00 秒；VAE decode 均值为 1146.46 秒，标准差约 0.85 秒。重复波动远小于混合精度带来的差异，短测性能方向具有较好的确定性。

Encoder fold、VAE 常驻、重复 upsample 等拓扑/驻留尝试只带来不足 1% 的变化，无法改变 VAE decode 主导的结构。

## 四、混合精度优化矩阵

| 方案 | 精度边界 | 总耗时/s | VAE decode/s | Decode 降幅 | 峰值分配/MiB | 结论 |
|---|---|---:|---:|---:|---:|---|
| FP32 baseline | FP32-effective | 1190.20 | 1144.41 | - | 25178 | 质量参考 |
| Global FP16 | 全部 VAE decode FP16 | 377.14 | 333.47 | 70.9% | 27595 | 质量不合格 |
| M0 | 仅早期路径 FP16 | 1184.34 | 1138.27 | 0.5% | 24667 | 收益很小 |
| M1 | 早期路径 + 1 个 up block | 1151.72 | 1106.83 | 3.3% | 24312 | 有效但有限 |
| M2 | 早期路径 + 2 个 up block | 1029.46 | 987.54 | 13.7% | 24069 | 明显改善 |
| M3 | 早期路径 + 3 个 up block | 756.04 | 712.45 | 37.7% | 22195 | 质量候选 |
| S1 | M3 + 最后 1 个 residual FP16 | 710.02 | 668.04 | 41.6% | 24410 | 可用短测 |
| S2 | M3 + 最后 2 个 residual FP16 | 672.33 | 630.31 | 44.9% | 24403 | 当前短测最优 |

主要规律：FP16 收益不是均匀分布的。M0/M1 几乎没有改变总耗时；从 M2 开始收益扩大，M3 和最终高分辨率 residual unit 才是主要加速来源。与此同时，继续把更早的高分辨率 residual 切到 FP16，会触发 5.80 GiB Conv3D 工作区申请失败。

## 五、生成质量验证

### 5.1 全局 FP16

全局 FP16 的三次成功输出具有相同 SHA-256，说明损坏具有确定性，而不是偶发噪声。首帧出现明显的水平条带、块状不连续、颜色漂移、对比度压缩和场景结构损坏。

| 指标 | Global FP16 | M3 | S2 | 方向 |
|---|---:|---:|---:|---|
| PSNR/dB | 14.254 | 39.376 | 39.361 | 越高越好 |
| Luma SSIM | 0.7019 | 0.9745 | 0.9743 | 越高越好 |
| RGB SSIM | 0.5996 | 0.9624 | 0.9617 | 越高越好 |
| MAE/255 | 38.50 | 1.66 | 1.66 | 越低越好 |
| P95 绝对误差 | 103 | 6 | 6 | 越低越好 |

### 5.2 M3 与 S2

M3 和 S2 的首帧与 FP32 工程参考高度接近，未观察到全局 FP16 的条带和颜色崩坏。两者的 PSNR、SSIM 和 MAE 几乎相同，说明将最终两个低风险 residual unit 继续下沉至 FP16，在该样本上没有造成明显新增损伤。

质量结论仍有严格边界：当前只有一个 prompt、一个 seed 和一个对齐首帧；没有检验 121 帧中的闪烁、运动一致性、时序漂移和跨场景稳定性。

## 六、完整 121 帧/20 步测试状态

| 路径 | 阶段 | 失败表现 | 判断 |
|---|---|---|---|
| S2 / TP2 | denoise | DMA 与 firmware semaphore 30 s timeout | 固件同步/长序列执行边界 |
| S2 / SP2 | denoise step 0 | 申请 160 MiB 失败，报告约 23.34 GiB 空闲 | 非普通容量耗尽，疑似 allocator/VA 问题 |
| S2 / SP4 | denoise step 0 | 申请 80 MiB 失败，报告约 23.15 GiB 空闲 | 扩大并行度未解决地址分配问题 |
| SP8 / ring | 模型加载 | native Diffusers fallback 不支持 sp=8/ring=8；Fantasy adapter 仅允许 Torch SDPA | 软件能力缺口 |

此外，SP2/SP4 的 sequence shard、offload，以及 SP8 的 allocator 配置等尝试均未形成有效 `perf.json`，不能计入性能结果。

## 七、高分辨率 Conv3D 深挖结果

为了继续提高 S2，测试了以下路径：

1. 只在目标 residual unit 的 Conv3D 内启用 FP16 autocast；
2. 预先把目标 Conv3D 权重转换为 FP16，避免 forward 临时复制权重；
3. 从 2 卡扩展为 4 卡，缩小空间分片；
4. 将整个倒数第 3 个高分辨率 residual unit 切换为 FP16。

四条路径均在同一高分辨率阶段失败。最完整错误显示：单次尝试申请约 5.80 GiB，GPU 报告仍有约 28.92 GiB 空闲，却返回 HIP OOM。四卡版本同样失败，并将短测去噪从约 57 秒增加到 86.88 秒，表明 TP4 通信成本抵消了扩卡价值。

当前软件栈下，S2 是可运行的最激进精度边界。继续通过参数扩大 FP16 范围的成功概率很低，下一阶段需要算子级 tiled Conv3D、低 workspace 算法或厂商后端修复。

## 八、与 RTX 5090 参考结果的关系

已有外部记录显示：8×RTX 5090、1280×704、121 帧、3 步时，U8+CFG1 为 57.206 秒，U4+CFG2 为 37.222 秒，后者快 1.5369 倍。

该数据可以用于确定通信拓扑优化方向，但不能与当前 GR308 短测相除：GR308 短测的有效帧数为 5，且完整 121 帧负载尚未跑通。当前最重要的差距不是百分之多少，而是 GR308 尚未形成同口径有效结果。

## 九、推荐配置与启动要点

当前仅建议将 S2 用于短测与后续回归：

```bash
export SGLANG_WAN_VAE_FP16_UP_BLOCKS=3
export SGLANG_WAN_VAE_FP16_FINAL_RESNETS=-2

python3 -c 'from sglang.cli.main import main; main()' generate \
  --model-path /model \
  --model-id Wan2.2-TI2V-5B-Diffusers \
  --num-gpus 2 --tp-size 2 \
  --performance-mode manual \
  --attention-backend torch_sdpa \
  --vae-cpu-offload=true \
  --vae-precision fp32 \
  --vae-decode-precision fp16 \
  --vae-config.use-parallel-decode true \
  --vae-config.parallel-decode-mode auto \
  --width 1280 --height 704 \
  --num-frames 1 --num-inference-steps 3 \
  --guidance-scale 5 --seed 42
```

该配置依赖定制 Wan VAE 混合精度实现，不能只依赖原生 `--vae-decode-precision fp16`；后者会进入全局 FP16 并造成质量损坏。

## 十、部署判断与后续路线

### 10.1 当前部署判断

- 不建议在当前软件版本上将 Wan2.2 121 帧任务直接投入生产；
- 可保留 S2 作为算子优化和固件回归的基准候选；
- 不应使用全局 FP16；
- 不应把短测 1.77× E2E 加速外推到长视频；
- 多卡数量增加并未自动改善性能或稳定性，必须同时解决后端的分布式执行与显存分配问题。

### 10.2 优先级

1. 构造最小化 FP16 Conv3D 复现，向厂商确认 5.80 GiB workspace 的算法来源，并增加 workspace limit/低内存算法选择；
2. 实现带 halo 的空间 tiled Conv3D，先在单 residual unit 上验证数值一致性和显存峰值；
3. 使用 `hipMemGetInfo`、同步点和 FHCCL `ALLOC/GRAPH/TUNING` 日志区分 PyTorch allocator、PAL VA manager 与通信分配；
4. 修复 121 帧 denoise 的 firmware semaphore timeout；
5. 完整负载跑通后，再进行 U8+CFG1、U4+CFG2、TP/SP/CFG 的正式矩阵；
6. 至少扩展到多 prompt、多 seed、121 帧视频，并加入时序一致性与人工盲检。

## 十一、限制

- 所有有效 GR308 性能结果来自短测，实际为 5 帧、3 步；
- 质量比较只有一个首帧样本，不支持统计推断；
- PSNR/SSIM 衡量的是相对 FP32 工程参考的一致性，而非真实视频的绝对审美质量；
- 失败路径可能同时包含 allocator、PAL、固件和 SGLang 分布式适配问题，当前日志不足以唯一归因；
- 121 帧/20 步尚未成功，因此本报告不能给出同口径 GR308/RTX 5090 性能比。

