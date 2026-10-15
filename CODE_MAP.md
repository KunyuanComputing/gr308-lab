# 代码导航

| 模块 | 入口 | 内容 |
|---|---|---|
| 早期基线 | [run_fant_graph_baseline.sh](run_fant_graph_baseline.sh)、[run_sglang_post.sh](run_sglang_post.sh) | Graph 基线、SGLang 运行与统计 |
| Qwen 多卡调优 | [scripts](projects/gr308-three-card-mtp1-graph32/scripts/) | 正式吞吐测试、35B MTP4 矩阵、80B 短测、TTFT 调优 |
| MoE/EP 实验 | [moe-campaign-20260917](experiments/moe-campaign-20260917/) | 后期 MoE 验证、EP4 实验和执行合同 |
| C32 补测 | [remote-bench](experiments/remote-bench/) | 9B/80B C32 测试及结果摘要 |
| SGLang 插件 | [sglang-fant-platform](sglang-fant-platform/) | 平台接口、启动器；平台测试需要厂商运行时 |
| SGLang 适配 | [sglang-fant-adaptation](sglang-fant-adaptation/) | FLA/GDN、Qwen 兼容源码、补丁和探测 |
| Wan2.2 | [fantasy_wanvae_mixed.py](tools/fantasy_wanvae_mixed.py)、[run_wan22_matrix_job.sh](tools/run_wan22_matrix_job.sh) | VAE 混合精度、分布式适配和实验启动 |
| Wan2.2 质量 | [compare_wan22_quality.py](tools/compare_wan22_quality.py) | PSNR、SSIM、MAE；可用本地图片复核 |
| Qwen-Image-2.1 | [README](projects/qwen-image21-omni-fant/README.md)、[tp2_full_dit.py](projects/qwen-image21-omni-fant/scripts/tp2_full_dit.py) | TP2、分阶段执行、分块 VAE、算子替换与基准 |
| SDPA 原型 | [native/fant_sdpa](projects/qwen-image21-omni-fant/native/fant_sdpa/) | C++/HIP 原型；没有完成端到端性能验收 |
| H3 | [h3-sglang-fant](projects/h3-sglang-gr308/h3-sglang-fant/) | 平台注册、原生 W4A8 路径检查及单元测试 |
| H3 历史恢复 | [history](archive/recovered/h3/history/) | 恢复的平台注册、算子探测和测试 |
| 图像推理历史恢复 | [history](archive/recovered/qwen-image-2.1/history/) | 环境、下载、RoPE/VAE 补丁及 CPU/GPU 解码的历史版本 |
| GRServe | [README](projects/grserve-mac-mvp/README.md)、[src](projects/grserve-mac-mvp/src/) | C++20 Engine、C11 HAL、RPC、Python API；本地原型 |
| 报告生成 | [tools/reporting](tools/reporting/)、[build_three_baseline_pdf.py](build_three_baseline_pdf.py) | PDF 和图表生成；不产生 GPU 测量结果 |

## 版本选择

- Qwen 9B/30B 正式配置位于 `gr308_final_high_throughput.py`。
- 35B MTP4/batch4096 候选对应 `gr308_35b_mtp4_short_matrix.py` 的 D 组；不要误用正式脚本中较早的 35B MTP3 配置。
- 80B 的早期调优脚本覆盖 MTP0/1；MTP3 的 C32 配置见 `experiments/remote-bench/gr308-9b-80b-c32-short-20260917.py`。
- Wan2.2 S2 需要混合精度 VAE 实现，设置 `SGLANG_WAN_VAE_FP16_UP_BLOCKS=3`、`SGLANG_WAN_VAE_FP16_FINAL_RESNETS=-2`；单独设置原生 FP16 参数不等价于 S2。
- `archive/recovered/` 中的补丁与解码尝试按历史版本保留，包含失败方案，不能全部顺序执行。

子项目的 `tests/` 分别保留；根目录验证器只运行不需要 GR308 的明确子集。`sglang-fant-platform/tests/test_platform.py`、H3 `test_platform.py`、Triton smoke 和远端探测需要硬件环境，未纳入 CPU 检查。
