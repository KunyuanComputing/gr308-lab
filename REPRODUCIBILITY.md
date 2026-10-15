# 环境与复现说明

## 本地分析

`requirements-dev.txt` 提供 CPU 测试的基础依赖；`requirements-analysis.txt` 提供图表、PDF 和质量计算依赖。子项目锁文件记录原实验环境，根目录依赖清单不替代厂商容器版本。

```bash
python3 -m pip install -r requirements-analysis.txt
python3 tools/compare_wan22_quality.py \
  output/wan22_quality/frames/fp32_baseline.png \
  output/wan22_quality/wave5/s2_frame.png
```

报告记录的 S2 首帧 PSNR 约 39.3606 dB、亮度 SSIM 约 0.974300、MAE 约 1.66397。这里只复核图片一致性，不测视频时序质量或 GPU 耗时。

## 服务器实验

脚本依赖 Linux、Docker、GR308 驱动、`torch_fant`、FHCCL、模型权重和具体框架版本。保留 `/data/infra/...` 与 `/data/models/...` 历史路径是为了定位实验，而非声明这些目录已随仓库交付。

启动前需按实际机器核对镜像、模型路径、逻辑/物理卡映射、CPU 亲和性、端口、动态库挂载和输出目录。历史脚本包含容器创建、停止和设备状态检查，应先阅读对应脚本和报告。所有 `192.0.2.x` 地址是公开副本的示例地址。

Qwen 文本模型报告主要使用 vLLM 0.18.0、vllm_fant 1.1.0.20 和 BF16；Wan2.2 报告记录驱动 1.1.0.21。不同报告有各自冻结环境，不能把这些版本合并成一个统一环境。

## GRServe 构建依赖

完整第三方仓库未复制。固定版本见 [upstream-dependencies.json](upstream-dependencies.json)。首次构建前显式恢复依赖：

```bash
python3 tools/fetch_grserve_dependencies.py
cd projects/grserve-mac-mvp
uv sync --locked
cmake --preset mac-arm64-debug
cmake --build --preset mac-arm64-debug
```

依赖恢复脚本仅在手动执行时访问上游；不会自动覆盖已有目录。CMake 中的 Abseil 仍由固定 commit 的 FetchContent 获取。GRServe 需要 Python 3.11、C++20、CMake 和相应构建工具；本次整理未重新进行完整 C++ 构建验收。

`models/mini-qwen38-hybrid/` 是约 32 KiB 的合成测试夹具，已包含；它不是真实 Qwen 权重。

## 图像模型依赖

Qwen-Image 的 `reference-existing/` 是历史打补丁的 Diffusers 文件，使用上游包的相对导入，不是独立 Python 包。完整 vLLM-Omni 仓库不随包分发，原 vendor commit 记录在依赖清单中。服务器的 Diffusers、torch_fant 与权重版本需按历史脚本恢复。

## 历史证据

报告与图表、CSV/JSON 摘要已经归档；并非每次实验的原始 `benchmark.json`、`launch.json`、`perf.json`、镜像和模型都在本地。部分旧 Markdown 链接与绝对路径指向未恢复的服务器证据，不代表仓库内可访问文件。

公开整理替换了内网地址和个人目录。历史 `SHA256SUMS`、交接清单及恢复记录里的原哈希用于识别原始文件；发布版的完整性以根目录 `CHECKSUMS.sha256` 为准。

## 报告再生成

报告生成器位于 `tools/reporting/`，保留两级目录结构以兼容其根目录定位。部分历史生成器包含原机器路径、macOS 中文字体或缺失的远端原始数据，未保证在任意系统一键再生成。已生成的 PDF、Markdown、CSV/JSON 和图表可直接阅读。
