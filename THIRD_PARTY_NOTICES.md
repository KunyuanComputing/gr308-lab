# 第三方代码与许可说明

本整理包没有为所有文件新增统一许可证。原有版权头、SPDX 标识和许可证引用均保留；未声明许可的自有代码需由权利人另行决定开放许可。

| 来源 | 本包位置或用途 | 说明 |
|---|---|---|
| SGLang | `sglang-fant-adaptation/remote-src/`、部分 `tools/fantasy_*.py` | 原框架代码及本地适配；保留各文件声明 |
| FastVideo / Wan Team / Hugging Face | `tools/fantasy_wanvae*.py` 等 | 文件头声明 Apache-2.0 与上游归属 |
| Diffusers / Qwen-Image Team | `projects/qwen-image21-omni-fant/reference-existing/` | 含本地兼容修改的上游模型文件；保留 Apache-2.0 声明 |
| vLLM-Omni | 图像推理参考工程 | 完整上游仓库未复制；版本见依赖清单 |
| GoogleTest | GRServe 构建依赖 | 完整源码未复制；固定 commit，附原许可证副本 |
| Protocol Buffers | GRServe 构建依赖与生成代码 | 完整源码未复制；固定 commit，附原许可证副本 |
| Abseil | GRServe 间接构建依赖 | CMake 固定 commit |

许可文本见 `licenses/`。厂商运行时、模型权重、模型代码和任何未标明许可的服务器片段，不因本整理行为获得额外授权。此文件用于记录来源，不替代各上游许可证。

## 未随包分发的参考资料

按用户选择，以下厂商手册和外部团队原始报告仅列名称：

- `FHCCL编程指南_v1.0.0.pdf`
- `FTUCA编程指南.pdf`
- `GR308S MAX_风华三号_V1p0_CN_PyTorch使用指南_20250908.pdf`
- `HIP_Runtime_API_Reference_v1.0.pdf`
- `风华 GPU 性能并发测试报告.pdf`
- `风华GPU测试.docx`

本包包含我们依据这些资料生成的分析和测试报告，并保留其证据限制；没有将外部团队测量结果标记为本次实测。
