# 整理包校验

校验日期：2026-10-15。环境：macOS arm64、Python 3.11.15、pytest 8.3.5。

## 本地检查结果

- 179 个 Python 文件通过语法解析；19 个 Shell 脚本通过 `bash -n`。
- 6 组 pytest：共 69 项通过、1 项跳过；另有 1 项独立稳定性脚本合同检查通过。
- 跳过项：Qwen-Image 的 `test_fant_sdpa_bench.py` 中需要 PyTorch 的测试，当前验证环境没有安装 `torch`。
- 环境中的 pytest-asyncio 插件产生默认事件循环配置弃用提示，没有导致测试失败。
- 501 个复制文件的来源 SHA-256 与原文件一致，发布版哈希与来源清单一致。
- 11 个 PDF 共 44 页：页数均与原件一致；排除地址/个人目录后，逐页非空白字符及其数量一致。处理个人目录的报告页已渲染检查。
- 顶层导航文档的本地链接有效。
- 全包文件校验值见 `CHECKSUMS.sha256`；ZIP 内容与目录逐文件比对。

| 测试组 | 结果 |
|---|---|
| 基线聚合与脚本 | 13 通过 |
| 稳定性脚本合同 | 1 项独立脚本检查通过 |
| Wan 混合精度 | 1 通过 |
| Qwen 文本模型矩阵 | 24 通过 |
| Qwen-Image | 22 通过，1 跳过 |
| H3 注册与 W4A8 合同 | 6 通过 |
| H3 恢复的能力探测 | 3 通过 |

原始检查输出见 [CPU 结果](docs/package/cpu-validation.json)，PDF 检查见 [PDF 结果](docs/package/pdf-validation.json)。

## 复核命令

```bash
python3 tools/verify_package.py
python3 tools/verify_package.py --cpu-tests
```

第一个命令检查语法、顶层导航、文件大小与已列文件 SHA-256。第二个额外运行明确列出的 CPU 测试；需要先安装 `requirements-dev.txt`。

## 未验证范围

没有重新运行 GPU 推理、性能基准、完整 C++ 构建、FANT 平台集成测试或 HIP 编译。CPU 检查通过不代表报告性能已重新复现，也不代表全部历史片段可独立运行。H3 缺失模块和不完整片段见恢复目录说明。
