# 历史代码恢复

本目录保存从 Codex 开发记录恢复的代码，来源、原始哈希和发布版哈希见 `manifest.json`。现有工程副本已去重，统一放在仓库的 `projects/` 下。

- H3：`h3/history/` 中 5 个文件在恢复时与旧交接索引 SHA-256 完全一致。公开整理可能替换个人路径和内部地址，因此发布版哈希另列。
- Qwen-Image-2.1：`qwen-image-2.1/history/` 中 23 个文件名、25 个历史版本。`Lxxxxx` 是原始 JSONL 行号，版本顺序和源记录见清单。
- `h3/fragments/fantasy_full_dit_probe.py.partial.txt` 只有前 300 行，在 finally 中途结束，不能执行。
- `h3/missing-index.json` 列出仍未恢复的 31 个 H3 历史顶层 Python 文件。

H3 仍缺少 `fantasy_quant_compat.py`、`fantasy_layerwise_offload.py`、`fantasy_parameter_bridge.py` 等关键模块。旧记录指向服务器 `/data/infra/benchmarks/gr308-h3-20260905/` 的三个证据包：`h3-evidence-20260905.tgz`、`h3-loader-evidence-20260906.tgz`、`h3-runtime-evidence-20260906.tgz`。本次没有确认这些远端包仍然存在。

历史脚本包含失败尝试、替代方案和旧环境路径。尤其不要将多个 VAE 补丁无条件顺序执行；优先阅读 `projects/qwen-image21-omni-fant/` 的当前适配代码和报告。
