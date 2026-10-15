# MiniMax H3 / SGLang / GR308 原生 W4A8 门禁报告

## 结论

本次 H3 独立验证在“原生 W4A8 算子”门禁处停止，未启动正式 H3 服务，也未产生视频性能数据。

SGLang diffusion 平台已经成功接入 Fantasy GR308，基础 FP16/BF16 算子可在 `fant:0` 正确执行；但本机的 Comfy Kitchen 没有 Fantasy W4A8 后端，H3 的 ConvRot W4A8 线性层自动选择 `eager` 实现，并通过 `aten::_int_mm` 的 PrivateUse1 通用 fallback 执行。该路径数值正确，但不能视为 GR308 原生 W4A8 加速。

## 已验证项目

- SGLang diffusion 当前平台：`FantasyDiffusionPlatform`。
- PrivateUse1 后端名称：`fant`。
- FP16/BF16 matmul、Torch SDPA、Conv3D 共六项均通过。
- 六项小算子相对 L2 误差范围：约 `0.0002` 至 `0.0017`。
- 实际 H3 权重：`blocks.0.attn.out_proj`。
- W4A8 输出设备：`fant:0`。
- W4A8 输出有限且相对 CPU 参考误差为 `0.0`。
- 单次代表性投影耗时：`0.4858196899` 秒；该数字只用于识别 fallback，不是性能成绩。

## 阻塞证据

- Comfy Kitchen 选择的后端：`eager`。
- 实际实现：`comfy_kitchen.backends.eager.w4a8_int8.w4a8_int8_linear`。
- `aten::_int_mm` 的 `PrivateUse1` 专用 kernel：不存在。
- `aten::_int_mm` 的 Fantasy 条目来自 `torch_fant/csrc/aten/ops/Copy.cpp:422`，类型为 backend fallback。
- `comfy_kitchen::int8_linear` 和 `comfy_kitchen::dequantize_int8_convrot_weight` 同样没有 PrivateUse1 专用 kernel。
- Comfy Kitchen 后端状态：CUDA 扩展不存在、HIP runtime 不可用、Triton 认为 CUDA/XPU 均不可用，只有 eager 可用。
- `iFormer.ops.gptq_gemm` 虽然存在，但 GPTQ 权重布局和当前 H3 ConvRot W4A8 序列化布局不是同一接口；没有格式转换与误差验证前不能替代当前算子。

因此原生门禁结果为：

```text
native_gate_passed = false
gate_reason = no native Fantasy W4A8/INT8 dispatch is proven
probe exit code = 2
```

## 解锁条件

继续正式 H3 服务和性能测试至少需要满足以下一种条件：

1. 厂商提供 `aten::_int_mm` 的 GR308 PrivateUse1 原生 INT8 GEMM kernel；或
2. 提供 Comfy Kitchen `w4a8_int8_linear` 的 Fantasy 后端，兼容当前 ConvRot W4A8 的 packed INT4、group scale、channel scale 和 codebook；或
3. 改用与 GR308 已有 `iFormer.ops.gptq_gemm` 明确兼容的 H3 权重格式，并完成转换、逐层误差及完整 DiT 正确性验证。

在上述任一条件满足前，继续运行完整 H3 只会测到兼容 fallback 的速度，不能回答 GR308 上 H3 的真实性能。

## 环境恢复

- 门禁容器和调度审计容器均使用 `--rm`，验证结束后无遗留容器。
- 8 个逻辑 GPU 最终均为 `540 MiB / 57344 MiB`，利用率 `0%`。
- Qwen baseline 和 SGLang upstream 均未修改。

## 证据索引

- `probe.json`：结构化门禁结果、后端选择、数值和耗时。
- `dispatch-audit.json`：三个关键算子的完整调度表。
- `probe.log`：容器标准输出与错误输出。
- `checkpoint.sha256`：H3 FL2VA W4A8 权重校验值。
- `gpu-before.csv`、`gpu-after.csv`、`gpu-final.csv`：门禁前后 GPU 状态。
- `SHA256SUMS`：服务器端证据文件校验清单。
