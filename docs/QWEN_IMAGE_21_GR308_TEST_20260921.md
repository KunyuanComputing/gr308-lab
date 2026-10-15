# Qwen-Image-2.1 在 GR308 上的实测记录

- 日期：2026-09-21
- 主机：`192.0.2.11:22`
- 设备：Fantasy III / GR308，主机共 4 张双芯卡（8 个逻辑设备）
- 本次测试设备：单逻辑芯片 `fant:0`，可见显存 55.48 GiB
- 容器：`fantasy/vllm-ftuca:0.18.0-ftuca20260920174921-torch2.7.1-py310-ubuntu22.04-amd64`
- PyTorch：`2.7.1+cpu` + `torch_fant 1.1.0+c263bd4`
- 模型：`Qwen/Qwen-Image-2.1`，ModelScope 权重约 30.9 GiB
- Diffusers：`0.41.0.dev0`，固定提交 `9f1246971270c84dcbe71233edb7a519596a5d02`

## 结论

Qwen-Image-2.1 的 BF16 扩散主干可以在单个 GR308 逻辑芯片上运行。原始 Diffusers 实现不能直接完成端到端推理，需要兼容补丁和分阶段卸载：扩散阶段保存潜变量，释放 Transformer 后再单独加载 VAE 解码。

## 已验证结果

### 40 步、1024×1024、BF16

- 扩散：40/40 成功，容器 exit 0
- 扩散时间：1622.44 秒（27 分 2 秒）
- 平均单步：约 40.56 秒
- 扩散峰值显存：40,595,459,072 bytes（约 37.81 GiB）
- VAE 独立解码：成功，容器 exit 0
- VAE 解码时间：1241.47 秒（20 分 41 秒）
- VAE 峰值显存：32,374,563,328 bytes（约 30.15 GiB）
- 两阶段合计计算时间：2863.92 秒（约 47 分 44 秒）
- 输出：`qwen-image-2.1-bfloat16-1024x1024-40steps.png`
- SHA-256：`fb636e4cd60ffcbde6d0563732b890ddfe81dfab005f8feead397225463453fb`

40 步图像主体、材质、背景均清晰，“风华三号”四个中文字完整且正确。

### 4 步、1024×1024、BF16

- 扩散：4/4 成功
- 单步时间：约 40.6 秒
- 扩散总时间：约 162.6 秒
- VAE 单独解码：成功
- VAE 解码时间：1241.66 秒
- VAE 峰值显存：32,374,563,328 bytes（约 30.15 GiB）
- 输出：`qwen-image-2.1-bfloat16-1024x1024-4steps.png`
- SHA-256：`4051d658f40e25b79f03c471ffbaf6c999a96c994456b8f1cfd267778ac0ac21`

4 步图像已形成红色茶壶主体，但细节和中文文字尚未收敛。这是低步数冒烟测试的预期表现。

## 必需兼容修改

1. QwenImage21 RoPE 的复数查表和复数旋转改为 CPU 查表后生成实数 cos/sin，并在 FANT 上使用实数旋转。
2. 修正缓存解码时 tuple 形式 rotary embedding 的切片。
3. 将 VAE `DupUp3D` 的 `repeat_interleave` 改为 CPU 构造索引后在 FANT 上 `index_select`。
4. 将原实现的 8 维 `permute` 拆成不超过 5 维的空间和时间重排。
5. FANT 不支持原生 `pixel_shuffle`；使用切片和 `copy_` 完成等价重排。

## 已发现限制

- FANT 后端对 `complex64`、8 维 `permute`、VAE `repeat_interleave` 和 `pixel_shuffle` 支持不完整。
- `aten::_upsample_nearest_exact2d.out` 缺少 FANT 原生实现，会走慢速路径。
- 将 Transformer 和 VAE 同时常驻单逻辑芯片时，扩散结束后约 47.45 GiB 已分配；VAE 后段一次额外 5.06 GiB 分配会触发 OOM。
- 分阶段运行解决显存问题，但当前 VAE 1024×1024 解码约 20.7 分钟，尚不适合作为低延迟服务。

## 服务器目录

所有权重、依赖、补丁脚本、日志和输出位于：

`/data/infra/qwen-image-2.1-probe-20260921`
