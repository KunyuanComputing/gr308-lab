# GR308 vLLM Baseline 实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 4 张物理 GR308 上，使用同一份本地 Qwen3.8-27B BF16 权重，建立可复现、可审计、可供后续自研推理框架直接对比的 vLLM 基线。

**Architecture:** 实验分为三条互不混算的证据链：上游原版 vLLM 只做 GR308 兼容性门禁；厂商适配版 vLLM eager 作为主要性能基线；厂商 Graph32 配置仅作为可选的“现有栈上限”参考。所有性能请求走同一个 OpenAI-compatible 流式客户端，原始请求、服务日志、1 Hz GPU 遥测、版本哈希和统计结果写入唯一运行目录。

**Tech Stack:** Ubuntu 22.04、4×Fantasy GR308、Qwen3.8-27B BF16、vLLM 0.28.0 upstream compatibility lane、vLLM 0.18.0+empty + `vllm_fant` 1.1.0.20 runnable lane、Docker、`fantgpu-smi`、Python 3 标准库 benchmark client。

---

## 0. 文档状态与范围

**状态：`PLANNED / NOT STARTED`。** 本文完成后仍不授权启动容器、占用 GPU、恢复生产服务或执行正式请求；开始实验需要用户再次明确说“开始执行 baseline”。

本轮范围：

- 只测试 vLLM；SGLang 不进入实验矩阵、统计表或本轮耗时预算。
- 使用同一份 `/data/models/Qwen3.8-27B` 本地 BF16 权重。
- 主要结果是厂商适配版 vLLM eager 的系统级 baseline。
- 上游原版 vLLM 0.28.0 只给出 `runnable` 或 `unsupported` 的兼容性结论，不以 CPU 数据代替 GR308 数据。
- 可选 Graph32 结果单列，不能和 eager 数据拼成同一条曲线。
- 不包含量化、投机解码、LoRA、LMCache、网关、路由、多模型共驻、开放到达容量、自研框架实现或生产验收。

截至 2026-09-07 16:47 CST 的最新直接检查：吞吐客户端与遥测已停止，`qwen38-27b-bf16-eager-baseline` 已退出，8000/30000 未监听。执行前仍必须重新采集现场状态，不能把这一时刻的状态当作未来保证。

先前预跑目录 `/home/USER/levius/baselines/20260907-qwen38-bf16` 必须保留但排除出正式统计：

- 正确性哨兵 12/12 通过，可作为工具链诊断证据。
- eager C1、128→64、N=8 完成，309.645 秒、1.654 output tok/s，只用于本计划的排期校准。
- C8 在用户要求暂停后中断，不是失败点，也不是正式样本。
- 该目录缺完整正式 manifest 和预注册顺序，任何数字不得进入正式 baseline 汇总。

当前本地工作区不是 Git 仓库，因此本文不要求也不伪造 Git commit；文件完整性使用 SHA256 清单保证。

## 1. 比较对象与命名

| Lane | 固定名称 | 用途 | 是否产生性能数字 |
|---|---|---|---|
| U | `vllm-upstream-0.28.0` | 验证上游原版能否识别并初始化 GR308 | 否；本计划只做兼容性门禁 |
| E | `vllm-fant-1.1.0.20-eager` | 主要 baseline；后续自研框架首先与此对比 | 是 |
| G | `vllm-fant-1.1.0.20-graph32` | 可选“厂商现有栈默认/优化配置”参考 | 是，但单列且不归因于 vLLM 核心框架 |

已知软件身份：

- Lane U：`/home/USER/levius/.venvs/vllm`，vLLM 0.28.0，当前 PyTorch 为 `2.13.0+cpu`；预跑报 `Failed to infer device type`。
- Lane E/G 镜像：`fantasy/vllm:1.1.0.20-py310-ubuntu22.04`。
- Docker image ID：`sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc`；RepoDigest 在每次运行的 `images.json` 中另行记录。
- 容器内：vLLM `0.18.0+empty`、`vllm_fant` 1.1.0.20、`torch_fant` 1.1.0、PyTorch `2.7.1+cpu`，实际设备后端名为 `fant`。
- Lane E 现有容器 `qwen38-27b-bf16-eager-baseline` 只作为启动参数核对模板；正式运行创建带唯一 run ID 的新容器，避免复用累计日志和历史状态。
- Lane G 现有容器：`qwen38-27b-bf16`，Graph bucket 为 1/2/4/8/16/32；C64 不在冻结 bucket 内，必须标记为 saturation/stress 点。

结论措辞必须准确：Lane E/G 是“基于 vLLM 的厂商适配发行版”，不能写成“上游原版 vLLM 在 GR308 上的性能”。

## 2. 成功标准

完成核心 baseline 必须同时满足：

- [ ] 新运行目录、计划、模型、容器、客户端、哨兵、采集器均有 SHA256。
- [ ] 4 张物理卡与逻辑设备 0/2/4/6 的 PCI 地址和序列号重新核对。
- [ ] Lane U 完成兼容性门禁并保留退出码与完整错误日志。
- [ ] Lane E engine-cold 启动一次通过，服务 ready 后无自动重启。
- [ ] 正式性能前、稳定性后各执行一次固定哨兵，两次均 12/12 通过。
- [ ] Rapid 五档并发全部完成并用于更新后续时长预算。
- [ ] Formal 五档并发至少完成 3 个 engine-cold block replicate。
- [ ] 128→64、C64、1024 请求的 60 分钟级稳定性阶段完成。
- [ ] 每个有效请求均满足 HTTP 200、SSE `[DONE]`、`stream_done=true`、`finish_reason=length`、输入/输出 token 数精确匹配。
- [ ] 无非实验 GPU 进程共驻，无容器重启、OOM、FHCCL/driver reset 或静默请求丢失。
- [ ] 形成机器可读汇总、中文报告和全目录 SHA256 清单。

Baseline 完成不等于生产资格。最终报告必须保留：

```text
production_promotion_allowed=false
business_quality_certified=false
sglang_in_scope=false
```

## 3. 硬件与模型冻结

### 3.1 主机与设备

| 项目 | 冻结/重验值 |
|---|---|
| OS | Ubuntu 22.04.5，kernel 5.15.0-190-generic |
| CPU | 2×AMD EPYC 9654；224 逻辑 CPU |
| 内存 | 约 1 TiB；2 个 NUMA node |
| NUMA0 CPU | 0–55、112–167 |
| NUMA1 CPU | 56–111、168–223 |
| 物理 GPU | 4×GR308，每张 57,344 MiB（56 GiB），物理总显存 224 GiB |
| 逻辑设备 | 8 个 PCI function；同卡 `.0/.2` 共享物理资源 |
| baseline 设备 | 逻辑 0/2/4/6，各取一张物理卡的 `.0` function |
| PCIe/NUMA | 4 卡均位于 NUMA0；设备报告为 Gen5 x16、32 GT/s |
| 驱动/SMI | driver 1.1.0.19；GPU-SMI 2.2.0.9-linux |

物理卡映射：

| 物理卡 | 逻辑设备 | PCI functions | 序列号末四位 |
|---:|---|---|---|
| 0 | 0/1 | 0000:2e:00.0 / 0000:2e:00.2 | 0073 |
| 1 | 2/3 | 0000:34:00.0 / 0000:34:00.2 | 0076 |
| 2 | 4/5 | 0000:37:00.0 / 0000:37:00.2 | 0278 |
| 3 | 6/7 | 0000:38:00.0 / 0000:38:00.2 | 0193 |

采集时保留全部 8 个逻辑 function 的原始行，统计时只用 0/2/4/6 代表 4 张物理卡；不得把 sibling 报出的容量、显存或功率重复相加。`.2` 的 `power.draw=N/A` 不能当作 0 W，SMI 功率也不是整机墙上功率。

### 3.2 模型身份

| 项目 | 冻结值 |
|---|---|
| 路径 | `/data/models/Qwen3.8-27B` |
| Served name | `qwen3.8-27b` |
| 实际 architecture | `Qwen3_5ForConditionalGeneration` |
| 参数量 | 27,781,427,952 |
| 权重 | BF16，18/18 safetensors，55,563,006,776 bytes |
| 层数/hidden/FFN | 64 / 5120 / 17408 |
| Attention | 24 Q heads、4 KV heads、head_dim 256；48 linear + 16 full attention |
| 原始 context | 262,144；baseline 服务上限固定为 32,768 |
| `config.json` SHA256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| `tokenizer.json` SHA256 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer_config.json` SHA256 | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` |
| `chat_template.jinja` SHA256 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| safetensors index SHA256 | `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` |

完整权重哈希以 `/data/infra/benchmarks/gr308-report-execution-20260905/qwen38-bf16-weight-sha256.txt` 为冻结清单。任何一项不匹配时终止实验，不自动改用 `/models` 镜像路径或其他 Qwen 模型。

## 4. 公平性约束

主要 Lane E 固定：

- BF16、TP=4、text-only、逻辑设备 0/2/4/6。
- `max-model-len=32768`、`gpu-memory-utilization=0.40`、`block-size=800`。
- eager execution；prefix caching=false；mamba cache mode=none。
- 禁用量化、speculative decoding、LoRA、外部 router、LMCache、网关和共驻模型。
- 直连 `/v1/completions`，不经 8080/8180 Envoy/EPP。
- 合成性能请求使用 token-ID prompt：`[64] × ISL`、`temperature=0`、`ignore_eos=true`、streaming、usage 回传。
- 客户端必须逐请求验证实际 prompt/completion token 数；chunk 数不能当 token 数。
- 每次正式 block 从新容器、新 engine 进程启动；同进程内的不同并发点只能称同一 block 内技术重复。
- 记录 CPU affinity、NUMA policy、CPU governor、CPU/内存/磁盘后台负载；本轮不临时绑核或更换 governor。
- warmup 配置预先固定，不能看到成绩后延长或补热。

未来自研框架复用本计划时，必须匹配模型哈希、dtype、TP、物理卡、KV 可用容量、上下文上限、缓存策略、输入/输出 token 和并发顺序。若双方一边 eager、一边 graph，只能称“各自冻结配置的系统对比”，不能把差异全部归因于调度器或框架本身。

## 5. 证据目录与文件职责

正式结果落在 `/data`，不落在本地空间较小且未证明容量充足的 `/home`：

```text
/data/infra/benchmarks/gr308-vllm-baseline-YYYYMMDDTHHMMSSZ/
├── .gr308-vllm-baseline-root
├── PLAN_SHA256
├── STATUS.events.jsonl
├── manifest/
│   ├── run-manifest.json
│   ├── host-before.txt
│   ├── host-after.txt
│   ├── model.sha256
│   ├── model-code-and-config.sha256
│   ├── host-bind-files.sha256
│   ├── tools.sha256
│   ├── containers.json
│   ├── images.json
│   ├── env-redacted.txt
│   └── excluded-prior-runs.json
├── tools/
│   ├── closed_loop_safe_client.py
│   ├── run_sentinels.py
│   └── sentinel_cases.json
├── upstream-gate/
│   ├── platform.log
│   ├── serve.log
│   └── result.json
├── eager/
│   ├── startup/
│   ├── correctness-pre/
│   ├── rapid/c{1,8,16,32,64}/
│   ├── formal/r{1,3,5}/c{1,8,16,32,64}/
│   ├── stability/c64/
│   └── correctness-post/
├── graph32/                 # 可选，未执行时不创建结果子目录
├── analysis/
│   ├── run-level.csv
│   ├── request-level.csv
│   ├── telemetry-physical.csv
│   ├── baseline-summary.json
│   └── validation.json
├── REPORT_ZH.md
└── SHA256SUMS
```

所有产生数据的工具必须“文件已存在即失败”，禁止覆盖旧结果。`env-redacted.txt` 不保存 SSH 密码、token、代理凭据或私有 registry 密码。

Task 2–13 的每个独立 shell 都先执行以下上下文恢复块；不得依赖前一个代理或 shell 的临时变量：

```bash
set -Eeuo pipefail
set -o noclobber
umask 002
GR_ACTIVE_LINK=/data/infra/benchmarks/gr308-vllm-baseline-active
test -L "$GR_ACTIVE_LINK"
GR_RUN_ROOT="$(readlink -f -- "$GR_ACTIVE_LINK")"
case "$GR_RUN_ROOT" in
  /data/infra/benchmarks/gr308-vllm-baseline-20??????T??????Z) ;;
  *) printf 'unsafe run root: %s\n' "$GR_RUN_ROOT" >&2; exit 90 ;;
esac
test -f "$GR_RUN_ROOT/.gr308-vllm-baseline-root"
GR_RUN_ID="${GR_RUN_ROOT##*/}"
: "${GR_RUN_ROOT:?}" "${GR_RUN_ID:?}"
```

`set -o noclobber` 使普通 `>` 在目标已存在时失败；需要追加的 `owned-containers.txt` 和 `STATUS.events.jsonl` 是两个预注册例外。复制文件前仍要显式 `test ! -e destination`。所有预期可能返回非零的命令必须放进 `if command; then ... else rc=$?; fi`，保证退出码先落盘再结束任务。

计划冻结的既有工具：

| 工具 | 路径 | SHA256 |
|---|---|---|
| 闭环客户端 | `/data/infra/benchmarks/gr308-throughput-20260905/closed_loop_safe_client.py` | `c8e98954be2ff311821a05195064485cfe018ed56d62eb0fa0f7c436bfadb369` |
| 哨兵 runner | `/data/infra/benchmarks/gr308-iteration-v2-20260905/run_sentinels.py` | `f5a00d8573fcc9a87a6a7f10343b586432f59b7c8a4e53da55ad2c687d820326` |
| 哨兵 cases | `/data/infra/benchmarks/gr308-iteration-v2-20260905/GR308_Iteration_v2_20260905/sentinel_cases.PROPOSED.json` | `5bf8c6c566ade4e890915c4b31694897f468bc6cad3f004e34a03d3a699dc2fa` |

`closed_loop_safe_client.py --warmup N` 实际是 N 个串行、最多输出 8 token 的轻预热，不是 N 轮目标并发预热。本计划不使用该参数模拟并发 warmup，而是创建独立且弃权的目标并发 warmup case。

## 6. 阶段与预计耗时

时间均为单个 vLLM Lane E 的墙钟预算。第一次 Rapid 后必须按实测更新 Formal 排期；下表是资源预约，不是性能预测。模型全量哈希会顺序读取约 52 GiB 权重并加热宿主 page cache，因此本文的“engine-cold”仅指新容器和新 engine 进程，不代表存储冷启动、驱动冷启动或 host-cache-cold；禁止为制造冷启动执行 `drop_caches`。

| 阶段 | 内容 | 预计耗时 | 结束条件 |
|---|---|---:|---|
| P0 | 计划评审与执行授权 | 15–30 分钟 | 用户明确授权开始，计划 SHA 冻结 |
| P1 | 独占性、端口、进程、磁盘、时钟预检 | 20–30 分钟 | 4 张物理卡无非实验进程，目标端口空闲 |
| P2 | 硬件、模型、镜像、容器、工具完整性冻结 | 20–40 分钟 | manifest 与 SHA256 完整 |
| P3 | 上游 vLLM 兼容性门禁 | 10–15 分钟 | `runnable` 或证据完整的 `unsupported` |
| P4 | 厂商 eager engine-cold 启动与 health | 15–45 分钟 | `/health`、`/v1/models` 通过且无 restart |
| P5 | 12 条固定正确性哨兵 | 10–20 分钟 | 12/12 精确通过 |
| P6 | Rapid 128→64 五档矩阵 | 36–50 分钟 | 五点有效或真实失败已记录 |
| P7 | Rapid 审计与 Formal 排期重估 | 15–25 分钟 | 预算公式和正式时隙固化 |
| P8 | Formal 1024→256，3 个 engine-cold block | 10–14.5 小时 | 每点 3 个 run-level block 样本 |
| P9 | engine-cold 启动、C64 稳定性、性能后哨兵 | 100–160 分钟 | ≥1024 终态、持续约 60 分钟、后哨兵通过 |
| P10 | 聚合、统计、报告、哈希 | 60–120 分钟 | validation 通过、报告可复算 |
| P11 | 停止实验容器并恢复“全部停止”状态 | 15–30 分钟 | 端口关闭、无自有测试进程 |

核心 vLLM baseline 总预算：**约 15–24 小时**。若 3 个 Formal block 后触发第 4 个 block，增加 **3–4.5 小时**；第 4、5 个都触发时增加 **6–9 小时**。可选 Graph32 全矩阵另需 **14–22 小时**，不计入核心预算。

### 6.1 Rapid 分档预算

Rapid：ISL=128、OSL=64。测量请求数固定为 8/16/32/64/128；每档先跑 1 个同 shape、同并发、请求数=C 的弃权 warmup wave。

| 并发 C | 正式请求 N | 预跑测量中央值 | 含 warmup/落盘建议时隙 |
|---:|---:|---:|---:|
| 1 | 8 | 5.16 分钟，已观测但排除 | 7–9 分钟 |
| 8 | 16 | 约 1.64 分钟 | 4–6 分钟 |
| 16 | 32 | 约 2.58 分钟 | 5–7 分钟 |
| 32 | 64 | 约 4.24 分钟 | 7–10 分钟 |
| 64 | 128 | 约 9.43 分钟 | 13–18 分钟 |

除 C1 外的中央值仅用于排期，来自 C1 预跑锚点和历史并发曲线的保守缩放，不得提前写入结果表。

### 6.2 Formal 分档预算

Formal：ISL=1024、OSL=256。每档正式请求数为 `max(12, 2×C)`。Lane E 每档预热 1 个同 ISL/并发、OSL=64、请求数=C 的弃权 wave；Lane G 若将来执行，必须预热完整 OSL=256 的对应 graph/bucket 路径。

| 并发 C | 正式请求 N | 单次测量排期中央值 | 含预热/落盘建议时隙 |
|---:|---:|---:|---:|
| 1 | 12 | 约 42 分钟 | 45–60 分钟 |
| 8 | 16 | 约 9 分钟 | 12–18 分钟 |
| 16 | 32 | 约 14 分钟 | 18–27 分钟 |
| 32 | 64 | 约 23 分钟 | 30–42 分钟 |
| 64 | 128 | 约 52 分钟 | 65–90 分钟 |

动态排期公式：

```text
T_formal_measurement(C) ≈ N_requests × 256 / (0.70 × q_rapid(C))
T_reserved(C) = 1.20 × T_formal_measurement(C) + warmup_and_flush_time
```

`0.70` 是根据旧 1024→256 数据得到的保守长度折损，只用于排期，不是对正式吞吐的预测或修正系数。

## 7. 执行任务

### Task 1：冻结授权、运行 ID 与后置状态

**Files:**

- Create: `/data/infra/benchmarks/gr308-vllm-baseline-YYYYMMDDTHHMMSSZ/STATUS.events.jsonl`
- Create: `/data/infra/benchmarks/gr308-vllm-baseline-YYYYMMDDTHHMMSSZ/manifest/run-manifest.json`

- [ ] **Step 1：确认执行授权**

只有用户在本文完成后再次明确要求开始，才把 manifest 中以下字段写为真：

```json
{
  "baseline_start_authorized": true,
  "scope": "vllm-only",
  "desired_post_state": "all-stopped",
  "production_promotion_allowed": false,
  "sglang_in_scope": false
}
```

- [ ] **Step 2：生成唯一 UTC run ID**

```bash
set -Eeuo pipefail
set -o noclobber
umask 002
GR_RUN_ID="gr308-vllm-baseline-$(date -u +%Y%m%dT%H%M%SZ)"
GR_RUN_ROOT="/data/infra/benchmarks/${GR_RUN_ID}"
GR_ACTIVE_LINK=/data/infra/benchmarks/gr308-vllm-baseline-active
test ! -e "$GR_RUN_ROOT"
test ! -e "$GR_ACTIVE_LINK"
mkdir "$GR_RUN_ROOT"
mkdir "$GR_RUN_ROOT"/{manifest,tools,upstream-gate,eager,analysis}
printf '%s\n' "$GR_RUN_ID" > "$GR_RUN_ROOT/.gr308-vllm-baseline-root"
ln -s "$GR_RUN_ROOT" "$GR_ACTIVE_LINK"
test ! -e "$GR_RUN_ROOT/plan.md"
cp /home/USER/levius/docs/superpowers/plans/2026-09-07-gr308-vllm-baseline.md "$GR_RUN_ROOT/plan.md"
sha256sum "$GR_RUN_ROOT/plan.md" > "$GR_RUN_ROOT/PLAN_SHA256"
```

Expected: `test` 返回 0，新目录唯一，active symlink 精确指向它，marker 与 plan SHA 均存在。

- [ ] **Step 3：登记排除的预跑目录**

```bash
GR_CREATED_AT="$(date -u +%FT%TZ)"
jq -n \
  --arg run_id "$GR_RUN_ID" \
  --arg run_root "$GR_RUN_ROOT" \
  --arg created_at "$GR_CREATED_AT" \
  '{schema_version:1,run_id:$run_id,run_root:$run_root,created_at_utc:$created_at,baseline_start_authorized:true,scope:"vllm-only",desired_post_state:"all-stopped",production_promotion_allowed:false,sglang_in_scope:false}' \
  > "$GR_RUN_ROOT/manifest/run-manifest.json"
jq -cn \
  --arg created_at "$GR_CREATED_AT" \
  '{schema_version:1,status:"planned-authorized",updated_at_utc:$created_at}' \
  > "$GR_RUN_ROOT/STATUS.events.jsonl"
jq -n \
  '{schema_version:1,excluded:[{path:"/home/USER/levius/baselines/20260907-qwen38-bf16",classification:"aborted_preplan",reason:"user_paused_before_plan_freeze",observations:{sentinel_passed:12,sentinel_total:12,c1_completed:true,c1_output_tok_s:1.6535,c8_interrupted:true},eligible_for_formal_aggregate:false}]}' \
  > "$GR_RUN_ROOT/manifest/excluded-prior-runs.json"
jq -e . "$GR_RUN_ROOT/manifest/run-manifest.json" > /dev/null
jq -e . "$GR_RUN_ROOT/STATUS.events.jsonl" > /dev/null
jq -e . "$GR_RUN_ROOT/manifest/excluded-prior-runs.json" > /dev/null
```

不删除、不改写原始预跑文件。

### Task 2：预检独占性与安全边界

**Files:**

- Create: `$GR_RUN_ROOT/manifest/host-before.txt`
- Create: `$GR_RUN_ROOT/manifest/preflight.json`

- [ ] **Step 1：采集主机、端口、容器和进程状态**

```bash
{
  date -u '+%FT%TZ'
  uname -a
  lscpu
  numactl --hardware
  df -B1 /data
  docker ps -a --no-trunc
  ss -ltnp
  fantgpu-smi
  pgrep -af 'vllm|sglang|closed_loop_safe_client|fantgpu-smi.*lms' || true
} > "$GR_RUN_ROOT/manifest/host-before.txt" 2>&1
```

- [ ] **Step 2：执行硬门**

必须同时满足：

- 8000、8001、8080、8180、8181、18000、30000 无监听。
- 全部逻辑 function 0–7 均无模型或测试进程；统计设备虽为 0/2/4/6，但 sibling 1/3/5/7 同样共享物理资源，任何一个都不能共驻。
- 不存在下载、转换或其他作业造成持续高 CPU、内存、磁盘 I/O。
- `/data` 可用空间不少于 200 GiB。
- NTP/系统时间可用，采集机和服务机是同一主机。

核对完成后用以下固定 schema 写结果：

```bash
GR_PREFLIGHT_STATUS=passed
GR_PREFLIGHT_REASON=all_exclusivity_and_safety_gates_passed
jq -n \
  --arg status "$GR_PREFLIGHT_STATUS" \
  --arg reason "$GR_PREFLIGHT_REASON" \
  --arg checked_at "$(date -u +%FT%TZ)" \
  '{schema_version:1,status:$status,reason:$reason,checked_at_utc:$checked_at,ports_free:true,all_gpu_functions_idle:true,no_running_containers:true,data_free_bytes_gte_200_gib:true,time_synchronized:true}' \
  > "$GR_RUN_ROOT/manifest/preflight.json"
jq -e '.status=="passed"' "$GR_RUN_ROOT/manifest/preflight.json" > /dev/null
```

只有证据确实满足全部硬门时才能使用上述 `passed` 内容；任一项不满足则把 status 写为 `blocked`、各布尔值写成实测值并停止。不得使用 `killall`、GPU reset、删除容器或修改驱动来强行通过。

### Task 3：冻结模型、工具和运行栈

**Files:**

- Create: `$GR_RUN_ROOT/manifest/model.sha256`
- Create: `$GR_RUN_ROOT/manifest/model-code-and-config.sha256`
- Create: `$GR_RUN_ROOT/manifest/model-weights-verify.log`
- Create: `$GR_RUN_ROOT/manifest/host-bind-files.sha256`
- Create: `$GR_RUN_ROOT/manifest/tools.sha256`
- Create: `$GR_RUN_ROOT/manifest/containers.json`
- Create: `$GR_RUN_ROOT/manifest/images.json`
- Create: `$GR_RUN_ROOT/manifest/env-redacted.txt`

- [ ] **Step 1：核对关键模型文件与完整权重清单**

```bash
sha256sum \
  /data/models/Qwen3.8-27B/config.json \
  /data/models/Qwen3.8-27B/tokenizer.json \
  /data/models/Qwen3.8-27B/tokenizer_config.json \
  /data/models/Qwen3.8-27B/chat_template.jinja \
  /data/models/Qwen3.8-27B/model.safetensors.index.json \
  > "$GR_RUN_ROOT/manifest/model.sha256"
(
  cd /data/models/Qwen3.8-27B
  sha256sum -c /data/infra/benchmarks/gr308-report-execution-20260905/qwen38-bf16-weight-sha256.txt
) > "$GR_RUN_ROOT/manifest/model-weights-verify.log" 2>&1
find /data/models/Qwen3.8-27B -maxdepth 1 -type f ! -name '*.safetensors' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  > "$GR_RUN_ROOT/manifest/model-code-and-config.sha256"
find \
  /usr/lib/x86_64-linux-gnu/fantgpu-fh3 \
  /usr/local/fant/include \
  /usr/local/fant/lib/llvm \
  -type f -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  > "$GR_RUN_ROOT/manifest/host-bind-files.sha256"
```

Expected: 所有权重显示 `OK`，关键哈希与第 3.2 节完全一致；模型目录中所有会被 `--trust-remote-code` 加载的源码/配置和所有镜像外 bind 文件均进入哈希清单。若模型无需 remote code，正式启动前应移除 `--trust-remote-code`；保留该参数时不得跳过这些哈希。

- [ ] **Step 2：复制并复核工具**

```bash
test ! -e "$GR_RUN_ROOT/tools/closed_loop_safe_client.py"
test ! -e "$GR_RUN_ROOT/tools/run_sentinels.py"
test ! -e "$GR_RUN_ROOT/tools/sentinel_cases.json"
cp /data/infra/benchmarks/gr308-throughput-20260905/closed_loop_safe_client.py "$GR_RUN_ROOT/tools/closed_loop_safe_client.py"
cp /data/infra/benchmarks/gr308-iteration-v2-20260905/run_sentinels.py "$GR_RUN_ROOT/tools/run_sentinels.py"
cp /data/infra/benchmarks/gr308-iteration-v2-20260905/GR308_Iteration_v2_20260905/sentinel_cases.PROPOSED.json "$GR_RUN_ROOT/tools/sentinel_cases.json"
sha256sum "$GR_RUN_ROOT"/tools/* > "$GR_RUN_ROOT/manifest/tools.sha256"
python3 "$GR_RUN_ROOT/tools/closed_loop_safe_client.py" --help > "$GR_RUN_ROOT/manifest/client-help.txt"
python3 "$GR_RUN_ROOT/tools/run_sentinels.py" --help > "$GR_RUN_ROOT/manifest/sentinel-help.txt"
fantgpu-smi --help > "$GR_RUN_ROOT/manifest/fantgpu-smi-help.txt" 2>&1
```

Expected: SHA256 与第 5 节一致。

- [ ] **Step 3：冻结 Docker 身份和启动模板**

```bash
docker image inspect fantasy/vllm:1.1.0.20-py310-ubuntu22.04 \
  > "$GR_RUN_ROOT/manifest/images.json"
docker inspect qwen38-27b-bf16-eager-baseline qwen38-27b-bf16 \
  > "$GR_RUN_ROOT/manifest/containers.json"
env \
  | LC_ALL=C sort \
  | sed -E '/(PASS|TOKEN|SECRET|CREDENTIAL|AUTH|COOKIE|KEY)/Id' \
  > "$GR_RUN_ROOT/manifest/env-redacted.txt"
```

保存完整 JSON；逐项核对 image ID、RepoDigest、模型挂载、served name、TP、dtype、设备可见性、eager/graph、max model length、block size、memory utilization、cache 和 restart policy。旧容器只供核对，不直接启动；配置有差异时停止，不在运行中临时改参数。

### Task 4：执行上游原版兼容性门禁

**Files:**

- Create: `$GR_RUN_ROOT/upstream-gate/platform.log`
- Create: `$GR_RUN_ROOT/upstream-gate/serve.log`
- Create: `$GR_RUN_ROOT/upstream-gate/result.json`

- [ ] **Step 1：记录上游环境**

```bash
{
  /home/USER/levius/.venvs/vllm/bin/python -VV
  /home/USER/levius/.venvs/vllm/bin/pip freeze
  /home/USER/levius/.venvs/vllm/bin/python -c 'import torch,vllm; print(torch.__version__); print(vllm.__version__)'
} > "$GR_RUN_ROOT/upstream-gate/platform.log" 2>&1
```

- [ ] **Step 2：运行 120 秒有界门禁**

```bash
GR_UPSTREAM_RC=0
if CUDA_VISIBLE_DEVICES=0,2,4,6 \
   timeout --signal=TERM --kill-after=10s 120s \
     /home/USER/levius/.venvs/vllm/bin/vllm serve /data/models/Qwen3.8-27B \
     --host 127.0.0.1 \
     --port 18000 \
     --dtype bfloat16 \
     --tensor-parallel-size 4 \
     --max-model-len 32768 \
     --trust-remote-code \
     > "$GR_RUN_ROOT/upstream-gate/serve.log" 2>&1; then
  GR_UPSTREAM_RC=0
else
  GR_UPSTREAM_RC=$?
fi
printf '%s\n' "$GR_UPSTREAM_RC" > "$GR_RUN_ROOT/upstream-gate/serve.rc"
```

判定：只有日志明确显示 GR308 worker 和 0/2/4/6 设备初始化后才是 `runnable`。`UnspecifiedPlatform`、`Failed to infer device type`、0 active drivers 或退回 CPU 均判为 `unsupported`。`unsupported` 是有效门禁结果，不进入吞吐阶段，也不尝试把厂商插件注入上游环境。

- [ ] **Step 3：写入结构化门禁结果**

`result.json` 固定包含 `lane`、`vllm_version`、`torch_version`、`exit_code`、`status`、`device_platform`、`ready_seen`、`reason` 和日志 SHA256。`status` 只能取 `runnable`、`unsupported`、`startup-failed`；根据完整日志判定，缺字段时判 `startup-failed`，不得把未知情况归为兼容。

- [ ] **Step 4：验证门禁进程完全退出**

```bash
{
  date -u '+%FT%TZ'
  ss -ltnp | grep -E ':18000([[:space:]]|$)' || true
  pgrep -af '/home/USER/levius/.venvs/vllm|vllm serve.*/data/models/Qwen3.8-27B' || true
  fantgpu-smi
} > "$GR_RUN_ROOT/upstream-gate/cleanup-check.txt" 2>&1
```

18000 仍监听、存在新 vLLM worker，或任一逻辑 GPU function 不再空闲时，Lane E 不得开始。只允许针对本门禁启动并已记录身份的进程做有界终止；身份无法证明时停止并请求人工处理。

### Task 5：启动 Lane E 并执行 engine-cold 门禁

**Files:**

- Create: `$GR_RUN_ROOT/eager/startup/container-id.txt`
- Create: `$GR_RUN_ROOT/eager/startup/container-ready.json`
- Create: `$GR_RUN_ROOT/eager/startup/server-ready.log`
- Create: `$GR_RUN_ROOT/eager/startup/ready.json`

- [ ] **Step 1：用冻结参数创建唯一的新容器**

```bash
mkdir "$GR_RUN_ROOT/eager/startup"
GR_CONTAINER="${GR_RUN_ID}-eager-rapid"
GR_START_NS="$(date +%s%N)"
printf '%s\n' "$GR_START_NS" > "$GR_RUN_ROOT/eager/startup/start-ns.txt"
printf '%s\n' "$GR_CONTAINER" > "$GR_RUN_ROOT/eager/startup/container-intent.txt"
GR_CONTAINER_CREATED=0
trap 'if [ "$GR_CONTAINER_CREATED" -eq 1 ]; then docker stop --time 120 "$GR_CONTAINER" >/dev/null 2>&1 || true; fi' ERR INT TERM
GR_DOCKER_RC=0
if docker run -d \
  --name "$GR_CONTAINER" \
  --hostname user \
  --label "gr308.baseline.run=$GR_RUN_ID" \
  --privileged \
  --runtime=fantasy \
  --ipc=host \
  --network=host \
  --restart=no \
  --security-opt label=disable \
  -e CUDA_VISIBLE_DEVICES=0,2,4,6 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
  -v /usr/local/fant/include:/usr/local/fant/include:ro \
  -v /usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro \
  -v /data/models/Qwen3.8-27B:/model:ro \
  sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc \
  vllm serve /model \
    --host 127.0.0.1 \
    --port 8000 \
    --served-model-name qwen3.8-27b \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.40 \
    --block-size 800 \
    --language-model-only \
    --trust-remote-code \
    --enforce-eager \
  > "$GR_RUN_ROOT/eager/startup/container-id.txt"; then
  GR_CONTAINER_CREATED=1
else
  GR_DOCKER_RC=$?
fi
printf '%s\n' "$GR_DOCKER_RC" > "$GR_RUN_ROOT/eager/startup/docker-run.rc"
test "$GR_DOCKER_RC" -eq 0
test "$(docker inspect -f '{{ index .Config.Labels "gr308.baseline.run" }}' "$GR_CONTAINER")" = "$GR_RUN_ID"
printf '%s\n' "$GR_CONTAINER" >> "$GR_RUN_ROOT/manifest/owned-containers.txt"
trap - ERR INT TERM
```

Expected: 创建并启动一个 `--restart=no` 的新容器；不得启动或修改旧模板容器。`--privileged` 与 Fantasy runtime 具有高宿主权限，只能在用户明确授权后执行。

- [ ] **Step 2：等待 ready，绝对上限 45 分钟**

每 5 秒请求一次 `http://127.0.0.1:8000/health`，同时跟踪容器状态与 `RestartCount`：

```bash
GR_READY=0
for GR_CHECK in $(seq 1 540); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/health \
      > /dev/null; then
    GR_READY=1
    date +%s%N > "$GR_RUN_ROOT/eager/startup/ready-ns.txt"
    break
  fi
  test "$(docker inspect -f '{{.State.Running}}' "$GR_CONTAINER")" = true || break
  sleep 5
done
test "$GR_READY" -eq 1
```

ready 后记录：

```bash
curl -fsS http://127.0.0.1:8000/health > "$GR_RUN_ROOT/eager/startup/health.txt"
curl -fsS http://127.0.0.1:8000/version > "$GR_RUN_ROOT/eager/startup/version.json"
curl -fsS http://127.0.0.1:8000/v1/models > "$GR_RUN_ROOT/eager/startup/models.json"
docker inspect "$GR_CONTAINER" > "$GR_RUN_ROOT/eager/startup/container-ready.json"
docker logs --timestamps "$GR_CONTAINER" > "$GR_RUN_ROOT/eager/startup/server-ready.log" 2>&1
jq -n \
  --arg container "$GR_CONTAINER" \
  --arg start_ns "$GR_START_NS" \
  --arg ready_ns "$(cat "$GR_RUN_ROOT/eager/startup/ready-ns.txt")" \
  '{schema_version:1,status:"ready",container:$container,start_ns:($start_ns|tonumber),ready_ns:($ready_ns|tonumber),engine_start_to_ready_s:((($ready_ns|tonumber)-($start_ns|tonumber))/1000000000)}' \
  > "$GR_RUN_ROOT/eager/startup/ready.json"
```

只允许一次预注册启动尝试。超时、自动 restart、OOM、FHCCL/driver reset 或模型身份不符都记为真实 `startup-failed`；不得静默重启后只保留成功一次。

### Task 6：正确性哨兵门禁

**Files:**

- Create: `$GR_RUN_ROOT/eager/correctness-pre/responses.jsonl`
- Create: `$GR_RUN_ROOT/eager/correctness-pre/summary.json`
- Create: `$GR_RUN_ROOT/eager/correctness-pre.run.log`
- Create: `$GR_RUN_ROOT/eager/correctness-pre.rc`

- [ ] **Step 1：运行固定 12 例**

```bash
GR_SENTINEL_RC=0
if python3 "$GR_RUN_ROOT/tools/run_sentinels.py" \
     --endpoint http://127.0.0.1:8000/v1/chat/completions \
     --model qwen3.8-27b \
     --cases "$GR_RUN_ROOT/tools/sentinel_cases.json" \
     --output "$GR_RUN_ROOT/eager/correctness-pre" \
     --max-tokens 64 \
     --timeout-s 90 \
     > "$GR_RUN_ROOT/eager/correctness-pre.run.log" 2>&1; then
  GR_SENTINEL_RC=0
else
  GR_SENTINEL_RC=$?
fi
printf '%s\n' "$GR_SENTINEL_RC" > "$GR_RUN_ROOT/eager/correctness-pre.rc"
test "$GR_SENTINEL_RC" -eq 0
```

- [ ] **Step 2：执行门禁**

必须是 12/12 精确匹配、HTTP 200、无空输出、无单字符异常重复。该集合只覆盖中文 JSON、整数排序和字段抽取，不得表述为英语、代码、复杂推理或业务质量认证。

### Task 7：执行 Rapid 矩阵

**Files:**

- Create: `$GR_RUN_ROOT/eager/rapid/telemetry.csv`
- Create: `$GR_RUN_ROOT/eager/rapid/c*/warmup-{requests.jsonl,summary.json}`
- Create: `$GR_RUN_ROOT/eager/rapid/c*/{requests.jsonl,summary.json,client.log}`

- [ ] **Step 1：启动独立 1 Hz 遥测**

```bash
mkdir "$GR_RUN_ROOT/eager/rapid"
stdbuf -oL fantgpu-smi \
  --query-gpu timestamp,index,pci.bus_id,serial_number,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
  --format csv,noheader,nounits \
  -lms 1000 \
  > "$GR_RUN_ROOT/eager/rapid/telemetry.csv" \
  2> "$GR_RUN_ROOT/eager/rapid/telemetry.err" &
GR_TELEMETRY_PID=$!
printf '%s\n' "$GR_TELEMETRY_PID" > "$GR_RUN_ROOT/eager/rapid/telemetry.pid"
trap 'kill "$GR_TELEMETRY_PID" 2>/dev/null || true; wait "$GR_TELEMETRY_PID" 2>/dev/null || true' INT TERM EXIT
```

保存采集器 PID，只终止该 PID；不得使用 broad process kill。遥测相邻时间戳超过 3 秒，或任一正式窗口缺首尾采样时，该点标为 telemetry incomplete。

- [ ] **Step 2：按 1、8、16、32、64 顺序执行**

每档先用相同 ISL/OSL、`requests=C`、`warmup=0` 运行弃权 warmup；warmup 进程退出后等待 10 秒并重新检查 `/health`，再运行正式 N 请求。每档变量和目录按以下方式生成：

```bash
for GR_PAIR in 1:8 8:16 16:32 32:64 64:128; do
  GR_CONCURRENCY="${GR_PAIR%%:*}"
  GR_REQUESTS="${GR_PAIR##*:}"
  GR_CASE_DIR="$GR_RUN_ROOT/eager/rapid/c${GR_CONCURRENCY}"
  test ! -e "$GR_CASE_DIR"
  mkdir -p "$GR_CASE_DIR"

  GR_WARMUP_RC=0
  if python3 "$GR_RUN_ROOT/tools/closed_loop_safe_client.py" \
       --endpoint http://127.0.0.1:8000/v1/completions \
       --model qwen3.8-27b \
       --deployment-id "vllm-fant-1.1.0.20-eager-rapid-c${GR_CONCURRENCY}-warmup" \
       --input-tokens 128 \
       --output-tokens 64 \
       --concurrency "$GR_CONCURRENCY" \
       --requests "$GR_CONCURRENCY" \
       --warmup 0 \
       --prompt-token-id 64 \
       --timeout-s 1200 \
       --output "$GR_CASE_DIR/warmup-requests.jsonl" \
       --summary "$GR_CASE_DIR/warmup-summary.json" \
       > "$GR_CASE_DIR/warmup-client.log" 2>&1; then
    GR_WARMUP_RC=0
  else
    GR_WARMUP_RC=$?
  fi
  printf '%s\n' "$GR_WARMUP_RC" > "$GR_CASE_DIR/warmup-client.rc"
  test "$GR_WARMUP_RC" -eq 0 || break

  sleep 10
  curl -fsS --max-time 3 http://127.0.0.1:8000/health > /dev/null

  GR_CASE_RC=0
  if timeout --signal=INT --kill-after=30s 1800s \
       python3 "$GR_RUN_ROOT/tools/closed_loop_safe_client.py" \
       --endpoint http://127.0.0.1:8000/v1/completions \
       --model qwen3.8-27b \
       --deployment-id "vllm-fant-1.1.0.20-eager-rapid-c${GR_CONCURRENCY}" \
       --input-tokens 128 \
       --output-tokens 64 \
       --concurrency "$GR_CONCURRENCY" \
       --requests "$GR_REQUESTS" \
       --warmup 0 \
       --prompt-token-id 64 \
       --timeout-s 1200 \
       --output "$GR_CASE_DIR/requests.jsonl" \
       --summary "$GR_CASE_DIR/summary.json" \
       > "$GR_CASE_DIR/client.log" 2>&1; then
    GR_CASE_RC=0
  else
    GR_CASE_RC=$?
  fi
  printf '%s\n' "$GR_CASE_RC" > "$GR_CASE_DIR/client.rc"
  test "$GR_CASE_RC" -eq 0 || break
done
```

矩阵结束或中止后，无论结果如何都只停止本轮 sampler PID：

```bash
kill "$GR_TELEMETRY_PID" 2>/dev/null || true
wait "$GR_TELEMETRY_PID" 2>/dev/null || true
trap - INT TERM EXIT
```

- [ ] **Step 3：逐点验收**

要求 exit code 0、`statuses={"ok":N}`、JSONL 恰好 N 行、每行 128/64 token、`stream_done=true`、`finish_reason=length`。任一 HTTP 错误、超时、token mismatch 或引擎 restart 都停止后续点并进入取证，不用重跑覆盖。

### Task 8：Rapid 后重新估时

**Files:**

- Create: `$GR_RUN_ROOT/analysis/formal-time-budget.json`

- [ ] **Step 1：读取每个并发点的 `raw_completed_output_tokens_per_s`**

- [ ] **Step 2：用第 6.2 节公式计算每点正式时隙**

- [ ] **Step 3：若某点预算超过表中上界 1.5 倍，暂停 Formal 并向用户报告新总时长**

这一步只改变预约时间，不改变请求数、顺序、warmup 或统计规则。

### Task 9：执行 Formal 三个 engine-cold block

**Files:**

- Create: `$GR_RUN_ROOT/eager/formal/r*/startup/`
- Create: `$GR_RUN_ROOT/eager/formal/r*/c*/warmup-*`
- Create: `$GR_RUN_ROOT/eager/formal/r*/c*/{requests.jsonl,summary.json,client.log,telemetry.csv}`

- [ ] **Step 1：冻结并发顺序**

基础三次使用：

```text
R1: 1, 8, 64, 16, 32
R3: 16, 32, 8, 64, 1
R5: 64, 1, 32, 8, 16
```

若触发补测，依次加入：

```text
R2: 8, 16, 1, 32, 64
R4: 32, 64, 16, 1, 8
```

完成五次后，每个并发恰好出现在每个序位一次。

- [ ] **Step 2：每个 repeat 从新容器和新 engine 进程开始**

先验证 rapid 容器 label 与 run ID，再停止并封存它；确认 8000 关闭、全部 GPU function 释放后创建 `formal` 根目录。每个 block 的新容器分别命名为 `${GR_RUN_ID}-eager-r1`、`${GR_RUN_ID}-eager-r3`、`${GR_RUN_ID}-eager-r5`；若触发补测则使用 `r2`、`r4`。这仍是 host-cache-warm 的操作性 block replicate，不宣称存储或驱动冷启动。

```bash
GR_RAPID_CONTAINER="${GR_RUN_ID}-eager-rapid"
test "$(docker inspect -f '{{ index .Config.Labels "gr308.baseline.run" }}' "$GR_RAPID_CONTAINER")" = "$GR_RUN_ID"
docker logs --timestamps "$GR_RAPID_CONTAINER" > "$GR_RUN_ROOT/eager/startup/server-final.log" 2>&1
docker inspect "$GR_RAPID_CONTAINER" > "$GR_RUN_ROOT/eager/startup/container-final.json"
docker stop --time 120 "$GR_RAPID_CONTAINER" > "$GR_RUN_ROOT/eager/startup/docker-stop.txt"
for GR_CHECK in $(seq 1 60); do
  if ! ss -ltnH | awk '{print $4}' | grep -Eq ':8000$'; then break; fi
  sleep 2
done
if ss -ltnH | awk '{print $4}' | grep -Eq ':8000$'; then exit 93; fi
fantgpu-smi > "$GR_RUN_ROOT/eager/startup/smi-after-stop.txt"
```

- [ ] **Step 3：每档执行固定预热和正式请求**

预热：ISL=1024、OSL=64、concurrency=C、requests=C，弃权。正式：ISL=1024、OSL=256、requests=12/16/32/64/128。以下是一个完整 block 内的可执行结构；`GR_REPEAT` 逐次取 `r1`、`r3`、`r5`，补测时取 `r2`、`r4`：

```bash
GR_REPEAT=r1
case "$GR_REPEAT" in
  r1) GR_ORDER='1 8 64 16 32' ;;
  r2) GR_ORDER='8 16 1 32 64' ;;
  r3) GR_ORDER='16 32 8 64 1' ;;
  r4) GR_ORDER='32 64 16 1 8' ;;
  r5) GR_ORDER='64 1 32 8 16' ;;
  *) printf 'invalid repeat: %s\n' "$GR_REPEAT" >&2; exit 91 ;;
esac

GR_REPEAT_ROOT="$GR_RUN_ROOT/eager/formal/$GR_REPEAT"
test -d "$GR_RUN_ROOT/eager/formal" || mkdir "$GR_RUN_ROOT/eager/formal"
test ! -e "$GR_REPEAT_ROOT"
mkdir "$GR_REPEAT_ROOT"
mkdir "$GR_REPEAT_ROOT/startup"
GR_CONTAINER="${GR_RUN_ID}-eager-${GR_REPEAT}"
printf '%s\n' "$GR_CONTAINER" > "$GR_REPEAT_ROOT/startup/container-intent.txt"
GR_START_NS="$(date +%s%N)"
printf '%s\n' "$GR_START_NS" > "$GR_REPEAT_ROOT/startup/start-ns.txt"
GR_CONTAINER_CREATED=0
GR_TELEMETRY_PID=''
trap 'if [ -n "$GR_TELEMETRY_PID" ]; then kill "$GR_TELEMETRY_PID" 2>/dev/null || true; wait "$GR_TELEMETRY_PID" 2>/dev/null || true; fi; if [ "$GR_CONTAINER_CREATED" -eq 1 ]; then docker stop --time 120 "$GR_CONTAINER" >/dev/null 2>&1 || true; fi' ERR INT TERM

GR_DOCKER_RC=0
if docker run -d \
  --name "$GR_CONTAINER" \
  --hostname user \
  --label "gr308.baseline.run=$GR_RUN_ID" \
  --privileged \
  --runtime=fantasy \
  --ipc=host \
  --network=host \
  --restart=no \
  --security-opt label=disable \
  -e CUDA_VISIBLE_DEVICES=0,2,4,6 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
  -v /usr/local/fant/include:/usr/local/fant/include:ro \
  -v /usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro \
  -v /data/models/Qwen3.8-27B:/model:ro \
  sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc \
  vllm serve /model \
    --host 127.0.0.1 \
    --port 8000 \
    --served-model-name qwen3.8-27b \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.40 \
    --block-size 800 \
    --language-model-only \
    --trust-remote-code \
    --enforce-eager \
  > "$GR_REPEAT_ROOT/startup/container-id.txt"; then
  GR_CONTAINER_CREATED=1
else
  GR_DOCKER_RC=$?
fi
printf '%s\n' "$GR_DOCKER_RC" > "$GR_REPEAT_ROOT/startup/docker-run.rc"
test "$GR_DOCKER_RC" -eq 0
test "$(docker inspect -f '{{ index .Config.Labels "gr308.baseline.run" }}' "$GR_CONTAINER")" = "$GR_RUN_ID"
printf '%s\n' "$GR_CONTAINER" >> "$GR_RUN_ROOT/manifest/owned-containers.txt"

GR_READY=0
for GR_CHECK in $(seq 1 540); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/health > /dev/null; then
    GR_READY=1
    date +%s%N > "$GR_REPEAT_ROOT/startup/ready-ns.txt"
    break
  fi
  test "$(docker inspect -f '{{.State.Running}}' "$GR_CONTAINER")" = true || break
  sleep 5
done
test "$GR_READY" -eq 1
curl -fsS http://127.0.0.1:8000/v1/models > "$GR_REPEAT_ROOT/startup/models.json"
docker inspect "$GR_CONTAINER" > "$GR_REPEAT_ROOT/startup/container-ready.json"

GR_BLOCK_FAILED=0
for GR_CONCURRENCY in $GR_ORDER; do
  case "$GR_CONCURRENCY" in
    1) GR_REQUESTS=12 ;;
    8) GR_REQUESTS=16 ;;
    16) GR_REQUESTS=32 ;;
    32) GR_REQUESTS=64 ;;
    64) GR_REQUESTS=128 ;;
    *) exit 92 ;;
  esac
  GR_CASE_DIR="$GR_REPEAT_ROOT/c${GR_CONCURRENCY}"
  mkdir "$GR_CASE_DIR"
  curl -fsS http://127.0.0.1:8000/metrics > "$GR_CASE_DIR/metrics-before.prom"
  docker inspect "$GR_CONTAINER" > "$GR_CASE_DIR/container-before.json"
  stdbuf -oL fantgpu-smi \
    --query-gpu timestamp,index,pci.bus_id,serial_number,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
    --format csv,noheader,nounits -lms 1000 \
    > "$GR_CASE_DIR/telemetry.csv" 2> "$GR_CASE_DIR/telemetry.err" &
  GR_TELEMETRY_PID=$!

  GR_WARMUP_RC=0
  if timeout --signal=INT --kill-after=30s 3600s \
       python3 "$GR_RUN_ROOT/tools/closed_loop_safe_client.py" \
       --endpoint http://127.0.0.1:8000/v1/completions \
       --model qwen3.8-27b \
       --deployment-id "vllm-fant-eager-${GR_REPEAT}-c${GR_CONCURRENCY}-warmup" \
       --input-tokens 1024 --output-tokens 64 \
       --concurrency "$GR_CONCURRENCY" --requests "$GR_CONCURRENCY" \
       --warmup 0 --prompt-token-id 64 --timeout-s 3600 \
       --output "$GR_CASE_DIR/warmup-requests.jsonl" \
       --summary "$GR_CASE_DIR/warmup-summary.json" \
       > "$GR_CASE_DIR/warmup-client.log" 2>&1; then
    GR_WARMUP_RC=0
  else
    GR_WARMUP_RC=$?
  fi
  printf '%s\n' "$GR_WARMUP_RC" > "$GR_CASE_DIR/warmup-client.rc"

  GR_CASE_RC=0
  if [ "$GR_WARMUP_RC" -eq 0 ]; then
    sleep 10
    curl -fsS --max-time 3 http://127.0.0.1:8000/health > /dev/null
    if timeout --signal=INT --kill-after=30s 7200s \
         python3 "$GR_RUN_ROOT/tools/closed_loop_safe_client.py" \
         --endpoint http://127.0.0.1:8000/v1/completions \
         --model qwen3.8-27b \
         --deployment-id "vllm-fant-eager-${GR_REPEAT}-c${GR_CONCURRENCY}" \
         --input-tokens 1024 --output-tokens 256 \
         --concurrency "$GR_CONCURRENCY" --requests "$GR_REQUESTS" \
         --warmup 0 --prompt-token-id 64 --timeout-s 7200 \
         --output "$GR_CASE_DIR/requests.jsonl" \
         --summary "$GR_CASE_DIR/summary.json" \
         > "$GR_CASE_DIR/client.log" 2>&1; then
      GR_CASE_RC=0
    else
      GR_CASE_RC=$?
    fi
  else
    GR_CASE_RC=125
  fi
  printf '%s\n' "$GR_CASE_RC" > "$GR_CASE_DIR/client.rc"
  kill "$GR_TELEMETRY_PID" 2>/dev/null || true
  wait "$GR_TELEMETRY_PID" 2>/dev/null || true
  GR_TELEMETRY_PID=''
  curl -fsS http://127.0.0.1:8000/metrics > "$GR_CASE_DIR/metrics-after.prom"
  docker inspect "$GR_CONTAINER" > "$GR_CASE_DIR/container-after.json"
  if [ "$GR_WARMUP_RC" -ne 0 ] || [ "$GR_CASE_RC" -ne 0 ]; then
    GR_BLOCK_FAILED=1
    break
  fi
done

docker logs --timestamps "$GR_CONTAINER" > "$GR_REPEAT_ROOT/startup/server.log" 2>&1
docker inspect "$GR_CONTAINER" > "$GR_REPEAT_ROOT/startup/container-final.json"
test "$(docker inspect -f '{{ index .Config.Labels "gr308.baseline.run" }}' "$GR_CONTAINER")" = "$GR_RUN_ID"
docker stop --time 120 "$GR_CONTAINER" > "$GR_REPEAT_ROOT/startup/docker-stop.txt"
GR_CONTAINER_CREATED=0
trap - ERR INT TERM
test "$GR_BLOCK_FAILED" -eq 0
```

每个 block 单独执行上面代码，并在开始前重新执行第 5 节的上下文恢复块。不得一次修改 `GR_REPEAT` 跳过已计划的 r1/r3/r5；触发补测时先完成三次统计，再依次执行 r2、r4。

- [ ] **Step 4：使用预注册的自适应重复规则**

3 次后，如任一并发点 run-level validated output throughput 的 CV>5%，增加一个完整 block；最多增加到 5 次。5 次后 CV 5–10% 标 `high-variance`，>10% 标 `unstable`，不得宣称稳定的性能领先。

### Task 10：执行稳定性阶段

**Files:**

- Create: `$GR_RUN_ROOT/eager/stability/c64/requests.jsonl`
- Create: `$GR_RUN_ROOT/eager/stability/c64/summary.json`
- Create: `$GR_RUN_ROOT/eager/stability/c64/telemetry.csv`
- Create: `$GR_RUN_ROOT/eager/stability/c64/server.log`

- [ ] **Step 1：从新容器和新 engine 进程开始**

创建 `$GR_RUN_ROOT/eager/stability/startup` 和 `c64`，并用以下冻结命令启动唯一 stability 容器：

```bash
test ! -e "$GR_RUN_ROOT/eager/stability"
mkdir "$GR_RUN_ROOT/eager/stability"
mkdir "$GR_RUN_ROOT/eager/stability/startup" "$GR_RUN_ROOT/eager/stability/c64"
GR_STABILITY_ROOT="$GR_RUN_ROOT/eager/stability"
GR_CASE_DIR="$GR_STABILITY_ROOT/c64"
GR_CONTAINER="${GR_RUN_ID}-eager-stability"
printf '%s\n' "$GR_CONTAINER" > "$GR_STABILITY_ROOT/startup/container-intent.txt"
GR_START_NS="$(date +%s%N)"
printf '%s\n' "$GR_START_NS" > "$GR_STABILITY_ROOT/startup/start-ns.txt"
GR_CONTAINER_CREATED=0
GR_TELEMETRY_PID=''
trap 'if [ -n "$GR_TELEMETRY_PID" ]; then kill "$GR_TELEMETRY_PID" 2>/dev/null || true; wait "$GR_TELEMETRY_PID" 2>/dev/null || true; fi; if [ "$GR_CONTAINER_CREATED" -eq 1 ]; then docker stop --time 120 "$GR_CONTAINER" >/dev/null 2>&1 || true; fi' ERR INT TERM

GR_DOCKER_RC=0
if docker run -d \
  --name "$GR_CONTAINER" \
  --hostname user \
  --label "gr308.baseline.run=$GR_RUN_ID" \
  --privileged --runtime=fantasy --ipc=host --network=host --restart=no \
  --security-opt label=disable \
  -e CUDA_VISIBLE_DEVICES=0,2,4,6 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
  -v /usr/local/fant/include:/usr/local/fant/include:ro \
  -v /usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro \
  -v /data/models/Qwen3.8-27B:/model:ro \
  sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc \
  vllm serve /model \
    --host 127.0.0.1 --port 8000 --served-model-name qwen3.8-27b \
    --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.40 --block-size 800 \
    --language-model-only --trust-remote-code --enforce-eager \
  > "$GR_STABILITY_ROOT/startup/container-id.txt"; then
  GR_CONTAINER_CREATED=1
else
  GR_DOCKER_RC=$?
fi
printf '%s\n' "$GR_DOCKER_RC" > "$GR_STABILITY_ROOT/startup/docker-run.rc"
test "$GR_DOCKER_RC" -eq 0
test "$(docker inspect -f '{{ index .Config.Labels "gr308.baseline.run" }}' "$GR_CONTAINER")" = "$GR_RUN_ID"
printf '%s\n' "$GR_CONTAINER" >> "$GR_RUN_ROOT/manifest/owned-containers.txt"

GR_READY=0
for GR_CHECK in $(seq 1 540); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/health > /dev/null; then
    GR_READY=1
    date +%s%N > "$GR_STABILITY_ROOT/startup/ready-ns.txt"
    break
  fi
  test "$(docker inspect -f '{{.State.Running}}' "$GR_CONTAINER")" = true || break
  sleep 5
done
test "$GR_READY" -eq 1
curl -fsS http://127.0.0.1:8000/v1/models > "$GR_STABILITY_ROOT/startup/models.json"
docker inspect "$GR_CONTAINER" > "$GR_STABILITY_ROOT/startup/container-ready.json"
```

- [ ] **Step 2：运行 128→64、C64、N=1024**

该请求量按当前保守速度约 75–100 分钟，并计划提供 1024 个终态用于描述 P99。执行命令：

```bash
GR_STABILITY_WARMUP_RC=0
if timeout --signal=INT --kill-after=30s 1800s \
     python3 "$GR_RUN_ROOT/tools/closed_loop_safe_client.py" \
     --endpoint http://127.0.0.1:8000/v1/completions \
     --model qwen3.8-27b \
     --deployment-id vllm-fant-eager-stability-c64-warmup \
     --input-tokens 128 --output-tokens 64 \
     --concurrency 64 --requests 64 \
     --warmup 0 --prompt-token-id 64 --timeout-s 1200 \
     --output "$GR_CASE_DIR/warmup-requests.jsonl" \
     --summary "$GR_CASE_DIR/warmup-summary.json" \
     > "$GR_CASE_DIR/warmup-client.log" 2>&1; then
  GR_STABILITY_WARMUP_RC=0
else
  GR_STABILITY_WARMUP_RC=$?
fi
printf '%s\n' "$GR_STABILITY_WARMUP_RC" > "$GR_CASE_DIR/warmup-client.rc"
test "$GR_STABILITY_WARMUP_RC" -eq 0
sleep 10
curl -fsS --max-time 3 http://127.0.0.1:8000/health > /dev/null

stdbuf -oL fantgpu-smi \
  --query-gpu timestamp,index,pci.bus_id,serial_number,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
  --format csv,noheader,nounits -lms 1000 \
  > "$GR_CASE_DIR/telemetry.csv" 2> "$GR_CASE_DIR/telemetry.err" &
GR_TELEMETRY_PID=$!
curl -fsS http://127.0.0.1:8000/metrics > "$GR_CASE_DIR/metrics-before.prom"
docker inspect "$GR_CONTAINER" > "$GR_CASE_DIR/container-before.json"

GR_STABILITY_RC=0
if timeout --signal=INT --kill-after=30s 9000s \
     python3 "$GR_RUN_ROOT/tools/closed_loop_safe_client.py" \
     --endpoint http://127.0.0.1:8000/v1/completions \
     --model qwen3.8-27b \
     --deployment-id vllm-fant-eager-stability-c64 \
     --input-tokens 128 --output-tokens 64 \
     --concurrency 64 --requests 1024 \
     --warmup 0 --prompt-token-id 64 --timeout-s 9000 \
     --output "$GR_CASE_DIR/requests.jsonl" \
     --summary "$GR_CASE_DIR/summary.json" \
     > "$GR_CASE_DIR/client.log" 2>&1; then
  GR_STABILITY_RC=0
else
  GR_STABILITY_RC=$?
fi
printf '%s\n' "$GR_STABILITY_RC" > "$GR_CASE_DIR/client.rc"

if [ "$GR_STABILITY_RC" -eq 0 ] && \
   [ "$(jq -r '.measurement_duration_s < 3600' "$GR_CASE_DIR/summary.json")" = true ]; then
  GR_EXTENSION_RC=0
  if timeout --signal=INT --kill-after=30s 4500s \
       python3 "$GR_RUN_ROOT/tools/closed_loop_safe_client.py" \
       --endpoint http://127.0.0.1:8000/v1/completions \
       --model qwen3.8-27b \
       --deployment-id vllm-fant-eager-stability-c64-extension \
       --input-tokens 128 --output-tokens 64 \
       --concurrency 64 --requests 512 \
       --warmup 0 --prompt-token-id 64 --timeout-s 4500 \
       --output "$GR_CASE_DIR/extension-requests.jsonl" \
       --summary "$GR_CASE_DIR/extension-summary.json" \
       > "$GR_CASE_DIR/extension-client.log" 2>&1; then
    GR_EXTENSION_RC=0
  else
    GR_EXTENSION_RC=$?
  fi
  printf '%s\n' "$GR_EXTENSION_RC" > "$GR_CASE_DIR/extension-client.rc"
fi

kill "$GR_TELEMETRY_PID" 2>/dev/null || true
wait "$GR_TELEMETRY_PID" 2>/dev/null || true
GR_TELEMETRY_PID=''
curl -fsS http://127.0.0.1:8000/metrics > "$GR_CASE_DIR/metrics-after.prom"
docker inspect "$GR_CONTAINER" > "$GR_CASE_DIR/container-after.json"
```

9000 秒到期时 `timeout` 先向客户端发送 SIGINT，30 秒后仍未退出则强制终止。客户端终止后不再发新请求；只统计已写入 JSONL 的终态，仍在途的请求标 `unknown-after-client-termination`，不得声称已成功或已正常取消。主段少于 1000 个成功终态时不发布 P99。

- [ ] **Step 3：检查漂移**

按 5 分钟窗口绘制 output tok/s、TTFT、TPOT、显存、物理卡利用率、温度与功率；比较首末各 10 分钟。5 分钟窗口只用于趋势图，不当作独立统计重复。

- [ ] **Step 4：运行性能后哨兵**

```bash
GR_POST_SENTINEL_RC=0
if python3 "$GR_RUN_ROOT/tools/run_sentinels.py" \
     --endpoint http://127.0.0.1:8000/v1/chat/completions \
     --model qwen3.8-27b \
     --cases "$GR_RUN_ROOT/tools/sentinel_cases.json" \
     --output "$GR_RUN_ROOT/eager/correctness-post" \
     --max-tokens 64 --timeout-s 90 \
     > "$GR_RUN_ROOT/eager/correctness-post.run.log" 2>&1; then
  GR_POST_SENTINEL_RC=0
else
  GR_POST_SENTINEL_RC=$?
fi
printf '%s\n' "$GR_POST_SENTINEL_RC" > "$GR_RUN_ROOT/eager/correctness-post.rc"
docker logs --timestamps "$GR_CONTAINER" > "$GR_STABILITY_ROOT/startup/server.log" 2>&1
docker inspect "$GR_CONTAINER" > "$GR_STABILITY_ROOT/startup/container-final.json"
test "$(docker inspect -f '{{ index .Config.Labels "gr308.baseline.run" }}' "$GR_CONTAINER")" = "$GR_RUN_ID"
docker stop --time 120 "$GR_CONTAINER" > "$GR_STABILITY_ROOT/startup/docker-stop.txt"
GR_CONTAINER_CREATED=0
trap - ERR INT TERM
test "$GR_STABILITY_RC" -eq 0
test "$GR_POST_SENTINEL_RC" -eq 0
```

后哨兵必须再次 12/12 通过；失败时所有性能 raw 仍保留，但 Lane E 总状态标为 `post-correctness-failed`。

### Task 11：聚合与统计

**Files:**

- Create: `$GR_RUN_ROOT/analysis/run-level.csv`
- Create: `$GR_RUN_ROOT/analysis/request-level.csv`
- Create: `$GR_RUN_ROOT/analysis/telemetry-physical.csv`
- Create: `$GR_RUN_ROOT/analysis/validation.json`
- Create: `$GR_RUN_ROOT/REPORT_ZH.md`

- [ ] **Step 1：计算主指标**

主指标为 validated output throughput：

```text
通过 HTTP/SSE/token/终态验证的输出 token 总数 / 整段 monotonic wall-clock 秒数
```

失败请求已生成的部分 token 不进分子，但失败前耗时保留在分母。另报 raw output tok/s 和 conditional-on-success tok/s，但不能用后者掩盖失败。

- [ ] **Step 2：计算次指标**

- request/s、input tok/s、total tok/s。
- TTFT、TPOT、E2E 的 P50/P95；P95 在同一 lane、workload、concurrency、phase 累计成功请求少于 400 时标为描述性，P99 只在同一组合内累计成功请求 ≥1000 时发布，禁止跨 workload 或 phase 拼样本。
- TPOT 固定为 `(E2E−TTFT)/(output_tokens−1)`；它是客户端近似，不是真实逐 token ITL。
- engine-start-to-ready、成功率、错误/超时/OOM/restart 计数。
- 4 张物理卡去重后的显存、利用率、温度和功率；按功率时间积分计算设备侧 J/有效输出 token，明确不是整机能耗。

- [ ] **Step 3：run-level 统计**

吞吐的统计单位是完整的 `engine × concurrency × engine-cold block`，不能把同一 block 内请求当成吞吐独立样本，也不能把这些 host-cache-warm block 夸大为完全独立的机器重复。n≥3 时报告全部原始点、mean、median、std、CV 和描述性区间：

```text
95% t-CI = mean ± t(0.975, n−1) × sample_std / sqrt(n)
```

n=3 时区间只作描述，不据此声称约 5% 的小差异。未来与自研框架比较时，使用同一 repeat 顺序的配对 log ratio，并在指数变换后报告比值区间。

- [ ] **Step 4：成功率区间**

成功率报告 Wilson 区间；零失败时附 rule-of-three：95% 失败率上界约为 `3/N`。慢值和长尾不做 IQR、Grubbs 或人工剔除。

### Task 12：异常分类与重跑规则

- [ ] **Step 1：仅允许替换外部测量故障**

客户端进程崩溃、结果文件损坏、非 baseline GPU 进程闯入、端口被外部进程占用、遥测采集器损坏，可将该 run 标为 `invalid-measurement` 后追加 replacement。原目录、原始证据和作废理由必须保留。

- [ ] **Step 2：真实系统失败不得删除或替换**

HTTP 错误、超时、token mismatch、OOM、FHCCL/driver reset、引擎 restart、错误输出、内存持续增长均属于 baseline 的可靠性结果。恢复后可追加 `diagnostic-rerun`，但原失败和诊断结果必须同时报告。

- [ ] **Step 3：执行停止条件**

发生以下任一项立即停止新请求、只等待/取消本轮自有请求并取证：

- driver reset、设备消失或 FHCCL crash；
- 容器自动 restart；
- 任一请求 token 契约错误；
- 单个 Rapid 点超过 30 分钟、单个 Formal 点超过 2 小时或稳定性阶段超过 150 分钟；长请求期间不使用“120 秒无完整请求”作为停止条件；
- 物理显存余量不足 2 GiB；
- 任一物理卡温度达到 80°C，或日志出现 thermal throttle；
- 结果盘可用空间低于 100 GiB。

禁止 GPU reset、删除历史 STOP/HALT、全局杀进程、升级驱动/torch/vLLM、修改路由或在失败后临时调参。

### Task 13：收尾与恢复

**Files:**

- Create: `$GR_RUN_ROOT/manifest/host-after.txt`
- Create: `$GR_RUN_ROOT/SHA256SUMS`

- [ ] **Step 1：停止 Lane E**

逐行读取 `$GR_RUN_ROOT/manifest/owned-containers.txt`，同时校验名称前缀、label 和 image ID，只停止仍在运行的本轮容器：

```bash
while IFS= read -r GR_OWNED_CONTAINER; do
  case "$GR_OWNED_CONTAINER" in
    "${GR_RUN_ID}"-eager-*) ;;
    *) printf 'unsafe container name: %s\n' "$GR_OWNED_CONTAINER" >&2; exit 94 ;;
  esac
  test "$(docker inspect -f '{{ index .Config.Labels "gr308.baseline.run" }}' "$GR_OWNED_CONTAINER")" = "$GR_RUN_ID"
  test "$(docker inspect -f '{{.Image}}' "$GR_OWNED_CONTAINER")" = 'sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc'
  if [ "$(docker inspect -f '{{.State.Running}}' "$GR_OWNED_CONTAINER")" = true ]; then
    docker stop --time 120 "$GR_OWNED_CONTAINER" > /dev/null
  fi
done < "$GR_RUN_ROOT/manifest/owned-containers.txt"
```

各阶段已在各自 startup 目录保存最终 inspect 和日志；若缺失，Task 13 以新的唯一文件名补采。证据审计前不执行 `docker rm`。

- [ ] **Step 2：确认回到用户指定的全停止状态**

确认目标端口无监听、无 benchmark/sampler 进程、4 张卡无实验进程。除非用户届时另行要求，不能自动恢复此前的 Qwen、R1、Envoy/EPP 或 LMCache 服务。保存证据：

```bash
{
  date -u '+%FT%TZ'
  docker ps -a --no-trunc
  ss -ltnp
  pgrep -af 'vllm|sglang|closed_loop_safe_client|fantgpu-smi.*lms' || true
  fantgpu-smi
  df -B1 /data
} > "$GR_RUN_ROOT/manifest/host-after.txt" 2>&1
GR_FINAL_STATUS="$(jq -r '.overall_status' "$GR_RUN_ROOT/analysis/validation.json")"
case "$GR_FINAL_STATUS" in
  complete|partial|unsupported|startup-failed|correctness-failed|post-correctness-failed|stability-failed) ;;
  *) printf 'invalid final status: %s\n' "$GR_FINAL_STATUS" >&2; exit 95 ;;
esac
jq -cn \
  --arg status "$GR_FINAL_STATUS" \
  --arg updated_at "$(date -u +%FT%TZ)" \
  '{schema_version:1,status:$status,updated_at_utc:$updated_at}' \
  >> "$GR_RUN_ROOT/STATUS.events.jsonl"
```

- [ ] **Step 3：生成完整性清单**

```bash
cd "$GR_RUN_ROOT"
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
```

Expected: 全部文件 `OK`；`STATUS.events.jsonl` 最后一行由 Task 13 使用 `jq -cn ... >>` 追加，状态为 `complete`、`partial`、`unsupported` 或明确失败状态之一，不能只有模糊的 `done`。最后验证 active link 仍精确指向本次 run，再执行 `unlink "$GR_ACTIVE_LINK"`；不得删除 run 目录或任何容器。

## 8. 报告必须包含的表格

### 8.1 环境与身份表

记录硬件、驱动、共享库、容器 ID、image ID、RepoDigest、完整 argv/env、模型/Tokenizer SHA、设备映射、CPU/NUMA、缓存与 graph/eager 状态。

### 8.2 Rapid 与 Formal 主结果表

每行至少包括：lane、workload、repeat、concurrency、offered/sent/completed/ok/failed、duration、validated output tok/s、raw output tok/s、request/s、TTFT/TPOT/E2E P50/P95、engine-start-to-ready、RestartCount。

### 8.3 资源表

按物理卡列出显存峰值、GPU 利用率 P50/P95、功率均值/峰值、温度均值/峰值、遥测覆盖率和设备侧 J/有效输出 token。

### 8.4 失败与排除表

逐项列出 upstream `unsupported`、startup/correctness/stability failure、invalid measurement、diagnostic rerun 和被排除的预跑数据。缺失结果显示 `N/A + reason`，不得填 0。

## 9. 可选 Graph32 Lane

Lane G 只在 Lane E 核心 baseline 完成且用户另行授权后执行：

- 使用同一模型、设备、dtype、TP、请求、顺序和证据格式。
- 保留 Graph bucket 1/2/4/8/16/32；不得偷偷加入 Graph64。
- C64 明确标为 bucket 外 saturation/stress，不与 Lane E 的 C64 做“仅 graph 单变量”因果解释。
- 完整重复 Rapid、Formal 和 stability；不把历史约 5.29/33.32/42.33/51.45/46.32 tok/s 混入新统计。
- 额外预算 14–22 小时；若触发一个额外 Formal block 增加 3–4.5 小时，触发两个增加 6–9 小时。

## 10. SGLang 未来接入条件

SGLang 当前不在本轮计划中。未来只有同时满足以下条件，才能复制本计划创建独立的 SGLang baseline 文档：

- PyTorch PrivateUse1/Fantasy 设备能被 `torch.get_device_module('fant')` 正确注册。
- SGLang worker 能完成 GR308 设备选择、模型加载、TP4 和 FHCCL 初始化。
- `/v1/completions` 支持 list-of-token-ID prompt、streaming、`ignore_eos` 和 usage 回传。
- radix/prefix cache 能显式关闭或用经审计的等价策略处理。
- 通过同一模型/Tokenizer SHA、12 条哨兵、128→64 Rapid 和 1024→256 Formal 门禁。
- 适配代码、wheel、镜像、共享库和启动参数均冻结；不能把调试版本与本次 vLLM 运行时跨日期直接比较。

## 11. 审阅清单

- [ ] 范围只有 vLLM，SGLang 仅是未来条件。
- [ ] 上游兼容性与厂商适配性能结果没有混称。
- [ ] 所有阶段都有耗时、输入、输出、并发、请求数和停止条件。
- [ ] C1 预跑只用于排期，整个旧目录明确排除。
- [ ] 4 个物理卡与 8 个逻辑 function 未重复计数。
- [ ] Rapid 不给置信区间；Formal 以 engine-cold block 为统计单位并只给描述性区间。
- [ ] P95/P99 的样本量限制已写明。
- [ ] 失败不会因重跑而被覆盖。
- [ ] 结果落 `/data` 且文件不覆盖，完整 SHA256 可复算。
- [ ] 实验结束后的默认状态是全部服务停止。
- [ ] 文档没有授权立即执行。

## 12. 执行交接

计划评审通过后有两种执行方式：

1. **Subagent-Driven（推荐）**：按 Task 1–13 分派独立子任务，主代理在兼容性、正确性、Rapid 重估、每个 Formal block 和收尾阶段逐次验收。
2. **Inline Execution**：在当前任务中按批次执行，每个阶段结束后输出 checkpoint，再继续下一阶段。

在用户选择并再次授权前，保持 `PLANNED / NOT STARTED`，不启动任何服务或 benchmark。
