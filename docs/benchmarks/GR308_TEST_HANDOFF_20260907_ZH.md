# GR308 测试交接文档

版本：2026-09-07；主体现场基准：北京时间 **11:11:17**，网络恢复后最新复核：**12:28:14**（UTC 04:28:14）。  
对象：`user@192.0.2.11`；本地工作区：`/Users/USER/Documents/infra`。  
交接性质：完整阶段交接，**不是生产验收通过报告，也不是自动续测或停服授权**。

交付时补充：12:24–12:26曾因网络中断导致同步失败；用户恢复连接后，12:28重新核实路由utun8、SSH认证、五端口健康、九容器身份与排空均通过。主体表格仍保留11:11采样时刻，最新恢复快照见第15节；不把恢复事实倒填到中断时刻。

## 0. 接手人先看这一页

- 现网仍为 Qwen BF16 TP4 和 R1 BF16 TP1。五个入口健康、九个受保护服务容器的身份/启动时间/配置哈希与冻结基线一致；两个后端连续空闲检查通过。
- 生产工作表共 **31 项：1 项按限定范围完成，22 项部分完成或已有问题，8 项待完成**。还有 30 项未完全闭环，不是 30 项完全没做。唯一闭环项是本轮限定范围的 XFER-01；XFER-02 的同步矩阵虽完成，原事件协议仍未闭环。
- 09-06/07 新生产套件累计 **567 个实际测试请求**，包括故意错误请求、主动取消和一个 R1 超时，不能写成 567 个正常回答全部成功。健康/metrics GET 不计入；早期不同实验的请求不并入此数。
- GPTQ 数值/打包路径、Attention 索引上下文故障、原 pinned 事件问题、FHCCL 历史初始化崩溃、Qwen 两次自动重启、监控重启循环均未关闭。所有 STOP/HALT 和原失败证据保留。
- 本次交接没有提交新推理请求或 native GPU 测试。11:11 快照未发现已登记的 P2P/路线对照/soak/Attention 测试进程，两个已完成路线控制器 PID 均不存在；P2P 复制运行目录无结果、完整实现冻结清单不存在。
- 用户已明确撤销额度查询和低于 50% 停止条件。额度自动任务此前已暂停；**不要重新启用额度检查或伪造旧执行器所需的新额度记录**。健康、身份、排空、温度/OOM、只取消自有请求/进程及绝对截止时间仍保留。
- 最近完整 CPU 回归：服务器 496 项通过；本次交接本地 **513 项通过，3.208 秒**。新增 17 项是 P2P 离线测试，不是 P2P 设备测试，尚未在服务器运行。
- 模型支线：H3 与 Wan2.2 的下载最终记录均为全文件验证通过；DPSKV4 下载最终记录不完整且下载器退出 2；H3 无完整合格 DiT、无视频成品，Wan2.2 无生成 E2E。文件齐全不等于模型可用。

入口文件：

1. 本交接文档——当前结论、历史沿革、未完项和接手顺序。
2. [现场只读快照](/Users/USER/Documents/infra/benchmarks/gr308-handoff-20260907/handoff-live-snapshot-20260907.json)——当前容器/库/端口/SMI/下载器状态；采集过程不加载 GPU runtime。
3. [证据索引](/Users/USER/Documents/infra/benchmarks/gr308-handoff-20260907/EVIDENCE_INDEX.json)——相关报告、源码、审计文件的实际路径、大小和 SHA256。
4. [生产原工作表](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/README.md)与[原机器进度](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/PROGRESS.json)——截至 09:51–09:56 的阶段记录；之后 P2P 准备和本次只读发现以本文补充，原文件不冒充 11:11 新快照。

## 1. 范围、证据等级与计数规则

本文覆盖 09-04 至 09-07 与当前 GR308 工作相关的服务部署、量化与模型适配、优化实验、基础 kernel/dense/通信/传输、生产 API/性能/资源/可靠性、异常与恢复、未执行准备、资产位置和后续验收。8×5090 历史工作仅作为方法来源，不把其 CUDA 路径、拓扑或速度当作 GR308 实测。

证据解释顺序：冻结原始请求/张量/日志及 manifest → 独立回放 → 阶段报告 → 本文汇总。发生矛盾时保留原记录、查实际时间及作用域；不把早期“未下载/只测 4 KiB”覆盖到后来完成的下载/64 MiB 矩阵，也不把后来的成功倒填进早期失败。

术语：

|术语|本文含义|
|---|---|
|执行完成|预定尝试已有终态，可以包含失败和未发送；不自动等于功能通过|
|回放通过|原始证据完整且被独立解析/复算；故障证据也可能回放有效|
|数值通过|在指定输入、dtype、布局和预定容差内通过，不是所有输入的误差保证|
|恢复通过|指定时点身份/健康/排空或恢复哨兵通过，不保证无业务影响或长期稳定|
|生产资格|业务质量、SLO、容量、可靠性和操作要求一起通过；当前没有获得|
|case / batch / request|设备用例、计时批次、HTTP 请求分开；批内操作次数不当独立样本|
|GB / GiB|GB 为十进制字节，GiB 为 2^30 字节；容量不能按 sibling 翻倍|

文件名时间有时为预设标签，实际生成时间以 JSON 内时间/事件为准。例如路线描述性汇总名含 01:44 UTC，实际生成 01:49:19 UTC。不同阶段不是同一个连续时间窗，不拼接生成未经预登记的 CI/P99。

## 2. 当前环境与服务交接

### 2.1 主机、设备和 NUMA

|项目|已核实值/限制|
|---|---|
|主机|Ubuntu 22.04.5，kernel 5.15.0-190-generic；历史集群为单节点 K3s，不是跨机集群|
|CPU / 内存|双路 AMD EPYC 9654，224 个逻辑 CPU，约 1 TiB RAM，两个 NUMA 节点|
|CPU NUMA 0|0–55、112–167|
|CPU NUMA 1|56–111、168–223|
|物理加速卡|4×GR308，每张 57,344 MiB＝56 GiB，总计 224 GiB GDDR6|
|逻辑设备|8 个 PCI function，同卡 .0/.2 共享物理资源；另有显示控制器不算加速卡|
|PCIe|既有 sysfs/SMI 报告 Gen5 x16、32 GT/s；四张 GPU 均归 NUMA 0|
|驱动 / SMI|11:11 再核实：1.1.0.19（2026-08-07）/2.2.0.9-linux|
|软件栈|既有现场记录：torch_fant 1.1.0、iFormer 1.1.0.20、vllm_fant 1.1.0.20、vLLM 0.18.0+empty；不得用包名代替实际库哈希|
|数据盘|11:11 可用 2,337,059,532,800 字节，约 2.126 TiB；总量约 3.5 TiB|
|本地盘|本次 df 快照约 28 GiB 可用；旧报告约 12 GiB 是历史值，本次未删除用户数据|

|物理卡|逻辑设备|PCI functions|11:11 各逻辑 memory.used|
|---|---|---|---|
|0|0 / 1|0000:2e:00.0 / .2|25,668 / 20,181 MiB|
|1|2 / 3|0000:34:00.0 / .2|25,668 / 540 MiB|
|2|4 / 5|0000:37:00.0 / .2|25,668 / 540 MiB|
|3|6 / 7|0000:38:00.0 / .2|25,436 / 540 MiB|

上表为设备报表读数，不能将每行两个 56 GiB 容量相加，也不把逻辑 memory.used 简单求和当去重后的物理分配账本。11:11 温度 48–51°C，GPU utilization 均报 0%；.0 功率为 261/272/274/267 W，.2 为 N/A，不当 0 W。不是整机墙上功率或性能负载测量。

### 2.2 入口和模型身份

|访问入口|后端与实际身份|用途/限制|
|---|---|---|
|8000|Qwen 直连，服务别名 qwen3.8-27b|实际架构 Qwen3_5ForConditionalGeneration；BF16 TP4，逻辑 0/2/4/6|
|8080|Envoy/EPP → 同一个 Qwen 8000|交互入口，不是独立模型副本|
|8180|独立 Envoy/EPP → 同一个 Qwen 8000|后台连接池 cap4；不是严格全局 4 请求硬上限|
|8001|R1 直连，deepseek-r1-distill-qwen-7b|Qwen2ForCausalLM，BF16 TP1，逻辑 1，与 Qwen 共享第一张物理卡|
|8181|独立 Envoy/EPP → R1 8001|R1 入口，不静默替代 Qwen|

五端口 /health 和 /v1/models 均返回 200，模型映射正确；前后两次身份核验通过。Qwen 为 64 层，48 linear attention＋16 full attention，hidden 5120、intermediate 17408；R1 为 28 层。二者都不是 MoE，不能验收 MoE 路径。

Qwen 冻结配置为 BF16 TP4、max-model-len 32768、block-size 800、gpu-memory-utilization 0.40、FULL_DECODE_ONLY Graph 桶 [1,2,4,8,16,32]。现存 cache_config 指标证明 Qwen prefix caching=False、mamba cache mode=none；R1 prefix caching=True、block-size 16。LMCache 没有被证明通过 connector 接入当前模型，更没有完成命中/驱逐/一致性验收。

镜像基线：`sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc`。完整参数/镜像/容器配置哈希见冻结 inventory 和 deployment-metadata，而不是根据本文重新创建容器。

### 2.3 受保护的九个容器

以下 PID 仅为 **11:11 快照**，不能拿旧 PID 直接发送信号。

|容器|宿主 PID|启动时间 UTC|RestartCount|
|---|---:|---|---:|
|qwen38-27b-bf16|2018111|2026-09-06 16:25:23.948|2|
|r1-distill-7b|3576956|2026-09-05 06:28:00.453|0|
|llm-d-envoy|2023422|2026-09-06 16:26:26.515|0|
|llm-d-envoy-qwen-throughput|2023524|2026-09-06 16:26:26.656|0|
|llm-d-envoy-r1|3577440|2026-09-05 06:28:01.037|0|
|llm-d-epp|2016929|2026-09-04 14:38:47.460|0|
|llm-d-epp-qwen-throughput|2372475|2026-09-04 18:01:33.186|0|
|llm-d-epp-r1|2220217|2026-09-04 16:26:41.029|0|
|lmcache-cpu|1962528|2026-09-04 14:04:43.147|0|

“九容器通过”不是全主机所有容器都健康。快照还存在旧开发容器、转换容器和已退出下载器；`glm53-gptq-convert` 仍为 Up，进程表有 bash/python/tee，本次未核实它是否在有效推进转换，不标转换完成。没有清理这些历史容器。

### 2.4 实际库身份与监控问题

11:11 对冻结库文件再次核验成功。宿主库与容器库有差异，运行前必须区分；`/proc/2018111/root/...` 只在该容器进程身份保持时有效，不能修改 PID 常量绕过身份门。

|库|SHA256|
|---|---|
|宿主 libfanthip.so.1|46a7e43215b56f30ae245c56438a0faf2d1aeb22aabfda18a4cbdf14373f9af4|
|容器 iFormer/_C|048bba02826dcf8aab61d4378ba640e6f48e53c88362f4568ecfa590466f19d9|
|容器 libfant_python.so.1|e238225b15eddc9c6ac94e357bf128affdad80299008922d73fd288b7b5be0c2|
|容器 libfhblas.so.4.0|e80c6a6efd5e023e2c55e3d872fd9f01009d69311f9171958fe306950c0841b6|
|容器 libfantdnn.so.0.1.0|6a164f60a7a07a4a8bb20e9a5f5d8e4aad7a6c35a6a32182eafb780c0670aedd|
|容器 libfhccl.so|141f7c6d21ffcbb3fc5d2fb415c2091d3c3ca5e3610745198baf66db1765dfee|

monitor_server.service 原绑定 192.0.2.10:9110，历史日志为 EADDRNOTAVAIL 循环。11:11 NRestarts=3278，随后只读复查为 3284，虽然两次瞬时状态均 active/running，**仍不能判为监控健康**。此前误采入诊断副本的明文凭据行已脱敏；原服务日志未改。本文不包含密码、令牌或原敏感日志；对外分享前仍需审查主机地址、模型许可和原始日志。

## 3. 已执行工作的时间线

以下均为北京时间，日期以事件记录为准。

|时间/阶段|完成的工作|保留的限制|
|---|---|---|
|09-04～09-05 清晨|Qwen/R1、三条 llm-d 数据面；MTP/prefix/TP2/混跑试验|HAMi 无 GPU device-plugin 资格；LMCache 仅 metrics；候选收益未自动推广|
|09-05 早期 GPTQ 资格|真实 checkpoint/血缘检查，BF16/GPTQ 200 题，启动可靠性|GPTQ 0/200，资格失败；未继续 TP2/TP1/24 h|
|09-05 v2 诊断|真实 7 类 GPTQ 层、FHCCL 冷启动失败、BF16 恢复及开放到达|原生崩溃/数值错误未修；临时 cap8 未上线|
|09-05 吞吐与 Graph64|436 条闭环吞吐；Graph32/64 隔离比较和两组 200 题|Graph64 C64 raw 吞吐提高，但延迟门不合格；恢复原 Graph32|
|09-05～09-06 模型支线|DPSKV4 真实专家/投影；H3 平台、loader、完整组件；Wan 下载|DPSKV4 loader/NaN/clamp 问题；H3 DiT 数值失败；无视频 E2E|
|09-06 基础层|9 native kernel、45 dense、8 GPTQ case、13 次最小 FHCCL 冷启动|GPTQ 全失败；峰值/全通信矩阵未完成|
|09-07 00:21:55～00:26:27|已授权的一次限定维护，原 pinned 事件复测、新顺序测试、恢复|约 272 秒；Qwen 两次自动重启后恢复；该维护授权已使用|
|09-07 01:28 起|冻结新生产基线，API/SSE/错误/取消、算子、长度/闭环与开放到达|R1 超时、错误格式、布局/参数/RoPE 失败均保留|
|09-07 06:29 / 06:56|pageable / pinned 64 case 全矩阵复核|同步墙钟；pinned 中断及原事件问题保留|
|09-07 07:18～07:28|Attention 首项故障，60 秒自有进程截止，UV91 恢复哨兵|0 数值通过、99 未测；不自动重试|
|09-07 08:24:03～08:39:03|900 秒低速 Qwen 混合长度观察|30 请求、450 快照；不是长稳态/双模型资格|
|09-07 09:00:40～约09:07:24|随机化路线对照原段 48 条|第 49 条发送前旧额度准入停止；72 终态/HALT 保留|
|09-07 09:06:56|用户撤销额度监测要求|新版本明确采用无额度策略，不改冻结历史|
|09-07 09:34:15～约09:42:01|仅续测原未发送 24 条|总 72 次网络完成，两个时间窗、96 个终态，不是连续 A/B|
|09-07 10:18～交接前|peer API 二进制只读分析、复制探针构建和离线测试|96 项计划未开始；完整 controller/auditor 尚未冻结|
|09-07 11:11|交接只读现场核验|没有新推理/设备测试；快照不代表未来持续健康|

## 4. 早期服务、质量与优化实验

### 4.1 Qwen BF16 与 GPTQ

- BF16 TP4 三轮 C1/C2/C4/C8 output tok/s 均值为 5.284/9.514/17.210/33.158；早期 C16/C32 为 42.356/54.183。不同轮次负载与计时口径不完全相同，不能与后续所有数字直接混合。
- 200 题机械式数据集的 expected 独立验算通过；BF16 strict/semantic 都为 **127/200（63.5%）**，73 个错答不是格式修复或输出上限可以消除。此数据集不是客户领域质量集。
- GPTQ 五文件实际总大小 19,559,451,000 字节；tensor payload 19,559,146,976，头部/前缀 304,024；2,399 个 tensor，其中 799 个非量化 BF16 tensor 与原模型一致，载荷 6,908,929,504 字节。不要继续使用旧 index.total_size 充当精确 payload 的口径。
- GPTQ HTTP 可返回 200、raw 吞吐可更高，但 200 题均为 128 个“!”、**0/200**。不能算业务 goodput。三次资格启动仅一次 ready，其他为 FHCCL 初始化失败。
- FP16 activation 配合显式 4 GiB KV 的 TP4 smoke 能启动；自动 KV 预算曾导致 OOM。没有完整 FP16 模型资格，更没有 GPTQ TP2/TP1、1×TP4/2×TP2/4×TP1、1,000 业务任务或 24 h soak。
- v2 七种真实层 × M=1/4/32/128 共 28 组 packed execution 全部严重偏离 CPU 解包参考，relL2 12.5348–23.4782；同解包 dense FP16 约 0.000262–0.000273。无 NaN/Inf，不能统称 FP16 overflow。日志有 dma/tma 两类派发，均存在失败。
- 后续 FTUCA 8-case INT4 定位见第 6 节。不同阶段 28 与 8 不混成一次实验；没有修复 native 库。

依据：[GPTQ 执行记录](/Users/USER/Documents/infra/benchmarks/gr308_report_execution_20260905.md)、[v2 执行报告](/Users/USER/Documents/infra/benchmarks/gr308-iteration-v2-20260905/GR308_v2_EXECUTION_REPORT_ZH.md)、[厂商复现材料](/Users/USER/Documents/infra/benchmarks/gr308-iteration-v2-20260905/VENDOR_REPRO_ZH.md)。材料尚不能自动发给外部厂商。

### 4.2 低延迟、网关与混跑

|候选/实验|观察|交接结论|
|---|---|---|
|MTP|Qwen3_5MTP 初始化 native worker 退出|已关闭，非可用收益|
|prefix caching|1024-token 公共前缀 TTFT P50 3.478→1.227 秒；随机短请求 TPOT +8.2%、E2E +7.6%|回退默认；新 cache lane 尚未验收|
|BF16 TP2|C1 TPOT P95 279.62 ms，对比 TP4 约 175.66 ms|低延迟候选退化，不能预设缩 TP 获益|
|停止空闲 R1 的 A-B-B-A|Qwen 差异约在测量噪声内|不因空闲 R1 自动频繁停服|
|同一 Qwen 混 C1+C32|低并发 TTFT 约19.3秒、TPOT约567ms|两个入口不是两个独立推理池|
|8080 C1＋8180 C8，连接 cap4|4/4＋8/8 成功，交互 TPOT P95 188.50 ms|有限小样本可用；不是服务等级隔离认证|
|临时 cap8 8182 对照|0.4 req/s 下 TTFT P95 34.420→17.050s，TPOT 275.4→308.2ms；raw 15.751→18.458 tok/s|排队/执行取舍；24/24 仅过合成 batch 门，未推广|
|早期双模型|Qwen C1＋R1 C8、Qwen C32＋R1 C1 小样本成功|保留历史成功，不替代当前 REL-01 完整双模型 soak|

早期 cap4 曾用 max_requests=4 导致 3/8 失败，已回退至连接池策略。v2 后端采样仍达到 running=6，说明“cap4”不能宣称全局严格最多四个请求。TTFT 包含 prefill/排队等，不等于独立网关 queue time。

v2 开放到达：8080 的 0.15/0.25/0.35/0.45/0.55 req/s 共 105 条全部 HTTP 成功；合成标准门通过数为 9/9、15/15、3/21、0/27、0/33。8180 的 0.4 req/s 24/24 HTTP 成功，batch 门21/24；临时 cap8 另24/24。共 **153 条**，不是稳定容量或可售配额；没有开放到达随机复测/长稳态资格。

依据：[双模型与 TPOT 报告](/Users/USER/Documents/infra/benchmarks/qwen38_dual_model_tpot_report_zh_20260905.md)、[架构报告](/Users/USER/Documents/infra/benchmarks/fantasy_xpu_architecture_test_report_zh_20260905.md)、v2 报告。

### 4.3 吞吐及 Graph64

固定输入 128、输出64时，后续吞吐试验 C1/C8/C16/C32/C64 分别约 5.29/33.32/42.33/51.45/46.32 tok/s；C32 复测51.60、网关51.53。1024→256、C32仅37.58 tok/s，TTFT P95 132.73秒。正式 **436/436** 请求通过完整 HTTP/SSE/token 契约，前后语义各12/12；不能据此认证合成 prompt 的答案质量。

Graph32→Graph64 隔离单变量比较：

|档位|Graph32 output tok/s|Graph64 output tok/s|限制|
|---|---:|---:|---|
|C32|51.2518|51.4134|+0.32%，未见实质收益|
|C64|46.2712|57.1200|+23.45%，但 P95 TTFT 42.783s、TPOT827.44ms，仍不过延迟门|

四档正式 384/384 HTTP 成功。两个新进程各200题，均127/200且逐题答案一致，不是200题全正确。通信默认网卡路径首轮第三次两卡启动 SIGSEGV 后中止；后续隔离容器精确限定 ens5f0，两卡6/6、四卡6/6启动及 BF16 AllReduce/AllGather 小矩阵通过。有限通过不能证明修好历史 ncclTopoGetLocalNet，更不覆盖 H3 AllToAll。

试验后恢复原 Graph32；临时 cap8 同样未推广。NUMA-local 新进程初始化、scheduler token budget 粗筛、同四卡预算2×TP2仍是候选，不是完成收益。既有部分线程已绑定 Node0，不能写成“所有线程都未绑核”；numa_maps 为映射页，不是独立物理内存或 FHCCL 热缓冲证据。

依据：[吞吐报告](/Users/USER/Documents/infra/benchmarks/gr308-throughput-20260905/run-20260905T044530Z/THROUGHPUT_REPORT_ZH.md)、[优化评审](/Users/USER/Documents/infra/benchmarks/gr308-optimization-review-20260905/OPTIMIZATION_REVIEW_ZH.md)、[Graph64 E2E 报告](/Users/USER/Documents/infra/benchmarks/gr308-optimization-e2e-20260905/OPTIMIZATION_E2E_REPORT_ZH.md)。

## 5. 模型下载与兼容支线

### 5.1 当前下载记录与真实 E2E 分开

11:11 只读核对了下载器状态和既有 download-final.json；**本次没有重新全量哈希模型大权重**。下面的“已验证”指下载完成时的记录，不是本次再次扫描每个权重字节。

|模型/制品|固定版本与下载记录|运行/缺口|
|---|---|---|
|MiniMax-H3 W4A8 ConvRot|koongrizzly/MiniMax_H3_int4_W4A8_ConvRot_Pruned，revision 3c8619a7838f31df789e066a8460599ef9d285aa；5/5、34,054,225,887 bytes验证通过；下载器退出0|完整组件部分通过，DiT 数值门未过，无视频 E2E|
|H3 官方辅助配置|MiniMaxAI/MiniMax-H3，revision 42ed227ee7df40d41602854ae760620d6eb651fe；28文件验证通过|不含官方全量大权重|
|Wan2.2-TI2V-5B|Wan-AI/Wan2.2-TI2V-5B，revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e；13/13、34,201,537,147 bytes验证通过；下载器退出0|无视频生成/质量/耗时/吞吐成绩；旧“仍下载”已过时|
|DPSKV4 Flash GPTQ|sokada4/DeepSeek-V4-Flash-0731-GPTQ-Int4-V1，revision 4375393f25f4d901712d9eebd2da610cfa27980e；最终记录24/36权重分片完整，含partial123,970,239,998/152,752,856,854 bytes；全文件33/45通过|all_files_verified=false；下载器退出2，不能写成继续下载或整模可启动；该最终记录的具体退出根因本次未追查|
|Qwen3.5-4B|revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a；13必要文件、9,342,905,899 bytes的下载方案就绪|无下载最终记录，无GR308 E2E；旧网络阻塞已不代表当前主机不可达|
|GLM-5.3 BF16|历史仓库约1.403TiB，原下载按当时明确要求停止并删除约1.1TB|没有可恢复BF16完整模型或真模型E2E；本次未删除任何文件|
|GLM INT4/INT8Mix→GPTQ|旧int4下载器退出137，转换容器仍Up；本次未完整核验分片/转换结果|tiny dummy 可达不代表真权重/CPU offload/容量与语义通过|

DPSKV4 未完成字节约28.78GB只是旧最终记录中的续传缺口；partial未验证，不能当可加载权重。退出137本身不等于已证实OOM，退出2本身不定位根因。H3/Wan/DPSKV4最终记录原文、SHA和采集时刻都在现场快照内。

LFM2.5-1.2B-Instruct、Gemma4 E2B、Z-Image-Turbo 仅为历史候选，未形成本任务下载/部署/性能证据；不在本次自动扩大到这些模型。不同模型可能要求不同软件栈、许可和硬件能力，接手前另核实，不按旧推荐直接上线。

### 5.2 DPSKV4 真实权重实验

早期 tiny dummy DeepSeek-V4 8→8 token HTTP E2E约6.274s，仅证明结构链。真实权重检查进一步发现：

- 第0/2/5层、每层专家0/1，18个真实gate/up/down投影；M=1/4/32、FP16/BF16共18组FFN对照全部过预定门，FP16 relL2约0.0000191–0.0001249，BF16约0.003696–0.004043。
- 普通GPTQ线性层自动选择报 PlatformEnum.OOT KeyError。手工Triton候选真实q_a_proj：FP16 6/6通过，BF16 **0/6且NaN**；尚未接入实际loader/repack/模型链。
- 模型swiglu_limit=10，旧通用MoE未传clamp，高幅两层relL2约3.431/3.478；仅探针内补clamp后约0.002644/0.002757。不能将全局硬编码clamp当通用修复。
- 原始FP8设备类型不可用时采用BF16 sparse attention和byte-packed KV兼容路径，不是原生FP8硬件加速。
- 所有结论都不证明全部专家、学习路由、TP、稀疏attention、KV codec或完整语义通过。真实整模试验还需下载完整、兼容门和新的独占维护范围。

依据：[DPSKV4报告](/Users/USER/Documents/infra/benchmarks/gr308-dsv4-20260905/DPSKV4_STATUS_ZH.md)、[可复算结果](/Users/USER/Documents/infra/benchmarks/gr308-dsv4-20260905/derived-results.json)。

### 5.3 H3：已到完整组件，但未到合格视频

实验固定SGLang/Comfy Kitchen源码，显式进程内Fantasy平台覆盖；FP8 scale在CPU解码为FP32，原INT8打包权重不改。原始依赖/checkpoint未改写，没有全局升级生产栈；这不是原生INT4/INT8/FP8加速资格。

|验证|结果|不能据此推断|
|---|---|---|
|基础FP16/BF16 MM、SDPA、Conv3D|6/6通过|不是完整视频算子链|
|实验平台/跨流Event/H3 RoPE|7/7；实际56-head配置补测通过|不是所有事件路径正常，不能抵消原pinned事件故障|
|block0 W4A8完整投影|4投影×M1/4/32＝12/12；同量化CPU参考relL2=0|不是对未量化BF16模型等价|
|原生loader＋逐层卸载线性层|12case、24前向循环通过|不是完整offload视频吞吐|
|完整DiT严格加载|1132key，50层、200个W4A8线性层|加载成功不是数值通过|
|合成输入完整DiT|CPU finite；设备50层完成但视频/音频relL2=1.3241/1.0792，门0.05|明确未通过|
|完整文本编码器|1953参数；两轮50层均通过，relL2=.00181479，输出一致|只编码10-token短句，无正式prompt模板、无视觉塔前向|
|首三层诊断|67模块；同输入叶子最大relL2=.000137283|只覆盖前三层|
|逐层喂CPU中间输入|单层最大relL2=.00785202|打断误差传播，不能计正常完整前向通过|
|真实embedding完整DiT|视频.0451071、音频.0600674|音频超.05，整体失败|
|Fantasy SDPA升FP32候选|102次调用，末端误差未降低|没有修复问题|
|完整音频VAE解码|1087key、两次通过，relL2约7.49e-7；输出[2,1,3200]、32kHz|合成latent解码，不是文本条件音频或视频成品|

文本编码器两轮约217.87/219.30s，分配峰值5,143,781,888 bytes；音频VAE约4.87/4.08s，峰值672,035,840 bytes。这些是组件测量，不是视频生成时间。完整DiT设备前向有限值也不能当数值合格。

已取回15个冻结run，逐文件manifest校验通过；29项H3工具/边界CPU测试通过。网络中断后已补取结果并确认临时容器清理。文本编码器曾出现“推理成功、收集器因root 0600输出失败”，只修该产物owner/mode，SHA保持，单独记录finalization，不把收尾错误隐去。

下一步仍为DiT误差界定/修正、视频VAE、正式模板、采样调度、最终视频/音频质量与E2E。依据：[H3最新组件报告](/Users/USER/Documents/infra/benchmarks/gr308-h3-20260905/H3_LOADER_OFFLOAD_PROGRESS_20260906_ZH.md)、[H3派生汇总](/Users/USER/Documents/infra/benchmarks/gr308-h3-20260905/derived-loader-progress.json)、[H3证据归档](/Users/USER/Documents/infra/benchmarks/gr308-h3-20260905/h3-loader-evidence-20260906.tgz)。

## 6. 基础 kernel、dense、数值和通信

### 6.1 Native与dense

|类别|覆盖与结果|代表数值/边界|
|---|---|---|
|native launch/copy/triad|9case、81轮通过；copy/triad各1/8/64/256MiB|launch事件5.493µs/次、墙钟11.459µs/次；256MiB copy/triad逻辑带宽379.360/350.599GB/s|
|dense|45case、405轮过预定门；宿主33＋容器12|三精度；one-hot精确相等；主矩阵K5120/N1024，M1到2048；M1仍为torch.mm而非专用GEMV|
|dense M2048宿主|FP16/BF16/FP32事件中位1991.660/1998.943/84530.426µs|墙钟6142.292/5553.523/89285.413µs；不是同一口径，不能混用|
|dense后续只读审计|45case/405轮及首末完整输出两端复算，冻结worker循环一致|FP32走Sgemm、半精度GemmEx；事件包装无额外比例转换；实际kernel/时钟/差额根因未闭环|

native copy是设备kernel的逻辑读写带宽，不是H2D/D2H/P2P DMA；没有硬件DRAM流量counter，重复数组可能受缓存影响。共享服务常驻、未完成独占跨进程重复，不认证硬件峰值。

### 6.2 GPTQ最新定位

真实Qwen layers.3.self_attn.k_proj，K5120/N1024、4bit、group128，M=1/2/8/32/128及3个合成case，共 **8/8数值错误**，进程可正常退出，不全是native crash。

合成one-hot的有效载荷区1088/1088输出精确符合signed-byte读取预测，同区仅1个符合正确INT4参考；同一小样本FP16 dense 1024/1024精确正确。证据指向位宽/打包契约，提示潜在越界风险；未捕获设备ISA或直接证明越界访存。该路径保持STOP，真实权重数组留服务器，本地离线包只含合成数据，不导出潜在越界内容。

### 6.3 FHCCL与P2P capability

- FTUCA最小FHCCL：两卡单卡可见、两卡共享列表、四卡共享列表各3次＝9次；再对两套实际库和卡对2/4、0/2追加4次，共13次冷启动成功。每次仅单元素FP32 sum AllReduce、同步回读和释放，两卡得3、四卡得10。
- SHM或P2P/IPC日志不是通信带宽。早期完整vLLM/PyFHccl初始化崩溃依然未闭环；另一次ens5f0受限的AR/AG成功有独立作用域，不能与13次混成无失败统一资格。
- 只读P2P查询8×8＝64次，56/56非对角报告可访问，其中48个为跨物理卡有向逻辑对。**未启用映射、未分配peer buffer、未实际复制**。四张物理卡12个有向代表对的复制计划见第10节，尚未开始。
- 大消息/完整collective、重复完整框架冷启动、跨机RDMA、拥塞故障、通信计算重叠均未完成。没有确认第二台可用同栈节点。

依据：[基础总结](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/GR308_RECENT_BASIC_TESTS_SUMMARY_ZH.md)、[native报告](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/NATIVE_PROBE_RESULTS_ZH.md)、[dense/通信报告](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/dense-comm-next/DENSE_COMM_RESULTS_ZH.md)、[GPTQ/FHCCL报告](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/gptq-repro/GPTQ_FHCCL_RESULTS_ZH.md)、[dense计时复核](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/DENSE_TIMING_ZH.md)。

## 7. 主机传输：旧故障和新矩阵并存

### 7.1 原始4KiB路径和维护

旧pageable H2D回读失败；同显式stream串联初始化/复制/回读后的H2D、D2H各1case/9batch通过，墙钟约132.013/128.850µs/次。改序方案通过，不等于旧故障唯一根因已定位。

原pinned首次数据正确但事件门失败；首次未存原事件值，不倒填为0。后来明确记录event API成功但0ms、位模式0x00000000；batch2000长等待后CPU限额终止，不当有效性能样本。每次同步、只用墙钟的4KiB规避两次通过，约126.577/126.917µs/次。

经一次明确授权的隔离维护，暂停Qwen、8080/8180和监控、保留R1；原事件仍为0。新顺序同时加“开始事件完成再copy”和“结束事件后先同步stream”，两次各9batch通过，约0.054272/0.054015ms事件每批，对应墙钟5.037477/5.144524ms每批。每批20次，**不是每次**；两个屏障未分别对照。

00:26:27恢复，停服到恢复约272秒。Qwen退出码1自动重启两次、第三次起来；轮转日志不足解释前两次。维护已用，不自动重用一次性授权。

只读二进制复核证明事件时间除数1e6，长异常值与float32(2^34/1e6)位模式相符；附近1537个整数同样映射到该float32，故不能唯一证明counter回绕。存在重复轮询分支，但无故障当时timestamp/分支trace，0值与长等待根因未关闭。

### 7.2 新同步墙钟矩阵

|项|Pageable|Pinned|
|---|---|---|
|大小|4KiB、1MiB、8MiB、64MiB|相同|
|方向 / NUMA / 重复|H2D/D2H × CPU0/1 × memory0/1 × 两独立进程|相同|
|结果|64/64case、576batch通过|64/64case、576batch通过|
|原始完整首末输出|128份，2,449,604,608 bytes|128份，相同字节数|
|独立uint32核验|612,401,152元素一致|相同|
|同配置两次中位最大比|2.174|1.077|
|额外验证|前后全页面NUMA、源/目标、显式清理|另核Host属性、NumaUser flag=536870912、pointer、非managed、释放|

同一stream最多一份复制在途、同步完成再提交；每case首次＋3warmup＋9batch，四大小每batch20/10/3/1次。每worker45秒、分配前至少1GiB设备余量；最大设备buffer64MiB。独立回放为首末完整raw；中间轮由冻结worker校验，不能称每次copy都独立离线重算。

代表性的64MiB case墙钟中位范围（四种NUMA×两重复，非CI）：

|方向|Pageable|Pinned|
|---|---:|---:|
|H2D|11.337–28.210ms|2.703–2.949ms|
|D2H|7.979–17.392ms|1.730–1.981ms|

不是随机交错A/B，且pageable波动显著，不直接报因果加速比或最佳NUMA；bytes/墙钟仅为同步有效载荷速率，不是异步DMA/PCIe峰值。

Pinned第一项native通过，旧审计器误要求pageable环境键而拒绝。旧源码/HALT/失败保留；CPU修正版重放原raw通过后，新阶段只执行剩余63项，**没有重跑第一项**。原事件故障不因此解除。

完整raw仅在服务器pageable-wall/runs、pinned-wall/runs、pinned-wall-followup/runs；本地只同步控制、manifest、日志、回放/统计，故本地不是这两矩阵的完整原始副本。

依据：[Pageable完整矩阵](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/PAGEABLE_MATRIX_RESULTS_ZH.md)、[Pinned完整矩阵](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/PINNED_MATRIX_RESULTS_ZH.md)、[事件二进制复核](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/RUNTIME_EVENT_REVIEW_ZH.md)、[维护恢复状态](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/transfer-next/maintenance/isolated/session-20260906T162155Z/state.json)。

## 8. 新生产套件：API、算子、性能与资源

### 8.1 567个请求如何组成

|请求类别|实际发送数|结论|
|---|---:|---|
|初始语义/SSE|74|73通过，1个R1客户端90秒超时；另10个R1未测不计数|
|错误参数/格式|18|18正确拒绝且token计数不变；14结构契约通过、4纯文本400失败|
|主动取消＋恢复|6|3条自有流取消＋3条完整恢复回答通过各自协议|
|长度/闭环矩阵|246|预定网络请求全部最终回放通过|
|算子事件后的恢复哨兵|5|布局、False模式、显式group、RoPE、Attention后各1条UV91|
|开放到达|116|通过；120offered另4未发送、后续24未测|
|900秒有界观察|30|30/30通过|
|随机化路线对照|72|原48＋新24全部通过；原未发送终态不重复计请求|
|合计|**567**|不是567个正常语义回答均合格|

Qwen三路径各12语义＋12SSE，共72通过。R1仅1语义成功、1超时；其余4语义＋6SSE未测。超时不等于已证实错答、设备死锁或服务宕机，也未通过延长截止时间反复重试。

四项错误契约问题是8080/8180各自的畸形JSON和缺messages返回纯文本400；不能因为HTTP拒绝正确就标结构化错误通过。三入口主动关闭自身SSE后约0.518–0.520s首次采样到空闲、每次增加2生成token；没有逐请求引擎abort trace，不能称精确引擎取消延迟。

### 8.2 算子覆盖与失败拆分

预定参考为NumPy FP64，BF16/FP16/FP32筛查门分别atol/rtol=.002/.02、.0002/.004、.000002/.00002；不是通用误差上界。安装wrapper前提必须执行，不因native schema有参数就假设所有取值支持。

|阶段|结果|故障/边界|
|---|---|---|
|RMSNorm、Gemma RMSNorm、SiLU×Mul|141case、1107计时batch通过；首末raw回放通过|18smoke无热态计时；123case×9batch×5调用，批内5次非独立样本|
|最初operator加载前置|首次调用前拒绝|未版本化/版本化libfant_python哈希相同inode不同；修正快照后独立v2，旧HALT留存|
|residual/gating/RoPE扩展|306计划：3通过、1失败、302未运行|非连续输入绕过require_contiguous，为测试器范围错误，不证明合法布局kernel缺陷|
|连续布局版|171计划：55通过、1拒绝、115未运行；210热态轮|norm_before_gate=False不支持，不能当数值失败|
|remaining版|99计划：2 BF16 gated True通过、1拒绝、96未运行|group_size=128被拒；要求None；通过2项中有1个历史重复|
|独立RoPE|81计划：首BF16数值门失败、80未运行|Q7/3584、K4/512越门，全有限、输入不变|
|Attention|100计划：首项失败、0数值通过、99未运行|索引内核MMU/上下文故障，60秒终止自有worker；未保存数值输出|

RMSNorm/Gemma维度128/256/3584/5120，SiLU输出128/4352/18944；行数1/7/32/128及零值smoke，三精度。全部只是这些合成shape的局部筛查。

非法布局case的CPU访问模型按紧密跨度128可精确复现两份896元素输出；没有设备trace，原padding只保存worker布尔记录，不补造raw。Gated编译断言与实测一致：只支持group_size=None、norm_before_gate=True；Qwen源码使用该组合，但未逐次跟踪生产调用。

RoPE最大绝对误差Q/K为.0244140625/.01513671875；逐乘法BF16舍入后再加减舍入可精确复现全部4096输出，解释此样例的算术差异。原FP64门和失败仍保留，不能事后调宽门槛标通过。

Attention首项为1token、1head、dim64、BF16、causal，理论输出等于V。故障日志指向测试PID2651560与LaunchIndexKernelOptVec；测试器backing[[0,-1]]边界检查是有证据的候选。故障地址偏移符合负偏移转32位无符号后的算术，但不是ISA级证明。驱动曾自动复位该上下文；本任务没有主动执行GPU reset，不能写“完全未发生复位”。60秒门后进程退出，117份采样后端空闲、token计数未变；1.475秒UV91恢复成功不解除HALT。

Attention离线候选通过16项fake-tensor测试：先回CPU再转换/索引、提前保存核心输出、异常checkpoint、禁止覆盖；两端CLI以64拒绝启动。它不是已启用worker，没有GPU回归。

依据：[生产阶段详表](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/PRODUCTION_TEST_STATUS_ZH.md)、[Attention事件报告](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/ATTENTION_INCIDENT_ZH.md)、[Attention离线候选](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/ATTENTION_OFFLINE_CANDIDATE_ZH.md)。

### 8.3 长度、并发、开放到达

长度桶：输入128/512/2048/8192，输出16/64/128，每组合3轮配对三入口；另3smoke与闭环并发1/2/4/8、每入口每档3×并发条。原阶段有额度准入中断，新阶段只补未发送，最终 **246条网络计划全完成**；不能改写原job失败或未发送记录。

独立重复性检查发现63种payload中60种重复发送，其中3组在temperature=0/seed=0下有两种token序列；nonce52/62/68分别最早在第3/9/16token分叉。36个长度配对组三入口一致，差异在闭环批量。没有logits/独立整模型参考，不直接归因某kernel或业务错答。

开放到达12/6/3秒间隔×三入口共108条通过；8000的1.5秒档8条通过、4次到达触及客户端上限未发；8080/8180该档共24次未测。当前阶段120offered、120terminal、116实际请求，停止后未放宽上限重试。客户端限制不等于已测得服务端饱和点或业务背压合格。

原52个性能job保存3000份离散资源采样；这些不是连续soak。不同阶段参数和早期0.15–0.55req/s pilot不同，不合并作一条负载曲线。

### 8.4 900秒Qwen观察

08:24:03.696–08:39:03.696，900.000106秒，8000每30秒一条、最多1在途，30/30通过。输入128/512/2048×输出16/64共6组各5个不同合成输入：

|输入/输出|E2E中位s|TTFT中位s|TPOT中位ms|
|---|---:|---:|---:|
|128/16|3.673|1.040|175.61|
|128/64|12.111|1.033|175.50|
|512/16|3.903|1.270|175.48|
|512/64|12.351|1.279|175.70|
|2048/16|9.840|7.210|175.27|
|2048/64|18.285|7.219|175.64|

450份快照，最大间隔2.024s；Qwen prompt/generation增量26,880/1,200，R1为0；无新增OOM/preemption，最高55°C，SMI显存恒定。Qwen/R1主机cgroup首尾+2,576,384/−204,800bytes，不能直接判泄漏或排除泄漏。不是双模型干扰、24h稳态或业务SLO；见[有界观察报告](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/BOUNDED_SOAK_ZH.md)。

### 8.5 固定随机化三入口对照

两档128/16、2048/64，各12配对组；每组同prompt三入口各一次，每档6种排列各出现两次，首次请求前固定随机顺序。总72个请求、24/24配对组token序列一致。

|输入/输出|端口|E2E中位s|TTFT中位s|TPOT中位ms|
|---|---:|---:|---:|---:|
|128/16|8000|3.66943|1.03470|175.440|
|128/16|8080|3.66961|1.03537|175.575|
|128/16|8180|3.66991|1.03338|175.500|
|2048/64|8000|18.28340|7.21393|175.737|
|2048/64|8080|18.29548|7.21658|175.699|
|2048/64|8180|18.28847|7.20449|175.673|

每格n12。8080−8000配对E2E均差短/长+2.036/+10.649ms，8180−8000为+1.597/−4.887ms；正负均有，不能称稳定加速。

原阶段48条成功后旧额度准入停止，24条未发；新策略只补原序号48–71。原72terminal＋新24terminal＝96终态，但仅72实际网络请求，无重复发送。两段402.838/464.348秒，中间1610.675秒；原201份、新233份资源快照分开。两端计数证据表明间隔内prompt/generation无增量，但无连续资源窗。

原阶段仍passed=false、新阶段passed=true。只有一个持续存活部署，没有多次独立部署重复，不输出CI、P99、因果排名或业务goodput。见[完整路线结果](/Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite/ROUTE_AB_RESULTS_ZH.md)。

## 9. 31项全范围交接矩阵

“完成”仅指本文明确限定范围；“部分/问题”仍要闭环；“待完成”可能已有历史参考，不代表从零开始。NUM-02有旧单层证据但完整同输入真实层/高层误差传播要求仍待完成；DEL-01虽有本文，最终全项资格审计仍未完成。

|ID|工作项|状态|已有证据 / 接手后关闭条件|
|---|---|---|---|
|ENV-01|环境/物理拓扑/身份/容量|部分/问题|当前快照通过；新阶段仍需持续复核，不能用旧PID/库代替|
|ENV-02|可观测性|部分/问题|SMI/metrics/cgroup可用；修监控绑定与敏感日志，完善进程/物理卡同期账本|
|KRN-01|launch/copy/triad|部分/问题|9case/81轮；补独占/跨进程重复与counter口径|
|DENSE-01|三精度、多M、两库|部分/问题|45case；补合法布局、边界shape、跨进程重复|
|DENSE-02|FP32慢路径/事件差额|部分/问题|只读复算与调用源码；补实际kernel/时钟校准及根因|
|NUM-01|GPTQ INT4故障|部分/问题|明确失败且STOP；需修正build/契约和安全回归|
|NUM-02|真实层/误差传播|待完成|完整模型相同输入权重、独立参考、层级误差链未齐|
|XFER-01|pageable有界矩阵|完成|4KiB–64MiB、双向、四NUMA、双进程，64case/576batch完整raw回放|
|XFER-02|pinned矩阵与事件|部分/问题|同步64case通过；原事件协议/异常清理根因未闭环|
|XFER-03|零事件/长等待根因|部分/问题|只读二进制算术线索；缺故障timestamp/分支trace及双屏障分别对照|
|XFER-04|异步DMA/多buffer/stream|待完成|先可信计时与清理，再新受控矩阵|
|P2P-01|卡对/方向/双向复制|部分/问题|64次capability完成；96项复制计划未启动，双向并发未实现|
|COMM-01|2/4卡完整collective|部分/问题|小消息/13次native与另轮AR/AG；补消息/collective/冷启动矩阵|
|COMM-02|完整框架初始化故障|部分/问题|ncclTopoGetLocalNet历史SIGSEGV；需新维护范围及修正对照|
|COMM-03|跨机RDMA/拥塞/重叠|待完成|需要第二同栈节点、网络范围与授权|
|OPS-01|attention/norm/RoPE/激活|部分/问题|141基础及部分融合通过；布局/参数/RoPE/Attention停止项未闭环|
|OPS-02|GDN/MoE/Graph对照|待完成|需绑定适用模型；当前dense模型不能冒充MoE|
|API-01|语义与完整响应|部分/问题|Qwen36、R1 1通过/1超时，R1另4未测；不是业务质量|
|API-02|SSE与usage/终止|部分/问题|Qwen36通过，R1 6未测；chunk非token|
|API-03|错误/JSON/工具/上下文|部分/问题|18拒绝中4错误格式；工具调用及其余声明能力待验证|
|API-04|取消/超时/恢复|部分/问题|Qwen三路径取消恢复；缺引擎abort关联与其他变体|
|QUAL-01|真实业务质量|待完成|需要代表数据、独立答案/评分、失败分类及长上下文口径|
|PERF-01|长度/并发/到达阶梯|部分/问题|246网络＋116开放到达通过；cap停测与重复性差异保留|
|PERF-02|随机化/重复/CI|部分/问题|72网络、24组一致；中断保留、缺独立部署重复和CI|
|PERF-03|真实SLO goodput|待完成|业务SLO/流量模型/全offered分母未齐|
|KV-01|prefix/LMCache|部分/问题|配置已核；真实connector/命中/驱逐/一致性未验收|
|MEM-01|同期内存账本/泄漏碎片|部分/问题|3000离散＋450连续低速采样等；无完整分配分类账/泄漏结论|
|REL-01|soak/混跑/恢复|部分/问题|900秒单模型通过，早期双模型小样本；缺长时间双模型业务工况|
|REL-02|冷启动/drain/回滚|部分/问题|有恢复与drain成功；Qwen两失败未解释，缺完整冷启动重复|
|REL-03|故障注入/降级|待完成|需对象、故障预算、恢复方法及授权|
|DEL-01|全证据与最终资格|待完成|本次交接完成不代表全部31项资格完成；不得改all_requirements_complete为true|

## 10. 中断时在做的P2P工作：尚未启动

### 10.1 已准备与未准备

|资产|当前位置/成熟度|
|---|---|
|capture_peer_runtime.py、peer-runtime-capture-20260907T021800Z.json|两端；只读6条ELF命令、头文件/库身份/健康/计数，无GPU调用|
|peer_copy_probe.cpp、peer_copy_protocol.h、peer_copy_plan.py、prepare_peer_copy.py|两端；native同步探针已构建|
|peer-copy-build-20260907T022721Z|两端；11项CPU/负参数自检，未请求GPU API|
|PEER_COPY_PREREGISTRATION_ZH.md|两端；实验设计，不是执行结果|
|peer_copy_evidence.py、peer_copy_guard.py、audit_peer_copy.py、test_peer_copy.py|本地未冻结候选；尚未同步服务器、未在服务器做17项新增回归|
|peer-copy-implementation.json|两端均不存在；完整实现未冻结，**不是launch-ready**|
|peer-copy-wall|服务器无结果文件；第一项s2-d4-b4096-r0未开始|

native binary SHA256：
`213baf8b8043d3c8d7c92a69baf38370fed779113311d3d4872c4a032d784703`。  
构建结果SHA256：
`2232229fc98e45e136b08025bce903a4f1074fa2a4b062c05166f6a5f686bd9a`。

### 10.2 固定计划与已知限制

四张物理卡取.0代表，逻辑2/4/0/6；12有向卡对×4大小（4KiB/1MiB/8MiB/64MiB）×两独立进程＝96项，先小后大。不是所有56个逻辑对，也没有双向同时复制实现。

每进程两份自有device allocation，payload两侧各256字节guard；两端分配前各至少1GiB余量。独立uint32输入、目标反码poison、source/destination不同guard；每轮完整回读核验，首末两端完整保存。同步hipMemcpyPeer以source current调用，然后两端device synchronize；首轮＋3warmup＋9batch，每batch20/10/3/1。记录API耗时与同步墙钟，不调用Event/Async/reset。

只启用自身source→destination mapping，若“已启用”则拒绝；成功时释放与禁用自身映射。失败路径不在析构中盲目重试HIP清理，进程退出后的上下文资源回收仍须证明。拟设worker60秒/阶段95秒，身份/全局锁/空闲计数/2秒SMI/10秒采样新鲜度/80°C/新OOM/只取消自有PID保护，当前controller还需完整审查和集成测试。

已读安装头文件将P2P标为实验性；同步/异步wrapper共享内部helper仅是静态线索，不能证明异步stream被忽略、无host staging或真实DMA重叠。未来即使回读正确，首轮也只能报“peer API端到端同步有效带宽”，不标链路峰值。

接手下一步：审查未冻结四文件与故障清理→补自有进程截止/取消集成测试→确认完整513项在服务器通过→生成新不可变实现manifest→重新取得当前健康/身份/空闲和资源证据→在明确续测范围内从第一项开始。本文不执行这一步，不给绕过准入的裸native启动命令。

## 11. 故障、停止项和不可越过的门

|问题|已知层级/当前状态|禁止误判/下一关闭条件|
|---|---|---|
|GPTQ整模及packed层|质量/数值明确失败，可能打包/位宽契约问题|禁止继续用吞吐绕过质量；先离线契约/修正build和有界安全回归|
|FHCCL冷启动SIGSEGV|完整框架路径失败，最小native/限定网卡部分成功|有限通过不等于根因修复；需完整日志/同框架版本对照|
|Qwen恢复两次退出1|第三次恢复、RestartCount2保持|恢复成功不等于冷启动稳定；日志缺口不能补造|
|原pinned事件0/长等待|数据与event时间协议分离；静态线索存在|不扩大batch追压；不把同步规避当驱动修复|
|非法非连续fused输入|测试器漏合法前提|不得写成已证实合法输入kernel缺陷|
|gated False/group128|实现明确不支持|不是数值误差；不能因为schema有参数就重试|
|RoPE FP64门失败|逐乘法BF16模型精确解释样例|失败不改写，需预定业务/数值接受标准和设备证据|
|Attention索引/MMU|测试器GPU负索引候选，驱动自动context reset|0数值输出；先离线修正，不能直接恢复故障路径|
|R1 90秒超时|客户端截止与恢复证据有效|不能当已证实错答或deadlock；新协议需另登记|
|网关4项错误格式|正确拒绝但纯文本400|需明确兼容契约与修正范围，不能算结构化错误通过|
|重复payload输出分叉|闭环3组有token差异|无logits/独立参考，不能直接归因某kernel|
|monitor循环及敏感日志|NRestarts持续增长，历史绑定不存在地址|瞬时active不算健康；配置/凭据整改需运维范围|
|DPSKV4下载/loader/NaN/clamp|下载不完整、真实链路仍有阻断|tiny成功不表示真模型可上线|
|H3完整DiT|finite但误差超门|诊断替换CPU中间输入不算正常前向成功|
|旧额度门停止|历史执行策略已撤销，新策略独立实现|不修改旧HALT/源码，不延长假观察，不重发已发送项|

本阶段没有主动停止模型/改路由/换库/重置设备。历史已授权维护有停服；Attention有驱动自动上下文复位；都应明确保留，不能统称“全过程无任何复位/重启”。

禁止删除旧HALT、STOP、once-used标记或改冻结SHA以让新实验启动。禁止杀模型PID假装取消本任务请求。不得用旧维护授权停服务、静默换模型、替换驱动、故障注入或跨机操作。

## 12. 证据、存储与完整性

### 12.1 路径映射

|用途|本地|服务器|
|---|---|---|
|工作根|/Users/USER/Documents/infra|/data/infra|
|基础测试|benchmarks/gr308-ftuca-20260906|同相对路径|
|生产套件|benchmarks/gr308-ftuca-20260906/production-suite|同相对路径|
|本次交接支持件|benchmarks/gr308-handoff-20260907|同相对路径|
|真实模型权重|不制作完整副本|/data/models 下各固定模型目录|

本文可点击链接以本地绝对路径为主；服务器阅读时按上表替换根前缀。证据索引同时列本地路径与标准服务器映射；“映射路径”本身不是已逐文件核实的服务器副本声明。

完整原始传输raw仅在服务器：pageable-wall/runs、pinned-wall/runs、pinned-wall-followup/runs，总首末输出4,899,209,216 bytes。旧GPTQ大张量/大量PAL日志也以各阶段报告说明的远端目录为准。HTTP、算子首末张量、Attention事件、soak与路线对照的已取回证据按阶段manifest保存；H3有15run轻量归档，不含大模型/完整依赖。

本次只生成交接文档、快照、索引与核验件；不删除用户数据，不重新打包多GB模型/重复日志。若需离线携走全部证据，应先确定目的盘容量和权限，再按manifest精确复制；没有本地raw时禁止伪造空文件使审计通过。

### 12.2 关键冻结锚点

|文件/清单|SHA256|
|---|---|
|inventory-20260906T172800Z.json|65c6f84fe11b7dd057a2f9199d72240934c2cae59e96e216e9c79361736b13e9|
|USER_EXECUTION_POLICY.json|5f0782b787998a8ea3420ca05a1a510e8eec08bc9c53985bb6fd95d5368cf773|
|pageable-implementation.json|75bb3e501846ffa287e8ef64474bcbe7e00d46da5b5cc42e39ea302683afd3a4|
|pinned-implementation.json|98243a0add02d7cf716f03e2cd70b05c01cf6cca1b6dd3efe42dcf746f22e095|
|pinned-followup-implementation.json|018cce1500a222ade1608f9e95eaa696c3cb72cb71c82694b9960345c9a3d6a1|
|attention-implementation.json|107f3953a70a898acb5e41121451541ad10cd74a2170ebd714ee3efc9faad333|
|soak-implementation.json|92d7e03847d6d9251b7a21178ea7aa587c5475607f8a3bdf9dcaad7d929c90fd|
|route-ab-followup-implementation.json|72cb1a411f8cba6fcbf3ae1dc93d1687d6e413d1c7365518c09eb1dd9f7ad0d2|
|原route evidence manifest|5061ad3c8fda8dd18930caab513754a015da27d5c02a37d801f14093db078cef|
|续测route evidence manifest|1390b31739d1aa63c378b5eab7a746b25fc0c273cbb0fd76c4dee17f46a41b49|
|本次现场快照|8f7fa5c56c70de5a7db763c800a8337f1b10fc4bdf5bedaeff1077b38113ddf7|

原始文件哈希一致仅证明字节完整，不能代替请求/数值/时间轴语义审计。很多回放测试专门验证“篡改后重签manifest”仍会因语义不符被拒绝。

### 12.3 常用证据入口

- API：api-audit-20260906T182000Z.json、protocol-audit-20260906T184000Z.json、api-runs/、qwen-stream/及协议/恢复目录。
- 算子：operator-audit-20260906T190500Z.json；operator-extension-reviewed-failure-20260906T202120Z.json；operator-contiguous-audit-20260906T205650Z.json；operator-remaining-audit-20260906T211650Z.json；rope-numerical-review-20260906T213825Z.json。
- Attention：attention-varlen/runs/000-single-key-bf16-r0/、attention-incident-review-20260906T232600Z.json、post-attention-recovery-audit-20260906T232800Z.json。
- 性能：performance/与performance-followup/；token-repeatability-20260906T205620Z.json；open-arrival-audit-20260906T220157Z.json。
- 传输：pageable-whole-matrix-20260906T222900Z.json；pinned-whole-matrix-20260906T225600Z.json；pinned-audit-correction-review-20260906T224300Z.json。
- 只读P2P：peer-capability-audit-20260906T215115Z.json；peer-runtime-capture-20260907T021800Z.json。
- Soak：bounded-soak/runs/qwen-low-rate-15m/、bounded-soak-audit-20260907T004000Z.json与remote审计。
- 路线：randomized-route-ab/runs/、route-ab-followup/runs/；route-ab-initial-audit-20260907T010900Z.json、route-ab-followup-audit-20260907T014200Z.json、route-ab-descriptive-summary-20260907T014400Z.json及对应remote文件。
- 历史维护：transfer-next/maintenance/preserved-markers.sha256、isolated/session-20260906T162155Z/、isolated/audit-20260906T163347Z/。
- 本次新增测试结果摘要：[本地513项CPU回归](/Users/USER/Documents/infra/benchmarks/gr308-handoff-20260907/local-cpu-regression-513.json)。
- 全部以上文件的绝对路径与存在性/哈希见[机器证据索引](/Users/USER/Documents/infra/benchmarks/gr308-handoff-20260907/EVIDENCE_INDEX.json)。索引不是所有原始raw的替代品。

## 13. 接手操作与安全复核

### 13.1 第一轮只读核验

1. 用现有安全认证方式连接主机，不把密码/令牌写进报告、命令行或日志。先核实路由/SSH；失联只说明访问失败，不能直接判远端服务宕机。
2. 阅读本交接、对应阶段报告和USER_EXECUTION_POLICY；确认未收到新的生产变更/维护授权。
3. 检查五端口、九容器ID/镜像/config hash/PID/start ticks/restart_count、boot ID、库SHA/inode、两后端running/waiting、磁盘、监控状态。
4. 核对旧STOP/HALT/used标记未变；核对已有自有测试进程和未完成claim。任何身份不匹配先解释原因，不修改基线“让它通过”。
5. 只在需要继续测试且保护门全部满足时进入新预登记阶段；未测项、失败项和待授权项分开，不把所有旧控制器批量启动。

下面只读采集会生成一个**新的证据文件**，不发推理、不调用GPU runtime、不改服务。需要已经存在的授权SSH及sudo环境；输出文件名必须没用过：

```bash
sudo -n env TORCH_DEVICE_BACKEND_AUTOLOAD=0 /usr/bin/python3 -B \
  /data/infra/benchmarks/gr308-handoff-20260907/capture_handoff_snapshot.py \
  --output /data/infra/benchmarks/gr308-handoff-20260907/handoff-live-snapshot-NEW.json
```

NEW须由接手人换成唯一时间标签，不覆盖本次快照。采集器若因库/身份/全局锁变化拒绝，保留错误并停止；不要去掉保护检查。

### 13.2 CPU回归与离线回放

本地当前513项：

```bash
cd /Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B -m unittest discover -p 'test_*.py'
```

测试中故意打印passed=false、usage拒绝、模拟超时是负向测试，不代表发生新现网事故；看最终unittest结论，不把模拟日志计成实测终态。旧版本315/349/381/443/465/492/496等是演进套件，不逐次累加；513不是513个GPU case。

完整路线离线回放（本地已包含所需raw；输出需新路径）：

```bash
cd /Users/USER/Documents/infra/benchmarks/gr308-ftuca-20260906/production-suite
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B summarize_route_ab.py \
  --output /Users/USER/Documents/infra/benchmarks/gr308-handoff-20260907/route-replay-NEW.json
```

这会重放原失败段及成功续段，不查询账号额度。正确结果应为network_requests=72、all_terminal_records=96、original_stage_passed=false、continuation_stage_passed=true、confidence_intervals=null。

**不要直接把summarize_pageable.py/summarize_pinned.py的历史CLI当可无条件执行的离线入口**：它们仍封存旧实时额度准入。不能刷新假额度来让它运行。若仅审计raw，可在服务器通过已冻结的audit_pageable.audit_job和audit_pinned_v2.audit_job文件回放接口重放历史证据；这些检查按记录时刻验证旧策略，不执行新的实时准入。单case命令格式已经核对：

```bash
cd /data/infra/benchmarks/gr308-ftuca-20260906/production-suite
sudo -n env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /proc/2018111/root/root/miniconda3/bin/python3.10 -B audit_pinned_v2.py \
  --job h2d-b4096-cpu0-mem0-r0 --phase review_original \
  --implementation-sha256 98243a0add02d7cf716f03e2cd70b05c01cf6cca1b6dd3efe42dcf746f22e095 \
  --output /data/infra/benchmarks/gr308-handoff-20260907/pinned-original-replay-NEW.json
```

该命令只审计已完成的第一项，不重跑GPU；需先核实PID/解释器身份仍对应本快照。其余63项用followup phase及pinned-followup清单SHA；不把第一项归到followup。完整raw缺失时必须先从服务器精确取证，而非删掉审计字段。

### 13.3 下一个阶段的优先顺序

|优先级|建议接手工作|进入条件|
|---|---|---|
|P0|保全现网/证据，登记监控循环、两次冷启动、危险native STOP|只读可做；配置修复/凭据整改/停服需另明确范围|
|P1|完成P2P候选controller/auditor审查、故障清理与服务器CPU验证|不触发GPU；当前不是launch-ready|
|P1|GPTQ、Attention索引、原事件/FHCCL根因离线定位与厂商修正build登记|不能裸重跑危险路径；对外发送材料需批准及脱敏|
|P2|API缺口、输出可重复性、GDN/Graph合法路径覆盖|先冻结明确能力/容差/请求截止与安全计划|
|P2|随机交错独立重复、真实长度到达轨迹、双模型有界观察|需确定负载范围、足够重复和真实业务门；不拼接旧实验CI|
|P3|KV命中/驱逐与内存分类账|需真实connector/模型兼容验证；不能只看metrics服务在线|
|独立支线|DPSKV4下载恢复/loader/clamp、H3 DiT/视频VAE、Wan短视频、Qwen4B|分别明确优先级/资源；不会因文件下载完成自动上线|
|待授权|重启/独占、故障注入、跨机、升级/换库、外发厂商材料|必须有对象、恢复基线、窗口、故障预算和明确授权|

### 13.4 需要用户/接手团队补齐的信息

- 真实业务数据集、可保存/脱敏范围、独立答案/人工评分规则、长上下文与工具调用的实际需求。
- TTFT、TPOT、E2E、超时/错误率等业务SLO；租户/流量混合、到达分布、目标成本和goodput口径。历史4秒/250ms、30秒/600ms只为合成诊断门。
- 新维护/独占窗口、允许停止的精确服务、回滚方法、失败预算；旧一次性维护额度不复用。
- 第二台同栈节点、RDMA网络/故障测试范围。
- 修正驱动/库build来源与SHA、厂商联系人和允许外发的材料；本交接不等于外发授权。
- 下一优先目标是基础栈闭环、现网服务验收还是模型支线；不同路线不能用同一“完成百分比”替代。

## 14. 交接完成标准

本次交接交付：本文、现场只读快照、当前关键证据索引、CPU回归摘要、文件校验清单。原始实验、失败、停止标记和历史报告保持原样；本次检查没有新推理或native GPU负载。

后续任务完成则另需：31项逐条有适用范围与原始证据；缺项不删、失败不洗成通过；真实业务门和可靠性门满足；回滚/监控/日志与资源账本可用；最后再做独立资格审计。**在此之前，all_requirements_complete=false、production_promotion_allowed=false保持不变。**

## 15. 交付时网络中断、恢复与重新同步

北京时间12:24–12:26，最后上传主文档和支持目录时SSH连接超时，rsync退出255；此前认证会话亦报告服务器无响应。TCP22三秒探测失败。只读路由为gateway=192.0.2.12、interface=en0，不再是先前可用的VPN路径。这只证明本机当前访问链路不可用，不能判定服务器、模型或VPN服务本身宕机。

中断时远端已保存11:11快照和采集器，但完整交接主文档、最新索引/校验件的远端副本当时未获确认。没有自动恢复网络、改路由/VPN、重启服务或重新开始测试。上传进程因超时退出；本任务仅终止仍等待的本地SSH校验进程，未向模型进程发送信号。随后本地tar目录检查成功不是远端校验成功。

恢复访问后，只需先做第13节只读身份/健康核验，再精确同步主文档及gr308-handoff-20260907支持目录，运行HANDOFF_SHA256SUMS检查。不要上传未冻结的P2P控制器以触发实测，也不要覆盖历史原始实验。网络诊断和同步失败摘要另存[交付网络记录](/Users/USER/Documents/infra/benchmarks/gr308-handoff-20260907/network-at-delivery-20260907.json)。

用户告知连接恢复后已完成上述只读核验：12:28:14五端口health/models均200、模型映射正确，九容器ID/启动时间/PID/restart_count/config hash未变，两后端连续空闲。没有已登记测试进程，没有peer-copy实现冻结清单或复制结果；没有新推理/native测试。监控NRestarts已到3671，瞬时active仍不代表健康。

[恢复后完整快照](/Users/USER/Documents/infra/benchmarks/gr308-handoff-20260907/handoff-restored-snapshot-20260907.json)的SHA256为b372034712be5164c9dcfb8bc210ae8960b6a797187eb733869fab1cfef5070c。主文档及支持目录在恢复后重新按最终HANDOFF_SHA256SUMS进行交付校验；原中断记录保留为历史，不再据其判断当前连接状态。轻量包不含未冻结P2P控制器或大型raw。
