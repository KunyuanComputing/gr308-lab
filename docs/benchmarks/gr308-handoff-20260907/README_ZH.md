# GR308 交接支持件

主文档：`../GR308_TEST_HANDOFF_20260907_ZH.md`。

这是轻量交接包，不是完整实验原始数据、模型权重或可独立运行环境。

12:24–12:26曾因网络访问超时导致同步失败，用户恢复连接后12:28已重新核实SSH、五端口和九容器身份/排空。handoff-restored-snapshot-20260907.json是最新快照；原失败记录仅描述中断时刻。最终交付按HANDOFF_SHA256SUMS对照本地和服务器文件，不自动恢复测试负载；详见主文档第15节。

- `handoff-live-snapshot-20260907.json`：北京时间11:11的只读现场状态；不发新推理。
- `monitor-followup-20260907.json`：随后监控重启数继续增长，以及GLM转换容器的只读进程名。
- `local-cpu-regression-513.json`：本地513项CPU回归摘要；服务器最近完整回归仍为496项。
- `route-replay-handoff-20260907.json`：交接时再次完整CPU回放72条路线请求，原中断/HALT保留；无新网络测试。
- `EVIDENCE_INDEX.json`：相关文件的本地路径、映射服务器路径、大小、SHA256；不包含递归全部raw。
- `HANDOFF_SHA256SUMS`：在infra工作根目录执行`shasum -a 256 -c benchmarks/gr308-handoff-20260907/HANDOFF_SHA256SUMS`复核本包文件；服务器可用`sha256sum -c`。
- `capture_handoff_snapshot.py`：服务器只读采集器，要求原冻结依赖/身份仍成立；只能写新的输出文件。
- `build_handoff_index.py`：完整本地infra工作区使用的只读索引生成器；轻量包缺原始工作区时不能独立重建索引。

路径映射：本地`/Users/USER/Documents/infra`，服务器`/data/infra`。主文档链接以本地绝对路径为主，换机需按根前缀映射。

原始传输矩阵约4.90GB的首末raw仅在服务器三个transfer运行目录中。真实模型大权重、旧GPTQ大型dump/PAL日志不随本包复制。包内不包含SSH密码或令牌；内部IP、容器身份及日志引用仍应在对外发送前审查。

P2P的四个未冻结候选文件仍只在原本地production-suite中，本包索引会注明，未因为制作交接文档而同步或授权运行。没有peer复制结果，没有生产晋级结论。
