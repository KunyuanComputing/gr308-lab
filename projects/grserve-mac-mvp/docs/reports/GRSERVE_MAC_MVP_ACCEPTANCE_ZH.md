# GRServe Mac MVP 验收报告

**结论：NOT ACCEPTED / BLOCKED**

本报告不构成 Mac MVP 通过认证；必须修复上表中的失败或不完整 gate，并用一次新的 full summary 重新生成报告。

## 证据绑定

- Run ID：`20260909T165459Z-456d1686`
- Git SHA：`c6edcf73d83383d4e4c0215e54eb1fe30d718261`
- Summary 精确路径：`/Users/USER/Desktop/Work/GR308/.worktrees/grserve-mac-mvp/artifacts/mac-mvp/20260909T165459Z-456d1686/summary.json`
- Summary SHA256：`28a43f27fefdf3ee79fb4964f328bed3f26036eafce8d3a3d99ec5392de6f5db`
- Summary 状态：`FAIL`
- Critical skip count：`0`

本报告不使用 `latest` 链接或目录修改时间定位证据；校验命令会依据上述精确 summary 路径重新渲染，并逐字节比较报告。

## Gate 结果

| Gate | 状态 | 原因 |
|---|---|---|
| preflight | PASS | — |
| Debug | PASS | — |
| Release | PASS | — |
| C11 ABI | PASS | — |
| C++ | PASS | — |
| Python | PASS | — |
| proto | PASS | — |
| manifest | PASS | — |
| golden | PASS | — |
| fault matrix | PASS | — |
| ASan/UBSan | PASS | — |
| TSan | PASS | — |
| Linux amd64 | FAIL | COMMAND_FAILED |
| API E2E | PASS | — |

## 跳过项

- 无

## 依赖 gitlink

| 依赖 | Gitlink | Resolved HEAD |
|---|---|---|
| googletest | `6910c9d9165801d8827d628cb72eb7ea9dd538c5` | `6910c9d9165801d8827d628cb72eb7ea9dd538c5` |
| protobuf | `43e1626812c1b543e56a7bec59dc09eb18248bd2` | `43e1626812c1b543e56a7bec59dc09eb18248bd2` |

- 其他锁定依赖（机器可读）：`{"abseil":{"expected":"9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f","presets":{"mac-arm64-asan-ubsan":{"clean":true,"head":"9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f","identity":{"clean":true,"head_tree":"254921002927a26dfa6581b058f34eba74f6c1aa","index_flags_clean":true,"index_tree":"254921002927a26dfa6581b058f34eba74f6c1aa","worktree_tree":"254921002927a26dfa6581b058f34eba74f6c1aa"},"tree_sha256":"eb8cab5bd24122810e80f28fc9642f21840fabd9e07b496a81dccca1abfd73c8"},"mac-arm64-debug":{"clean":true,"head":"9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f","identity":{"clean":true,"head_tree":"254921002927a26dfa6581b058f34eba74f6c1aa","index_flags_clean":true,"index_tree":"254921002927a26dfa6581b058f34eba74f6c1aa","worktree_tree":"254921002927a26dfa6581b058f34eba74f6c1aa"},"tree_sha256":"eb8cab5bd24122810e80f28fc9642f21840fabd9e07b496a81dccca1abfd73c8"},"mac-arm64-release":{"clean":true,"head":"9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f","identity":{"clean":true,"head_tree":"254921002927a26dfa6581b058f34eba74f6c1aa","index_flags_clean":true,"index_tree":"254921002927a26dfa6581b058f34eba74f6c1aa","worktree_tree":"254921002927a26dfa6581b058f34eba74f6c1aa"},"tree_sha256":"eb8cab5bd24122810e80f28fc9642f21840fabd9e07b496a81dccca1abfd73c8"},"mac-arm64-tsan":{"clean":true,"head":"9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f","identity":{"clean":true,"head_tree":"254921002927a26dfa6581b058f34eba74f6c1aa","index_flags_clean":true,"index_tree":"254921002927a26dfa6581b058f34eba74f6c1aa","worktree_tree":"254921002927a26dfa6581b058f34eba74f6c1aa"},"tree_sha256":"eb8cab5bd24122810e80f28fc9642f21840fabd9e07b496a81dccca1abfd73c8"}},"resolved":"9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f","tree_sha256":"eb8cab5bd24122810e80f28fc9642f21840fabd9e07b496a81dccca1abfd73c8"},"apt":{"build-essential":"12.9","git":"1:2.39.5-0+deb12u2","lock_sha256":"829d9e8c7bca1d736822b0832a6c932cf38c219d86c8c5bbfa99a62aaa1c1ca9","snapshot":"20250601T000000Z"}}`

## Manifest 与 Golden

| 资产 | SHA256 |
|---|---|
| `models/mini-qwen38-hybrid/manifest.textproto` | `243fbda41cf8698164fe25db4596b37cf0ecdb2d493ce4c5d57b87ec158a997e` |
| `models/mini-qwen38-hybrid/golden.json` | `3106862ea0c44d464138b751b76fe83199814ba8c2fe4c508618ec80c1be8f5b` |

## 范围边界与服务器后续

即使结论为 ACCEPTED，也只说明本地协议、状态机、故障边界以及 CPU/模拟接口在该次证据中通过；不证明 GR308 性能或硬件正确性，也不代表服务器上线资格。

服务器实验前仍必须完成独立的 Fant HAL/P0 资格验证：设备身份、内存、stream/event、kernel launch、copy、错误传播、资源回收和 FHCCL 冷启动。
