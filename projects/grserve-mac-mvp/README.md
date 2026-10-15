# GRServe Mac MVP

GRServe Mac MVP 是一个仅用于 Apple Silicon 本地开发与验收的最小系统。它验证
C++20 Engine Core、纯 C11 HAL ABI、两 rank 事务状态机、CPU/Mock/Fault 后端、
Unix Domain Socket RPC，以及预分词的最小 OpenAI-compatible HTTP/SSE 接口。

它不加载真实 Qwen3.8-27B，不包含 Fant HAL、FHCCL、GR308 kernel 或生产网关。
因此，本项目不证明 GR308 性能，不证明 GR308 硬件正确性，也不代表服务器上线资格。

## 本地启动

先构建 Debug 版本：

```bash
cmake --preset mac-arm64-debug --fresh
cmake --build --preset mac-arm64-debug
```

在第一个终端启动本地 Engine（以下示例使用 CPU Reference HAL）：

```bash
mkdir -p /private/tmp/grserve-local
build/mac-arm64-debug/grserve-engine \
  --socket /private/tmp/grserve-local/engine.sock \
  --model-dir models/mini-qwen38-hybrid \
  --hal-plugin build/mac-arm64-debug/libgrserve_cpu_hal.dylib \
  --ranks=2
```

在第二个终端启动仅监听 loopback 的 API：

```bash
uv run --locked grserve-api \
  --engine-socket /private/tmp/grserve-local/engine.sock \
  --host 127.0.0.1 \
  --port 8000
```

## 请求示例

接口范围仅包括 `GET /health`、`GET /v1/models` 和接受预分词 token 的
`POST /v1/completions`。MVP 不提供 tokenizer、Chat Completions、鉴权或 TLS。

```bash
curl --fail --silent http://127.0.0.1:8000/health
curl --fail --silent http://127.0.0.1:8000/v1/models
curl --no-buffer --fail-with-body \
  -H 'content-type: application/json' \
  -d '{"model":"mini-qwen38-hybrid","prompt_token_ids":[1,4,7],"max_tokens":8,"stream":true,"temperature":0,"top_p":1,"n":1}' \
  http://127.0.0.1:8000/v1/completions
```

## 单命令验收与证据

从无 tracked/untracked 改动的干净 checkout、锁定依赖和可用的本地 Docker
daemon 执行：

```bash
./tools/verify_mac_mvp.sh
```

该命令固定运行 `uv run --locked python tools/verify.py --profile full`。它会输出
唯一的 `GRSERVE_SUMMARY=/absolute/path/to/summary.json` 定位行。只有该精确路径
指向的 full summary 通过严格 gate contract，Mac MVP 才可标记为 accepted；禁止
用 `latest` 链接或目录修改时间猜测证据。

报告必须由同一次 summary 生成并复核：

```bash
uv run --locked python tools/evidence.py render-report \
  --summary /absolute/path/to/summary.json \
  --output docs/reports/GRSERVE_MAC_MVP_ACCEPTANCE_ZH.md
uv run --locked python tools/evidence.py validate-report \
  --summary /absolute/path/to/summary.json \
  --report docs/reports/GRSERVE_MAC_MVP_ACCEPTANCE_ZH.md
```

即使所有本地 gate 通过，证据也只覆盖本地协议、状态机、错误边界、sanitizer
和 Linux amd64 可移植性。服务器实验仍需先完成独立 Fant HAL/P0 资格验证。
