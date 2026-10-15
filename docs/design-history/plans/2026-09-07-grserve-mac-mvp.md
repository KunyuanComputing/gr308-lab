# GRServe Mac MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
**Goal:** 在 Apple Silicon Mac 上交付可重复构建、可故障注入、具备 OpenAI-compatible 预分词接口的 GRServe 双 rank 事务型推理框架 MVP，并证明未提交 token 永不发布。

**Architecture:** C++20 Engine Controller 是请求、事务、状态与发布决定的唯一逻辑所有者；两个 Rank Worker 通过稳定 C11 HAL ABI 调用 CPU、MockAsync 或 FaultInjection 后端。Python FastAPI 适配器经“4 字节大端长度 + protobuf”的 Unix Domain Socket RPC 访问 Engine，只负责 HTTP/SSE 协议转换。

**Tech Stack:** C++20、C11 ABI、CMake/Ninja、GoogleTest、protobuf、POSIX UDS、Python 3.11、uv、FastAPI、uvicorn、pytest/httpx、ASan/UBSan、Docker linux/amd64。

---

## 0. 计划使用说明

设计依据：[GRServe Mac MVP 设计规格](../specs/2026-09-07-grserve-mac-mvp-design.md)。规格已于 2026-09-07 获用户确认。

当前现场：

- 工作目录 `/Users/USER/Desktop/Work/GR308` 不是 Git 仓库。
- 已有 Apple Clang 16、Homebrew、uv 0.11.7、Docker CLI 29.5.3。
- 未安装宿主 CMake、Ninja、Python 3.11、protoc；本计划通过 uv 与源码依赖解决，不使用系统 Python 3.9。
- Docker daemon 当前未运行。最终 Linux amd64 门禁必须在 Docker Desktop 启动后执行；daemon 不可用时 full profile 必须以 `DOCKER_DAEMON_UNAVAILABLE` 失败，不能 skip。
- 所有开发都在 Mac 本地进行。本计划不登录、不修改、不启停 `192.0.2.11` 上的任何内容。

### 0.1 里程碑

| 里程碑 | 任务 | 独立验收结果 |
|---|---:|---|
| M1 工具链与 ABI | 1–4 | 可复现构建、强类型、错误模型、C11 HAL 与动态装载门禁 |
| M2 本地异步后端 | 5–6 | 虚拟时钟、乱序完成、可重复故障注入 |
| M3 事务 Engine | 7–10 | rank shadow/canonical、原子 admission、commit barrier、取消/fatal |
| M4 微型模型 | 11–12 | 冻结 manifest/权重、确定性 CPU 计算、三后端协议一致性 |
| M5 RPC 与 API | 13–18 | protobuf/UDS、Engine 进程、Python 客户端、HTTP/SSE、断线语义 |
| M6 可审计验收 | 19–21 | JSONL 证据、sanitizer、Linux amd64、单命令验收报告 |

每个任务严格执行 RED → GREEN → REFACTOR。除工具链配置和冻结数据外，不得先写生产代码再补测试。每次 commit 只添加任务列出的文件，禁止顺手纳入现有无关文档或测试资产。

Task 1 的首次提交完成后，Task 2–21 全部从专用 worktree 根目录执行；实施过程不修改本计划中的 checkbox，进度由执行会话记录，避免计划文件本身造成无关 dirty state。

### 0.2 冻结的事务语义

全局状态：

```text
NEW -> RESERVED -> PREPARED -> EXECUTED -> COMMIT_DECIDED
     -> LOCAL_COMMITTED -> PUBLISHED

NEW/RESERVED/PREPARED/EXECUTED -> ABORTED
COMMIT_DECIDED/部分COMMITTED + 无法确认全部提交 -> FATAL
```

per-rank 状态固定为 `IDLE/RESERVED/PREPARED/EXECUTED/COMMITTED/ABORTED/LOST`。`LOCAL_COMMITTED` 表示所有 rank 已确认，不表示单个 rank 的 ack。两个 rank 是持有相同微型状态的确定性事务参与者，只用于验证协议，不宣称 tensor parallel。

- 相同 epoch、相同 payload 的重复命令返回原 ack，不重复修改资源或状态。
- 相同 epoch、不同 payload 的重复命令返回 `INTERNAL`，并将 replica 置为 fatal。
- 旧 epoch 与迟到响应返回稳定 stale 结果，不修改计数、digest 或 canonical state。
- commit decision 必须先写入 Controller 内存日志，再向任一 rank 发送 Commit。
- decision 前取消、deadline 或 rank failure：Abort，零发布，释放全部 reservation。
- decision 后取消或 transport detach：继续完成 commit，不开始下一 token；逻辑发布写入 discard sink，记录 `delivery=dropped_client_disconnect`。
- decision 后 commit-before-apply、commit-after-apply-but-ack-lost 或 timeout：事务和 replica 均 fatal，零发布；MVP 只允许重启 Engine 进程恢复。

### 0.3 冻结的 RPC/HTTP 契约

- RPC version 固定 `1.0`。major 不同或 client minor 大于 server minor 时，在触碰 Controller 前返回 `INCOMPATIBLE_VERSION`。
- 每帧为 4 字节 network byte order 的无符号长度，后接 protobuf payload；长度 `0` 或大于 `4 MiB` 时关闭连接并报告协议错误。
- 每个 UDS connection 只承载一个 unary 操作或一个 completion stream；completion stream 可追加同 request ID 的 Cancel。
- UDS 只接受 UTF-8 绝对路径，`os.fsencode(path)` 最多 100 字节；父目录 `0700`、socket `0600`。只删除同 uid、已确认不可连接的 stale socket；普通文件、符号链接和 active socket 永不删除。
- HTTP completion 请求只接受 `model`、`prompt_token_ids`、`max_tokens`、`stream:true`、`temperature:0`、`top_p:1`、`n:1`。这是 GRServe 的 pretokenized MVP extension，不宣称完整 OpenAI 文本接口。
- Engine 接收相对 `timeout_ms`，入口立即用注入的 steady clock 固化 deadline；跨进程不传 monotonic timestamp。
- API 必须先收到 `CompletionAccepted` 才返回 HTTP 200。此前的错误是 JSON；此后的错误是 SSE error frame，随后仍发送精确的 `data: [DONE]\n\n`。
- token chunk 的 `choices[0].text` 固定为空字符串，扩展字段 `token_ids:[id]`，顶层含 `sequence`；最终 chunk 含 `finish_reason:"length"` 与精确 usage。
- HTTP 错误映射：`INVALID_ARGUMENT/UNSUPPORTED_CAPABILITY→400`，`RESOURCE_EXHAUSTED→429`，`DEADLINE_EXCEEDED→504`，`INCOMPATIBLE_VERSION/RANK_FAILURE/REPLICA_FATAL/UDS unavailable→503`，`INTERNAL→500`。Pydantic 422 统一改写为该 JSON 错误形状的 400。

### 0.4 文件责任图

```text
include/grserve/         稳定公共 C/C++ 接口
src/common/              Status、ID、clock、hash 等无业务依赖基础件
src/hal/                 动态装载与 RAII，不含具体后端逻辑
src/hal_mock/            虚拟异步队列与 scripted 后端
src/hal_fault/           fault plan 与事务阶段 hook
src/hal_cpu/             MiniQwen 确定性 CPU HAL
src/model/               manifest、资产校验、微型模型
src/engine/              rank、admission、transaction、controller
src/rpc/                 framing、UDS、session、Engine 可执行程序
proto/                   唯一 wire schema
python/grserve_api/      无状态 HTTP/SSE adapter
tests/c/                 真正用 C11 编译的 ABI 测试
tests/cpp/               Engine/HAL/RPC 单元与集成测试
tests/python/            RPC/API 黑盒测试
models/mini-qwen38-hybrid/  冻结小模型资产
tools/                   codegen、preflight、统一验收与证据生成
docker/                  linux/amd64 构建门禁
```

## Task 1: 初始化 Git、uv 与跨平台构建骨架

**Files:**

- Create: `.gitignore`
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `CMakeLists.txt`
- Create: `CMakePresets.json`
- Create: `cmake/Dependencies.cmake`
- Create: `cmake/Sanitizers.cmake`
- Create: `python/grserve_api/__init__.py`
- Create: `tests/cpp/smoke_test.cc`
- Modify: `docs/superpowers/specs/2026-09-07-grserve-mac-mvp-design.md`

- [ ] **Step 1: 建立仓库，但不纳入无关现有文件**

Run:

```bash
git init -b main
git status --short
```

Expected: 初始化成功；现有 `docs/` 显示为 untracked，没有文件被自动提交。

- [ ] **Step 2: 写 smoke test 并观察构建入口缺失**

Create `tests/cpp/smoke_test.cc`:

```cpp
#include <gtest/gtest.h>

TEST(BuildSmoke, TestProcessRuns) { EXPECT_EQ(2 + 2, 4); }
```

Run:

```bash
uv run cmake --preset mac-arm64-debug
```

Expected: FAIL，指出不存在 `pyproject.toml` 或 CMake preset；这是工具链 RED，不是测试通过。

- [ ] **Step 3: 固定 Python/CMake 依赖与忽略规则**

`.python-version` 固定为 `3.11`。`pyproject.toml` 固定：

```toml
[project]
name = "grserve-api"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "fastapi==0.115.12",
  "httpx==0.28.1",
  "protobuf==6.30.2",
  "uvicorn==0.34.2",
]

[dependency-groups]
dev = [
  "cmake==4.0.2",
  "mypy==1.15.0",
  "ninja==1.13.2",
  "pytest==8.3.5",
  "pytest-asyncio==0.26.0",
  "ruff==0.11.8",
]

[build-system]
requires = ["hatchling==1.27.0"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["python/grserve_api"]

[tool.pytest.ini_options]
asyncio_mode = "strict"
testpaths = ["tests/python"]

[tool.uv]
package = true
```

`python/grserve_api/__init__.py` 固定为：

```python
__version__ = "0.1.0"
```

`.gitignore` 必须包含：

```gitignore
.DS_Store
~$*.docx
.venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
__pycache__/
*.py[cod]
build/
artifacts/
.worktrees/
```

Run:

```bash
uv python install 3.11
uv lock
uv sync --locked
```

Expected: `uv.lock` 生成；`uv run python --version` 输出 Python 3.11.x。

- [ ] **Step 4: 固定 CMake presets 与源码依赖**

将 GoogleTest `v1.16.0` 与 protobuf `v30.2` 作为 Git submodule 加入；gitlink 本身冻结实际 commit：

```bash
git submodule add -b v1.16.0 https://github.com/google/googletest.git third_party/googletest
git -C third_party/googletest checkout v1.16.0
git submodule add -b v30.2 https://github.com/protocolbuffers/protobuf.git third_party/protobuf
git -C third_party/protobuf checkout v30.2
git submodule status
```

`CMakeLists.txt` 固定 C11/C++20、`CMAKE_C_EXTENSIONS=OFF`、`CMAKE_CXX_EXTENSIONS=OFF`、`Threads::Threads`、CTest，并从 `cmake/Dependencies.cmake` 以 `add_subdirectory` 构建两个 submodule。Presets 至少定义 `mac-arm64-debug`、`mac-arm64-release`、`mac-arm64-asan-ubsan` 和 `mac-arm64-tsan`，所有构建目录放在 `build/<preset>`。

初始 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.28)
project(grserve VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CTest)
include(Dependencies)
include(Sanitizers)
find_package(Threads REQUIRED)

add_executable(grserve_smoke_test tests/cpp/smoke_test.cc)
target_link_libraries(grserve_smoke_test PRIVATE GTest::gtest_main Threads::Threads)
add_test(NAME cpp.smoke COMMAND grserve_smoke_test)
```

`cmake/Dependencies.cmake`：

```cmake
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/protobuf EXCLUDE_FROM_ALL)
```

`cmake/Sanitizers.cmake`：

```cmake
if(GRSERVE_SANITIZERS STREQUAL "address,undefined")
  add_compile_options(-fno-omit-frame-pointer -fsanitize=address,undefined)
  add_link_options(-fno-omit-frame-pointer -fsanitize=address,undefined)
elseif(GRSERVE_SANITIZERS STREQUAL "thread")
  add_compile_options(-fno-omit-frame-pointer -fsanitize=thread)
  add_link_options(-fno-omit-frame-pointer -fsanitize=thread)
elseif(NOT GRSERVE_SANITIZERS STREQUAL "")
  message(FATAL_ERROR "unsupported GRSERVE_SANITIZERS=${GRSERVE_SANITIZERS}")
endif()
```

`CMakePresets.json` 用 version 6，hidden `base` preset 固定 generator `Ninja`、`CMAKE_OSX_ARCHITECTURES=arm64`；四个 mac configure preset 分别设置 Debug、Release、Debug+`address,undefined`、Debug+`thread`，并各有同名 build/test preset。Test preset 固定 `outputOnFailure=true`。

Run:

```bash
uv run --locked cmake --preset mac-arm64-debug
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.smoke$' --output-on-failure
```

Expected: PASS，`100% tests passed, 0 tests failed`。

- [ ] **Step 5: 提交可复现骨架**

```bash
git add .gitignore .python-version pyproject.toml uv.lock CMakeLists.txt CMakePresets.json cmake python/grserve_api/__init__.py tests/cpp/smoke_test.cc .gitmodules third_party/googletest third_party/protobuf docs/superpowers/specs/2026-09-07-grserve-mac-mvp-design.md docs/superpowers/plans/2026-09-07-grserve-mac-mvp.md
git commit -m "build: bootstrap reproducible GRServe workspace"
```

Expected: commit 成功；`git status --short` 只显示未纳入的原有文档，不显示本任务文件。

- [ ] **Step 6: 创建并切换到专用实施 worktree**

```bash
git worktree add .worktrees/grserve-mac-mvp -b feat/grserve-mac-mvp
cd .worktrees/grserve-mac-mvp
git status --short --branch
```

Expected: 当前分支为 `feat/grserve-mac-mvp`，工作树干净。Task 2–21 的所有相对路径和命令均以此目录为根。

## Task 2: 强类型 ID、版本与稳定错误模型

**Files:**

- Create: `include/grserve/types.h`
- Create: `include/grserve/error.h`
- Create: `include/grserve/version.h`
- Create: `src/common/error.cc`
- Create: `tests/cpp/types_test.cc`
- Create: `tests/cpp/error_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写不可混用 ID 与错误矩阵测试**

核心测试：

```cpp
static_assert(!std::is_convertible_v<grserve::RequestId, grserve::RankId>);
static_assert(!std::is_convertible_v<std::uint64_t, grserve::RequestId>);

TEST(Error, RetryabilityIsStable) {
  using grserve::ErrorCode;
  EXPECT_FALSE(grserve::Error::From(ErrorCode::kInvalidArgument, "bad").retryable());
  EXPECT_TRUE(grserve::Error::From(ErrorCode::kResourceExhausted, "full").retryable());
  EXPECT_FALSE(grserve::Error::From(ErrorCode::kReplicaFatal, "fatal").retryable());
}
```

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
```

Expected: FAIL，缺少 `grserve/types.h` 与 `grserve/error.h`。

- [ ] **Step 2: 实现最小公共类型**

公开 API 固定为：

```cpp
template <class Tag, class Rep = std::uint64_t>
class StrongId {
 public:
  explicit constexpr StrongId(Rep value) : value_(value) {}
  [[nodiscard]] constexpr Rep value() const noexcept { return value_; }
  auto operator<=>(const StrongId&) const = default;
 private:
  Rep value_;
};

using RequestId = StrongId<struct RequestIdTag>;
using ReplicaId = StrongId<struct ReplicaIdTag>;
using RankId = StrongId<struct RankIdTag, std::uint32_t>;
using TransactionId = StrongId<struct TransactionIdTag>;
using Epoch = StrongId<struct EpochTag>;
using SequenceNumber = StrongId<struct SequenceNumberTag>;
using TokenId = StrongId<struct TokenIdTag, std::uint32_t>;
```

`ErrorCode` 必须完整覆盖设计规格八类错误；实现 C++20 `Status` 与 move-aware `StatusOr<T>`，不使用 C++23 `std::expected`。`version.h` 固定 Engine `0.1.0`、RPC `1.0`、HAL ABI `1.0`。

retryable 矩阵固定为：InvalidArgument=false、IncompatibleVersion=false、UnsupportedCapability=false、ResourceExhausted=true、DeadlineExceeded=true、RankFailure=true、ReplicaFatal=false、Internal=false。

- [ ] **Step 3: 运行局部与全量测试**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.(types|error)$' --output-on-failure
uv run --locked ctest --preset mac-arm64-debug --output-on-failure
```

Expected: 两个局部测试及 smoke 全部 PASS。

- [ ] **Step 4: 提交**

```bash
git add include/grserve/types.h include/grserve/error.h include/grserve/version.h src/common/error.cc tests/cpp/types_test.cc tests/cpp/error_test.cc CMakeLists.txt
git commit -m "feat: define stable GRServe core types"
```

## Task 3: 建立纯 C11 HAL ABI

**Files:**

- Create: `include/grserve/hal.h`
- Create: `tests/c/hal_abi_smoke.c`
- Create: `tests/cpp/hal_abi_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 先写 C11 编译与布局测试**

`hal_abi_smoke.c` 必须由 C compiler 编译：

```c
#include "grserve/hal.h"
#include <stdint.h>

int main(void) {
  grserve_hal_api_request_t request = {0};
  grserve_hal_api_t api = {0};
  request.struct_size = (uint32_t)sizeof(request);
  request.abi_major = GRSERVE_HAL_ABI_MAJOR;
  request.min_abi_minor = GRSERVE_HAL_ABI_MINOR;
  api.struct_size = (uint32_t)sizeof(api);
  return request.struct_size == 0u || api.struct_size == 0u;
}
```

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target hal_abi_smoke
```

Expected: FAIL，`grserve/hal.h` 不存在。

- [ ] **Step 2: 定义 ABI vtable 与唯一入口**

`hal.h` 只能使用固定宽度整数、C pointer、opaque handle 和 caller-sized struct；禁止 `bool`、`size_t`、C++ enum、STL 或异常。唯一导出符号：

```c
GRSERVE_HAL_EXPORT grserve_hal_status_t grserve_hal_get_api(
    const grserve_hal_api_request_t* request,
    grserve_hal_api_t* out_api);
```

`grserve_hal_api_t` 的稳定前缀依次包含：`struct_size`、ABI major/minor、capabilities、backend/build 字符串函数、runtime/device/stream/event/buffer/executable 的 create/destroy、copy、submit、record/wait/query/synchronize/load/launch 与 TLS last-error copy。所有 destroy 接受 pointer-to-handle，`destroy(NULL)` 和空 handle 成功；首次销毁置空，已释放裸地址不得再次传入。

- [ ] **Step 3: 验证 C/C++ 两侧的 struct-size 兼容规则**

测试必须覆盖：完整 struct、截短到稳定前缀、尾部扩展、更小于必需前缀、major 不同、minor 不足。Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^c.hal_header|^cpp.hal_abi$' --output-on-failure
```

Expected: C11 smoke 与 C++ ABI 测试全部 PASS。

- [ ] **Step 4: 提交**

```bash
git add include/grserve/hal.h tests/c/hal_abi_smoke.c tests/cpp/hal_abi_test.cc CMakeLists.txt
git commit -m "feat: freeze GRServe HAL C ABI"
```

## Task 4: 动态 HAL loader 与 move-only RAII

**Files:**

- Create: `src/hal/hal_library.h`
- Create: `src/hal/hal_library.cc`
- Create: `src/hal/hal_raii.h`
- Create: `src/hal/hal_raii.cc`
- Create: `tests/fixtures/hal_good.cc`
- Create: `tests/fixtures/hal_bad_major.cc`
- Create: `tests/fixtures/hal_missing_capability.cc`
- Create: `tests/cpp/hal_loader_test.cc`
- Create: `tests/cpp/hal_raii_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写真实动态库装载失败测试**

测试通过 CMake 的 `$<TARGET_FILE:hal_good>` 注入路径，禁止硬编码 `.dylib`、`.so` 或环境库路径。断言 bad major、缺 capability 与过短 vtable 都在 `runtime_create_calls==0` 时失败。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
```

Expected: FAIL，缺少 `HalLibrary::Open`。

- [ ] **Step 2: 实现 loader 生命周期**

API 固定为：

```cpp
class HalLibrary {
 public:
  static StatusOr<std::shared_ptr<HalLibrary>> Open(
      const std::filesystem::path& path,
      std::uint64_t required_capabilities);
  const grserve_hal_api_t& api() const noexcept;
  HalLibrary(const HalLibrary&) = delete;
  HalLibrary& operator=(const HalLibrary&) = delete;
 private:
  void* native_library_;
  grserve_hal_api_t api_;
};
```

macOS 使用 `dlopen/dlsym/dlclose`，Linux 使用同一 POSIX 接口。每个 `extern "C"` fixture 捕获全部异常并复制 TLS error。所有 RAII handle 为 move-only，持有 `shared_ptr<HalLibrary>`，保证最后一个 handle 销毁前不会 `dlclose`。

- [ ] **Step 3: 验证销毁与装载次序**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.hal_(loader|raii)$' --output-on-failure
```

Expected: PASS；成功、create 失败、move、显式 reset 和作用域退出的 create/destroy 计数严格相等。

- [ ] **Step 4: 提交**

```bash
git add src/hal tests/fixtures/hal_good.cc tests/fixtures/hal_bad_major.cc tests/fixtures/hal_missing_capability.cc tests/cpp/hal_loader_test.cc tests/cpp/hal_raii_test.cc CMakeLists.txt
git commit -m "feat: load and own HAL plugins safely"
```

## Task 5: 虚拟时钟与 MockAsync 完成队列

**Files:**

- Create: `src/common/clock.h`
- Create: `src/hal_mock/mock_scheduler.h`
- Create: `src/hal_mock/mock_scheduler.cc`
- Create: `src/hal_mock/scripted_result.h`
- Create: `src/hal_mock/scripted_result.cc`
- Create: `src/hal_mock/mock_hal.cc`
- Create: `tests/cpp/mock_async_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写不依赖 sleep 的异步顺序测试**

测试创建 `ManualClock`，提交 rank0/rank1 两个 operation，分别以 `0→1` 与 `1→0` 推进，验证 pending/ready/failed 和 event ordinal；任何测试不得调用 `sleep_for`。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
```

Expected: FAIL，缺少 `ManualClock` 与 `MockScheduler`。

- [ ] **Step 2: 实现 deterministic scheduler 与 scripted result**

接口固定为：

```cpp
class ManualClock final : public Clock {
 public:
  time_point Now() const noexcept override;
  void Advance(duration delta);
};

class MockScheduler {
 public:
  OperationId Submit(RankId rank, ScriptedResult result);
  Status Complete(OperationId id);
  Status Fail(OperationId id, Error error);
  std::vector<Completion> DrainReady();
};
```

`ScriptedResult` 明确携带 candidate token、canonical state digest 与虚拟耗时。Mock HAL 对外仍只暴露 Task 3 的 C vtable。

- [ ] **Step 3: 重复执行乱序测试**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.mock_async$' --repeat until-fail:100 --output-on-failure
```

Expected: 100 次全部 PASS，事件序列摘要一致。

- [ ] **Step 4: 提交**

```bash
git add src/common/clock.h src/hal_mock tests/cpp/mock_async_test.cc CMakeLists.txt
git commit -m "feat: add deterministic asynchronous mock HAL"
```

## Task 6: 声明式 FaultPlan 与事务阶段故障 hook

**Files:**

- Create: `src/hal_fault/fault_plan.h`
- Create: `src/hal_fault/fault_plan.cc`
- Create: `src/hal_fault/fault_hook.h`
- Create: `src/hal_fault/fault_hal.cc`
- Create: `tests/cpp/fault_plan_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写精确 occurrence 与可重复摘要测试**

测试冻结 schema：

```cpp
FaultRule rule{
    .rank = RankId{1},
    .phase = FaultPhase::kCommit,
    .occurrence = 2,
    .action = FaultAction::kDropAck,
    .delay_ticks = 0,
};
FaultPlan plan(/*seed=*/17, {rule});
EXPECT_FALSE(plan.Match(RankId{1}, FaultPhase::kCommit).has_value());
EXPECT_EQ(plan.Match(RankId{1}, FaultPhase::kCommit)->action,
          FaultAction::kDropAck);
EXPECT_EQ(plan.CanonicalDigest(),
          "seed=17;rank=1,phase=commit,occurrence=2,action=drop_ack,delay=0");
```

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
```

Expected: FAIL，缺少 `FaultPlan`。

- [ ] **Step 2: 实现 fault plan 与 hook 边界**

action 固定为 `FAIL/BLOCK/TIMEOUT/DROP_ACK/RANK_LOSS`；phase 固定为 `RESERVE/PREPARE/EXECUTE/COMMIT/ABORT/HAL_LAUNCH`。HAL launch 故障由 `fault_hal.cc` 包装 Mock HAL；prepare/commit 等事务阶段通过以下接口注入，不能污染稳定 HAL ABI：

```cpp
class IFaultHook {
 public:
  virtual ~IFaultHook() = default;
  virtual std::optional<FaultEffect> Before(
      RankId rank, FaultPhase phase, std::uint64_t ordinal) = 0;
  virtual std::optional<FaultEffect> After(
      RankId rank, FaultPhase phase, std::uint64_t ordinal) = 0;
};
```

规范化摘要排除 wall/monotonic timestamp 与 run ID，只包含 seed、规则和稳定事件序列。

- [ ] **Step 3: 验证五类 action 与复现性**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.fault_plan$' --repeat until-fail:100 --output-on-failure
```

Expected: PASS；相同 seed 的规范化摘要 100 次相同，不同 seed 不改变显式 occurrence 规则。

- [ ] **Step 4: 提交**

```bash
git add src/hal_fault tests/cpp/fault_plan_test.cc CMakeLists.txt
git commit -m "feat: add reproducible fault injection"
```

## Task 7: 聚合 Transaction 状态机

**Files:**

- Create: `src/engine/rank_protocol.h`
- Create: `src/engine/transaction.h`
- Create: `src/engine/transaction.cc`
- Create: `tests/cpp/transaction_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写合法路径、非法跳转与幂等测试**

测试直接表达预期 API：

```cpp
std::array ranks{RankId{0}, RankId{1}};
auto tx = Transaction::Begin(TransactionId{7}, Epoch{3},
                             ranks).value();
EXPECT_TRUE(tx.AcknowledgePrepared(PreparedAck{RankId{1}, Epoch{3}}).ok());
EXPECT_TRUE(tx.AcknowledgePrepared(PreparedAck{RankId{0}, Epoch{3}}).ok());
EXPECT_EQ(tx.state(), TransactionState::kPrepared);
EXPECT_FALSE(tx.DecideCommit().ok());
EXPECT_EQ(tx.state(), TransactionState::kPrepared);
```

另测 complete path、重复相同 ack、冲突重复、旧 epoch、decision 前 Abort、decision 后 Abort 被拒绝、publish twice 被拒绝。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
```

Expected: FAIL，缺少 `Transaction`。

- [ ] **Step 2: 实现纯值状态机**

API 固定为：

```cpp
class Transaction {
 public:
  static StatusOr<Transaction> Begin(
      TransactionId id, Epoch epoch, std::span<const RankId> ranks);
  Status ReserveAll();
  Status AcknowledgePrepared(const PreparedAck& ack);
  Status AcknowledgeExecuted(const ExecutedAck& ack);
  Status DecideCommit();
  Status AcknowledgeCommitted(const CommittedAck& ack);
  Status MarkPublished(SequenceNumber sequence);
  Status Abort(const Error& reason);
  Status MarkFatal(const Error& reason);
  [[nodiscard]] TransactionState state() const noexcept;
  [[nodiscard]] bool may_abort() const noexcept;
};
```

ExecutedAck 的 candidate token 在所有 rank 间必须相同；state digest 作为 per-rank 值分别记录，不要求两个 shard digest 相等。

- [ ] **Step 3: 穷举状态转换表**

用参数化测试对每个全局状态尝试每种事件，断言状态、错误码和 mutation count。Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.transaction$' --output-on-failure
```

Expected: PASS，且无未覆盖的 state/event 组合。

- [ ] **Step 4: 提交**

```bash
git add src/engine/rank_protocol.h src/engine/transaction.h src/engine/transaction.cc tests/cpp/transaction_test.cc CMakeLists.txt
git commit -m "feat: enforce transaction state invariants"
```

## Task 8: RankWorker shadow/canonical 与全-rank admission

**Files:**

- Create: `src/engine/rank_worker.h`
- Create: `src/engine/rank_worker.cc`
- Create: `src/engine/completion_queue.h`
- Create: `src/engine/completion_queue.cc`
- Create: `src/engine/admission.h`
- Create: `src/engine/admission.cc`
- Create: `tests/cpp/rank_worker_test.cc`
- Create: `tests/cpp/admission_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写 canonical 隔离与原子 reservation 测试**

核心断言：Prepare/Execute/Abort 不改变 canonical digest；Commit 只交换一次；rank1 满载时 rank0/rank1 的 slot 计数在 admission 前后完全相同。

```cpp
auto before = worker.canonical_digest();
ASSERT_TRUE(worker.Enqueue(prepare).ok());
ASSERT_TRUE(completions.WaitPop().has_value());
ASSERT_TRUE(worker.Enqueue(execute).ok());
ASSERT_TRUE(completions.WaitPop().has_value());
EXPECT_EQ(worker.canonical_digest(), before);
ASSERT_TRUE(worker.Enqueue(commit).ok());
ASSERT_TRUE(completions.WaitPop().has_value());
EXPECT_NE(worker.canonical_digest(), before);
```

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
```

Expected: FAIL，缺少 `RankWorker` 与 `AdmissionController`。

- [ ] **Step 2: 实现值队列、worker 与 move-only lease**

```cpp
using RankCommand =
    std::variant<PrepareCommand, ExecuteCommand, CommitCommand, AbortCommand>;
using RankCompletion =
    std::variant<PreparedAck, ExecutedAck, CommittedAck, AbortedAck, RankError>;

class RankWorker {
 public:
  Status Enqueue(RankCommand command);
  [[nodiscard]] StateDigest canonical_digest() const;
};

class AdmissionController {
 public:
  StatusOr<ReservationLease> ReserveAll(RequestId request,
                                        std::uint32_t state_slots);
};
```

一个 Worker 线程只写本 rank 状态，通过 `CompletionQueue<RankCompletion>` 回传值对象。ReservationLease move-only，析构释放且每 rank 恰好一次。

- [ ] **Step 3: 覆盖重复、迟到与并发关闭**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.(rank_worker|admission)$' --repeat until-fail:100 --output-on-failure
```

Expected: PASS；shutdown 后 enqueue 返回稳定错误；全部线程 join 后 handle 与 slot 计数为零。

- [ ] **Step 4: 提交**

```bash
git add src/engine/rank_worker.h src/engine/rank_worker.cc src/engine/completion_queue.h src/engine/completion_queue.cc src/engine/admission.h src/engine/admission.cc tests/cpp/rank_worker_test.cc tests/cpp/admission_test.cc CMakeLists.txt
git commit -m "feat: isolate rank state and reservations"
```

## Task 9: 两 rank Controller 正常提交与发布屏障

**Files:**

- Create: `src/engine/request.h`
- Create: `src/engine/request_event_sink.h`
- Create: `src/engine/replica_gate.h`
- Create: `src/engine/replica_gate.cc`
- Create: `src/engine/controller.h`
- Create: `src/engine/controller.cc`
- Create: `tests/cpp/controller_success_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写两个完成顺序的零提前发布测试**

`RecordingSink` 在每个 rank ack 后检查 token count；只有两个 commit ack 后为 1。反转 prepare、execute 和 commit 完成顺序，最终 token/sequence/digest 必须相同。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
```

Expected: FAIL，缺少 `Controller::Submit`。

- [ ] **Step 2: 实现单写者 Controller API**

```cpp
class RequestEventSink {
 public:
  virtual ~RequestEventSink() = default;
  virtual bool TryAccepted(RequestId) noexcept = 0;
  virtual bool TryToken(const TokenEvent&) noexcept = 0;
  virtual bool TryFinished(const FinishedEvent&) noexcept = 0;
  virtual bool TryError(const Error&) noexcept = 0;
};

class Controller {
 public:
  Status Submit(GenerateRequest request,
                std::shared_ptr<RequestEventSink> sink);
  Status Cancel(RequestId request, CancelReason reason);
  Status DetachTransport(RequestId request);
  void PollOnce();
};
```

Controller 线程是 request/transaction/sequence/replica gate 的唯一写者。Sink 方法只做 bounded nonblocking enqueue，Controller 不执行 socket 或文件 I/O。每个 token 使用新 TransactionId，SequenceNumber 从 1 严格递增。

- [ ] **Step 3: 验证 barrier 与顺序不变性**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.controller_success$' --repeat until-fail:100 --output-on-failure
```

Expected: PASS；两种完成顺序的规范化事件序列完全相同。

- [ ] **Step 4: 提交**

```bash
git add src/engine/request.h src/engine/request_event_sink.h src/engine/replica_gate.h src/engine/replica_gate.cc src/engine/controller.h src/engine/controller.cc tests/cpp/controller_success_test.cc CMakeLists.txt
git commit -m "feat: commit tokens behind an all-rank barrier"
```

## Task 10: deadline、取消、rank-loss 与 fatal 门禁

**Files:**

- Create: `tests/cpp/controller_faults_test.cc`
- Create: `tests/cpp/controller_cancel_test.cc`
- Modify: `src/engine/controller.cc`
- Modify: `src/engine/replica_gate.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写 decision 两侧的故障矩阵**

参数化 case 固定为：prepare fail、execute fail、pre-decision timeout、rank-loss、cancel before decision、commit fail before apply、commit apply then ack loss、post-decision timeout、detach after decision。每个 case 断言 transaction terminal、canonical 是否前进、logical publish、network delivery、reservation、replica health。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.controller_(faults|cancel)$' --output-on-failure
```

Expected: FAIL；当前 Controller 尚未实现完整故障边界。

- [ ] **Step 2: 实现相对 deadline 和粘性 ReplicaGate**

入口将 `timeout_ms` 转成 `Clock::Now()+duration`，之后只查注入 Clock。ReplicaGate API：

```cpp
class ReplicaGate {
 public:
  Status CheckAdmission() const;
  void MarkFatal(Error error);
  [[nodiscard]] ReplicaHealth health() const noexcept;
  [[nodiscard]] std::optional<Error> fatal_error() const;
};
```

`MarkFatal` 粘性且幂等；fatal 后所有 Submit 返回 `REPLICA_FATAL`，不会调用 RankWorker 或 HAL。decision 后 transport detach 完成 commit、逻辑 publish 到 discard sink、`delivery=dropped_client_disconnect`，并停止下一 token。

- [ ] **Step 3: 重复执行完整故障矩阵**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.controller_(faults|cancel)$' --repeat until-fail:100 --output-on-failure
```

Expected: 100 次 PASS；所有 decision 前失败均零发布且资源归零，所有不确定 commit 均零发布且 replica fatal。

- [ ] **Step 4: 运行 M3 回归并提交**

```bash
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.(transaction|rank_worker|admission|controller_)' --output-on-failure
git add src/engine/controller.cc src/engine/replica_gate.cc tests/cpp/controller_faults_test.cc tests/cpp/controller_cancel_test.cc CMakeLists.txt
git commit -m "feat: make cancellation and commit failures safe"
```

## Task 11: 生成并校验冻结的 MiniQwen 模型资产

**Files:**

- Create: `proto/model_manifest.proto`
- Create: `src/common/sha256.h`
- Create: `src/common/sha256.cc`
- Create: `src/model/manifest.h`
- Create: `src/model/manifest.cc`
- Create: `tools/generate_mini_model.py`
- Create: `models/mini-qwen38-hybrid/manifest.textproto`
- Create: `models/mini-qwen38-hybrid/weights.i32le`
- Create: `models/mini-qwen38-hybrid/golden.json`
- Create: `tests/cpp/sha256_test.cc`
- Create: `tests/cpp/manifest_test.cc`
- Create: `tests/python/test_model_assets.py`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写 SHA、严格 manifest 与重复生成测试**

C++ 先测 NIST 空串与 `abc` 向量；Python 测生成器连续运行两次产生 byte-for-byte 相同的三个资产，并验证 manifest 中 SHA256 等于 `weights.i32le` 实际哈希。将未知字段追加到 manifest 副本时，C++ loader 必须拒绝。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked pytest -q tests/python/test_model_assets.py
```

Expected: FAIL，缺少资产生成器与 manifest loader。

- [ ] **Step 2: 冻结模型尺寸、算术和资产格式**

`ModelManifest` 固定字段：schema version `1`、model ID `mini-qwen38-hybrid`、model format `1`、topology `GDN,GDN,GDN,FULL`、dtype `Q16_16`、vocab `16`、hidden `8`、max context `64`、weight SHA256、golden version `1`、HAL ABI `1.0`、required capability `EXECUTABLE_LAUNCH`。textproto parser 不允许未知字段。

生成器不使用 NumPy，权重公式固定为：

```python
def coefficient(index: int) -> int:
    return (((index * 17 + 3) % 23) - 11) * 1024
```

所有整数按 little-endian signed int32 写入 `weights.i32le`。模型运算使用 Q16.16、int64 中间值、饱和到 int32；round-shift 定义为：

```python
def round_shift(value: int, bits: int) -> int:
    magnitude = (abs(value) + (1 << (bits - 1))) >> bits
    return magnitude if value >= 0 else -magnitude
```

每个 GDN 层按固定 diagonal gate/input/state 系数更新 recurrent state；Full 层将 key/value 追加到 FullKV，并用所有 value 的逐维整数平均更新 hidden；输出用冻结矩阵计算 logits，argmax tie 选择最小 token ID。xorshift64 RNG 每个已提交 token 前进一次，但不参与 greedy argmax。

- [ ] **Step 3: 实现运行时完整性门禁**

Loader 顺序固定为：解析 manifest → 验 schema/模型字段 → 验文件大小 → 流式 SHA256 → 验 HAL ABI 范围与 capability → 返回 immutable `ModelAssets`。任一步失败时 HAL runtime/create/launch spy 均为零。

Run:

```bash
uv run --locked python tools/generate_mini_model.py --output models/mini-qwen38-hybrid
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.(sha256|manifest)$' --output-on-failure
uv run --locked pytest -q tests/python/test_model_assets.py
```

Expected: 全部 PASS，第二次生成后 `git diff --exit-code -- models/mini-qwen38-hybrid` 返回 0。

- [ ] **Step 4: 提交**

```bash
git add proto/model_manifest.proto src/common/sha256.h src/common/sha256.cc src/model/manifest.h src/model/manifest.cc tools/generate_mini_model.py models/mini-qwen38-hybrid tests/cpp/sha256_test.cc tests/cpp/manifest_test.cc tests/python/test_model_assets.py CMakeLists.txt
git commit -m "feat: freeze deterministic MiniQwen assets"
```

## Task 12: CPU Reference HAL、状态 golden 与三后端一致性

**Files:**

- Create: `src/model/mini_qwen.h`
- Create: `src/model/mini_qwen.cc`
- Create: `src/model/state_codec.h`
- Create: `src/model/state_codec.cc`
- Create: `src/hal_cpu/cpu_hal.cc`
- Create: `tests/reference_model.py`
- Create: `tests/cpp/mini_qwen_test.cc`
- Create: `tests/cpp/backend_parity_test.cc`
- Modify: `tools/generate_mini_model.py`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 用独立 Python oracle 冻结 token 与状态 golden**

`tests/reference_model.py` 只依赖 Python 标准库，按 Task 11 公式计算 prompt `[1, 4, 7]` 后连续 8 个 token。`golden.json` 每一步固定 `token_id`、FullKV、三个 GDN state、RNG 与 canonical digest。

Run:

```bash
uv run --locked python tests/reference_model.py --check models/mini-qwen38-hybrid/golden.json
```

Expected: FAIL，当前 golden 尚未包含完整 step states 或 `--check` 不存在。

- [ ] **Step 2: 写 C++ golden 与事务回滚测试**

测试加载真实冻结资产，逐 token 与 Python golden 比较；digest 输入固定为 model format version、little-endian vector lengths、所有 int32 FullKV/GDN 值和 uint64 RNG。另测 Execute 产生 next state 后 Abort，下一次执行仍从最后 canonical commit 继续。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
```

Expected: FAIL，缺少 `MiniQwen::ExecuteNext` 与 state codec。

- [ ] **Step 3: 实现 CPU HAL 与 scripted Mock parity**

公开模型 API：

```cpp
class MiniQwen {
 public:
  static StatusOr<MiniQwen> Load(const ModelAssets& assets);
  StatusOr<CandidateStep> ExecuteNext(std::span<const TokenId> context,
                                      const CanonicalState& state) const;
};
```

CPU HAL 通过稳定 vtable load/launch 模型；Mock scripted result 从同一冻结 golden 读取 token/digest，但不调用 CPU 模型。Fault HAL 包装 Mock。三者对同一请求的 accepted/token/finished/error 协议序列必须一致，内部 timing event 可不同。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.(mini_qwen|backend_parity)$' --output-on-failure
uv run --locked python tests/reference_model.py --check models/mini-qwen38-hybrid/golden.json
```

Expected: PASS，arm64 上 token 和 digest 精确匹配；数值中间值不使用 float、libm 或 FMA。

- [ ] **Step 4: 提交**

```bash
git add src/model/mini_qwen.h src/model/mini_qwen.cc src/model/state_codec.h src/model/state_codec.cc src/hal_cpu/cpu_hal.cc tests/reference_model.py tests/cpp/mini_qwen_test.cc tests/cpp/backend_parity_test.cc tools/generate_mini_model.py models/mini-qwen38-hybrid/golden.json CMakeLists.txt
git commit -m "feat: execute deterministic MiniQwen on CPU"
```

## Task 13: 冻结 protobuf schema 与双语言 codegen

**Files:**

- Create: `proto/engine.proto`
- Create: `proto/CMakeLists.txt`
- Create: `tools/proto_codegen.py`
- Modify: `python/grserve_api/__init__.py`
- Create: `python/grserve_api/_generated/__init__.py`
- Create: `python/grserve_api/_generated/engine_pb2.py`
- Create: `tests/cpp/rpc/proto_contract_test.cc`
- Create: `tests/python/test_proto_codegen.py`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写字段号与生成物同步测试**

Python 测试执行 `tools/proto_codegen.py --check`；C++ 测试从 descriptor 断言所有 enum 值、oneof 和关键 field number。Run:

```bash
uv run --locked pytest -q tests/python/test_proto_codegen.py
```

Expected: FAIL，schema 与 codegen 工具不存在。

- [ ] **Step 2: 创建完整 RPC v1 schema**

`proto/engine.proto` 固定为：

```proto
syntax = "proto3";
package grserve.rpc.v1;

message ProtocolVersion { uint32 major = 1; uint32 minor = 2; }
enum ErrorCode {
  ERROR_CODE_UNSPECIFIED = 0;
  ERROR_CODE_INVALID_ARGUMENT = 1;
  ERROR_CODE_INCOMPATIBLE_VERSION = 2;
  ERROR_CODE_UNSUPPORTED_CAPABILITY = 3;
  ERROR_CODE_RESOURCE_EXHAUSTED = 4;
  ERROR_CODE_DEADLINE_EXCEEDED = 5;
  ERROR_CODE_RANK_FAILURE = 6;
  ERROR_CODE_REPLICA_FATAL = 7;
  ERROR_CODE_INTERNAL = 8;
}
message RpcError {
  ErrorCode code = 1;
  string message = 2;
  string request_id = 3;
  string transaction_id = 4;
  bool retryable = 5;
}
message HealthRequest {}
enum ReplicaState {
  REPLICA_STATE_UNSPECIFIED = 0;
  REPLICA_STATE_STARTING = 1;
  REPLICA_STATE_HEALTHY = 2;
  REPLICA_STATE_FATAL = 3;
  REPLICA_STATE_STOPPING = 4;
}
message HealthResponse {
  string engine_build_id = 1;
  bool accepting_requests = 2;
  ReplicaState replica_state = 3;
  RpcError fatal_error = 4;
}
message ListModelsRequest {}
message ModelInfo {
  string id = 1;
  string manifest_sha256 = 2;
  uint32 max_context_tokens = 3;
}
message ListModelsResponse { repeated ModelInfo models = 1; }
enum DecodeMode {
  DECODE_MODE_UNSPECIFIED = 0;
  DECODE_MODE_GREEDY = 1;
}
message CompletionRequest {
  string request_id = 1;
  string model_id = 2;
  repeated uint32 prompt_token_ids = 3 [packed = true];
  uint32 max_tokens = 4;
  uint64 timeout_ms = 5;
  DecodeMode decode_mode = 6;
}
enum CancelReason {
  CANCEL_REASON_UNSPECIFIED = 0;
  CANCEL_REASON_CLIENT_DISCONNECTED = 1;
  CANCEL_REASON_DEADLINE = 2;
  CANCEL_REASON_SERVER_SHUTDOWN = 3;
}
message CancelRequest { string request_id = 1; CancelReason reason = 2; }
message CompletionAccepted { string request_id = 1; }
message TokenEvent {
  string request_id = 1;
  string transaction_id = 2;
  uint64 sequence_number = 3;
  uint32 token_id = 4;
}
message Usage {
  uint32 prompt_tokens = 1;
  uint32 completion_tokens = 2;
  uint32 total_tokens = 3;
}
enum FinishReason {
  FINISH_REASON_UNSPECIFIED = 0;
  FINISH_REASON_LENGTH = 1;
  FINISH_REASON_CANCELLED = 2;
}
message CompletionFinished {
  string request_id = 1;
  FinishReason finish_reason = 2;
  Usage usage = 3;
  uint64 last_sequence_number = 4;
}
enum CancelDisposition {
  CANCEL_DISPOSITION_UNSPECIFIED = 0;
  CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION = 1;
  CANCEL_DISPOSITION_COMMITTING_DETACHED = 2;
  CANCEL_DISPOSITION_ALREADY_TERMINAL = 3;
  CANCEL_DISPOSITION_NOT_FOUND = 4;
}
message CancelAck { string request_id = 1; CancelDisposition disposition = 2; }
message ClientEnvelope {
  ProtocolVersion protocol_version = 1;
  oneof payload {
    HealthRequest health = 10;
    ListModelsRequest list_models = 11;
    CompletionRequest completion = 12;
    CancelRequest cancel = 13;
  }
}
message ServerEnvelope {
  ProtocolVersion protocol_version = 1;
  oneof payload {
    HealthResponse health = 10;
    ListModelsResponse list_models = 11;
    CompletionAccepted accepted = 12;
    TokenEvent token = 13;
    CompletionFinished finished = 14;
    CancelAck cancel_ack = 15;
    RpcError error = 16;
  }
}
```

- [ ] **Step 3: 用同一个 pinned protoc 生成 C++ 与 Python**

C++ 文件只生成到 `build/<preset>/generated`；Python `engine_pb2.py` 提交仓库。`proto_codegen.py --check` 在临时目录重生成并 byte compare，不修改工作树。

Run:

```bash
uv run --locked cmake --preset mac-arm64-debug
uv run --locked cmake --build --preset mac-arm64-debug --target check_proto_generated
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.rpc.proto_contract$' --output-on-failure
uv run --locked pytest -q tests/python/test_proto_codegen.py
```

Expected: 全部 PASS；Python runtime 版本为 `6.30.2`，protoc 源自 `third_party/protobuf` 的 v30.2 build。

- [ ] **Step 4: 提交**

```bash
git add proto/engine.proto proto/CMakeLists.txt tools/proto_codegen.py python/grserve_api/__init__.py python/grserve_api/_generated tests/cpp/rpc/proto_contract_test.cc tests/python/test_proto_codegen.py CMakeLists.txt
git commit -m "feat: freeze Engine RPC protocol v1"
```

## Task 14: 长度帧 codec 与安全 UDS ownership

**Files:**

- Create: `src/rpc/frame_codec.h`
- Create: `src/rpc/frame_codec.cc`
- Create: `src/rpc/unix_socket.h`
- Create: `src/rpc/unix_socket.cc`
- Create: `tests/cpp/rpc/frame_codec_test.cc`
- Create: `tests/cpp/rpc/unix_socket_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写 framing 的每个 split point 测试**

覆盖 header/body 的每个拆分点、两帧粘包、EINTR、header 中 EOF、body 中 EOF、长度 0、`4 MiB+1`、invalid protobuf 与 partial write。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
```

Expected: FAIL，缺少 `FrameReader` 与 `FrameWriter`。

- [ ] **Step 2: 实现 bounded codec**

```cpp
inline constexpr std::uint32_t kMaxFrameBytes = 4U * 1024U * 1024U;

class FrameReader {
 public:
  StatusOr<grserve::rpc::v1::ClientEnvelope> ReadOne(int fd);
};
class FrameWriter {
 public:
  Status WriteOne(int fd, const grserve::rpc::v1::ServerEnvelope& envelope);
};
```

读取循环处理 EINTR 与半包；0/超限在分配 body 前失败。macOS 对 socket 设置 `SO_NOSIGPIPE`，Linux send 使用 `MSG_NOSIGNAL`；全部 fd 设置 `FD_CLOEXEC`。

- [ ] **Step 3: 写 UDS 破坏性边界测试并实现**

测试使用短路径 `/tmp/grserve-test.<pid>/engine.sock`，覆盖 100/101 字节边界、相对路径/NUL、权限、active listener、stale socket、普通文件、符号链接和退出前 inode 被替换。实现只在 `lstat` 为同 uid socket、connect 明确拒绝时 unlink；析构时必须比对启动时 device/inode。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.rpc.(frame_codec|unix_socket)$' --output-on-failure
```

Expected: PASS；普通文件、符号链接与 active socket 内容/身份保持不变。

- [ ] **Step 4: 提交**

```bash
git add src/rpc/frame_codec.h src/rpc/frame_codec.cc src/rpc/unix_socket.h src/rpc/unix_socket.cc tests/cpp/rpc/frame_codec_test.cc tests/cpp/rpc/unix_socket_test.cc CMakeLists.txt
git commit -m "feat: frame protobuf safely over Unix sockets"
```

## Task 15: RpcSession、Controller bridge 与 Engine 进程

**Files:**

- Create: `src/rpc/rpc_session.h`
- Create: `src/rpc/rpc_session.cc`
- Create: `src/rpc/engine_rpc_server.h`
- Create: `src/rpc/engine_rpc_server.cc`
- Create: `src/rpc/engine_main.cc`
- Create: `tests/cpp/rpc/rpc_session_test.cc`
- Create: `tests/cpp/rpc/disconnect_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写协议顺序、背压与断开竞态测试**

Fake Controller 断言 version/admission error 不触发 rank/HAL；completion 首帧必须 Accepted；token sequence 严格 +1；重复/错 request Cancel 稳定失败；不读 socket 的客户端不能阻塞 Controller PollOnce。断开测试以可控 barrier 排列 `Detach` 与 final ExecuteAck，不使用 sleep。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
```

Expected: FAIL，缺少 `RpcSession` 与 `EngineRpcServer`。

- [ ] **Step 2: 实现 session reader/writer 与非阻塞 sink**

每个 session 有 reader、bounded outbound queue 和 writer；`RequestEventSink::Try*` 只 enqueue。队列耗尽在 Accepted 前返回 `RESOURCE_EXHAUSTED`；Accepted 后无法排入 token 是 transport failure，向 Controller 投递一次 Detach。EOF、EPIPE 与显式 Cancel 都归一为同一幂等 Controller event。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.rpc.session$' --output-on-failure
```

Expected: session 测试 PASS。

- [ ] **Step 3: 实现可管理的 Engine 进程生命周期**

`grserve-engine` 参数固定为 `--socket`、`--model-dir`、`--hal-plugin`、`--ranks=2`、`--max-concurrent-requests`、`--max-state-slots-per-rank`。启动顺序：解析参数 → 安全 bind → manifest/HAL 门禁 → 创建 rank/Controller → 开始 accept；SIGINT/SIGTERM 先停止 admission，再 bounded drain、join、按 inode 安全删 socket。

Run:

```bash
build/mac-arm64-debug/bin/grserve-engine --help
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.rpc.(session|disconnect)$' --repeat until-fail:100 --output-on-failure
```

Expected: `--help` 退出 0；RPC 测试 100 次 PASS；线程、fd、HAL handle 计数全部归零。

- [ ] **Step 4: 提交**

```bash
git add src/rpc/rpc_session.h src/rpc/rpc_session.cc src/rpc/engine_rpc_server.h src/rpc/engine_rpc_server.cc src/rpc/engine_main.cc tests/cpp/rpc/rpc_session_test.cc tests/cpp/rpc/disconnect_test.cc CMakeLists.txt
git commit -m "feat: serve Engine RPC over Unix sockets"
```

## Task 16: Python async UDS RPC client

**Files:**

- Create: `python/grserve_api/config.py`
- Create: `python/grserve_api/rpc_client.py`
- Create: `tests/python/support/__init__.py`
- Create: `tests/python/support/fake_engine.py`
- Create: `tests/python/test_rpc_client.py`

- [ ] **Step 1: 写半包、协议顺序与取消锁测试**

Fake Engine 可按任意 byte split 返回 frame。测试覆盖 `readexactly`、0/超限 frame、EOF、首事件不是 Accepted、request ID 不匹配、sequence 跳号、同连接 Cancel 与 writer lock、`aclose()` 两次、连接失败。

Run:

```bash
uv run --locked pytest -q tests/python/test_rpc_client.py
```

Expected: FAIL，无法导入 `grserve_api.rpc_client`。

- [ ] **Step 2: 实现严格配置与 EngineClient**

公开行为固定为：

```python
@dataclass(frozen=True, slots=True)
class ApiConfig:
    engine_socket: Path
    request_timeout_ms: int = 30_000
    cancel_timeout_ms: int = 250

class EngineClient:
    async def health(self) -> HealthResponse:
        response = await self._unary(client_envelope_for_health())
        return require_payload(response, "health")

    async def list_models(self) -> ListModelsResponse:
        response = await self._unary(client_envelope_for_models())
        return require_payload(response, "list_models")

    async def start_completion(
        self, request: CompletionRequest
    ) -> "CompletionStream":
        reader, writer = await asyncio.open_unix_connection(
            str(self._config.engine_socket)
        )
        await write_client_envelope(writer, client_envelope_for_completion(request))
        return await CompletionStream.create(reader, writer, request.request_id)

class CompletionStream:
    accepted: CompletionAccepted

    def __aiter__(self) -> AsyncIterator[ServerEnvelope]:
        return self._events()

    async def cancel(self, reason: CancelReason) -> CancelAck | None:
        async with self._writer_lock:
            if self._closed:
                return None
            await write_client_envelope(
                self._writer,
                client_envelope_for_cancel(self.accepted.request_id, reason),
            )
        return await asyncio.wait_for(
            asyncio.shield(self._cancel_ack_future),
            timeout=self._config.cancel_timeout_ms / 1000,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.close()
        await self._writer.wait_closed()
```

`CompletionStream.create()` 启动唯一 reader pump；pump 先验证 Accepted，再把 token/finished/error 放入 event queue，把 CancelAck 完成到 `_cancel_ack_future`。任何时候都只有 reader pump 读取 StreamReader。辅助函数与 `_events()` 负责 4-byte framing、4 MiB 上限、protocol 1.0、RpcError、request ID 与 sequence 校验。配置拒绝相对/NUL/>100-byte socket path。所有 writer 操作由一个 `asyncio.Lock` 串行，cancel/aclose 幂等；`aclose()` 还必须 cancel 并 await reader task。

- [ ] **Step 3: 运行 RPC client 测试**

```bash
uv run --locked pytest -q tests/python/test_rpc_client.py
uv run --locked ruff check python tests/python/test_rpc_client.py
uv run --locked mypy python/grserve_api
```

Expected: pytest、ruff、mypy 全部退出 0。

- [ ] **Step 4: 提交**

```bash
git add python/grserve_api/config.py python/grserve_api/rpc_client.py tests/python/support tests/python/test_rpc_client.py
git commit -m "feat: connect API adapter to Engine RPC"
```

## Task 17: 严格 HTTP schema、错误映射与 SSE codec

**Files:**

- Create: `python/grserve_api/schemas.py`
- Create: `python/grserve_api/errors.py`
- Create: `python/grserve_api/sse.py`
- Create: `tests/python/test_schemas.py`
- Create: `tests/python/test_errors.py`
- Create: `tests/python/test_sse.py`

- [ ] **Step 1: 写请求矩阵与逐字节 SSE 测试**

合法请求固定为：

```json
{
  "model": "mini-qwen38-hybrid",
  "prompt_token_ids": [1, 4, 7],
  "max_tokens": 8,
  "stream": true,
  "temperature": 0,
  "top_p": 1,
  "n": 1
}
```

测试拒绝额外字段、空 prompt、bool token、token≥16、`stream=false`、非零 temperature、非 1 top_p/n、max_tokens 超 context。所有八种 ErrorCode 映射必须固定。把 SSE 输出按每个字节切分后仍能解析 token/final/error 与 `[DONE]`。

Run:

```bash
uv run --locked pytest -q tests/python/test_schemas.py tests/python/test_errors.py tests/python/test_sse.py
```

Expected: FAIL，三个模块不存在。

- [ ] **Step 2: 实现 schema 与错误脱敏**

Pydantic model 使用 `ConfigDict(extra="forbid", strict=True)`；对整数字段额外拒绝 `bool`。公开错误 JSON 固定为：

```json
{
  "error": {
    "message": "stable public message",
    "type": "grserve_error",
    "code": "RESOURCE_EXHAUSTED",
    "request_id": "req-1",
    "retryable": true
  }
}
```

后端堆栈、socket 路径、动态库路径和 fault plan 不进入公开 message。

- [ ] **Step 3: 实现精确 SSE 输出**

首个 token 与结束帧示例：

```text
data: {"id":"req-1","object":"text_completion","model":"mini-qwen38-hybrid","sequence":1,"choices":[{"index":0,"text":"","token_ids":[9],"finish_reason":null}]}

data: {"id":"req-1","object":"text_completion","model":"mini-qwen38-hybrid","sequence":8,"choices":[{"index":0,"text":"","token_ids":[],"finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":8,"total_tokens":11}}

data: [DONE]

```

JSON 编码使用 compact separators，测试比较解析后的 JSON 和精确 DONE bytes，不依赖 key 顺序。

Run:

```bash
uv run --locked pytest -q tests/python/test_schemas.py tests/python/test_errors.py tests/python/test_sse.py
uv run --locked ruff check python tests/python
```

Expected: 全部 PASS。

- [ ] **Step 4: 提交**

```bash
git add python/grserve_api/schemas.py python/grserve_api/errors.py python/grserve_api/sse.py tests/python/test_schemas.py tests/python/test_errors.py tests/python/test_sse.py
git commit -m "feat: freeze pretokenized completion contract"
```

## Task 18: FastAPI routes、断线语义与真实 E2E

**Files:**

- Create: `python/grserve_api/app.py`
- Create: `python/grserve_api/main.py`
- Create: `tests/python/support/engine_process.py`
- Create: `tests/python/test_api_contract.py`
- Create: `tests/python/test_disconnect.py`
- Create: `tests/python/test_api_e2e.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: 写 app factory 与 pre-Accept 测试**

注入 Fake EngineClient。测试 `/health`、`/v1/models`、固定 request schema、Pydantic error 改写、media type `text/event-stream`、`Cache-Control:no-cache`、`X-Accel-Buffering:no`。Fake 在 Accepted 前返回各错误时，响应必须是真实 400/429/500/503/504 JSON，不能先发 200。

Run:

```bash
uv run --locked pytest -q tests/python/test_api_contract.py
```

Expected: FAIL，缺少 `create_app`。

- [ ] **Step 2: 实现 routes 与 post-Accept streaming**

```python
def create_app(config: ApiConfig, engine: EngineClient | None = None) -> FastAPI:
    app = FastAPI()
    client = engine or EngineClient(config)
    # 注册 health、models、completions 与统一 validation handler。
    return app
```

实际文件不得保留注释代替实现。Completion route 在创建 `StreamingResponse` 前 await `start_completion()` 并取得 Accepted。Accepted 后的 RpcError 编码为 SSE error，再发送 DONE，HTTP status 保持 200。

- [ ] **Step 3: 分层验证 disconnect boundary**

Python 测试直接驱动 ASGI `http.disconnect`，在生成器 `CancelledError/finally` 中以 `asyncio.shield` 和 250ms bound 调用 cancel 恰好一次并关闭 UDS。C++ `cpp.rpc.disconnect` 分别验证：Detach 先入 Controller queue 时 Abort/零发布/canonical 不变；final ExecuteAck 先导致 decision 时完成所有 commit、canonical 前进、network delivered=0、delivery=dropped。EOF/EPIPE/explicit Cancel 三入口幂等。

Run:

```bash
uv run --locked pytest -q tests/python/test_disconnect.py
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.rpc.disconnect$' --repeat until-fail:100 --output-on-failure
```

Expected: 全部 PASS；测试不依赖真实 sleep。

- [ ] **Step 4: 运行真实 engine + uvicorn E2E**

`engine_process.py` 启动 `$GRSERVE_ENGINE_BIN`，通过真实 health RPC 判断 ready，禁止只检查 socket 文件。E2E 分别加载 CPU HAL 与 scripted Mock HAL，对 `[1,4,7]→8 tokens` 断言 token sequence、usage、finish_reason 和 DONE 完全一致；测试结束必须回收子进程与 socket。

Run:

```bash
GRSERVE_ENGINE_BIN="$PWD/build/mac-arm64-debug/bin/grserve-engine" uv run --locked pytest -q tests/python/test_api_e2e.py
```

Expected: CPU/Mock 两参数 case 全部 PASS，无残留 engine/uvicorn 进程或 socket。

- [ ] **Step 5: 提交**

```bash
git add python/grserve_api/app.py python/grserve_api/main.py tests/python/support/engine_process.py tests/python/test_api_contract.py tests/python/test_disconnect.py tests/python/test_api_e2e.py pyproject.toml uv.lock
git commit -m "feat: expose safe streaming completion API"
```

## Task 19: 结构化事件、规范化摘要与证据清单

**Files:**

- Create: `src/common/event_log.h`
- Create: `src/common/event_log.cc`
- Create: `proto/event.proto`
- Create: `tools/normalize_events.py`
- Create: `tools/evidence.py`
- Create: `tests/cpp/event_log_test.cc`
- Create: `tests/python/test_evidence.py`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写脱敏、规范化与不可覆盖测试**

测试记录 request/transaction/rank/epoch/state/error/sequence/delivery；确认 prompt token 内容、socket 路径、动态库路径和 backend private message 不出现。两次运行 timestamp/run ID 不同但规范化 digest 相同。Evidence writer 遇到既有 run directory 必须失败，不能覆盖。

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked pytest -q tests/python/test_evidence.py
```

Expected: FAIL，event/evidence 模块不存在。

- [ ] **Step 2: 实现 protobuf→JSONL 与 canonical digest**

事件先构造成 typed protobuf，再用 protobuf JSON util 写单行 JSON。每行含 monotonic tick 供诊断；`normalize_events.py` 删除 tick/run ID，只保留有序语义字段，并对 compact canonical JSON 计算 SHA256。

证据目录固定：

```text
artifacts/mac-mvp/<UTC-basic-run-id>/
├── preflight.json
├── build.json
├── cpp-tests.json
├── python-tests.json
├── sanitizer.json
├── linux-amd64.json
├── events.jsonl
├── normalized-events.sha256
├── assets.sha256
└── summary.json
```

- [ ] **Step 3: 运行脱敏与哈希测试**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.event_log$' --output-on-failure
uv run --locked pytest -q tests/python/test_evidence.py
```

Expected: PASS；所有文件 mode 为 owner-readable，已有目录测试返回稳定 `ARTIFACT_DIR_EXISTS`。

- [ ] **Step 4: 提交**

```bash
git add src/common/event_log.h src/common/event_log.cc proto/event.proto tools/normalize_events.py tools/evidence.py tests/cpp/event_log_test.cc tests/python/test_evidence.py CMakeLists.txt
git commit -m "feat: produce reproducible local evidence"
```

## Task 20: sanitizer 与 Linux amd64 必过门禁

**Files:**

- Create: `docker/Dockerfile.linux-amd64`
- Create: `docker/base-image.lock`
- Create: `tools/verify.py`
- Create: `tests/python/test_verify_tool.py`
- Modify: `CMakePresets.json`
- Modify: `cmake/Sanitizers.cmake`

- [ ] **Step 1: 写 preflight 与状态汇总测试**

Fake command runner 覆盖 uv/Xcode/submodule/Docker daemon 可用与不可用。`--profile full` 中 Docker 不可用必须退出非零、reason code `DOCKER_DAEMON_UNAVAILABLE`；不能产生 Linux `SKIP` 或整体 PASS。

Run:

```bash
uv run --locked pytest -q tests/python/test_verify_tool.py
```

Expected: FAIL，`tools/verify.py` 不存在。

- [ ] **Step 2: 实现 Debug/Release/ASan+UBSan 流程**

`verify.py --profile mac` 依次运行 configure/build/ctest Debug、Release、ASan+UBSan，随后 ruff、mypy、pytest、proto check、model asset check。任一子命令非零立即将该 gate 标为 FAIL，但仍落盘已完成 gate 的 stdout/stderr 与退出码。

TSan 单独做 compile+run capability probe：通过后运行 `mac-arm64-tsan`；Apple runtime 不支持时只允许 reason code `TSAN_RUNTIME_UNSUPPORTED_APPLE_CLANG`，不能跳过任何普通 C++/Python/事务测试。

- [ ] **Step 3: 冻结 Linux base image digest 并构建**

先启动 Docker Desktop，再解析一次 immutable digest：

```bash
docker pull --platform linux/amd64 python:3.11.12-slim-bookworm
docker image inspect python:3.11.12-slim-bookworm --format '{{index .RepoDigests 0}}' > docker/base-image.lock
grep -Eq '^python@sha256:[0-9a-f]{64}$' docker/base-image.lock
```

Dockerfile 以 `ARG BASE_IMAGE` + `FROM ${BASE_IMAGE}` 接收 lock 中的 digest；安装 C/C++ build essentials 后用 `pip install uv==0.11.7`、`uv sync --locked`，构建源码中的 pinned submodule。Run:

```bash
uv run --locked python tools/verify.py --profile linux-amd64
```

Expected: 容器内 `uname -m` 为 `x86_64`，Debug C/C++、Python、proto、model golden 与 API E2E 全部 PASS。

- [ ] **Step 4: 执行 sanitizer 与 Linux 回归**

```bash
uv run --locked python tools/verify.py --profile mac
uv run --locked python tools/verify.py --profile linux-amd64
```

Expected: 两个 profile 退出 0；ASan/UBSan 报告零错误；Linux 不是 skip。

- [ ] **Step 5: 提交**

```bash
git add docker/Dockerfile.linux-amd64 docker/base-image.lock tools/verify.py tests/python/test_verify_tool.py CMakePresets.json cmake/Sanitizers.cmake
git commit -m "test: gate GRServe on sanitizers and linux amd64"
```

## Task 21: 单命令验收与 Mac MVP 报告

**Files:**

- Create: `tools/verify_mac_mvp.sh`
- Create: `docs/reports/GRSERVE_MAC_MVP_ACCEPTANCE_ZH.md`
- Create: `tests/python/test_full_acceptance.py`
- Create: `README.md`
- Create: `tools/acceptance.py`

- [ ] **Step 1: 写 full summary contract 测试**

`test_full_acceptance.py` 导入尚不存在的 `tools.acceptance.validate_full_summary`，读取 fixture summary，要求以下 gate 恰好存在且为 PASS：preflight、Debug、Release、C11 ABI、C++、Python、proto、manifest、golden、fault matrix、ASan/UBSan、Linux amd64、API E2E。`critical_skip_count` 必须为 0；TSan 只能 PASS 或唯一允许 reason code。

Run:

```bash
uv run --locked pytest -q tests/python/test_full_acceptance.py
```

Expected: FAIL，无法导入 `tools.acceptance`。

- [ ] **Step 2: 建立唯一权威验收入口**

`tools/verify_mac_mvp.sh` 只负责严格 shell 入口：

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
exec uv run --locked python tools/verify.py --profile full
```

`tools/acceptance.py` 实现上述 exact gate-set validator；任何缺项、额外未知 gate、关键 skip 或非 PASS 都返回非零。`README.md` 明确本地接口测试范围、启动命令、请求示例和“不代表 GR308 性能/上线资格”。验收报告模板固定记录 git SHA、依赖 gitlink、manifest/golden SHA、各 gate 状态、跳过原因、证据目录和服务器上线前仍需完成的 Fant P0。

- [ ] **Step 3: 从干净构建执行 full 验收**

先确认 Docker daemon 可用，然后运行：

```bash
test -z "$(git status --porcelain --untracked-files=no)"
GR_VERIFY_LOG="$(mktemp -t grserve-verify.XXXXXX)"
./tools/verify_mac_mvp.sh | tee "$GR_VERIFY_LOG"
GR_SUMMARY_PATH="$(awk -F= '$1 == "GRSERVE_SUMMARY" {print $2}' "$GR_VERIFY_LOG")"
test -f "$GR_SUMMARY_PATH"
mkdir -p artifacts/mac-mvp
printf '%s\n' "$GR_SUMMARY_PATH" > artifacts/mac-mvp/full-summary.path
```

Expected: 退出码 0；标准输出包含且仅包含一个形如 `GRSERVE_SUMMARY=/absolute/path/to/summary.json` 的定位行，指向唯一新证据目录；summary 中全部关键 gate 为 PASS、`critical_skip_count=0`，无 sanitizer 错误。

- [ ] **Step 4: 由证据生成中文验收报告并复核**

```bash
GR_SUMMARY_PATH="$(LC_ALL=C sed -n '1p' artifacts/mac-mvp/full-summary.path)"
test -f "$GR_SUMMARY_PATH"
uv run --locked python tools/evidence.py render-report --summary "$GR_SUMMARY_PATH" --output docs/reports/GRSERVE_MAC_MVP_ACCEPTANCE_ZH.md
uv run --locked python tools/evidence.py validate-report --summary "$GR_SUMMARY_PATH" --report docs/reports/GRSERVE_MAC_MVP_ACCEPTANCE_ZH.md
```

Expected: 定位行恰好一条，validate 退出 0，报告所有数字均可回指该次 summary；不得用 `latest` symlink 或目录 mtime 猜测。

- [ ] **Step 5: 最终提交**

```bash
git add tools/verify_mac_mvp.sh tools/acceptance.py docs/reports/GRSERVE_MAC_MVP_ACCEPTANCE_ZH.md tests/python/test_full_acceptance.py README.md
git commit -m "docs: certify the GRServe Mac MVP"
git status --short
```

Expected: 任务文件无未提交改动；现有未纳入的无关文档保持原样。

## 最终验收矩阵

| Gate | 权威命令 | 必须结果 |
|---|---|---|
| C11 ABI | `ctest --preset mac-arm64-debug -R '^c.hal_header$'` | PASS |
| Core/Engine | `ctest --preset mac-arm64-debug -R '^cpp.'` | PASS |
| Fault repeat | `ctest --preset mac-arm64-debug -R '^cpp.controller_(faults|cancel)$' --repeat until-fail:100` | 100 次 PASS |
| Python API | `uv run --locked pytest -q tests/python` | PASS |
| Proto sync | `cmake --build --preset mac-arm64-debug --target check_proto_generated` | PASS |
| Asset/golden | `python tests/reference_model.py --check models/mini-qwen38-hybrid/golden.json` | 精确匹配 |
| ASan/UBSan | `ctest --preset mac-arm64-asan-ubsan` | PASS、零报告 |
| Linux amd64 | `python tools/verify.py --profile linux-amd64` | x86_64、全部 PASS |
| Full | `./tools/verify_mac_mvp.sh` | exit 0、关键 skip=0 |

## 规格覆盖自审

| 规格要求 | 实施任务 | 验证证据 |
|---|---:|---|
| C++20 核心不依赖 Python/GPU runtime | 1–2、7–10 | Core target link map、C++ tests |
| 纯 C HAL ABI、版本和 capability 门禁 | 3–4 | C11 smoke、ABI/loader tests |
| CPU/Mock/Fault 三后端 | 5–6、11–12 | backend parity、fault repeat |
| 两 rank 事务与 commit 后发布 | 7–10 | transaction/controller matrix |
| manifest、权重与 golden 完整性 | 11–12 | SHA、strict parser、oracle check |
| versioned protobuf over UDS | 13–15 | descriptor、framing、socket/session tests |
| health/models/pretokenized completions | 16–18 | Python contract 与真实 E2E |
| deadline、取消与断线竞态 | 10、15、18 | ManualClock、queue-order、ASGI disconnect |
| JSONL 与可重复故障摘要 | 6、19 | normalized digest tests |
| mac Debug/Release/sanitizer | 1、20–21 | summary gates |
| Linux amd64 可移植性 | 20–21 | Docker x86_64 gate |
| 单命令、机器可读验收 | 19–21 | `summary.json` 与中文报告 |
| 不干扰服务器 baseline | 全计划 | 无 SSH/服务器命令；仅本地路径 |

## 实施后仍不成立的结论

即使本计划全部通过，也只能说明本地协议、状态机、故障边界和 CPU/模拟接口正确。它不证明 GR308 HAL 可用，不证明真实 Qwen3.8-27B 数值正确，不证明 FHCCL/Graph/kernel 稳定，也不能给出任何 tok/s、TTFT、TPOT、显存或功耗结论。下一阶段必须先写并执行 Fant HAL/P0 资格规格，再进入服务器实验。
