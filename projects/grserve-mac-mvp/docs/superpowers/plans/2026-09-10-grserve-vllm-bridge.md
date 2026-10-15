# GRServe vLLM Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clearly labelled GRServe control-plane bridge that drives the vendor vLLM executor in eager and FANT Graph modes while preserving GRServe admission, lifecycle, cancellation, sequencing, and evidence ownership.

**Architecture:** The existing OpenAI API and Engine UDS RPC remain the public path. A new concurrent C++ `BridgeController` owns accepted requests and communicates over a separately versioned UDS/protobuf protocol with a stateless Python sidecar; the sidecar translates each execution into one streaming request to the already-running vendor vLLM deployment. The existing rank/HAL Controller remains unchanged for native MiniQwen mode and is never used to disguise vLLM as a rank-local HAL.

**Tech Stack:** C++20, C11-compatible project boundaries, CMake/Ninja, protobuf 6.30.2, POSIX Unix sockets, Python 3.11, asyncio, httpx, FastAPI/uvicorn, pytest, GoogleTest, ASan/UBSan, Docker with the Fantasy runtime.

---

## 0. Scope and execution rules

This plan implements the approved design in
`docs/superpowers/specs/2026-09-10-grserve-vllm-bridge-design.md`.

The bridge result must always identify itself as:

```json
{
  "executor": "vendor-vllm",
  "native_fant_hal": false,
  "formal_baseline": false
}
```

The three frozen baseline directories are read-only inputs and must never be
modified:

```text
/data/infra/benchmarks/gr308-vllm-baseline-20260907T095637Z/
/data/infra/benchmarks/gr308-sglang-baseline-20260908T053931Z/
/data/infra/benchmarks/gr308-vllm-fant-graph-20260909T141559Z/
```

All server artifacts go to a new exact directory whose basename is
`grserve-vllm-bridge-smoke-` followed by `date -u +%Y%m%dT%H%M%SZ`. A failed run
is kept; the runner creates a different directory for a retry.

## 1. File map

### Protocol and generated code

- Create `proto/bridge.proto`: independent Bridge RPC v1 schema.
- Modify `proto/CMakeLists.txt`: generate/link C++ Bridge protobuf.
- Modify `tools/proto_codegen.py`: generate and attest `bridge_pb2.py`.
- Create `python/grserve_bridge/_generated/bridge_pb2.py`: committed Python output.
- Modify `python/grserve_bridge/_generated/__init__.py`: generated package marker.

### C++ control plane

- Create `src/engine/request_controller.h`: RPC-facing controller interface.
- Modify `src/engine/controller.h`: implement the interface without changing native semantics.
- Modify `src/rpc/rpc_session.{h,cc}` and `src/rpc/engine_rpc_server.{h,cc}`: depend on the interface.
- Create `src/bridge/deployment_profile.{h,cc}`: strict bridge profile loader.
- Create `src/bridge/bridge_types.h`: execution IDs, completions, and state enums.
- Create `src/bridge/bridge_request.{h,cc}`: one-request lifecycle and sequence validator.
- Create `src/bridge/bridge_frame_codec.{h,cc}`: bounded Bridge protobuf framing.
- Create `src/bridge/bridge_executor_client.{h,cc}`: one UDS connection per execution.
- Create `src/bridge/bridge_controller.{h,cc}`: concurrent request table and authoritative lifecycle.
- Modify `src/rpc/engine_main.cc`: select `native` or `vllm-bridge` mode.

### Python sidecar

- Create `python/grserve_bridge/__init__.py`.
- Create `python/grserve_bridge/config.py`: strict CLI/runtime configuration.
- Create `python/grserve_bridge/protocol.py`: framing helpers.
- Create `python/grserve_bridge/vllm_client.py`: streaming vendor HTTP client.
- Create `python/grserve_bridge/server.py`: UDS sidecar server.
- Create `python/grserve_bridge/main.py`: `grserve-vllm-sidecar` entry point.
- Modify `pyproject.toml`: package and executable declaration.

### Tests, launch, and evidence

- Create `tests/cpp/bridge/deployment_profile_test.cc`.
- Create `tests/cpp/bridge/bridge_request_test.cc`.
- Create `tests/cpp/bridge/bridge_frame_codec_test.cc`.
- Create `tests/cpp/bridge/bridge_executor_client_test.cc`.
- Create `tests/cpp/bridge/bridge_controller_test.cc`.
- Create `tests/python/support/fake_vllm.py`.
- Create `tests/python/test_bridge_config.py`.
- Create `tests/python/test_bridge_protocol.py`.
- Create `tests/python/test_bridge_vllm_client.py`.
- Create `tests/python/test_bridge_sidecar.py`.
- Create `tests/python/test_bridge_e2e.py`.
- Create `tools/run_bridge_smoke.py`: guarded server orchestrator and matrix runner.
- Create `tools/bridge_evidence.py`: strict summary validator/report renderer.
- Create `tests/python/test_bridge_smoke_tool.py`.
- Modify `tools/verify.py`, `docker/Dockerfile.linux-amd64`, `README.md`, and `CMakeLists.txt`.

---

## Task 1: Freeze Bridge RPC v1 and generated outputs

**Files:**
- Create: `proto/bridge.proto`
- Modify: `proto/CMakeLists.txt`
- Modify: `tools/proto_codegen.py`
- Create: `python/grserve_bridge/_generated/__init__.py`
- Create: `python/grserve_bridge/_generated/bridge_pb2.py`
- Modify: `tests/cpp/rpc/proto_contract_test.cc`
- Modify: `tests/python/test_proto_codegen.py`

- [ ] **Step 1: Write failing schema/codegen tests**

Add a Python assertion that the committed output set includes Bridge and add a C++
descriptor test for the frozen package and field numbers:

```python
def test_bridge_generated_output_is_managed() -> None:
    assert "bridge_pb2.py" in proto_codegen.GENERATED_FILES
    assert (ROOT / "python/grserve_bridge/_generated/bridge_pb2.py").is_file()
```

```cpp
TEST(ProtoContract, BridgeV1FieldNumbersAreFrozen) {
  const auto* start = grserve::bridge::v1::BridgeStart::descriptor();
  EXPECT_EQ(start->file()->package(), "grserve.bridge.v1");
  EXPECT_EQ(start->FindFieldByName("execution_id")->number(), 1);
  EXPECT_EQ(start->FindFieldByName("prompt_token_ids")->number(), 3);
  EXPECT_EQ(start->FindFieldByName("max_tokens")->number(), 4);
  EXPECT_EQ(start->FindFieldByName("deadline_budget_ms")->number(), 5);
}
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run --locked pytest -q tests/python/test_proto_codegen.py -k bridge
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.rpc.proto_contract$' --output-on-failure
```

Expected: Python fails because `bridge_pb2.py` is unmanaged; C++ compilation fails
because `bridge.pb.h` does not exist.

- [ ] **Step 3: Add the exact Bridge schema**

Create `proto/bridge.proto` with no imports and this complete v1 surface:

```proto
syntax = "proto3";
package grserve.bridge.v1;

message BridgeVersion { uint32 major = 1; uint32 minor = 2; }
enum BridgeMode { BRIDGE_MODE_UNSPECIFIED = 0; BRIDGE_MODE_EAGER = 1; BRIDGE_MODE_FANT_GRAPH = 2; }
enum BridgeErrorCode {
  BRIDGE_ERROR_CODE_UNSPECIFIED = 0;
  BRIDGE_ERROR_CODE_INVALID_ARGUMENT = 1;
  BRIDGE_ERROR_CODE_RESOURCE_EXHAUSTED = 2;
  BRIDGE_ERROR_CODE_DEADLINE_EXCEEDED = 3;
  BRIDGE_ERROR_CODE_EXECUTOR_FAILURE = 4;
  BRIDGE_ERROR_CODE_PROTOCOL = 5;
}
message BridgeStart {
  string execution_id = 1;
  string model_id = 2;
  repeated uint32 prompt_token_ids = 3 [packed = true];
  uint32 max_tokens = 4;
  uint64 deadline_budget_ms = 5;
  string deployment_id = 6;
}
message BridgeAccepted { string execution_id = 1; }
message BridgeToken { string execution_id = 1; uint64 sequence_number = 2; uint32 token_id = 3; }
message BridgeFinished {
  string execution_id = 1;
  string finish_reason = 2;
  uint32 prompt_tokens = 3;
  uint32 completion_tokens = 4;
}
message BridgeCancel { string execution_id = 1; string reason = 2; }
message BridgeCancelAck { string execution_id = 1; bool stopped = 2; }
message BridgeError {
  string execution_id = 1;
  BridgeErrorCode code = 2;
  string message = 3;
  bool retryable = 4;
}
message BridgeHealthRequest {}
message BridgeHealth {
  string build_id = 1;
  string deployment_id = 2;
  BridgeMode mode = 3;
  bool accepting_requests = 4;
  string executor = 5;
  bool native_fant_hal = 6;
}
message BridgeDeploymentProfile {
  uint32 schema_version = 1;
  string deployment_id = 2;
  string model_id = 3;
  string model_manifest_sha256 = 4;
  string sidecar_socket = 5;
  BridgeMode mode = 6;
  string executor = 7;
  bool native_fant_hal = 8;
  uint32 max_context_tokens = 9;
  uint32 max_concurrent_requests = 10;
  uint32 vocab_size = 11;
  string event_log_path = 12;
}
message BridgeClientEnvelope {
  BridgeVersion protocol_version = 1;
  oneof payload { BridgeStart start = 10; BridgeCancel cancel = 11; BridgeHealthRequest health = 12; }
}
message BridgeServerEnvelope {
  BridgeVersion protocol_version = 1;
  oneof payload {
    BridgeAccepted accepted = 10;
    BridgeToken token = 11;
    BridgeFinished finished = 12;
    BridgeCancelAck cancel_ack = 13;
    BridgeError error = 14;
    BridgeHealth health = 15;
  }
}
```

Update `proto/CMakeLists.txt` to generate `bridge.pb.cc/.h`, build
`grserve_bridge_proto`, and add that target to `check_proto_generated` dependencies.
Refactor `GENERATED_FILES` in `tools/proto_codegen.py` into an explicit generated-name
to repository-destination mapping so `engine_pb2.py` and `event_pb2.py` remain under
`python/grserve_api/_generated/`, while `bridge_pb2.py` is copied and checked under
`python/grserve_bridge/_generated/`. One pinned protoc invocation generates all three.

- [ ] **Step 4: Generate and verify outputs**

Run:

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target grserve_protoc_attestation
uv run --locked python tools/proto_codegen.py \
  --protoc "$PWD/build/mac-arm64-debug/third_party/protobuf/protoc-30.2.0" \
  --attestation "$PWD/build/mac-arm64-debug/proto/Debug/protoc.attestation" \
  --config Debug
uv run --locked cmake --build --preset mac-arm64-debug --target check_proto_generated
```

Expected: `protobuf Python output is up to date` and no unmanaged generated file.

- [ ] **Step 5: Run tests and commit**

```bash
uv run --locked pytest -q tests/python/test_proto_codegen.py
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.rpc.proto_contract$' --output-on-failure
git add proto tools/proto_codegen.py python/grserve_bridge tests/cpp/rpc/proto_contract_test.cc tests/python/test_proto_codegen.py
git commit -m "feat: freeze bridge RPC protocol"
```

Expected: all focused checks pass.

---

## Task 2: Decouple Engine RPC from the native Controller

**Files:**
- Create: `src/engine/request_controller.h`
- Modify: `src/engine/controller.h`
- Modify: `src/rpc/rpc_session.h`
- Modify: `src/rpc/rpc_session.cc`
- Modify: `src/rpc/engine_rpc_server.h`
- Modify: `src/rpc/engine_rpc_server.cc`
- Modify: `tests/cpp/rpc/rpc_session_test.cc`
- Modify: `tests/cpp/rpc/engine_process_test.cc`

- [ ] **Step 1: Write a failing interface substitution test**

Define a test-only controller that records mailbox operations and prove an RPC session
can use it without constructing rank workers:

```cpp
class RecordingRequestController final : public grserve::RequestController {
 public:
  grserve::Status Submit(grserve::GenerateRequest request,
                         std::shared_ptr<grserve::RequestEventSink> sink) override {
    submitted.push_back(request.request_id);
    return sink->TryAccepted(request.request_id) ? grserve::Status::Ok()
        : grserve::Status{grserve::Error::From(
              grserve::ErrorCode::kResourceExhausted, "sink full",
              request.request_id)};
  }
  grserve::StatusOr<grserve::CancelDisposition> CancelTransport(
      grserve::RequestId, grserve::CancelReason) override {
    return grserve::CancelDisposition::kAbortedBeforeDecision;
  }
  grserve::Status DetachTransport(grserve::RequestId) override { return grserve::Status::Ok(); }
  grserve::Status Shutdown() override { return grserve::Status::Ok(); }
  bool PollOnce() override { return false; }
  bool idle() const noexcept override { return true; }
  std::vector<grserve::RequestId> submitted;
};
```

- [ ] **Step 2: Verify RED**

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target grserve_rpc_session_test
```

Expected: compilation fails because `RequestController` does not exist and
`ControllerMailbox::PollOne` requires concrete `Controller&`.

- [ ] **Step 3: Introduce the narrow interface**

Create:

```cpp
class RequestController {
 public:
  virtual ~RequestController() = default;
  virtual Status Submit(GenerateRequest, std::shared_ptr<RequestEventSink>) = 0;
  virtual StatusOr<CancelDisposition> CancelTransport(RequestId, CancelReason) = 0;
  virtual Status DetachTransport(RequestId) = 0;
  virtual Status Shutdown() = 0;
  virtual bool PollOnce() = 0;
  [[nodiscard]] virtual bool idle() const noexcept = 0;
};
```

Make the existing `Controller final : public RequestController`; add `override` to the
six methods. Change only RPC/mailbox/server parameter and member types from
`Controller` to `RequestController`. Do not change native state-machine logic.

- [ ] **Step 4: Run native regression tests**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp\.(controller_|rpc\.)' --output-on-failure
```

Expected: the substitution test and all existing native Controller/RPC tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/engine/request_controller.h src/engine/controller.h src/rpc tests/cpp/rpc
git commit -m "refactor: abstract RPC request controller"
```

---

## Task 3: Validate an immutable bridge deployment profile

**Files:**
- Create: `src/bridge/deployment_profile.h`
- Create: `src/bridge/deployment_profile.cc`
- Create: `tests/cpp/bridge/deployment_profile_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing strict-profile tests**

Cover a valid profile, missing fields, unknown fields, relative/non-socket paths,
invalid SHA256, unsupported mode, zero capacity, and `native_fant_hal=true`:

```cpp
TEST(BridgeDeploymentProfile, AcceptsFrozenSmokeProfile) {
  auto profile = LoadProfile(WriteProfile(R"({
    "schema_version":1,"deployment_id":"graph-smoke",
    "model_id":"qwen3.8-27b","model_manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "sidecar_socket":"/private/tmp/grserve-bridge.sock","mode":"BRIDGE_MODE_FANT_GRAPH",
    "executor":"vendor-vllm","native_fant_hal":false,
    "max_context_tokens":32768,"max_concurrent_requests":64,"vocab_size":248320,
    "event_log_path":"/private/tmp/grserve-bridge-events.jsonl"
  })"));
  ASSERT_TRUE(profile.ok());
  EXPECT_EQ(profile.value().mode, BridgeMode::kFantGraph);
}
```

- [ ] **Step 2: Verify RED**

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target grserve_bridge_profile_test
```

Expected: target/source is missing.

- [ ] **Step 3: Implement the strict value type and loader**

Expose:

```cpp
enum class BridgeMode { kEager, kFantGraph };
struct DeploymentProfile {
  std::string deployment_id;
  std::string model_id;
  std::string model_manifest_sha256;
  std::filesystem::path sidecar_socket;
  BridgeMode mode;
  std::uint32_t max_context_tokens;
  std::uint32_t max_concurrent_requests;
  std::uint32_t vocab_size;
  std::filesystem::path event_log_path;
};
[[nodiscard]] StatusOr<DeploymentProfile> LoadDeploymentProfile(
    const std::filesystem::path& path);
```

Parse `BridgeDeploymentProfile` with
`google::protobuf::util::JsonStringToMessage` and
`JsonParseOptions{.ignore_unknown_fields = false}`; reject symlinks, non-regular files,
files larger than 64 KiB, non-absolute socket/event-log paths, socket paths over 100
encoded bytes, and any identity other than `executor=vendor-vllm` plus
`native_fant_hal=false`. The event log parent must already exist, be owned by the
current uid, and not be group/world writable.

- [ ] **Step 4: Run tests**

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target grserve_bridge_profile_test
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.bridge.profile$' --output-on-failure
```

Expected: all strict parsing cases pass.

- [ ] **Step 5: Commit**

```bash
git add src/bridge/deployment_profile.* tests/cpp/bridge/deployment_profile_test.cc CMakeLists.txt
git commit -m "feat: validate bridge deployment profiles"
```

---

## Task 4: Implement the per-request Bridge lifecycle

**Files:**
- Create: `src/bridge/bridge_types.h`
- Create: `src/bridge/bridge_request.h`
- Create: `src/bridge/bridge_request.cc`
- Create: `tests/cpp/bridge/bridge_request_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the lifecycle transition tests**

Test accepted → running → token sequence → finished, cancellation, deadline, duplicate
terminal, unknown execution identity, and sequence gaps:

```cpp
TEST(BridgeRequest, PublishesOnlyStrictMonotonicTokens) {
  BridgeRequest request = MakeAcceptedRequest(RequestId{7}, "exec-7", 3);
  ASSERT_TRUE(request.MarkExecutorAccepted().ok());
  ASSERT_TRUE(request.AcceptToken(1, TokenId{11}).ok());
  EXPECT_EQ(request.AcceptToken(1, TokenId{12}).error()->code(), ErrorCode::kInternal);
  EXPECT_EQ(request.AcceptToken(3, TokenId{13}).error()->code(), ErrorCode::kInternal);
  ASSERT_TRUE(request.AcceptToken(2, TokenId{12}).ok());
}
```

- [ ] **Step 2: Verify RED**

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target grserve_bridge_request_test
```

Expected: target/source is missing.

- [ ] **Step 3: Implement an explicit state machine**

Define:

```cpp
enum class BridgeRequestState { kAccepted, kRunning, kCancelling, kFinished, kFailed };
struct BridgeCompletion {
  enum class Kind { kAccepted, kToken, kFinished, kCancelAck, kError };
  Kind kind;
  std::string execution_id;
  std::uint64_t sequence_number{0};
  TokenId token{0};
  std::uint32_t prompt_tokens{0};
  std::uint32_t completion_tokens{0};
  Error error{Error::From(ErrorCode::kInternal, "bridge completion")};
};
struct BridgeStartCommand {
  std::string execution_id;
  GenerateRequest request;
  std::uint64_t deadline_budget_ms;
  std::string deployment_id;
};
```

`BridgeRequest` owns the lease, sink, deadline, execution identity, next sequence, token
count, state, detached flag, and exactly-once terminal flag. Transition methods return
stable errors without throwing and never release the lease before terminal cleanup.

- [ ] **Step 4: Run lifecycle and sanitizer tests**

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target grserve_bridge_request_test
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.bridge.request$' --output-on-failure
uv run --locked cmake --build --preset mac-arm64-asan-ubsan --target grserve_bridge_request_test
uv run --locked ctest --preset mac-arm64-asan-ubsan -R '^cpp.bridge.request$' --output-on-failure
```

Expected: both presets pass.

- [ ] **Step 5: Commit**

```bash
git add src/bridge/bridge_types.h src/bridge/bridge_request.* tests/cpp/bridge/bridge_request_test.cc CMakeLists.txt
git commit -m "feat: add bridge request lifecycle"
```

---

## Task 5: Add bounded Bridge framing and executor transport

**Files:**
- Create: `src/bridge/bridge_frame_codec.h`
- Create: `src/bridge/bridge_frame_codec.cc`
- Create: `src/bridge/bridge_executor_client.h`
- Create: `src/bridge/bridge_executor_client.cc`
- Create: `tests/cpp/bridge/bridge_frame_codec_test.cc`
- Create: `tests/cpp/bridge/bridge_executor_client_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing framing and transport tests**

Use `socketpair()` and a fake UDS listener to cover partial reads/writes, 4 MiB cap,
zero frames, incompatible protocol version, disconnect, start/cancel, and shutdown:

```cpp
TEST(BridgeFrameCodec, RejectsOversizedBodyBeforeAllocation) {
  auto pair = SocketPair();
  WriteBigEndianLength(pair.writer(), kBridgeMaxFrameBytes + 1);
  auto result = BridgeFrameReader{}.ReadOne(pair.reader());
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), ErrorCode::kInvalidArgument);
}
```

```cpp
TEST(BridgeExecutorClient, DisconnectProducesCorrelatedFailure) {
  FakeBridgeServer server(FakeAction::kCloseAfterAccepted);
  CompletionQueue<BridgeCompletion> completions;
  auto client = BridgeExecutorClient::Connect(server.path(), completions);
  ASSERT_TRUE(client.ok());
  ASSERT_TRUE(client.value()->Start(MakeStart("exec-9")).ok());
  EXPECT_EQ(WaitCompletion(completions).execution_id, "exec-9");
  EXPECT_EQ(WaitCompletion(completions).error.code(), ErrorCode::kRankFailure);
}
```

- [ ] **Step 2: Verify RED**

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target grserve_bridge_frame_codec_test grserve_bridge_executor_client_test
```

Expected: targets/sources are missing.

- [ ] **Step 3: Implement the codec and client**

Use a four-byte big-endian length with `kBridgeMaxFrameBytes = 4U * 1024U * 1024U`.
Each executor connection handles one execution so HTTP-style head-of-line blocking cannot
stall unrelated requests. Expose:

```cpp
class BridgeExecutor {
 public:
  virtual ~BridgeExecutor() = default;
  virtual Status Start(const BridgeStartCommand&) = 0;
  virtual Status Cancel(std::string_view execution_id, CancelReason) = 0;
  virtual void Shutdown() noexcept = 0;
};

class BridgeExecutorClient final : public BridgeExecutor {
 public:
  static StatusOr<std::unique_ptr<BridgeExecutorClient>> Connect(
      const std::filesystem::path&, CompletionQueue<BridgeCompletion>&);
  Status Start(const BridgeStartCommand&) override;
  Status Cancel(std::string_view execution_id, CancelReason) override;
  void Shutdown() noexcept override;
};
```

Define the injectable factory used by the Controller tests:

```cpp
using BridgeExecutorFactory = std::function<StatusOr<std::unique_ptr<BridgeExecutor>>(
    CompletionQueue<BridgeCompletion>&)>;
```

The reader thread validates version `1.0`, oneof presence, execution identity, frame
size, and protobuf recursion limit. It pushes values only; it never calls a Controller
or sink from the reader thread. Each connection has a 64-event bounded queue; when full,
the reader stops consuming the UDS so kernel socket backpressure reaches the Sidecar.

- [ ] **Step 4: Run transport tests under normal and sanitizer builds**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.bridge.(frame_codec|executor_client)$' --output-on-failure
uv run --locked cmake --build --preset mac-arm64-asan-ubsan
uv run --locked ctest --preset mac-arm64-asan-ubsan -R '^cpp.bridge.(frame_codec|executor_client)$' --output-on-failure
```

Expected: all cases pass without leaks or sanitizer findings.

- [ ] **Step 5: Commit**

```bash
git add src/bridge/bridge_frame_codec.* src/bridge/bridge_executor_client.* tests/cpp/bridge CMakeLists.txt
git commit -m "feat: add bridge executor transport"
```

---

## Task 6: Build the concurrent authoritative BridgeController

**Files:**
- Create: `src/bridge/bridge_controller.h`
- Create: `src/bridge/bridge_controller.cc`
- Create: `tests/cpp/bridge/bridge_controller_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing concurrency/ownership tests**

Cover 64 simultaneously running requests, the 65th admission rejection, out-of-order
completion across request IDs, per-request sequence enforcement, deadline, cancel,
detach, executor failure, replica fatal correlation, shutdown, and zero leaked leases:

```cpp
TEST(BridgeController, HoldsSixtyFourIndependentRunningRequests) {
  Fixture fixture(/*capacity=*/64);
  for (std::uint64_t id = 1; id <= 64; ++id) {
    ASSERT_TRUE(fixture.Submit(id).ok());
  }
  EXPECT_EQ(fixture.controller.running_for_testing(), 64U);
  EXPECT_EQ(fixture.Submit(65).error()->code(), ErrorCode::kResourceExhausted);
  fixture.FinishInReverseOrder();
  EXPECT_TRUE(fixture.controller.idle());
  EXPECT_EQ(fixture.admission_used(), 0U);
}
```

- [ ] **Step 2: Verify RED**

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target grserve_bridge_controller_test
```

Expected: target/source is missing.

- [ ] **Step 3: Implement the controller**

Implement `RequestController` with owner-thread mutation and a map keyed by `RequestId`:

```cpp
class BridgeController final : public RequestController {
 public:
  static StatusOr<std::unique_ptr<BridgeController>> Create(
      AdmissionController, std::shared_ptr<ReplicaGate>,
      CompletionQueue<BridgeCompletion>&, BridgeExecutorFactory, Clock = {});
  Status Submit(GenerateRequest, std::shared_ptr<RequestEventSink>) override;
  StatusOr<CancelDisposition> CancelTransport(RequestId, CancelReason) override;
  Status DetachTransport(RequestId) override;
  Status Shutdown() override;
  bool PollOnce() override;
  bool idle() const noexcept override;
 private:
  std::map<RequestId, BridgeRequest> requests_;
};
```

On Submit, validate model/max tokens/deadline, reserve admission, publish Accepted, create
a unique execution ID containing engine epoch and request ID, and start one executor
connection. `PollOnce()` drains a bounded number of completions, checks deadlines for
all requests, publishes only validated tokens, and erases only terminally cleaned
requests. Never auto-retry a request after a token was published.

Reuse `EventLog` to write bounded structured lifecycle records to the profile's exact
`event_log_path`. Startup and shutdown records include `active_requests` and
`admission_leases`; the final shutdown record is valid only when both are zero.

- [ ] **Step 4: Run repeated concurrency tests**

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target grserve_bridge_controller_test
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.bridge.controller$' --repeat until-fail:100 --output-on-failure
```

Expected: 100 repetitions pass and every fixture ends with zero active leases.

- [ ] **Step 5: Commit**

```bash
git add src/bridge/bridge_controller.* tests/cpp/bridge/bridge_controller_test.cc CMakeLists.txt
git commit -m "feat: add concurrent bridge controller"
```

---

## Task 7: Implement the strict Python Bridge protocol and configuration

**Files:**
- Create: `python/grserve_bridge/__init__.py`
- Create: `python/grserve_bridge/config.py`
- Create: `python/grserve_bridge/protocol.py`
- Create: `tests/python/test_bridge_config.py`
- Create: `tests/python/test_bridge_protocol.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing Python tests**

```python
def test_config_rejects_public_vendor_endpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        BridgeConfig(
            socket_path=tmp_path / "bridge.sock",
            vendor_base_url="http://192.0.2.11:8000",
            deployment_id="eager-smoke",
            mode="eager",
            max_executions=64,
        )

@pytest.mark.asyncio
async def test_read_frame_rejects_more_than_four_mibibytes() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack(">I", MAX_FRAME_BYTES + 1))
    with pytest.raises(BridgeProtocolError, match="exceeds"):
        await read_client_envelope(reader)
```

- [ ] **Step 2: Verify RED**

```bash
uv run --locked pytest -q tests/python/test_bridge_config.py tests/python/test_bridge_protocol.py
```

Expected: import failure because `grserve_bridge` does not exist.

- [ ] **Step 3: Implement strict config and framing**

Use frozen dataclasses and explicit type/range checks:

```python
@dataclass(frozen=True, slots=True)
class BridgeConfig:
    socket_path: Path
    vendor_base_url: str
    deployment_id: str
    mode: Literal["eager", "fant-graph"]
    max_executions: int = 64
    connect_timeout_s: float = 5.0
    shutdown_timeout_s: float = 5.0
```

Require an absolute UDS path of at most 100 UTF-8 bytes, a vendor URL with scheme
`http`, host `127.0.0.1` or `localhost`, no credentials/query/fragment, a nonempty
deployment ID, and `1 <= max_executions <= 1024`. The protocol module serializes only
nonempty protobuf envelopes and enforces Bridge version `1.0`.

Update Hatch packaging to:

```toml
[tool.hatch.build.targets.wheel]
packages = ["python/grserve_api", "python/grserve_bridge"]

[project.scripts]
grserve-api = "grserve_api.main:main"
grserve-vllm-sidecar = "grserve_bridge.main:main"
```

- [ ] **Step 4: Run Python checks**

```bash
uv run --locked pytest -q tests/python/test_bridge_config.py tests/python/test_bridge_protocol.py
uv run --locked ruff check python/grserve_bridge tests/python/test_bridge_config.py tests/python/test_bridge_protocol.py
MYPYPATH=python uv run --locked mypy --explicit-package-bases python/grserve_bridge
```

Expected: tests, Ruff, and mypy pass.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_bridge tests/python/test_bridge_config.py tests/python/test_bridge_protocol.py pyproject.toml
git commit -m "feat: add bridge sidecar protocol"
```

---

## Task 8: Translate vendor vLLM streaming responses safely

**Files:**
- Create: `python/grserve_bridge/vllm_client.py`
- Create: `tests/python/support/fake_vllm.py`
- Create: `tests/python/test_bridge_vllm_client.py`

- [ ] **Step 1: Write failing streaming tests**

The fake server must emit known token IDs, fragmented SSE bytes, `[DONE]`, 400, 429,
500, timeout, invalid JSON, missing token IDs, duplicate terminal, and disconnect:

```python
@pytest.mark.asyncio
async def test_stream_yields_exact_token_ids_and_usage(fake_vllm: FakeVllm) -> None:
    fake_vllm.enqueue_tokens([17, 23, 42], prompt_tokens=4)
    events = [event async for event in VllmClient(fake_vllm.url).stream(request())]
    assert [event.token_id for event in events if isinstance(event, VendorToken)] == [17, 23, 42]
    assert events[-1] == VendorFinished("length", prompt_tokens=4, completion_tokens=3)
```

- [ ] **Step 2: Verify RED**

```bash
uv run --locked pytest -q tests/python/test_bridge_vllm_client.py
```

Expected: import failure because `VllmClient` is missing.

- [ ] **Step 3: Implement the vendor client**

Build one `httpx.AsyncClient(trust_env=False)` per Sidecar and one streamed POST per
execution. The JSON request is exact:

```python
payload = {
    "model": request.model_id,
    "prompt_token_ids": request.prompt_token_ids,
    "max_tokens": request.max_tokens,
    "temperature": 0,
    "top_p": 1,
    "stream": True,
    "stream_options": {"include_usage": True},
    "ignore_eos": True,
}
```

Accept only HTTP 200 SSE. Require one token ID per token event, exactly one `[DONE]`,
`finish_reason=length`, and usage matching observed completion tokens. Map 400/404 to
invalid argument, 429 to resource exhausted, timeout to deadline exceeded, and all
malformed/5xx/disconnect cases to executor failure. Never include response bodies or
request contents in public errors.

- [ ] **Step 4: Run the complete vendor-client matrix**

```bash
uv run --locked pytest -q tests/python/test_bridge_vllm_client.py
uv run --locked ruff check python/grserve_bridge/vllm_client.py tests/python/support/fake_vllm.py tests/python/test_bridge_vllm_client.py
MYPYPATH=python uv run --locked mypy --explicit-package-bases python/grserve_bridge/vllm_client.py
```

Expected: all streaming/error tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_bridge/vllm_client.py tests/python/support/fake_vllm.py tests/python/test_bridge_vllm_client.py
git commit -m "feat: translate vendor vLLM streams"
```

---

## Task 9: Build the concurrent vLLM Bridge Sidecar

**Files:**
- Create: `python/grserve_bridge/server.py`
- Create: `python/grserve_bridge/main.py`
- Create: `tests/python/test_bridge_sidecar.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing sidecar lifecycle tests**

Test health identity, 64 concurrent executions, 65th rejection, duplicate execution ID,
cancel, disconnect, bounded shutdown, UDS ownership, and zero execution leaks:

```python
@pytest.mark.asyncio
async def test_sidecar_runs_sixty_four_independent_streams(server: SidecarHarness) -> None:
    streams = [server.start(f"exec-{index}") for index in range(64)]
    await server.wait_until_vendor_running(64)
    rejected = await server.start("exec-65")
    assert await rejected.error_code() == bridge_pb2.BRIDGE_ERROR_CODE_RESOURCE_EXHAUSTED
    server.vendor.finish_all()
    assert all((await stream.finished()).completion_tokens == 8 for stream in streams)
    assert server.execution_count == 0
```

- [ ] **Step 2: Verify RED**

```bash
uv run --locked pytest -q tests/python/test_bridge_sidecar.py
```

Expected: missing server/main implementation.

- [ ] **Step 3: Implement the server**

`BridgeSidecar` owns an asyncio UDS server, one task per execution, a semaphore, and an
execution map guarded by one `asyncio.Lock`. Each connection may carry one Start plus
zero or more Cancel messages. Use `TaskGroup` for server-owned tasks, shield bounded
cleanup, and unlink the socket only when device/inode/uid still match the owned socket.

Expose a CLI:

```text
grserve-vllm-sidecar --socket /run/user/UID/grserve-bridge.sock \
  --vendor-base-url http://127.0.0.1:8000 \
  --deployment-id qwen38-eager-smoke --mode eager --max-executions 64
```

Write readiness JSON to stdout only after bind succeeds; diagnostics go to stderr.

- [ ] **Step 4: Run tests and repeated cancellation stress**

```bash
uv run --locked pytest -q tests/python/test_bridge_sidecar.py
for i in $(seq 1 100); do uv run --locked pytest -q tests/python/test_bridge_sidecar.py -k cancel || exit 1; done
```

Expected: every repetition passes and no socket/task remains.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_bridge/server.py python/grserve_bridge/main.py tests/python/test_bridge_sidecar.py pyproject.toml
git commit -m "feat: add vLLM bridge sidecar"
```

---

## Task 10: Wire bridge mode into `grserve-engine`

**Files:**
- Modify: `src/rpc/engine_main.cc`
- Modify: `CMakeLists.txt`
- Modify: `tests/cpp/rpc/engine_process_test.cc`
- Create: `tests/fixtures/bridge-eager-profile.json`

- [ ] **Step 1: Write failing CLI and process tests**

Test that native flags retain existing behavior, bridge flags are mutually exclusive
with HAL flags, profile identity appears in health/models, and a dead sidecar prevents
readiness:

```cpp
TEST(EngineProcess, BridgeModeRejectsNativeHalArguments) {
  auto result = RunEngine({"--execution-mode=vllm-bridge",
                           "--bridge-profile=tests/fixtures/bridge-eager-profile.json",
                           "--hal-plugin=/tmp/not-allowed"});
  EXPECT_EQ(result.exit_code, 2);
  EXPECT_THAT(result.stderr_text, HasSubstr("cannot be combined"));
}
```

- [ ] **Step 2: Verify RED**

```bash
uv run --locked cmake --build --preset mac-arm64-debug --target grserve_engine_process_test
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.rpc.engine_process$' --output-on-failure
```

Expected: bridge CLI is rejected as unknown.

- [ ] **Step 3: Implement explicit execution-mode startup**

Extend options with:

```cpp
enum class ExecutionMode { kNative, kVllmBridge };
ExecutionMode execution_mode{ExecutionMode::kNative};
std::filesystem::path bridge_profile;
```

Native mode keeps the current exact two-rank MiniQwen/HAL path. Bridge mode requires a
strict profile, connects and health-checks the Sidecar before accepting clients, creates
one-rank logical admission capacity from the profile, constructs `BridgeController`, and
passes it through `RequestController&` to `EngineRpcServer`. It does not open a HAL,
load MiniQwen assets, or claim native Fant capabilities.

- [ ] **Step 4: Run native and bridge process tests**

```bash
uv run --locked cmake --build --preset mac-arm64-debug
uv run --locked ctest --preset mac-arm64-debug -R '^cpp.rpc.engine_process$' --output-on-failure
GRSERVE_ENGINE_BIN="$PWD/build/mac-arm64-debug/bin/grserve-engine" uv run --locked pytest -q tests/python/test_api_e2e.py
```

Expected: all existing native tests and new bridge startup tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rpc/engine_main.cc CMakeLists.txt tests/cpp/rpc/engine_process_test.cc tests/fixtures/bridge-eager-profile.json
git commit -m "feat: start engine in vLLM bridge mode"
```

---

## Task 11: Prove full local GRServe-to-Fake-vLLM E2E

**Files:**
- Create: `tests/python/test_bridge_e2e.py`
- Modify: `tests/python/support/engine_process.py`
- Modify: `tests/python/support/fake_vllm.py`

- [ ] **Step 1: Write failing end-to-end tests**

Start Fake-vLLM, real Sidecar, real C++ Engine bridge mode, and real FastAPI/uvicorn.
Cover success, 64 concurrent requests, post-Accept error, cancel, deadline, disconnect,
Sidecar death, vLLM death, and shutdown:

```python
@pytest.mark.asyncio
async def test_real_bridge_control_plane_handles_c64(stack: BridgeStack) -> None:
    responses = await asyncio.gather(
        *(stack.complete(request_id=index, max_tokens=8) for index in range(1, 65))
    )
    assert all(response.status_code == 200 for response in responses)
    assert all(parse_sse(response.content)[-1] == "[DONE]" for response in responses)
    assert stack.fake_vllm.max_simultaneous_requests == 64
    assert await stack.engine_active_requests() == 0
    assert stack.sidecar.execution_count == 0
```

- [ ] **Step 2: Verify RED**

```bash
GRSERVE_ENGINE_BIN="$PWD/build/mac-arm64-debug/bin/grserve-engine" \
  uv run --locked pytest -q tests/python/test_bridge_e2e.py
```

Expected: stack fixture or bridge behavior is missing.

- [ ] **Step 3: Implement only the test harness support needed by the public stack**

Add `BridgeStack` async context management with private temporary sockets, subprocess
stderr files rather than unread pipes, readiness deadlines, PID ownership, and bounded
SIGTERM/SIGKILL cleanup. Do not add test-only RPC behavior to production binaries;
query active counts from structured metrics emitted by the Controller/Sidecar.

- [ ] **Step 4: Run E2E and stress**

```bash
GRSERVE_ENGINE_BIN="$PWD/build/mac-arm64-debug/bin/grserve-engine" \
  uv run --locked pytest -q tests/python/test_bridge_e2e.py
for i in $(seq 1 20); do \
  GRSERVE_ENGINE_BIN="$PWD/build/mac-arm64-debug/bin/grserve-engine" \
    uv run --locked pytest -q tests/python/test_bridge_e2e.py -k c64 || exit 1; \
done
```

Expected: full E2E passes; C64 passes 20/20 with no orphan process/socket.

- [ ] **Step 5: Commit**

```bash
git add tests/python/test_bridge_e2e.py tests/python/support/engine_process.py tests/python/support/fake_vllm.py
git commit -m "test: prove bridge control plane end to end"
```

---

## Task 12: Create the guarded server smoke runner and evidence validator

**Files:**
- Create: `tools/run_bridge_smoke.py`
- Create: `tools/bridge_evidence.py`
- Create: `tests/python/test_bridge_smoke_tool.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing dry-run, guard, and validation tests**

```python
def test_matrix_is_exact_and_c64_is_marked_outside_graph_bucket() -> None:
    plan = build_plan(mode="fant-graph")
    assert [case.concurrency for case in plan.cases] == [1, 2, 4, 8, 16, 32, 64]
    assert plan.cases[-1].graph_bucket is None
    assert plan.formal_baseline is False

def test_preflight_refuses_any_running_container(fake_host: FakeHost) -> None:
    fake_host.running_containers = ["unrelated-service"]
    with pytest.raises(SmokeBlocked, match="running container"):
        preflight(fake_host, config())
```

Also test occupied ports, active GPU processes, wrong model hash, missing Fantasy runtime,
existing output directory, invalid summary, cleanup target mismatch, and exact report
labelling.

- [ ] **Step 2: Verify RED**

```bash
uv run --locked pytest -q tests/python/test_bridge_smoke_tool.py
```

Expected: imports fail because both tools are missing.

- [ ] **Step 3: Implement orchestration with explicit ownership**

The runner accepts:

```text
bridge_run_id="grserve-vllm-bridge-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
bridge_sha="$(git rev-parse HEAD)"
python tools/run_bridge_smoke.py \
  --mode eager|fant-graph|both \
  --output "/data/infra/benchmarks/${bridge_run_id}" \
  --engine-bin "/data/infra/grserve/${bridge_sha}/build/linux-amd64-debug/bin/grserve-engine" \
  --source-root "/data/infra/grserve/${bridge_sha}/src" \
  --model /data/models/Qwen3.8-27B \
  --image fantasy/vllm:1.1.0.20-py310-ubuntu22.04 \
  --devices 0,2,4,6 --dry-run
```

It creates unique container/process/socket/event-log names, records intents before launch, starts
vendor vLLM → Sidecar → Engine → API, runs the exact seven-case matrix through the
GRServe endpoint, executes cancel/deadline/disconnect at C8, validates zero active
requests, and cleans up only identities created by that run. All subprocess calls use
argument vectors, bounded deadlines, `start_new_session=True`, and log files with a
size cap or post-run gzip.

The vendor launch vector shares BF16, TP=4, devices `0,2,4,6`,
`--max-model-len 32768`, `--gpu-memory-utilization 0.40`, `--block-size 800`,
`--language-model-only`, and no prefix caching. The exact mode delta is:

```python
EAGER_FLAGS = ["--enforce-eager"]
GRAPH_FLAGS = [
    "--compilation-config",
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",'
    '"cudagraph_capture_sizes":[1,2,4,8,16,32]}',
]
```

`bridge_evidence.py` rejects unknown/missing cases, non-integer counts, booleans in
integer fields, non-finite metrics, any request error, nonzero restart count, leaked
request/execution/lease counts, residual owned processes, and misleading identity flags.
Controller leak counts come from the final structured Controller event; Sidecar leak
counts come from its final structured shutdown record. Missing final records are a FAIL.

- [ ] **Step 4: Run tool tests and inspect dry-run commands**

```bash
uv run --locked pytest -q tests/python/test_bridge_smoke_tool.py
uv run --locked python tools/run_bridge_smoke.py \
  --mode both --output /data/infra/benchmarks/grserve-vllm-bridge-smoke-DRYRUN \
  --engine-bin /data/infra/grserve/DRYRUN/bin/grserve-engine \
  --source-root "$PWD" --model /data/models/Qwen3.8-27B \
  --image fantasy/vllm:1.1.0.20-py310-ubuntu22.04 \
  --devices 0,2,4,6 --dry-run
```

Expected: tests pass; dry-run prints two deployments, seven cases each, and performs no
filesystem/container/process mutation.

- [ ] **Step 5: Commit**

```bash
git add tools/run_bridge_smoke.py tools/bridge_evidence.py tests/python/test_bridge_smoke_tool.py pyproject.toml
git commit -m "feat: add guarded bridge smoke workflow"
```

---

## Task 13: Add an authoritative bridge-local verification profile

**Files:**
- Modify: `tools/verify.py`
- Modify: `docker/Dockerfile.linux-amd64`
- Modify: `tests/python/test_verify_tool.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing gate-membership tests**

Require a separate `bridge-local` profile with exact gates `Bridge C++`,
`Bridge Sidecar`, and `Bridge E2E`; require bridge sources in Ruff/mypy and Linux tests.
The existing Mac MVP `full` profile and its historical exact-gate reports must remain
unchanged. Also expose the existing sealed source-context builder through an explicit
export command so server builds retain the provenance format already verified by
`proto_codegen.py`:

```python
def test_bridge_local_profile_contains_exact_bridge_gates() -> None:
    runner = RecordingRunner()
    outcome = verify.run_profile("bridge-local", runner=runner)
    assert set(outcome.gates) == {"Bridge C++", "Bridge Sidecar", "Bridge E2E"}
    assert any("python/grserve_bridge" in command for command in runner.rendered_commands())

def test_mac_mvp_full_gate_set_is_not_rewritten_by_bridge() -> None:
    assert verify.required_gates("full") == MAC_MVP_REQUIRED_GATES

def test_exported_source_context_is_sealed(tmp_path: Path) -> None:
    destination = tmp_path / "context"
    verify.export_source_context(ROOT, destination)
    manifest = json.loads((destination / "source-provenance.json").read_text())
    assert manifest["root_head"] == git_text(ROOT, "rev-parse", "HEAD")
    assert manifest["protobuf_head"] == manifest["protobuf_gitlink"]
    assert manifest["files"]["root"]
```

- [ ] **Step 2: Verify RED**

```bash
uv run --locked pytest -q tests/python/test_verify_tool.py -k bridge
```

Expected: missing bridge gates and source coverage.

- [ ] **Step 3: Implement exact verification coverage**

Add focused CTest, Sidecar pytest, and real bridge E2E commands to the separate profile.
Do not change `tools/acceptance.py`, `tools/evidence.py`, or the Mac MVP exact gate set.
Add bridge files to Ruff/mypy and the Linux Docker command. README must include three
separate capability
tables:

```text
native MiniQwen path: local correctness only
vLLM bridge path: GRServe control plane + vendor executor
native GR308 path: not implemented
```

Add the non-overwriting CLI form:

```text
uv run --locked python tools/verify.py --export-source-context /absolute/new/directory
```

It calls the same clean-tree, gitlink, symlink, index-flag, file-count, byte-count, and
per-file SHA checks used by the Docker source context. It rejects an existing destination
and writes `source-provenance.json` inside the exported root.

- [ ] **Step 4: Run the full local gate**

```bash
uv run --locked python tools/verify.py --profile bridge-local
```

Expected on the Mac host: all three bridge-local gates pass. The existing full profile
continues to retain its separately documented Linux host-emulation block.

- [ ] **Step 5: Run direct fresh checks and commit**

```bash
uv run --locked ruff check python tools tests/python
MYPYPATH=python uv run --locked mypy --explicit-package-bases python/grserve_api python/grserve_bridge tools/verify.py tools/evidence.py tools/acceptance.py tools/bridge_evidence.py tools/run_bridge_smoke.py
GRSERVE_ENGINE_BIN="$PWD/build/mac-arm64-debug/bin/grserve-engine" uv run --locked pytest -q tests/python
uv run --locked ctest --preset mac-arm64-debug --no-tests=error
git add tools/verify.py tests/python/test_verify_tool.py docker/Dockerfile.linux-amd64 README.md
git commit -m "test: gate the vLLM bridge workflow"
```

Expected: Ruff/mypy pass, all Python/C++ tests pass, and any platform block is explicit.

---

## Task 14: Deploy immutably and run the server smoke matrix

**Files:**
- Create on server: `/data/infra/grserve/${bridge_sha}/src/source-provenance.json`
- Create on server: `/data/infra/grserve/${bridge_sha}/src/`
- Create on server: `/data/infra/benchmarks/${bridge_run_id}/`
- Do not modify: the three frozen baseline directories listed in Section 0

- [ ] **Step 1: Verify server preconditions without mutation**

Run over SSH:

```bash
test -d /data/models/Qwen3.8-27B
test "$(sha256sum /data/models/Qwen3.8-27B/config.json | awk '{print $1}')" = \
  191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab
test -z "$(docker ps -q)"
! ss -ltn | grep -Eq ':(8000|8010)\b'
fantgpu-smi
docker image inspect fantasy/vllm:1.1.0.20-py310-ubuntu22.04 >/dev/null
```

Expected: model/image exist, no running container/listener, and no accelerator process.

- [ ] **Step 2: Export a content-addressed source archive**

From the Mac worktree, create an archive containing tracked root files plus exact
submodule trees, calculate SHA256, and upload it to a new path under
`/data/infra/grserve`. Use these exact local commands:

```bash
bridge_sha="$(git rev-parse HEAD)"
test "$(git status --porcelain=v1)" = ""
bridge_stage="$(mktemp -d)"
uv run --locked python tools/verify.py --export-source-context "${bridge_stage}/context"
python3 - "${bridge_stage}/context/source-provenance.json" "${bridge_sha}" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert document["root_head"] == sys.argv[2]
assert document["protobuf_head"] == document["protobuf_gitlink"]
assert document["files"]["root"] and document["files"]["protobuf"]
PY
bridge_archive="/tmp/grserve-${bridge_sha}.tar.gz"
tar -C "${bridge_stage}/context" -czf "${bridge_archive}" .
bridge_archive_sha="$(shasum -a 256 "${bridge_archive}" | awk '{print $1}')"
bridge_id_file="/tmp/grserve-source-id"
printf '%s %s\n' "${bridge_sha}" "${bridge_archive_sha}" > "${bridge_id_file}"
test "${#bridge_sha}" -eq 40
test "${#bridge_archive_sha}" -eq 64
scp "${bridge_archive}" "${bridge_id_file}" user@192.0.2.11:/data/infra/
```

On the server, verify before extraction:

```bash
read -r bridge_sha expected_archive_sha < /data/infra/grserve-source-id
bridge_archive="/data/infra/grserve-${bridge_sha}.tar.gz"
bridge_archive_sha="$(sha256sum "${bridge_archive}" | awk '{print $1}')"
case "${bridge_sha}" in (*[!0-9a-f]*|'') exit 71;; esac
case "${bridge_archive_sha}" in (*[!0-9a-f]*|'') exit 72;; esac
test "${#bridge_sha}" -eq 40
test "${#bridge_archive_sha}" -eq 64
test "${bridge_archive_sha}" = "${expected_archive_sha}"
bridge_root="/data/infra/grserve/${bridge_sha}"
test ! -e "${bridge_root}"
mkdir -p "${bridge_root}/src"
tar -xzf "${bridge_archive}" -C "${bridge_root}/src"
python3 - "${bridge_root}/src/source-provenance.json" "${bridge_sha}" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert document["root_head"] == sys.argv[2]
assert document["protobuf_head"] == document["protobuf_gitlink"]
assert document["files"]["root"] and document["files"]["protobuf"]
PY
```

The exported provenance manifest remains inside the source root and is consumed by the
existing codegen/verification path when `.git` is absent.

- [ ] **Step 3: Build and test on native Linux before touching GPUs**

Use `/home/USER/levius/.bootstrap/bin/uv` and an isolated environment rooted under the
content-addressed source tree; do not install into the system Python or existing
baseline virtual environments:

```bash
uv_bin=/home/USER/levius/.bootstrap/bin/uv
test -x "${uv_bin}"
cd "/data/infra/grserve/${bridge_sha}/src"
UV_PROJECT_ENVIRONMENT="/data/infra/grserve/${bridge_sha}/venv" "${uv_bin}" sync --locked
UV_PROJECT_ENVIRONMENT="/data/infra/grserve/${bridge_sha}/venv" "${uv_bin}" run --locked cmake --preset linux-amd64-debug --fresh
UV_PROJECT_ENVIRONMENT="/data/infra/grserve/${bridge_sha}/venv" "${uv_bin}" run --locked cmake --build --preset linux-amd64-debug
UV_PROJECT_ENVIRONMENT="/data/infra/grserve/${bridge_sha}/venv" "${uv_bin}" run --locked ctest --preset linux-amd64-debug --no-tests=error
GRSERVE_ENGINE_BIN="$PWD/build/linux-amd64-debug/bin/grserve-engine" \
  UV_PROJECT_ENVIRONMENT="/data/infra/grserve/${bridge_sha}/venv" \
  "${uv_bin}" run --locked pytest -q tests/python/test_bridge_e2e.py
```

Expected: native Linux C++ tests and bridge E2E pass before vendor vLLM starts.

- [ ] **Step 4: Execute eager smoke through the GRServe endpoint**

```bash
bridge_run_id="grserve-vllm-bridge-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
bridge_output="/data/infra/benchmarks/${bridge_run_id}"
python tools/run_bridge_smoke.py \
  --mode eager \
  --output "${bridge_output}" \
  --engine-bin "$PWD/build/linux-amd64-debug/bin/grserve-engine" \
  --source-root "$PWD" --model /data/models/Qwen3.8-27B \
  --image fantasy/vllm:1.1.0.20-py310-ubuntu22.04 \
  --devices 0,2,4,6
```

Expected: C1/C2/C4/C8/C16/C32/C64 plus C8 cancel/deadline/disconnect pass, all requests
flow through the GRServe API port, and all owned processes/container are stopped.

- [ ] **Step 5: Execute FANT Graph smoke in the same run identity**

Resume only the exact output path created by Step 4 using the runner's explicit
`--resume-exact` option:

```bash
python tools/run_bridge_smoke.py \
  --mode fant-graph --resume-exact \
  --output "${bridge_output}" \
  --engine-bin "$PWD/build/linux-amd64-debug/bin/grserve-engine" \
  --source-root "$PWD" --model /data/models/Qwen3.8-27B \
  --image fantasy/vllm:1.1.0.20-py310-ubuntu22.04 \
  --devices 0,2,4,6
```

Expected: the same seven cases pass; C64 is labelled outside the configured capture
buckets rather than as a 64-size Graph capture.

- [ ] **Step 6: Validate, render, and freeze evidence**

```bash
python tools/bridge_evidence.py validate \
  --summary "${bridge_output}/summary.json"
python tools/bridge_evidence.py render-report \
  --summary "${bridge_output}/summary.json" \
  --output "${bridge_output}/REPORT_ZH.md"
cd "${bridge_output}"
find . -type f ! -name FINAL_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > FINAL_SHA256SUMS
```

Expected: validator returns `BRIDGE_SMOKE_ACCEPTED`; report explicitly states vendor
execution and no native Fant HAL.

- [ ] **Step 7: Confirm post-state and record final result**

```bash
test -z "$(docker ps -q)"
! ss -ltn | grep -Eq ':(8000|8010)\b'
fantgpu-smi
```

Expected: no owned service/container remains, no unexpected GPU process exists, and
memory has returned to the recorded idle range. If cleanup cannot be proven, the final
status is BLOCKED even when requests succeeded.

---

## Task 15: Review the bridge result without overstating capability

**Files:**
- Modify: `docs/reports/GRSERVE_VLLM_BRIDGE_SMOKE_ZH.md`
- Modify: `README.md`

- [ ] **Step 1: Bind the repository report to exact server evidence**

Copy only the validated summary and rendered report text into the repository report.
Record the exact server path, summary SHA256, GRServe Git SHA, image ID/digest, model
hash, mode matrix, request counts, cancellation outcomes, restart counts, leak counts,
and post-state.

- [ ] **Step 2: Run an overclaim scan**

```bash
rg -n '原生 Fant|替代 vLLM|性能提升|生产可用|正式 baseline' \
  docs/reports/GRSERVE_VLLM_BRIDGE_SMOKE_ZH.md README.md
```

Expected: every match is inside an explicit negation/scope-boundary sentence. Revise any
ambiguous positive claim before continuing.

- [ ] **Step 3: Validate links and evidence identity**

```bash
uv run --locked python tools/bridge_evidence.py validate-report \
  --summary docs/reports/evidence/grserve-vllm-bridge-smoke-summary.json \
  --report docs/reports/GRSERVE_VLLM_BRIDGE_SMOKE_ZH.md
git diff --check
```

Expected: `REPORT_VALID` and no whitespace errors.

- [ ] **Step 4: Run final fresh local verification**

```bash
uv run --locked ruff check python tools tests/python
MYPYPATH=python uv run --locked mypy --explicit-package-bases python/grserve_api python/grserve_bridge tools/verify.py tools/evidence.py tools/acceptance.py tools/bridge_evidence.py tools/run_bridge_smoke.py
GRSERVE_ENGINE_BIN="$PWD/build/mac-arm64-debug/bin/grserve-engine" uv run --locked pytest -q tests/python
uv run --locked ctest --preset mac-arm64-debug --no-tests=error
git status --short
```

Expected: all checks pass and the only uncommitted files are the intended report/evidence
changes.

- [ ] **Step 5: Commit the evidence-bound report**

```bash
git add README.md docs/reports/GRSERVE_VLLM_BRIDGE_SMOKE_ZH.md docs/reports/evidence/grserve-vllm-bridge-smoke-summary.json
git commit -m "docs: record vLLM bridge smoke result"
```

---

## Completion definition

The plan is complete only when:

- all 15 tasks are committed in order;
- native MiniQwen behavior remains green;
- local Fake-vLLM bridge C64 passes 20 consecutive runs;
- server-native Linux tests pass;
- eager and FANT Graph each pass C1/C2/C4/C8/C16/C32/C64 through the GRServe endpoint;
- C8 cancel, deadline, and disconnect cases match internal terminal state;
- both Controller and Sidecar report zero remaining requests/executions/leases;
- all owned processes and containers are stopped;
- the report states that vLLM is the executor and native Fant HAL is absent;
- no frozen baseline directory was modified.
