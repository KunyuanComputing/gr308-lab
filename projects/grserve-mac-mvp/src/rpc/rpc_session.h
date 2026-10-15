#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>

#include "engine.pb.h"
#include "engine/controller.h"
#include "engine/request_event_sink.h"
#include "engine/replica_gate.h"
#include "rpc/unix_socket.h"

namespace grserve::rpc {

class CancelResponseSink {
 public:
  virtual ~CancelResponseSink() = default;
  virtual bool TryComplete(RequestId request,
                           CancelDisposition disposition) noexcept = 0;
  virtual bool TryError(const Error& error) noexcept = 0;
};

class OwnerBoundRequestSink : public RequestEventSink {
 public:
  virtual bool BindInternal(RequestId request) noexcept = 0;
  [[nodiscard]] virtual std::optional<RequestId> internal_request() const noexcept = 0;
};

class ControllerMailbox {
 public:
  struct Submit {
    GenerateRequest request;
    std::shared_ptr<OwnerBoundRequestSink> sink;
  };
  struct Detach { std::shared_ptr<OwnerBoundRequestSink> request; };
  struct Cancel {
    std::shared_ptr<OwnerBoundRequestSink> request;
    CancelReason reason;
    std::shared_ptr<CancelResponseSink> response;
  };
  using Event = std::variant<Submit, Detach, Cancel>;

  [[nodiscard]] bool Post(Event event) noexcept;
  // Must only be called by the Controller owner thread.
  [[nodiscard]] bool PollOne(Controller& controller);
  [[nodiscard]] bool accepting() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  void StopAdmission() noexcept;
  void Close() noexcept;
#if defined(GRSERVE_RPC_TESTING)
  void WaitForEventForTesting();
  void WaitForEventCountForTesting(std::size_t count);
  [[nodiscard]] std::size_t size_for_testing() const noexcept;
  [[nodiscard]] static constexpr std::size_t capacity_for_testing() noexcept {
    return kSubmitCapacity + kDetachCapacity + kCancelCapacity;
  }
#endif

 private:
  static constexpr std::size_t kSubmitCapacity = 1024;
  static constexpr std::size_t kDetachCapacity = 2048;
  static constexpr std::size_t kCancelCapacity = 1024;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::array<std::optional<Submit>, kSubmitCapacity> submits_;
  std::array<std::optional<Detach>, kDetachCapacity> detaches_;
  std::array<std::optional<Cancel>, kCancelCapacity> cancels_;
  std::array<std::uint64_t, kSubmitCapacity> submit_sequences_{};
  std::array<std::uint64_t, kDetachCapacity> detach_sequences_{};
  std::array<std::uint64_t, kCancelCapacity> cancel_sequences_{};
  std::size_t submit_head_{0}, submit_size_{0};
  std::size_t detach_head_{0}, detach_size_{0};
  std::size_t cancel_head_{0}, cancel_size_{0};
  std::uint64_t next_event_sequence_{1};
  bool closed_{false};
  bool accepting_{true};
  std::uint64_t next_request_id_{1};
};

#if defined(GRSERVE_RPC_TESTING)
class RpcSessionTestBarrier {
 public:
  void BlockWriter();
  void WaitUntilWriterBlocked();
  void ReleaseWriter();
 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool blocked_{false};
  bool released_{false};
};
#endif

struct RpcSessionConfig {
  std::string model_id;
  std::size_t outbound_capacity{64};
  std::string engine_build_id{"0.1.0"};
  std::string manifest_sha256;
  std::uint32_t max_context_tokens{0};
  std::uint32_t vocab_size{0};
  std::shared_ptr<ReplicaGate> replica_gate;
};

class RpcSession final : public std::enable_shared_from_this<RpcSession> {
 public:
  [[nodiscard]] static StatusOr<std::shared_ptr<RpcSession>> Start(
      UniqueFd socket, std::shared_ptr<ControllerMailbox> mailbox,
      RpcSessionConfig config);
  RpcSession(const RpcSession&) = delete;
  RpcSession& operator=(const RpcSession&) = delete;
  ~RpcSession();
  void Shutdown() noexcept;
  [[nodiscard]] bool stopped() const noexcept;
#if defined(GRSERVE_RPC_TESTING)
  [[nodiscard]] std::shared_ptr<RequestEventSink> SinkForTesting(RequestId request);
  [[nodiscard]] static StatusOr<std::shared_ptr<RpcSession>> StartForTesting(
      UniqueFd socket, std::shared_ptr<ControllerMailbox> mailbox,
      RpcSessionConfig config, RpcSessionTestBarrier* writer_barrier);
#endif

 private:
  class Sink;
  class CancelSink;
  struct RequestTransportState {
    bool accepted{false};
    bool detached{false};
    bool cancel_pending{false};
    std::size_t pending_duplicate_cancels{0};
    std::uint64_t next_sequence{1};
    std::shared_ptr<OwnerBoundRequestSink> sink;
  };

  RpcSession(UniqueFd socket, std::shared_ptr<ControllerMailbox> mailbox,
             RpcSessionConfig config
#if defined(GRSERVE_RPC_TESTING)
             , RpcSessionTestBarrier* writer_barrier = nullptr
#endif
             );
  void ReaderLoop() noexcept;
  void WriterLoop() noexcept;
  void Handle(const v1::ClientEnvelope& envelope) noexcept;
  [[nodiscard]] std::shared_ptr<OwnerBoundRequestSink> MakeSink(
      std::string wire_request);
  [[nodiscard]] std::shared_ptr<CancelResponseSink> MakeCancelSink(
      std::string wire_request);
  [[nodiscard]] bool Enqueue(v1::ServerEnvelope envelope,
                             bool terminal = false) noexcept;
  [[nodiscard]] bool EnqueueLocked(v1::ServerEnvelope envelope,
                                   bool terminal);
  void EnqueueError(Error error, std::string wire_request = {}) noexcept;
  void FailTransport() noexcept;
  [[nodiscard]] bool Detach(const std::shared_ptr<OwnerBoundRequestSink>& request) noexcept;
  void DetachAll() noexcept;
  [[nodiscard]] static bool ValidWireRequestId(const std::string& text) noexcept;

  UniqueFd socket_;
  std::shared_ptr<ControllerMailbox> mailbox_;
  RpcSessionConfig config_;
  mutable std::mutex mutex_;
  std::condition_variable outbound_cv_;
  std::deque<v1::ServerEnvelope> outbound_;
  std::unordered_map<std::string, RequestTransportState> requests_;
  std::atomic<bool> stopping_{false};
  std::thread reader_;
  std::thread writer_;
#if defined(GRSERVE_RPC_TESTING)
  RpcSessionTestBarrier* writer_barrier_{nullptr};
#endif
};

}  // namespace grserve::rpc
