#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "engine/admission.h"
#include "engine/completion_queue.h"
#include "engine/rank_worker.h"
#include "engine/replica_gate.h"
#include "engine/request_event_sink.h"
#include "engine/transaction.h"

namespace grserve {

struct ControllerRank {
  RankId rank_id;
  std::function<Status(RankCommand)> enqueue;
};

// A Controller is thread-affine: Create binds it to the constructing thread.
// All stateful API calls must run there; foreign-thread PollOnce is a no-op.
// The completion queue and every captured rank endpoint must outlive the
// Controller. Call Shutdown (or destroy the Controller) before destroying or
// joining those dependencies so outstanding shadows can be terminated.
class Controller {
 public:
  using TokenPlanner =
      std::function<TokenId(const GenerateRequest&, SequenceNumber)>;
  using TimePoint = std::chrono::steady_clock::time_point;
  using Clock = std::function<TimePoint()>;
  using StartupHook = std::function<void(const GenerateRequest&)>;

  [[nodiscard]] static StatusOr<std::unique_ptr<Controller>> Create(
      AdmissionController admission, std::shared_ptr<ReplicaGate> replica_gate,
      CompletionQueue<RankCompletion>& completions, std::vector<ControllerRank> ranks,
      TokenPlanner token_planner = {}, Clock clock = {},
      StartupHook startup_hook = {});

  Controller(const Controller&) = delete;
  Controller& operator=(const Controller&) = delete;
  ~Controller() noexcept;

  Status Submit(GenerateRequest request, std::shared_ptr<RequestEventSink> sink);
  Status Cancel(RequestId request, CancelReason reason);
  // Owner-thread correlated cancellation for transports that need a decision
  // phase acknowledgement. The request is detached from network delivery.
  [[nodiscard]] StatusOr<CancelDisposition> CancelTransport(
      RequestId request, CancelReason reason);
  Status DetachTransport(RequestId request);
  Status Shutdown();
  bool PollOnce();
  [[nodiscard]] bool idle() const noexcept {
    return !active_ && pending_.empty();
  }

  [[nodiscard]] std::optional<TransactionState>
  last_terminal_state_for_testing() const noexcept {
    return last_terminal_transaction_
               ? std::optional{last_terminal_transaction_->state()}
               : std::nullopt;
  }
  [[nodiscard]] std::size_t logical_publish_count() const noexcept {
    return logical_publish_count_;
  }
  [[nodiscard]] std::size_t network_delivery_count() const noexcept {
    return network_delivery_count_;
  }
  [[nodiscard]] DeliveryOutcome last_delivery_outcome() const noexcept {
    return last_delivery_outcome_;
  }
  [[nodiscard]] std::optional<TimePoint> active_deadline_for_testing() const noexcept {
    return active_ ? std::optional{active_->deadline} : std::nullopt;
  }
  [[nodiscard]] std::size_t control_replay_size_for_testing() const noexcept {
    return control_replay_count_;
  }

 private:
  struct RequestState {
    GenerateRequest request;
    std::shared_ptr<RequestEventSink> sink;
    ReservationLease reservation;
    TimePoint deadline;
    SequenceNumber sequence{0};
    std::optional<Transaction> transaction;
    TokenId candidate{0};
    bool detached{false};
    bool stop_after_current{false};
    bool control_frozen{false};
    bool rank_lost{false};
    std::vector<bool> prepare_dispatched;
    Error startup_error;
    Error shutdown_error;

    RequestState(GenerateRequest request, std::shared_ptr<RequestEventSink> sink,
                 ReservationLease reservation, TimePoint deadline)
        : request(std::move(request)),
          sink(std::move(sink)),
          reservation(std::move(reservation)),
          deadline(deadline),
          startup_error(Error::From(ErrorCode::kInternal, "startup",
                                    this->request.request_id)),
          shutdown_error(Error::From(ErrorCode::kInternal, "shutdown",
                                     this->request.request_id)) {}
  };

  Controller(AdmissionController admission, std::shared_ptr<ReplicaGate> replica_gate,
             CompletionQueue<RankCompletion>& completions,
             std::vector<ControllerRank> ranks, TokenPlanner token_planner,
             Clock clock, StartupHook startup_hook);

  [[nodiscard]] Status StartNextRequest();
  [[nodiscard]] Status StartToken(RequestState& state);
  [[nodiscard]] Status CheckDeadline(const RequestState& state) const;
  [[nodiscard]] TimePoint SaturatingDeadline(TimePoint now,
                                             std::uint64_t timeout_ms) const noexcept;
  [[nodiscard]] Status DispatchPrepare(const Transaction& transaction);
  [[nodiscard]] Status DispatchExecute(const Transaction& transaction,
                                       TokenId candidate,
                                       const GenerateRequest& request,
                                       SequenceNumber sequence);
  [[nodiscard]] Status DispatchCommit(const Transaction& transaction);
  [[nodiscard]] Status HandleCompletion(const RankCompletion& completion);
  [[nodiscard]] Status PublishCommitted(RequestState& state);
  [[nodiscard]] StateDigest AggregateDigest(
      const Transaction& transaction) const noexcept;
  [[nodiscard]] Status CheckOwner(const char* operation) const;
  void AbortPreparedRanks(RequestState& state) noexcept;
  void PreserveActiveTransaction() noexcept;
  [[nodiscard]] bool TerminateActive(const Error& error);
  void DrainPending(const Error* fatal_reason) noexcept;
  void ShutdownNoexcept() noexcept;
  void FinishActive();
  void FailActive(Error error);
  void CancelActive(CancelReason reason, bool detach);
  enum class ControlKind { kCancel, kDetach };
  struct ControlReplay {
    RequestId request{0};
    ControlKind kind{ControlKind::kCancel};
    CancelReason reason{CancelReason::kClientDisconnected};
  };
  [[nodiscard]] bool WasControlled(RequestId request,
                                   ControlKind kind,
                                   CancelReason reason) const noexcept;
  void RememberControl(RequestId request, ControlKind kind,
                       CancelReason reason) noexcept;
  [[nodiscard]] bool ContainsRequest(RequestId request) const noexcept;
  [[nodiscard]] Error RequestError(ErrorCode code, const char* message,
                                   RequestId request) const;

  AdmissionController admission_;
  std::shared_ptr<ReplicaGate> replica_gate_;
  CompletionQueue<RankCompletion>* completions_;
  std::vector<ControllerRank> ranks_;
  std::vector<RankId> rank_ids_;
  TokenPlanner token_planner_;
  Clock clock_;
  StartupHook startup_hook_;
  std::deque<RequestState> pending_;
  std::optional<RequestState> active_;
  std::optional<Transaction> last_terminal_transaction_;
  std::uint64_t next_transaction_id_{1};
  std::uint64_t next_epoch_{1};
  // Request IDs form a controller-lifetime generation: only a successful
  // TryAccepted advances this watermark, and later accepted IDs must increase.
  // This makes bounded control replay safe after old tombstones are evicted.
  std::uint64_t highest_accepted_request_id_{0};
  std::thread::id owner_thread_;
  bool shutdown_{false};
  bool completion_abort_handled_{false};
  Error destructor_fatal_error_;
  std::size_t logical_publish_count_{0};
  std::size_t network_delivery_count_{0};
  DeliveryOutcome last_delivery_outcome_{DeliveryOutcome::kNone};
  static constexpr std::size_t kControlReplayLimit = 256;
  std::array<ControlReplay, kControlReplayLimit> control_replay_{};
  std::size_t control_replay_count_{0};
  std::size_t control_replay_cursor_{0};
};

}  // namespace grserve
