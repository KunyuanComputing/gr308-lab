#include "engine/controller.h"

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace grserve {
namespace {

TokenId DefaultToken(const GenerateRequest& request, SequenceNumber sequence) {
  const std::uint64_t seed = request.prompt_token_ids.empty()
                                 ? 0
                                 : request.prompt_token_ids.back().value();
  return TokenId{static_cast<std::uint32_t>(seed + sequence.value())};
}

Controller::TimePoint SteadyNow() { return std::chrono::steady_clock::now(); }

Epoch CompletionEpoch(const RankCompletion& completion) noexcept {
  return std::visit([](const auto& value) { return value.epoch; }, completion);
}

RankId CompletionRank(const RankCompletion& completion) noexcept {
  return std::visit([](const auto& value) { return value.rank_id; }, completion);
}

Status CorrelateDispatchFailure(const Status& status, RequestId request,
                                TransactionId transaction) {
  if (status.ok()) return Status::Ok();
  return Status{Error::From(status.error()->code(), status.error()->message(),
                            request, transaction)};
}

}  // namespace

StatusOr<std::unique_ptr<Controller>> Controller::Create(
    AdmissionController admission, std::shared_ptr<ReplicaGate> replica_gate,
    CompletionQueue<RankCompletion>& completions, std::vector<ControllerRank> ranks,
    TokenPlanner token_planner, Clock clock, StartupHook startup_hook) {
  if (!replica_gate) {
    return StatusOr<std::unique_ptr<Controller>>{Error::From(
        ErrorCode::kInvalidArgument, "controller requires a replica gate")};
  }
  if (ranks.empty()) {
    return StatusOr<std::unique_ptr<Controller>>{Error::From(
        ErrorCode::kInvalidArgument, "controller requires at least one rank")};
  }
  for (std::size_t index = 0; index < ranks.size(); ++index) {
    if (!ranks[index].enqueue) {
      return StatusOr<std::unique_ptr<Controller>>{Error::From(
          ErrorCode::kInvalidArgument, "controller rank endpoint is empty")};
    }
    if (!admission.capacity(ranks[index].rank_id).ok()) {
      return StatusOr<std::unique_ptr<Controller>>{Error::From(
          ErrorCode::kInvalidArgument, "controller rank is absent from admission")};
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (ranks[prior].rank_id == ranks[index].rank_id) {
        return StatusOr<std::unique_ptr<Controller>>{Error::From(
            ErrorCode::kInvalidArgument, "controller ranks must be unique")};
      }
    }
  }
  if (!token_planner) token_planner = DefaultToken;
  if (!clock) clock = SteadyNow;
  return StatusOr<std::unique_ptr<Controller>>{std::unique_ptr<Controller>(
      new Controller(std::move(admission), std::move(replica_gate), completions,
                     std::move(ranks), std::move(token_planner), std::move(clock),
                     std::move(startup_hook)))};
}

Controller::Controller(AdmissionController admission,
                       std::shared_ptr<ReplicaGate> replica_gate,
                       CompletionQueue<RankCompletion>& completions,
                       std::vector<ControllerRank> ranks,
                       TokenPlanner token_planner, Clock clock,
                       StartupHook startup_hook)
    : admission_(std::move(admission)),
      replica_gate_(std::move(replica_gate)),
      completions_(&completions),
      ranks_(std::move(ranks)),
      token_planner_(std::move(token_planner)),
      clock_(std::move(clock)),
      startup_hook_(std::move(startup_hook)),
      owner_thread_(std::this_thread::get_id()),
      destructor_fatal_error_(
          Error::From(ErrorCode::kReplicaFatal, "shutdown")) {
  rank_ids_.reserve(ranks_.size());
  for (const auto& rank : ranks_) rank_ids_.push_back(rank.rank_id);
}

Controller::~Controller() noexcept { ShutdownNoexcept(); }

Status Controller::Submit(GenerateRequest request,
                          std::shared_ptr<RequestEventSink> sink) {
  const Status owner = CheckOwner("submit");
  if (!owner.ok()) return owner;
  if (shutdown_) {
    return Status{RequestError(ErrorCode::kInternal, "controller is shut down",
                               request.request_id)};
  }
  const Status gate = replica_gate_->CheckAdmission();
  if (!gate.ok()) {
    return Status{RequestError(ErrorCode::kReplicaFatal,
                               "replica is unavailable after a fatal transaction",
                               request.request_id)};
  }
  if (request.request_id.value() == 0 || request.model_id.empty() ||
      request.prompt_token_ids.empty() || request.max_tokens == 0 ||
      request.timeout_ms == 0 || request.state_slots == 0 || !sink) {
    return Status{RequestError(ErrorCode::kInvalidArgument,
                               "generate request fields must be nonempty and nonzero",
                               request.request_id)};
  }
  if (request.request_id.value() <= highest_accepted_request_id_) {
    return Status{RequestError(
        ErrorCode::kInvalidArgument,
        "request id must increase within the controller lifetime",
        request.request_id)};
  }
  if (ContainsRequest(request.request_id)) {
    return Status{RequestError(ErrorCode::kInvalidArgument,
                               "request id is already active", request.request_id)};
  }

  TimePoint now;
  try {
    now = clock_();
  } catch (...) {
    return Status{RequestError(ErrorCode::kInternal, "controller clock threw",
                               request.request_id)};
  }
  const TimePoint deadline = SaturatingDeadline(now, request.timeout_ms);

  auto reservation = admission_.ReserveAll(request.request_id, request.state_slots);
  if (!reservation.ok()) return reservation.status();
  const RequestId request_id = request.request_id;
  pending_.emplace_back(std::move(request), std::move(sink),
                        std::move(reservation).value(), deadline);
  if (!pending_.back().sink->TryAccepted(request_id)) {
    pending_.pop_back();
    return Status{RequestError(ErrorCode::kResourceExhausted,
                               "accepted event queue is full", request_id)};
  }
  highest_accepted_request_id_ = request_id.value();
  if (!active_) {
    const Status started = StartNextRequest();
    if (!started.ok()) {
      FailActive(*started.error());
    }
  }
  return Status::Ok();
}

Status Controller::Cancel(RequestId request, CancelReason reason) {
  const Status owner = CheckOwner("cancel");
  if (!owner.ok()) return owner;
  if (request.value() == 0) {
    return Status{RequestError(ErrorCode::kInvalidArgument,
                               "request id must be nonzero", request)};
  }
  if (WasControlled(request, ControlKind::kCancel, reason)) return Status::Ok();
  if (active_ && active_->request.request_id == request) {
    RememberControl(request, ControlKind::kCancel, reason);
    CancelActive(reason, false);
    return Status::Ok();
  }
  const auto pending = std::find_if(
      pending_.begin(), pending_.end(), [request](const RequestState& state) {
        return state.request.request_id == request;
      });
  if (pending == pending_.end()) {
    return Status{RequestError(ErrorCode::kInvalidArgument,
                               "request is not active", request)};
  }
  RememberControl(request, ControlKind::kCancel, reason);
  if (reason != CancelReason::kClientDisconnected) {
    static_cast<void>(pending->sink->TryError(RequestError(
        reason == CancelReason::kDeadline ? ErrorCode::kDeadlineExceeded
                                          : ErrorCode::kInternal,
        reason == CancelReason::kDeadline ? "request deadline exceeded"
                                          : "request cancelled",
        request)));
  }
  pending_.erase(pending);
  return Status::Ok();
}

StatusOr<CancelDisposition> Controller::CancelTransport(
    RequestId request, CancelReason reason) {
  const Status owner = CheckOwner("cancel transport");
  if (!owner.ok()) {
    return StatusOr<CancelDisposition>(*owner.error());
  }
  if (request.value() == 0) {
    return StatusOr<CancelDisposition>(RequestError(
        ErrorCode::kInvalidArgument, "request id must be nonzero", request));
  }
  if (WasControlled(request, ControlKind::kCancel, reason)) {
    return StatusOr<CancelDisposition>(CancelDisposition::kAlreadyTerminal);
  }
  if (active_ && active_->request.request_id == request) {
    const bool post_decision =
        active_->transaction && !active_->transaction->may_abort();
    RememberControl(request, ControlKind::kCancel, reason);
    CancelActive(reason, true);
    return StatusOr<CancelDisposition>(
        post_decision ? CancelDisposition::kCommittingDetached
                      : CancelDisposition::kAbortedBeforeDecision);
  }
  const auto pending = std::find_if(
      pending_.begin(), pending_.end(), [request](const RequestState& state) {
        return state.request.request_id == request;
      });
  if (pending == pending_.end()) {
    return StatusOr<CancelDisposition>(RequestError(
        ErrorCode::kInvalidArgument, "request is not active", request));
  }
  RememberControl(request, ControlKind::kCancel, reason);
  pending_.erase(pending);
  return StatusOr<CancelDisposition>(
      CancelDisposition::kAbortedBeforeDecision);
}

Status Controller::DetachTransport(RequestId request) {
  const Status owner = CheckOwner("detach transport");
  if (!owner.ok()) return owner;
  if (request.value() == 0) {
    return Status{RequestError(ErrorCode::kInvalidArgument,
                               "request id must be nonzero", request)};
  }
  if (WasControlled(request, ControlKind::kDetach,
                    CancelReason::kClientDisconnected)) {
    return Status::Ok();
  }
  if (active_ && active_->request.request_id == request) {
    RememberControl(request, ControlKind::kDetach,
                    CancelReason::kClientDisconnected);
    CancelActive(CancelReason::kClientDisconnected, true);
    return Status::Ok();
  }
  const auto pending = std::find_if(
      pending_.begin(), pending_.end(), [request](const RequestState& state) {
        return state.request.request_id == request;
      });
  if (pending == pending_.end()) {
    return Status{RequestError(ErrorCode::kInvalidArgument,
                               "request is not active", request)};
  }
  RememberControl(request, ControlKind::kDetach,
                  CancelReason::kClientDisconnected);
  pending_.erase(pending);
  return Status::Ok();
}

Status Controller::Shutdown() {
  const Status owner = CheckOwner("shutdown");
  if (!owner.ok()) return owner;
  ShutdownNoexcept();
  return Status::Ok();
}

bool Controller::PollOnce() {
  if (std::this_thread::get_id() != owner_thread_) return false;
  if (completions_->aborted() && !completion_abort_handled_) {
    completion_abort_handled_ = true;
    const Error fatal = Error::From(
        ErrorCode::kReplicaFatal, "rank completion queue aborted",
        active_ ? std::optional{active_->request.request_id} : std::nullopt,
        active_ && active_->transaction
            ? std::optional{active_->transaction->id()}
            : std::nullopt);
    if (active_) {
      active_->rank_lost = true;
      FailActive(fatal);
    } else {
      replica_gate_->MarkFatal(fatal);
      DrainPending(&fatal);
    }
    return true;
  }
  if (active_) {
    const Status deadline = CheckDeadline(*active_);
    if (!deadline.ok()) {
      FailActive(*deadline.error());
      return true;
    }
  }
  auto completion = completions_->TryPop();
  if (!completion) return false;
  try {
    const Status status = HandleCompletion(*completion);
    if (!status.ok()) FailActive(*status.error());
  } catch (...) {
    FailActive(Error::From(ErrorCode::kInternal,
                           "controller completion callback threw",
                           active_ ? std::optional{active_->request.request_id}
                                   : std::nullopt,
                           active_ && active_->transaction
                               ? std::optional{active_->transaction->id()}
                               : std::nullopt));
  }
  return true;
}

Status Controller::StartNextRequest() {
  while (!shutdown_ && !active_ && !pending_.empty()) {
    active_.emplace(std::move(pending_.front()));
    pending_.pop_front();
    Status started;
    try {
      started = StartToken(*active_);
    } catch (...) {
      started = Status{active_->startup_error};
    }
    if (started.ok()) return Status::Ok();
    if (TerminateActive(*started.error())) {
      DrainPending(started.error());
      return replica_gate_->CheckAdmission();
    }
  }
  return Status::Ok();
}

Status Controller::StartToken(RequestState& state) {
  Status deadline = CheckDeadline(state);
  if (!deadline.ok()) return deadline;
  if (startup_hook_) startup_hook_(state.request);
  if (next_transaction_id_ == 0 || next_epoch_ == 0 ||
      next_transaction_id_ == std::numeric_limits<std::uint64_t>::max() ||
      next_epoch_ == std::numeric_limits<std::uint64_t>::max()) {
    return Status{RequestError(ErrorCode::kInternal,
                               "transaction id space is exhausted",
                               state.request.request_id)};
  }
  const TransactionId id{next_transaction_id_++};
  const Epoch epoch{next_epoch_++};
  auto transaction = Transaction::Begin(id, epoch, rank_ids_);
  if (!transaction.ok()) return transaction.status();
  Status reserved = transaction.value().ReserveAll();
  if (!reserved.ok()) return reserved;
  state.transaction.emplace(std::move(transaction).value());
  state.prepare_dispatched.assign(ranks_.size(), false);
  return DispatchPrepare(*state.transaction);
}

Status Controller::CheckDeadline(const RequestState& state) const {
  try {
    if (clock_() < state.deadline) return Status::Ok();
  } catch (...) {
    return Status{Error::From(ErrorCode::kInternal, "controller clock threw",
                              state.request.request_id,
                              state.transaction
                                  ? std::optional{state.transaction->id()}
                                  : std::nullopt)};
  }
  return Status{Error::From(ErrorCode::kDeadlineExceeded,
                            "request deadline exceeded",
                            state.request.request_id,
                            state.transaction
                                ? std::optional{state.transaction->id()}
                                : std::nullopt)};
}

Controller::TimePoint Controller::SaturatingDeadline(
    TimePoint now, std::uint64_t timeout_ms) const noexcept {
  using Duration = TimePoint::duration;
  using Wide = __int128_t;
  const Wide numerator = static_cast<Wide>(timeout_ms) * Duration::period::den;
  const Wide denominator = static_cast<Wide>(1000) * Duration::period::num;
  Wide ticks = numerator / denominator;
  if (numerator % denominator != 0) ++ticks;
  const Wide now_ticks = static_cast<Wide>(now.time_since_epoch().count());
  const Wide max_ticks =
      static_cast<Wide>(TimePoint::max().time_since_epoch().count());
  if (ticks >= max_ticks - now_ticks) return TimePoint::max();
  return TimePoint{Duration{static_cast<Duration::rep>(now_ticks + ticks)}};
}

Status Controller::DispatchPrepare(const Transaction& transaction) {
  for (std::size_t index = 0; index < ranks_.size(); ++index) {
    const Status deadline = CheckDeadline(*active_);
    if (!deadline.ok()) return deadline;
    const auto& rank = ranks_[index];
    Status status;
    // A callback that fails or throws may have accepted the command before
    // reporting failure. Conservatively include that rank in the abort set.
    active_->prepare_dispatched[index] = true;
    try {
      status = rank.enqueue(
          Prepare{transaction.id(), rank.rank_id, transaction.epoch()});
    } catch (...) {
      return Status{Error::From(ErrorCode::kInternal,
                                "rank prepare enqueue threw",
                                active_->request.request_id, transaction.id())};
    }
    if (!status.ok()) {
      return CorrelateDispatchFailure(status, active_->request.request_id,
                                      transaction.id());
    }
  }
  return Status::Ok();
}

Status Controller::DispatchExecute(const Transaction& transaction,
                                   TokenId candidate,
                                   const GenerateRequest& request,
                                   SequenceNumber sequence) {
  for (const auto& rank : ranks_) {
    const Status deadline = CheckDeadline(*active_);
    if (!deadline.ok()) return deadline;
    Status status;
    try {
      status = rank.enqueue(Execute{transaction.id(), rank.rank_id,
                                    transaction.epoch(), candidate,
                                    request.prompt_token_ids, sequence});
    } catch (...) {
      return Status{Error::From(ErrorCode::kInternal,
                                "rank execute enqueue threw",
                                active_->request.request_id, transaction.id())};
    }
    if (!status.ok()) {
      return CorrelateDispatchFailure(status, active_->request.request_id,
                                      transaction.id());
    }
  }
  return Status::Ok();
}

Status Controller::DispatchCommit(const Transaction& transaction) {
  for (const auto& rank : ranks_) {
    const Status deadline = CheckDeadline(*active_);
    if (!deadline.ok()) return deadline;
    Status status;
    try {
      status = rank.enqueue(
          Commit{transaction.id(), rank.rank_id, transaction.epoch()});
    } catch (...) {
      return Status{Error::From(ErrorCode::kInternal,
                                "rank commit enqueue threw",
                                active_->request.request_id, transaction.id())};
    }
    if (!status.ok()) {
      return CorrelateDispatchFailure(status, active_->request.request_id,
                                      transaction.id());
    }
  }
  return Status::Ok();
}

Status Controller::HandleCompletion(const RankCompletion& completion) {
  if (!active_ || !active_->transaction) {
    return Status{Error::From(ErrorCode::kInternal,
                              "completion has no active transaction")};
  }
  Transaction& transaction = *active_->transaction;
  if (std::find(rank_ids_.begin(), rank_ids_.end(), CompletionRank(completion)) ==
      rank_ids_.end()) {
    return Status::Ok();
  }
  const Epoch completion_epoch = CompletionEpoch(completion);
  if (completion_epoch != transaction.epoch()) {
    return Status::Ok();
  }
  if (const auto* rank_error = std::get_if<RankError>(&completion);
      rank_error != nullptr &&
      rank_error->transaction_id != transaction.id()) {
    return Status::Ok();
  }
  return std::visit(
      [this, &transaction](const auto& value) -> Status {
        using Completion = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Completion, PreparedAck>) {
          const TransactionState before = transaction.state();
          Status status = transaction.AcknowledgePrepared(value);
          if (!status.ok()) return status;
          if (before != TransactionState::kPrepared &&
              transaction.state() == TransactionState::kPrepared) {
            const SequenceNumber sequence{active_->sequence.value() + 1};
            try {
              active_->candidate = token_planner_(active_->request, sequence);
            } catch (...) {
              return Status{Error::From(ErrorCode::kInternal,
                                        "token planner threw",
                                        active_->request.request_id,
                                        transaction.id())};
            }
            return DispatchExecute(transaction, active_->candidate,
                                   active_->request, sequence);
          }
          return Status::Ok();
        } else if constexpr (std::is_same_v<Completion, ExecutedAck>) {
          Status status = transaction.AcknowledgeExecuted(value);
          if (!status.ok()) return status;
          if (transaction.state() != TransactionState::kExecuted) {
            return Status::Ok();
          }
          status = transaction.DecideCommit();
          return status.ok() ? DispatchCommit(transaction) : status;
        } else if constexpr (std::is_same_v<Completion, CommittedAck>) {
          Status status = transaction.AcknowledgeCommitted(value);
          if (!status.ok()) return status;
          return transaction.state() == TransactionState::kLocalCommitted
                     ? PublishCommitted(*active_)
                     : Status::Ok();
        } else if constexpr (std::is_same_v<Completion, RankError>) {
          active_->rank_lost = value.rank_lost;
          return Status{Error::From(value.error.code(), value.error.message(),
                                    active_->request.request_id,
                                    transaction.id())};
        } else {
          return Status{Error::From(ErrorCode::kInternal,
                                    "unexpected abort acknowledgement",
                                    active_->request.request_id, transaction.id())};
        }
      },
      completion);
}

Status Controller::PublishCommitted(RequestState& state) {
  Transaction& transaction = *state.transaction;
  const SequenceNumber sequence{state.sequence.value() + 1};
  Status status = transaction.MarkPublished(sequence);
  if (!status.ok()) return status;
  ++logical_publish_count_;
  const StateDigest digest = AggregateDigest(transaction);
  const TokenEvent event{state.request.request_id, transaction.id(), sequence,
                         *transaction.candidate_token(), digest};
  const bool delivered = !state.detached && !state.stop_after_current &&
                         state.sink->TryToken(event);
  if (delivered) {
    ++network_delivery_count_;
    last_delivery_outcome_ = DeliveryOutcome::kDelivered;
  } else {
    last_delivery_outcome_ = DeliveryOutcome::kDroppedClientDisconnect;
  }
  state.sequence = sequence;
  if (state.stop_after_current) {
    FinishActive();
    return Status::Ok();
  }
  if (!delivered) {
    state.detached = true;
    FinishActive();
    return Status::Ok();
  }
  if (state.sequence.value() == state.request.max_tokens) {
    const FinishedEvent finished{
        state.request.request_id, FinishReason::kLength,
        static_cast<std::uint32_t>(state.request.prompt_token_ids.size()),
        static_cast<std::uint32_t>(state.sequence.value()), state.sequence};
    static_cast<void>(state.sink->TryFinished(finished));
    FinishActive();
    return Status::Ok();
  }
  state.transaction.reset();
  return StartToken(state);
}

StateDigest Controller::AggregateDigest(
    const Transaction& transaction) const noexcept {
  // The rank list is immutable and preserves replica configuration order, so
  // completion arrival order cannot affect this digest.
  std::uint64_t value = 1469598103934665603ULL;
  const auto mix = [&value](std::uint64_t input) noexcept {
    value ^= input;
    value *= 1099511628211ULL;
  };
  for (const RankId rank : rank_ids_) {
    mix(rank.value());
    const auto digest = transaction.state_digest(rank);
    mix(digest ? digest->value() : 0);
  }
  return StateDigest{value};
}

Status Controller::CheckOwner(const char* operation) const {
  if (std::this_thread::get_id() == owner_thread_) return Status::Ok();
  return Status{Error::From(ErrorCode::kInternal,
                            std::string{"controller "} + operation +
                                " called from non-owner thread")};
}

void Controller::AbortPreparedRanks(RequestState& state) noexcept {
  if (!state.transaction) return;
  for (std::size_t index = 0; index < ranks_.size(); ++index) {
    if (index >= state.prepare_dispatched.size() ||
        !state.prepare_dispatched[index]) {
      continue;
    }
    try {
      static_cast<void>(ranks_[index].enqueue(
          Abort{state.transaction->id(), ranks_[index].rank_id,
                state.transaction->epoch()}));
    } catch (...) {
      // Abort is best effort after the logical transaction is already terminal.
    }
  }
}

void Controller::PreserveActiveTransaction() noexcept {
  if (!active_ || !active_->transaction) return;
  last_terminal_transaction_.emplace(std::move(*active_->transaction));
  active_->transaction.reset();
}

void Controller::FinishActive() {
  PreserveActiveTransaction();
  active_.reset();
  static_cast<void>(StartNextRequest());
}

bool Controller::TerminateActive(const Error& error) {
  if (!active_) return false;
  const bool post_decision =
      active_->transaction && !active_->transaction->may_abort();
  const bool loses_replica = post_decision || active_->rank_lost;
  if (post_decision) {
    static_cast<void>(active_->transaction->MarkFatal(error));
    replica_gate_->MarkFatal(error);
  } else if (active_->transaction) {
    static_cast<void>(active_->transaction->Abort(error));
    AbortPreparedRanks(*active_);
  }
  if (!post_decision && loses_replica) replica_gate_->MarkFatal(error);
  if (!active_->detached) static_cast<void>(active_->sink->TryError(error));
  PreserveActiveTransaction();
  active_.reset();
  return loses_replica;
}

void Controller::CancelActive(CancelReason reason, bool detach) {
  if (!active_) return;
  RequestState& state = *active_;
  if (state.transaction && !state.transaction->may_abort()) {
    if (reason == CancelReason::kDeadline) {
      if (detach) state.detached = true;
      FailActive(Error::From(ErrorCode::kDeadlineExceeded,
                             "request deadline exceeded",
                             state.request.request_id,
                             state.transaction->id()));
      return;
    }
    if (state.control_frozen) return;
    state.stop_after_current = true;
    state.control_frozen = true;
    const bool silent = detach || reason == CancelReason::kClientDisconnected;
    if (!silent) {
      static_cast<void>(state.sink->TryError(RequestError(
          ErrorCode::kInternal, "request cancelled", state.request.request_id)));
    }
    state.detached = true;
    return;
  }
  const Error error = RequestError(
      reason == CancelReason::kDeadline ? ErrorCode::kDeadlineExceeded
                                        : ErrorCode::kInternal,
      reason == CancelReason::kDeadline ? "request deadline exceeded"
                                        : "request cancelled",
      state.request.request_id);
  if (state.transaction) {
    static_cast<void>(state.transaction->Abort(error));
    AbortPreparedRanks(state);
  }
  if (!detach && reason != CancelReason::kClientDisconnected) {
    static_cast<void>(state.sink->TryError(error));
  }
  PreserveActiveTransaction();
  active_.reset();
  static_cast<void>(StartNextRequest());
}

bool Controller::WasControlled(RequestId request, ControlKind kind,
                               CancelReason reason) const noexcept {
  for (std::size_t index = 0; index < control_replay_count_; ++index) {
    if (control_replay_[index].request == request &&
        control_replay_[index].kind == kind &&
        control_replay_[index].reason == reason) {
      return true;
    }
  }
  return false;
}

void Controller::RememberControl(RequestId request, ControlKind kind,
                                 CancelReason reason) noexcept {
  if (control_replay_count_ < kControlReplayLimit) {
    control_replay_[control_replay_count_++] =
        ControlReplay{request, kind, reason};
    return;
  }
  control_replay_[control_replay_cursor_] =
      ControlReplay{request, kind, reason};
  control_replay_cursor_ = (control_replay_cursor_ + 1) % kControlReplayLimit;
}

void Controller::DrainPending(const Error* fatal_reason) noexcept {
  for (auto& pending : pending_) {
    if (!pending.detached) {
      if (fatal_reason != nullptr) {
        const Error scoped = fatal_reason->RecodeForRequest(
            ErrorCode::kReplicaFatal, pending.request.request_id);
        static_cast<void>(pending.sink->TryError(scoped));
      } else {
        static_cast<void>(pending.sink->TryError(pending.shutdown_error));
      }
    }
  }
  pending_.clear();
}

void Controller::FailActive(Error error) {
  if (!active_) return;
  if (TerminateActive(error)) {
    DrainPending(&error);
    return;
  }
  static_cast<void>(StartNextRequest());
}

void Controller::ShutdownNoexcept() noexcept {
  if (shutdown_) return;
  shutdown_ = true;
  bool fatal = false;
  if (active_) {
    fatal =
        active_->transaction && !active_->transaction->may_abort();
    if (fatal) {
      if (active_->transaction) {
        try {
          static_cast<void>(
              active_->transaction->MarkFatal(active_->shutdown_error));
        } catch (...) {
        }
      }
    } else if (active_->transaction) {
      try {
        static_cast<void>(active_->transaction->Abort(active_->shutdown_error));
      } catch (...) {
      }
      AbortPreparedRanks(*active_);
    }
    if (!active_->detached) {
      static_cast<void>(active_->sink->TryError(active_->shutdown_error));
    }
    PreserveActiveTransaction();
    active_.reset();
  }
  DrainPending(fatal ? &destructor_fatal_error_ : nullptr);
  if (fatal) {
    try {
      replica_gate_->MarkFatal(std::move(destructor_fatal_error_));
    } catch (...) {
    }
  }
}

bool Controller::ContainsRequest(RequestId request) const noexcept {
  if (active_ && active_->request.request_id == request) return true;
  return std::any_of(pending_.begin(), pending_.end(), [request](const auto& state) {
    return state.request.request_id == request;
  });
}

Error Controller::RequestError(ErrorCode code, const char* message,
                               RequestId request) const {
  return Error::From(code, message, request);
}

}  // namespace grserve
