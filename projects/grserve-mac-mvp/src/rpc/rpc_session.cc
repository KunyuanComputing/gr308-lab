#include "rpc/rpc_session.h"

#include <sys/socket.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

#include "grserve/version.h"
#include "rpc/frame_codec.h"

namespace grserve::rpc {
namespace {

v1::ErrorCode ToProto(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kInvalidArgument: return v1::ERROR_CODE_INVALID_ARGUMENT;
    case ErrorCode::kIncompatibleVersion: return v1::ERROR_CODE_INCOMPATIBLE_VERSION;
    case ErrorCode::kUnsupportedCapability: return v1::ERROR_CODE_UNSUPPORTED_CAPABILITY;
    case ErrorCode::kResourceExhausted: return v1::ERROR_CODE_RESOURCE_EXHAUSTED;
    case ErrorCode::kDeadlineExceeded: return v1::ERROR_CODE_DEADLINE_EXCEEDED;
    case ErrorCode::kRankFailure: return v1::ERROR_CODE_RANK_FAILURE;
    case ErrorCode::kReplicaFatal: return v1::ERROR_CODE_REPLICA_FATAL;
    case ErrorCode::kInternal: return v1::ERROR_CODE_INTERNAL;
  }
  return v1::ERROR_CODE_INTERNAL;
}

v1::CancelDisposition ToProto(CancelDisposition disposition) noexcept {
  switch (disposition) {
    case CancelDisposition::kAbortedBeforeDecision:
      return v1::CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION;
    case CancelDisposition::kCommittingDetached:
      return v1::CANCEL_DISPOSITION_COMMITTING_DETACHED;
    case CancelDisposition::kAlreadyTerminal:
      return v1::CANCEL_DISPOSITION_ALREADY_TERMINAL;
    case CancelDisposition::kNotFound:
      return v1::CANCEL_DISPOSITION_NOT_FOUND;
  }
  return v1::CANCEL_DISPOSITION_UNSPECIFIED;
}

v1::ServerEnvelope BaseEnvelope() {
  v1::ServerEnvelope envelope;
  envelope.mutable_protocol_version()->set_major(kRpcVersion.major);
  envelope.mutable_protocol_version()->set_minor(kRpcVersion.minor);
  return envelope;
}

Error Invalid(std::string message, std::optional<RequestId> request = std::nullopt) {
  return Error::From(ErrorCode::kInvalidArgument, std::move(message), request);
}

}  // namespace

bool ControllerMailbox::Post(Event event) noexcept {
  try {
    std::lock_guard lock(mutex_);
    if (closed_) return false;
    if (next_event_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      if (submit_size_ != 0 || detach_size_ != 0 || cancel_size_ != 0) {
        return false;
      }
      next_event_sequence_ = 1;
    }
    const std::uint64_t sequence = next_event_sequence_++;
    if (auto* submit = std::get_if<Submit>(&event)) {
      if (!accepting_ || submit_size_ == submits_.size()) return false;
      const std::size_t tail =
          (submit_head_ + submit_size_) % submits_.size();
      submits_[tail].emplace(std::move(*submit));
      submit_sequences_[tail] = sequence;
      ++submit_size_;
    } else if (auto* detach = std::get_if<Detach>(&event)) {
      if (detach_size_ == detaches_.size()) return false;
      const std::size_t tail =
          (detach_head_ + detach_size_) % detaches_.size();
      detaches_[tail].emplace(std::move(*detach));
      detach_sequences_[tail] = sequence;
      ++detach_size_;
    } else {
      if (cancel_size_ == cancels_.size()) return false;
      const std::size_t tail =
          (cancel_head_ + cancel_size_) % cancels_.size();
      cancels_[tail].emplace(std::move(std::get<Cancel>(event)));
      cancel_sequences_[tail] = sequence;
      ++cancel_size_;
    }
    cv_.notify_one();
    return true;
  } catch (...) {
    return false;
  }
}

bool ControllerMailbox::accepting() const noexcept {
  try {
    std::lock_guard lock(mutex_);
    return accepting_ && !closed_;
  } catch (...) { return false; }
}

bool ControllerMailbox::empty() const noexcept {
  try {
    std::lock_guard lock(mutex_);
    return submit_size_ == 0 && detach_size_ == 0 && cancel_size_ == 0;
  } catch (...) { return false; }
}

void ControllerMailbox::StopAdmission() noexcept {
  try {
    std::lock_guard lock(mutex_);
    accepting_ = false;
  } catch (...) {}
}

bool ControllerMailbox::PollOne(Controller& controller) {
  std::optional<Event> event;
  {
    std::lock_guard lock(mutex_);
    const std::uint64_t submit_sequence = submit_size_ == 0
        ? std::numeric_limits<std::uint64_t>::max()
        : submit_sequences_[submit_head_];
    const std::uint64_t detach_sequence = detach_size_ == 0
        ? std::numeric_limits<std::uint64_t>::max()
        : detach_sequences_[detach_head_];
    const std::uint64_t cancel_sequence = cancel_size_ == 0
        ? std::numeric_limits<std::uint64_t>::max()
        : cancel_sequences_[cancel_head_];
    if (submit_sequence == std::numeric_limits<std::uint64_t>::max() &&
        detach_sequence == std::numeric_limits<std::uint64_t>::max() &&
        cancel_sequence == std::numeric_limits<std::uint64_t>::max()) {
      return false;
    }
    if (detach_sequence < submit_sequence &&
        detach_sequence < cancel_sequence) {
      auto& slot = detaches_[detach_head_];
      event.emplace(std::move(*slot));
      slot.reset();
      detach_head_ = (detach_head_ + 1) % detaches_.size();
      --detach_size_;
    } else if (cancel_sequence < submit_sequence) {
      auto& slot = cancels_[cancel_head_];
      event.emplace(std::move(*slot));
      slot.reset();
      cancel_head_ = (cancel_head_ + 1) % cancels_.size();
      --cancel_size_;
    } else {
      auto& slot = submits_[submit_head_];
      event.emplace(std::move(*slot));
      slot.reset();
      submit_head_ = (submit_head_ + 1) % submits_.size();
      --submit_size_;
    }
  }
  std::visit([&](auto& value) {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, Submit>) {
      if (next_request_id_ == 0 ||
          next_request_id_ == std::numeric_limits<std::uint64_t>::max()) {
        static_cast<void>(value.sink->TryError(Error::From(
            ErrorCode::kResourceExhausted, "request id space is exhausted")));
        return;
      }
      const RequestId request{next_request_id_++};
      value.request.request_id = request;
      if (!value.sink->BindInternal(request)) return;
      const Status status = controller.Submit(std::move(value.request), value.sink);
      if (!status.ok()) static_cast<void>(value.sink->TryError(*status.error()));
    } else if constexpr (std::is_same_v<T, Detach>) {
      const auto request = value.request->internal_request();
      if (request) static_cast<void>(controller.DetachTransport(*request));
    } else {
      const auto request = value.request->internal_request();
      if (!request) {
        static_cast<void>(value.response->TryError(
            Error::From(ErrorCode::kInvalidArgument, "request is not active")));
        return;
      }
      auto disposition = controller.CancelTransport(*request, value.reason);
      if (disposition.ok()) {
        static_cast<void>(value.response->TryComplete(*request,
                                                       disposition.value()));
      } else {
        static_cast<void>(value.response->TryError(*disposition.error()));
      }
    }
  }, *event);
  return true;
}

void ControllerMailbox::Close() noexcept {
  try {
    std::lock_guard lock(mutex_);
    closed_ = true;
    for (auto& event : submits_) event.reset();
    for (auto& event : detaches_) event.reset();
    for (auto& event : cancels_) event.reset();
    submit_head_ = submit_size_ = detach_head_ = detach_size_ = 0;
    cancel_head_ = cancel_size_ = 0;
    next_event_sequence_ = 1;
    cv_.notify_all();
  } catch (...) {}
}

#if defined(GRSERVE_RPC_TESTING)
void ControllerMailbox::WaitForEventForTesting() {
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [this] {
    return closed_ || submit_size_ != 0 || detach_size_ != 0 ||
           cancel_size_ != 0;
  });
}
void ControllerMailbox::WaitForEventCountForTesting(std::size_t count) {
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [this, count] {
    return closed_ || submit_size_ + detach_size_ + cancel_size_ >= count;
  });
}
std::size_t ControllerMailbox::size_for_testing() const noexcept {
  try {
    std::lock_guard lock(mutex_);
    return submit_size_ + detach_size_ + cancel_size_;
  } catch (...) { return capacity_for_testing(); }
}
void RpcSessionTestBarrier::BlockWriter() {
  std::unique_lock lock(mutex_);
  blocked_ = true;
  cv_.notify_all();
  cv_.wait(lock, [this] { return released_; });
}
void RpcSessionTestBarrier::WaitUntilWriterBlocked() {
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [this] { return blocked_; });
}
void RpcSessionTestBarrier::ReleaseWriter() {
  std::lock_guard lock(mutex_);
  released_ = true;
  cv_.notify_all();
}
#endif

class RpcSession::Sink final : public OwnerBoundRequestSink {
 public:
  Sink(std::weak_ptr<RpcSession> session, std::string wire_request)
      : session_(std::move(session)), wire_request_(std::move(wire_request)) {}
  bool BindInternal(RequestId request) noexcept override {
    auto session = session_.lock();
    if (!session) return false;
    try {
      std::lock_guard lock(session->mutex_);
      auto it = session->requests_.find(wire_request_);
      if (it == session->requests_.end() || it->second.detached ||
          internal_request_) return false;
      internal_request_ = request;
      return true;
    } catch (...) { return false; }
  }
  std::optional<RequestId> internal_request() const noexcept override {
    auto session = session_.lock();
    if (!session) return std::nullopt;
    try {
      std::lock_guard lock(session->mutex_);
      return internal_request_;
    } catch (...) { return std::nullopt; }
  }
  bool TryAccepted(RequestId request) noexcept override {
    auto session = session_.lock();
    if (!session || !internal_request_ || request != *internal_request_) return false;
    try {
      auto envelope = BaseEnvelope();
      envelope.mutable_accepted()->set_request_id(wire_request_);
      std::lock_guard lock(session->mutex_);
      auto it = session->requests_.find(wire_request_);
      if (it == session->requests_.end() || it->second.accepted ||
          it->second.detached) return false;
      if (!session->EnqueueLocked(std::move(envelope), false)) {
        session->requests_.erase(it);
        return false;
      }
      it->second.accepted = true;
      return true;
    } catch (...) {
      try {
        std::lock_guard lock(session->mutex_);
        session->requests_.erase(wire_request_);
      } catch (...) {}
      session->FailTransport();
      return false;
    }
  }
  bool TryToken(const TokenEvent& token) noexcept override {
    auto session = session_.lock();
    if (!session || !internal_request_ || token.request_id != *internal_request_) return false;
    bool detach = false;
    bool fail_transport = false;
    try {
      auto envelope = BaseEnvelope();
      auto* output = envelope.mutable_token();
      output->set_request_id(wire_request_);
      output->set_transaction_id(std::to_string(token.transaction_id.value()));
      output->set_sequence_number(token.sequence_number.value());
      output->set_token_id(token.token_id.value());
      std::lock_guard lock(session->mutex_);
      auto it = session->requests_.find(wire_request_);
      if (it == session->requests_.end() || !it->second.accepted ||
          it->second.detached ||
          token.sequence_number.value() != it->second.next_sequence) {
        if (it != session->requests_.end() && !it->second.detached) {
          detach = true;
          fail_transport = true;
        }
      } else {
        if (session->EnqueueLocked(std::move(envelope), false)) {
          ++it->second.next_sequence;
          return true;
        }
        detach = true;
        fail_transport = true;
      }
    } catch (...) {
      std::shared_ptr<OwnerBoundRequestSink> sink;
      {
        std::lock_guard lock(session->mutex_);
        const auto it = session->requests_.find(wire_request_);
        if (it != session->requests_.end()) sink = it->second.sink;
      }
      static_cast<void>(session->Detach(sink));
      session->FailTransport();
      return false;
    }
    if (detach) {
      std::shared_ptr<OwnerBoundRequestSink> sink;
      {
        std::lock_guard lock(session->mutex_);
        const auto it = session->requests_.find(wire_request_);
        if (it != session->requests_.end()) sink = it->second.sink;
      }
      static_cast<void>(session->Detach(sink));
    }
    if (fail_transport) session->FailTransport();
    return false;
  }
  bool TryFinished(const FinishedEvent& finished) noexcept override {
    auto session = session_.lock();
    if (!session || !internal_request_ || finished.request_id != *internal_request_) return false;
    bool detach = false;
    bool fail_transport = false;
    try {
      auto envelope = BaseEnvelope();
      auto* output = envelope.mutable_finished();
      output->set_request_id(wire_request_);
      output->set_finish_reason(finished.finish_reason == FinishReason::kLength
                                    ? v1::FINISH_REASON_LENGTH
                                    : v1::FINISH_REASON_CANCELLED);
      output->mutable_usage()->set_prompt_tokens(finished.prompt_tokens);
      output->mutable_usage()->set_completion_tokens(finished.completion_tokens);
      output->mutable_usage()->set_total_tokens(finished.prompt_tokens +
                                                finished.completion_tokens);
      output->set_last_sequence_number(finished.last_sequence_number.value());
      std::lock_guard lock(session->mutex_);
      auto it = session->requests_.find(wire_request_);
      if (it == session->requests_.end() || !it->second.accepted ||
          it->second.detached || finished.last_sequence_number.value() + 1 !=
                                     it->second.next_sequence) {
        fail_transport = it != session->requests_.end() && !it->second.detached;
      } else {
        const bool queued = session->EnqueueLocked(std::move(envelope), true);
        if (!queued) detach = true;
        if (queued) session->requests_.erase(it);
        if (queued) return true;
      }
    } catch (...) {
      std::shared_ptr<OwnerBoundRequestSink> sink;
      {
        std::lock_guard lock(session->mutex_);
        const auto it = session->requests_.find(wire_request_);
        if (it != session->requests_.end()) sink = it->second.sink;
      }
      static_cast<void>(session->Detach(sink));
      session->FailTransport();
      return false;
    }
    if (detach || fail_transport) session->FailTransport();
    return false;
  }
  bool TryError(const Error& error) noexcept override {
    auto session = session_.lock();
    if (!session) return false;
    bool queued = false;
    try {
      auto envelope = BaseEnvelope();
      auto* output = envelope.mutable_error();
      output->set_code(ToProto(error.code()));
      output->set_message(error.message());
      output->set_request_id(wire_request_);
      if (error.transaction_id()) output->set_transaction_id(std::to_string(error.transaction_id()->value()));
      output->set_retryable(error.retryable());
      {
        std::lock_guard lock(session->mutex_);
        auto it = session->requests_.find(wire_request_);
        if (it != session->requests_.end() && it->second.detached) return false;
        queued = session->EnqueueLocked(std::move(envelope), true);
        if (queued && it != session->requests_.end()) session->requests_.erase(it);
      }
      if (!queued) session->FailTransport();
      return queued;
    } catch (...) {
      session->FailTransport();
      return false;
    }
  }
 private:
  std::weak_ptr<RpcSession> session_;
  std::string wire_request_;
  std::optional<RequestId> internal_request_;
};

class RpcSession::CancelSink final : public CancelResponseSink {
 public:
  CancelSink(std::weak_ptr<RpcSession> session, std::string wire_request)
      : session_(std::move(session)), wire_request_(std::move(wire_request)) {}

  bool TryComplete(RequestId request,
                   CancelDisposition disposition) noexcept override {
    auto session = session_.lock();
    if (!session) return false;
    bool queued = false;
    std::size_t duplicate_count = 0;
    try {
      auto envelope = BaseEnvelope();
      auto* ack = envelope.mutable_cancel_ack();
      ack->set_request_id(wire_request_);
      ack->set_disposition(ToProto(disposition));
      {
        std::lock_guard lock(session->mutex_);
        auto it = session->requests_.find(wire_request_);
        if (it == session->requests_.end() || !it->second.cancel_pending) {
          return false;
        }
        duplicate_count = it->second.pending_duplicate_cancels;
        queued = session->EnqueueLocked(std::move(envelope), true);
        if (queued) session->requests_.erase(it);
      }
      if (!queued) session->FailTransport();
      for (std::size_t index = 0; queued && index < duplicate_count; ++index) {
        session->EnqueueError(Invalid("request is not active", request),
                              wire_request_);
        queued = !session->stopped();
      }
      return queued;
    } catch (...) {
      session->FailTransport();
      return false;
    }
  }

  bool TryError(const Error& error) noexcept override {
    auto session = session_.lock();
    if (!session) return false;
    try {
      {
        std::lock_guard lock(session->mutex_);
        session->requests_.erase(wire_request_);
      }
      session->EnqueueError(error, wire_request_);
      return !session->stopped();
    } catch (...) {
      session->FailTransport();
      return false;
    }
  }

 private:
  std::weak_ptr<RpcSession> session_;
  std::string wire_request_;
};

StatusOr<std::shared_ptr<RpcSession>> RpcSession::Start(
    UniqueFd socket, std::shared_ptr<ControllerMailbox> mailbox,
    RpcSessionConfig config) {
  if (!socket || !mailbox || config.model_id.empty() ||
      config.outbound_capacity == 0 ||
      config.outbound_capacity == std::numeric_limits<std::size_t>::max()) {
    return StatusOr<std::shared_ptr<RpcSession>>(Invalid("invalid RPC session configuration"));
  }
  try {
    auto session = std::shared_ptr<RpcSession>(
        new RpcSession(std::move(socket), std::move(mailbox), std::move(config)));
    session->reader_ = std::thread(&RpcSession::ReaderLoop, session.get());
    session->writer_ = std::thread(&RpcSession::WriterLoop, session.get());
    return StatusOr<std::shared_ptr<RpcSession>>(std::move(session));
  } catch (...) {
    return StatusOr<std::shared_ptr<RpcSession>>(
        Error::From(ErrorCode::kInternal, "could not start RPC session"));
  }
}

#if defined(GRSERVE_RPC_TESTING)
StatusOr<std::shared_ptr<RpcSession>> RpcSession::StartForTesting(
    UniqueFd socket, std::shared_ptr<ControllerMailbox> mailbox,
    RpcSessionConfig config, RpcSessionTestBarrier* writer_barrier) {
  if (!socket || !mailbox || config.model_id.empty() ||
      config.outbound_capacity == 0 || writer_barrier == nullptr) {
    return StatusOr<std::shared_ptr<RpcSession>>(
        Invalid("invalid RPC session configuration"));
  }
  try {
    auto session = std::shared_ptr<RpcSession>(new RpcSession(
        std::move(socket), std::move(mailbox), std::move(config), writer_barrier));
    session->reader_ = std::thread(&RpcSession::ReaderLoop, session.get());
    session->writer_ = std::thread(&RpcSession::WriterLoop, session.get());
    return StatusOr<std::shared_ptr<RpcSession>>(std::move(session));
  } catch (...) {
    return StatusOr<std::shared_ptr<RpcSession>>(
        Error::From(ErrorCode::kInternal, "could not start RPC session"));
  }
}
#endif

RpcSession::RpcSession(UniqueFd socket, std::shared_ptr<ControllerMailbox> mailbox,
                       RpcSessionConfig config
#if defined(GRSERVE_RPC_TESTING)
                       , RpcSessionTestBarrier* writer_barrier
#endif
                       )
    : socket_(std::move(socket)), mailbox_(std::move(mailbox)),
      config_(std::move(config))
#if defined(GRSERVE_RPC_TESTING)
      , writer_barrier_(writer_barrier)
#endif
      {}
RpcSession::~RpcSession() { Shutdown(); }
bool RpcSession::stopped() const noexcept { return stopping_.load(); }

void RpcSession::Shutdown() noexcept {
  const bool first = !stopping_.exchange(true);
  if (first && socket_) shutdown(socket_.get(), SHUT_RDWR);
  outbound_cv_.notify_all();
  if (reader_.joinable() && reader_.get_id() != std::this_thread::get_id()) reader_.join();
#if defined(GRSERVE_RPC_TESTING)
  if (writer_barrier_) writer_barrier_->ReleaseWriter();
#endif
  if (writer_.joinable() && writer_.get_id() != std::this_thread::get_id()) writer_.join();
  DetachAll();
  socket_.reset();
}

void RpcSession::ReaderLoop() noexcept {
  try {
    FrameReader reader;
    while (!stopping_) {
      auto frame = reader.ReadOne(socket_.get());
      if (!frame.ok() || !frame.value().has_value()) break;
      Handle(*frame.value());
    }
  } catch (...) {}
  FailTransport();
}

void RpcSession::WriterLoop() noexcept {
  try {
    FrameWriter writer;
    while (true) {
      v1::ServerEnvelope envelope;
      {
        std::unique_lock lock(mutex_);
        outbound_cv_.wait(lock, [this] {
          return stopping_.load(std::memory_order_acquire) || !outbound_.empty();
        });
        if (outbound_.empty()) {
          if (stopping_) return;
          continue;
        }
        envelope = std::move(outbound_.front());
        outbound_.pop_front();
      }
#if defined(GRSERVE_RPC_TESTING)
      if (writer_barrier_) writer_barrier_->BlockWriter();
#endif
      if (!writer.WriteOne(socket_.get(), envelope).ok()) break;
    }
  } catch (...) {}
  FailTransport();
}

void RpcSession::Handle(const v1::ClientEnvelope& envelope) noexcept {
  try {
    const auto& version = envelope.protocol_version();
    if (version.major() != kRpcVersion.major || version.minor() > kRpcVersion.minor) {
      EnqueueError(Error::From(ErrorCode::kIncompatibleVersion,
                               "RPC protocol version is incompatible"));
      return;
    }
    if (envelope.has_health()) {
      auto response = BaseEnvelope();
      auto* health = response.mutable_health();
      health->set_engine_build_id(config_.engine_build_id);
      const bool healthy = !config_.replica_gate ||
          config_.replica_gate->health() == ReplicaHealth::kHealthy;
      health->set_accepting_requests(mailbox_->accepting() && healthy);
      health->set_replica_state(healthy ? v1::REPLICA_STATE_HEALTHY
                                        : v1::REPLICA_STATE_FATAL);
      if (!healthy) {
        const auto fatal = config_.replica_gate->fatal_error();
        if (fatal) {
          auto* output = health->mutable_fatal_error();
          output->set_code(ToProto(fatal->code()));
          output->set_message(fatal->message());
          // Controller request IDs are process-internal. A health probe has no
          // unambiguous session-local wire ID to echo, so omit this field.
          if (fatal->transaction_id()) {
            output->set_transaction_id(
                std::to_string(fatal->transaction_id()->value()));
          }
          output->set_retryable(fatal->retryable());
        }
      }
      if (!Enqueue(std::move(response))) FailTransport();
      return;
    }
    if (envelope.has_list_models()) {
      auto response = BaseEnvelope();
      auto* model = response.mutable_list_models()->add_models();
      model->set_id(config_.model_id);
      model->set_manifest_sha256(config_.manifest_sha256);
      model->set_max_context_tokens(config_.max_context_tokens);
      if (!Enqueue(std::move(response))) FailTransport();
      return;
    }
    if (envelope.has_completion()) {
      const auto& input = envelope.completion();
      const std::string wire_request = input.request_id();
      if (!ValidWireRequestId(wire_request)) {
        EnqueueError(Invalid("request id must contain 1 to 128 bytes"), wire_request);
        return;
      }
      bool token_out_of_range = false;
      if (config_.vocab_size != 0) {
        for (auto token : input.prompt_token_ids()) {
          if (token >= config_.vocab_size) { token_out_of_range = true; break; }
        }
      }
      const std::uint64_t total_context =
          static_cast<std::uint64_t>(input.prompt_token_ids_size()) + input.max_tokens();
      if (input.model_id() != config_.model_id || input.prompt_token_ids().empty() ||
          input.max_tokens() == 0 || input.timeout_ms() == 0 ||
          input.decode_mode() != v1::DECODE_MODE_GREEDY || token_out_of_range ||
          (config_.max_context_tokens != 0 &&
           total_context > config_.max_context_tokens)) {
        EnqueueError(Invalid("invalid completion request fields"), wire_request);
        return;
      }
      bool duplicate = false;
      {
        std::lock_guard lock(mutex_);
        duplicate = requests_.contains(wire_request);
        if (!duplicate) requests_.emplace(wire_request, RequestTransportState{});
      }
      if (duplicate) {
        EnqueueError(Invalid("request is already active"), wire_request);
        return;
      }
      GenerateRequest request{RequestId{0}, input.model_id(), {}, input.max_tokens(),
                              input.timeout_ms(), 1};
      request.prompt_token_ids.reserve(input.prompt_token_ids_size());
      for (auto token : input.prompt_token_ids()) request.prompt_token_ids.emplace_back(token);
      auto sink = MakeSink(wire_request);
      {
        std::lock_guard lock(mutex_);
        requests_.at(wire_request).sink = sink;
      }
      if (!mailbox_->Post(ControllerMailbox::Submit{std::move(request), sink})) {
        {
          std::lock_guard lock(mutex_);
          requests_.erase(wire_request);
        }
        EnqueueError(Error::From(ErrorCode::kResourceExhausted,
                                 "controller event queue is unavailable"),
                     wire_request);
      }
      return;
    }
    if (envelope.has_cancel()) {
      const std::string wire_request = envelope.cancel().request_id();
      if (!ValidWireRequestId(wire_request)) {
        EnqueueError(Invalid("request id must contain 1 to 128 bytes"), wire_request);
        return;
      }
      CancelReason reason = CancelReason::kClientDisconnected;
      switch (envelope.cancel().reason()) {
        case v1::CANCEL_REASON_CLIENT_DISCONNECTED:
          reason = CancelReason::kClientDisconnected;
          break;
        case v1::CANCEL_REASON_DEADLINE:
          reason = CancelReason::kDeadline;
          break;
        case v1::CANCEL_REASON_SERVER_SHUTDOWN:
          reason = CancelReason::kServerShutdown;
          break;
        default:
          EnqueueError(Invalid("invalid cancel reason"), wire_request);
          return;
      }
      auto response = MakeCancelSink(wire_request);
      std::shared_ptr<OwnerBoundRequestSink> request_sink;
      bool active = false;
      bool duplicate_pending = false;
      bool duplicate_overflow = false;
      {
        std::lock_guard lock(mutex_);
        auto it = requests_.find(wire_request);
        if (it != requests_.end() && !it->second.detached) {
          request_sink = it->second.sink;
          if (it->second.cancel_pending) {
            duplicate_pending = true;
            if (it->second.pending_duplicate_cancels <
                config_.outbound_capacity) {
              ++it->second.pending_duplicate_cancels;
            } else {
              duplicate_overflow = true;
            }
          } else {
            active = true;
            it->second.cancel_pending = true;
          }
        }
      }
      if (duplicate_overflow) { FailTransport(); return; }
      if (duplicate_pending) return;
      if (!active) {
        EnqueueError(Invalid("request is not active"), wire_request);
        return;
      }
      if (!mailbox_->Post(
              ControllerMailbox::Cancel{std::move(request_sink), reason,
                                        std::move(response)})) {
        {
          std::lock_guard lock(mutex_);
          auto it = requests_.find(wire_request);
          if (it != requests_.end()) it->second.cancel_pending = false;
        }
        EnqueueError(Error::From(ErrorCode::kResourceExhausted,
                                 "controller event queue is unavailable"),
                     wire_request);
        return;
      }
      return;
    }
    EnqueueError(Invalid("RPC payload is missing"));
  } catch (...) {
    try {
      EnqueueError(Error::From(ErrorCode::kInternal,
                               "RPC request handling failed"));
    } catch (...) { FailTransport(); }
  }
}

std::shared_ptr<OwnerBoundRequestSink> RpcSession::MakeSink(
    std::string wire_request) {
  return std::make_shared<Sink>(weak_from_this(), std::move(wire_request));
}

std::shared_ptr<CancelResponseSink> RpcSession::MakeCancelSink(
    std::string wire_request) {
  return std::make_shared<CancelSink>(weak_from_this(), std::move(wire_request));
}

bool RpcSession::Enqueue(v1::ServerEnvelope envelope, bool terminal) noexcept {
  try {
    std::lock_guard lock(mutex_);
    return EnqueueLocked(std::move(envelope), terminal);
  } catch (...) {
    return false;
  }
}

bool RpcSession::EnqueueLocked(v1::ServerEnvelope envelope, bool terminal) {
  const std::size_t limit = config_.outbound_capacity + (terminal ? 1U : 0U);
  if (stopping_ || outbound_.size() >= limit) return false;
  outbound_.push_back(std::move(envelope));
  outbound_cv_.notify_one();
  return true;
}

void RpcSession::EnqueueError(Error error, std::string wire_request) noexcept {
  try {
    auto envelope = BaseEnvelope();
    auto* output = envelope.mutable_error();
    output->set_code(ToProto(error.code()));
    output->set_message(error.message());
    if (!wire_request.empty()) output->set_request_id(std::move(wire_request));
    else if (error.request_id()) output->set_request_id(std::to_string(error.request_id()->value()));
    if (error.transaction_id()) output->set_transaction_id(std::to_string(error.transaction_id()->value()));
    output->set_retryable(error.retryable());
    if (!Enqueue(std::move(envelope), true)) FailTransport();
  } catch (...) { FailTransport(); }
}

void RpcSession::FailTransport() noexcept {
  stopping_.store(true, std::memory_order_release);
  outbound_cv_.notify_all();
  DetachAll();
  if (socket_) shutdown(socket_.get(), SHUT_RDWR);
}

bool RpcSession::Detach(
    const std::shared_ptr<OwnerBoundRequestSink>& request) noexcept {
  if (!request) return false;
  try {
    std::lock_guard lock(mutex_);
    auto it = std::find_if(requests_.begin(), requests_.end(),
                           [&](const auto& entry) {
                             return entry.second.sink == request;
                           });
    if (it != requests_.end() && !it->second.detached) {
      if (!mailbox_->Post(ControllerMailbox::Detach{request})) return false;
      it->second.detached = true;
      return true;
    }
    return it != requests_.end();
  } catch (...) { return false; }
}

void RpcSession::DetachAll() noexcept {
  try {
    std::lock_guard lock(mutex_);
    for (auto& [wire_request, state] : requests_) {
      if (!state.detached && !state.cancel_pending && mailbox_->Post(
              ControllerMailbox::Detach{state.sink})) {
        state.detached = true;
      }
    }
  } catch (...) { return; }
}

bool RpcSession::ValidWireRequestId(const std::string& text) noexcept {
  return !text.empty() && text.size() <= 128;
}

#if defined(GRSERVE_RPC_TESTING)
std::shared_ptr<RequestEventSink> RpcSession::SinkForTesting(RequestId request) {
  const std::string wire_request = std::to_string(request.value());
  auto sink = MakeSink(wire_request);
  {
    std::lock_guard lock(mutex_);
    requests_.try_emplace(wire_request, RequestTransportState{});
    requests_.at(wire_request).sink = sink;
  }
  static_cast<void>(sink->BindInternal(request));
  return sink;
}
#endif

}  // namespace grserve::rpc
