#include "hal_mock/mock_scheduler.h"

#include <new>
#include <type_traits>
#include <utility>

namespace grserve::hal_mock {
namespace {

static_assert(std::is_nothrow_move_constructible_v<Completion>);

#if defined(GRSERVE_MOCK_SCHEDULER_TESTING)
thread_local bool g_fail_next_ready_queue_append = false;
#endif

}  // namespace

Completion Completion::Success(OperationId operation_id, RankId rank_id,
                               EventOrdinal ordinal, ScriptedResult result) {
  return Completion{operation_id, rank_id, ordinal, std::move(result)};
}

Completion Completion::Failure(OperationId operation_id, RankId rank_id,
                               EventOrdinal ordinal, Error error) {
  return Completion{operation_id, rank_id, ordinal, std::move(error)};
}

Clock::time_point MockScheduler::EligibleAt(Clock::time_point now,
                                            Clock::duration duration) noexcept {
  if (duration <= Clock::duration::zero()) {
    return now;
  }
  if (duration > Clock::time_point::max() - now) {
    return Clock::time_point::max();
  }
  return now + duration;
}

Status MockScheduler::TransitionError(const char* message) const {
  return Status{Error::From(ErrorCode::kInvalidArgument, message)};
}

void MockScheduler::AppendReady(Completion completion) {
#if defined(GRSERVE_MOCK_SCHEDULER_TESTING)
  if (g_fail_next_ready_queue_append) {
    g_fail_next_ready_queue_append = false;
    throw std::bad_alloc{};
  }
#endif
  ready_.push_back(std::move(completion));
}

OperationId MockScheduler::Submit(RankId rank, ScriptedResult result) {
  const Clock::time_point now = clock_.Now();
  std::lock_guard<std::mutex> lock(mutex_);
  const OperationId id{next_operation_id_++};
  operations_.emplace(id, Operation{rank, result,
                                    EligibleAt(now, result.virtual_duration())});
  return id;
}

Status MockScheduler::Complete(OperationId id) {
  const Clock::time_point now = clock_.Now();
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = operations_.find(id);
  if (found == operations_.end()) {
    return TransitionError("unknown operation");
  }
  Operation& operation = found->second;
  if (operation.state != OperationState::kPending) {
    return TransitionError("operation is not pending");
  }
  if (now < operation.eligible_at) {
    return Status{Error::From(ErrorCode::kDeadlineExceeded,
                              "operation is not eligible")};
  }
  const EventOrdinal ordinal{next_ordinal_};
  Completion completion =
      Completion::Success(id, operation.rank, ordinal, operation.result);
  AppendReady(std::move(completion));
  operation.state = OperationState::kCompleted;
  ++next_ordinal_;
  return Status::Ok();
}

Status MockScheduler::Fail(OperationId id, Error error) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = operations_.find(id);
  if (found == operations_.end()) {
    return TransitionError("unknown operation");
  }
  Operation& operation = found->second;
  if (operation.state != OperationState::kPending) {
    return TransitionError("operation is not pending");
  }
  const EventOrdinal ordinal{next_ordinal_};
  Completion completion =
      Completion::Failure(id, operation.rank, ordinal, std::move(error));
  AppendReady(std::move(completion));
  operation.state = OperationState::kFailed;
  ++next_ordinal_;
  return Status::Ok();
}

std::vector<Completion> MockScheduler::DrainReady() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Completion> drained;
  drained.swap(ready_);
  return drained;
}

bool MockScheduler::IsPending(OperationId id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = operations_.find(id);
  return found != operations_.end() &&
         found->second.state == OperationState::kPending;
}

}  // namespace grserve::hal_mock

#if defined(GRSERVE_MOCK_SCHEDULER_TESTING)
namespace grserve::hal_mock::testing {

void FailNextReadyQueueAppend() noexcept {
  g_fail_next_ready_queue_append = true;
}

}  // namespace grserve::hal_mock::testing
#endif
