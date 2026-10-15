#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <mutex>
#include <variant>
#include <vector>

#include "common/clock.h"
#include "grserve/error.h"
#include "hal_mock/scripted_result.h"

namespace grserve::hal_mock {

using OperationId = StrongId<struct OperationIdTag, std::uint64_t>;
using EventOrdinal = StrongId<struct EventOrdinalTag, std::uint64_t>;

class Completion {
 public:
  [[nodiscard]] static Completion Success(OperationId operation_id, RankId rank_id,
                                           EventOrdinal ordinal,
                                           ScriptedResult result);
  [[nodiscard]] static Completion Failure(OperationId operation_id, RankId rank_id,
                                           EventOrdinal ordinal, Error error);

  [[nodiscard]] OperationId operation_id() const noexcept { return operation_id_; }
  [[nodiscard]] RankId rank_id() const noexcept { return rank_id_; }
  [[nodiscard]] EventOrdinal ordinal() const noexcept { return ordinal_; }
  [[nodiscard]] bool success() const noexcept {
    return std::holds_alternative<ScriptedResult>(outcome_);
  }
  [[nodiscard]] const ScriptedResult& result() const noexcept {
    assert(success());
    return std::get<ScriptedResult>(outcome_);
  }
  [[nodiscard]] const Error& error() const noexcept {
    assert(!success());
    return std::get<Error>(outcome_);
  }

 private:
  Completion(OperationId operation_id, RankId rank_id, EventOrdinal ordinal,
             std::variant<ScriptedResult, Error> outcome) noexcept
      : operation_id_(operation_id),
        rank_id_(rank_id),
        ordinal_(ordinal),
        outcome_(std::move(outcome)) {}

  OperationId operation_id_;
  RankId rank_id_;
  EventOrdinal ordinal_;
  std::variant<ScriptedResult, Error> outcome_;
};

class MockScheduler {
 public:
  explicit MockScheduler(Clock& clock) noexcept : clock_(clock) {}

  [[nodiscard]] OperationId Submit(RankId rank, ScriptedResult result);
  [[nodiscard]] Status Complete(OperationId id);
  [[nodiscard]] Status Fail(OperationId id, Error error);
  [[nodiscard]] std::vector<Completion> DrainReady();
  [[nodiscard]] bool IsPending(OperationId id) const;

 private:
  enum class OperationState { kPending, kCompleted, kFailed };
  struct Operation {
    RankId rank;
    ScriptedResult result;
    Clock::time_point eligible_at;
    OperationState state = OperationState::kPending;
  };

  [[nodiscard]] static Clock::time_point EligibleAt(Clock::time_point now,
                                                     Clock::duration duration) noexcept;
  [[nodiscard]] Status TransitionError(const char* message) const;
  void AppendReady(Completion completion);

  Clock& clock_;
  mutable std::mutex mutex_;
  std::map<OperationId, Operation> operations_;
  std::vector<Completion> ready_;
  std::uint64_t next_operation_id_ = 1U;
  std::uint64_t next_ordinal_ = 1U;
};

}  // namespace grserve::hal_mock

#if defined(GRSERVE_MOCK_SCHEDULER_TESTING)
namespace grserve::hal_mock::testing {

// Test-only scoped fault injection for proving transition strong safety.
void FailNextReadyQueueAppend() noexcept;

}  // namespace grserve::hal_mock::testing
#endif
