#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "engine/admission.h"
#include "engine/controller.h"
#include "engine/replica_gate.h"
#include "engine/rank_worker.h"

namespace {
using namespace grserve;

enum class FaultCase {
  kPrepareFail,
  kExecuteFail,
  kPreDecisionTimeout,
  kRankLoss,
  kCancelBeforeDecision,
  kCommitFailBeforeApply,
  kCommitApplyThenAckLoss,
  kPostDecisionTimeout,
  kDetachAfterDecision,
};

const char* Name(FaultCase value) {
  switch (value) {
    case FaultCase::kPrepareFail: return "PrepareFail";
    case FaultCase::kExecuteFail: return "ExecuteFail";
    case FaultCase::kPreDecisionTimeout: return "PreDecisionTimeout";
    case FaultCase::kRankLoss: return "RankLoss";
    case FaultCase::kCancelBeforeDecision: return "CancelBeforeDecision";
    case FaultCase::kCommitFailBeforeApply: return "CommitFailBeforeApply";
    case FaultCase::kCommitApplyThenAckLoss: return "CommitApplyThenAckLoss";
    case FaultCase::kPostDecisionTimeout: return "PostDecisionTimeout";
    case FaultCase::kDetachAfterDecision: return "DetachAfterDecision";
  }
}

class Sink final : public RequestEventSink {
 public:
  bool TryAccepted(RequestId) noexcept override { ++accepted; return true; }
  bool TryToken(const TokenEvent&) noexcept override { ++tokens; return true; }
  bool TryFinished(const FinishedEvent&) noexcept override { ++finished; return true; }
  bool TryError(const Error& value) noexcept override {
    ++errors;
    last_error = value.code();
    return true;
  }
  int accepted{0};
  int tokens{0};
  int finished{0};
  int errors{0};
  std::optional<ErrorCode> last_error;
};

struct ManualClock {
  Controller::TimePoint now{};
  std::size_t calls{0};
  Controller::TimePoint operator()() noexcept {
    ++calls;
    return now;
  }
  void Advance(std::chrono::milliseconds value) { now += value; }
};

struct ManualRank {
  RankId id;
  FaultCase scenario;
  std::vector<RankCommand> commands;
  std::uint64_t canonical{0};

  ControllerRank endpoint() {
    return {id, [this](RankCommand command) {
      commands.push_back(command);
      if (scenario == FaultCase::kPrepareFail && id == RankId{0} &&
          std::holds_alternative<Prepare>(command)) {
        return Status{Error::From(ErrorCode::kInternal, "prepare failed")};
      }
      if (scenario == FaultCase::kExecuteFail && id == RankId{0} &&
          std::holds_alternative<Execute>(command)) {
        return Status{Error::From(ErrorCode::kInternal, "execute failed")};
      }
      if (scenario == FaultCase::kCommitFailBeforeApply && id == RankId{0} &&
          std::holds_alternative<Commit>(command)) {
        return Status{Error::From(ErrorCode::kRankFailure, "commit unavailable")};
      }
      if ((scenario == FaultCase::kCommitApplyThenAckLoss ||
           scenario == FaultCase::kDetachAfterDecision) &&
          std::holds_alternative<Commit>(command)) {
        ++canonical;
      }
      return Status::Ok();
    }};
  }
};

class ControllerFaultMatrix : public testing::TestWithParam<FaultCase> {};

TEST_P(ControllerFaultMatrix, PreservesTheDecisionBoundaryInEveryFailureMode) {
  const FaultCase scenario = GetParam();
  ManualClock clock;
  ManualRank rank0{RankId{0}, scenario};
  ManualRank rank1{RankId{1}, scenario};
  auto admission = AdmissionController::Create(
      {{RankId{0}, 1}, {RankId{1}, 1}}).value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(
      admission, gate, completions, {rank0.endpoint(), rank1.endpoint()},
      [](const GenerateRequest&, SequenceNumber) { return TokenId{19}; },
      [&clock] { return clock(); });
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink = std::make_shared<Sink>();
  ASSERT_TRUE(controller->Submit(
      {RequestId{10}, "model", {TokenId{1}}, 2, 10, 1}, sink).ok());

  ASSERT_FALSE(rank0.commands.empty());
  const auto prepare = std::get<Prepare>(rank0.commands.front());
  const auto push = [&](RankCompletion completion) {
    ASSERT_TRUE(completions.Push(std::move(completion)));
    controller->PollOnce();
  };

  if (scenario == FaultCase::kPreDecisionTimeout) {
    clock.Advance(std::chrono::milliseconds{10});
    controller->PollOnce();
  } else if (scenario == FaultCase::kRankLoss) {
    push(RankError{RankId{0}, prepare.transaction_id, prepare.epoch,
                   Error::From(ErrorCode::kRankFailure, "rank lost"), true});
  } else if (scenario == FaultCase::kCancelBeforeDecision) {
    ASSERT_TRUE(controller->Cancel(RequestId{10}, CancelReason::kServerShutdown).ok());
  } else if (scenario != FaultCase::kPrepareFail) {
    push(PreparedAck{RankId{0}, prepare.epoch});
    push(PreparedAck{RankId{1}, prepare.epoch});
    if (scenario != FaultCase::kExecuteFail) {
      push(ExecutedAck{RankId{0}, prepare.epoch, TokenId{19}, StateDigest{101}});
      push(ExecutedAck{RankId{1}, prepare.epoch, TokenId{19}, StateDigest{102}});
      if (scenario == FaultCase::kCommitApplyThenAckLoss) {
        push(CommittedAck{RankId{0}, prepare.epoch});
        clock.Advance(std::chrono::milliseconds{10});
        controller->PollOnce();
      } else if (scenario == FaultCase::kPostDecisionTimeout) {
        clock.Advance(std::chrono::milliseconds{10});
        controller->PollOnce();
      } else if (scenario == FaultCase::kDetachAfterDecision) {
        ASSERT_TRUE(controller->DetachTransport(RequestId{10}).ok());
        push(CommittedAck{RankId{0}, prepare.epoch});
        push(CommittedAck{RankId{1}, prepare.epoch});
      }
    }
  }

  const bool published = scenario == FaultCase::kDetachAfterDecision;
  const bool fatal = scenario == FaultCase::kRankLoss ||
                     scenario == FaultCase::kCommitFailBeforeApply ||
                     scenario == FaultCase::kCommitApplyThenAckLoss ||
                     scenario == FaultCase::kPostDecisionTimeout;
  const bool canonical_advanced =
      scenario == FaultCase::kCommitApplyThenAckLoss || published;
  EXPECT_EQ(controller->last_terminal_state_for_testing(),
            published ? std::optional{TransactionState::kPublished}
                      : std::optional{fatal && scenario != FaultCase::kRankLoss
                                          ? TransactionState::kFatal
                                          : TransactionState::kAborted});
  EXPECT_EQ(rank0.canonical != 0, canonical_advanced);
  EXPECT_EQ(rank1.canonical != 0, canonical_advanced);
  EXPECT_EQ(controller->logical_publish_count(), published ? 1U : 0U);
  EXPECT_EQ(controller->network_delivery_count(), 0U);
  EXPECT_EQ(controller->last_delivery_outcome(),
            published ? DeliveryOutcome::kDroppedClientDisconnect
                      : DeliveryOutcome::kNone);
  EXPECT_EQ(sink->tokens, 0);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  EXPECT_EQ(gate->health(), fatal ? ReplicaHealth::kFatal
                                 : ReplicaHealth::kHealthy);
  if (scenario == FaultCase::kDetachAfterDecision) {
    EXPECT_EQ(sink->errors, 0);
  } else {
    ASSERT_EQ(sink->errors, 1);
    const ErrorCode expected =
        scenario == FaultCase::kPreDecisionTimeout ||
                scenario == FaultCase::kCommitApplyThenAckLoss ||
                scenario == FaultCase::kPostDecisionTimeout
            ? ErrorCode::kDeadlineExceeded
            : scenario == FaultCase::kRankLoss ||
                      scenario == FaultCase::kCommitFailBeforeApply
                  ? ErrorCode::kRankFailure
                  : ErrorCode::kInternal;
    EXPECT_EQ(sink->last_error, expected);
  }

  const auto rank0_calls = rank0.commands.size();
  const auto rank1_calls = rank1.commands.size();
  const auto clock_calls = clock.calls;
  auto later_sink = std::make_shared<Sink>();
  const Status later = controller->Submit(
      {RequestId{11}, "model", {TokenId{2}}, 1, 10, 1}, later_sink);
  if (fatal) {
    ASSERT_FALSE(later.ok());
    EXPECT_EQ(later.error()->code(), ErrorCode::kReplicaFatal);
    EXPECT_EQ(rank0.commands.size(), rank0_calls);
    EXPECT_EQ(rank1.commands.size(), rank1_calls);
    EXPECT_EQ(later_sink->accepted, 0);
    EXPECT_EQ(admission.active_reservation_count(), 0U);
    auto invalid_sink = std::make_shared<Sink>();
    const Status invalid = controller->Submit(
        {RequestId{12}, "", {}, 0, 0, 0}, invalid_sink);
    ASSERT_FALSE(invalid.ok());
    EXPECT_EQ(invalid.error()->code(), ErrorCode::kReplicaFatal);
    EXPECT_EQ(clock.calls, clock_calls);
    EXPECT_EQ(rank0.commands.size(), rank0_calls);
    EXPECT_EQ(rank1.commands.size(), rank1_calls);
  } else {
    EXPECT_GT(rank0.commands.size(), rank0_calls);
    static_cast<void>(controller->Cancel(RequestId{11},
                                         CancelReason::kServerShutdown));
  }
}

INSTANTIATE_TEST_SUITE_P(
    RequiredCases, ControllerFaultMatrix,
    testing::Values(FaultCase::kPrepareFail, FaultCase::kExecuteFail,
                    FaultCase::kPreDecisionTimeout, FaultCase::kRankLoss,
                    FaultCase::kCancelBeforeDecision,
                    FaultCase::kCommitFailBeforeApply,
                    FaultCase::kCommitApplyThenAckLoss,
                    FaultCase::kPostDecisionTimeout,
                    FaultCase::kDetachAfterDecision),
    [](const testing::TestParamInfo<FaultCase>& info) { return Name(info.param); });

TEST(ControllerDeadline, SaturatesAnOverflowingRelativeTimeout) {
  ManualClock clock{Controller::TimePoint::max() - std::chrono::milliseconds{1}};
  ManualRank rank{RankId{0}, FaultCase::kPreDecisionTimeout};
  auto admission = AdmissionController::Create({{RankId{0}, 1}}).value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions, {rank.endpoint()}, {},
                                 [&clock] { return clock(); });
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink = std::make_shared<Sink>();
  EXPECT_TRUE(controller->Submit(
      {RequestId{20}, "model", {TokenId{1}}, 1,
       std::numeric_limits<std::uint64_t>::max(), 1}, sink).ok());
  EXPECT_EQ(controller->active_deadline_for_testing(), Controller::TimePoint::max());
}

TEST(ReplicaGate, FatalCauseIsStickyAndAdmissionIsAlwaysReplicaFatal) {
  ReplicaGate gate;
  const Error first = Error::From(ErrorCode::kRankFailure, "rank 0 lost",
                                  RequestId{3}, TransactionId{4});
  gate.MarkFatal(first);
  gate.MarkFatal(Error::From(ErrorCode::kInternal, "later"));
  ASSERT_EQ(gate.health(), ReplicaHealth::kFatal);
  ASSERT_TRUE(gate.fatal_error().has_value());
  EXPECT_EQ(gate.fatal_error()->code(), ErrorCode::kRankFailure);
  EXPECT_EQ(gate.fatal_error()->message(), "rank 0 lost");
  const Status admission = gate.CheckAdmission();
  ASSERT_FALSE(admission.ok());
  EXPECT_EQ(admission.error()->code(), ErrorCode::kReplicaFatal);
}

TEST(ControllerFaults, AbortedCompletionQueueTerminatesActiveRequest) {
  ManualRank rank0{RankId{0}, FaultCase::kRankLoss};
  ManualRank rank1{RankId{1}, FaultCase::kRankLoss};
  auto admission = AdmissionController::Create(
      {{RankId{0}, 1}, {RankId{1}, 1}}).value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions(1);
  auto made = Controller::Create(
      admission, gate, completions, {rank0.endpoint(), rank1.endpoint()});
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink = std::make_shared<Sink>();
  ASSERT_TRUE(controller->Submit(
      {RequestId{30}, "model", {TokenId{1}}, 1, 1000, 1}, sink).ok());
  completions.Abort();
  controller->PollOnce();
  EXPECT_EQ(sink->errors, 1);
  EXPECT_EQ(sink->last_error, ErrorCode::kReplicaFatal);
  EXPECT_EQ(controller->last_terminal_state_for_testing(),
            TransactionState::kAborted);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
}

TEST(ControllerFaults, LateQueueAbortAfterCancelFatalsReplicaExactlyOnce) {
  CompletionQueue<RankCompletion> completions(1);
  std::barrier release_prepare(3);
  std::atomic<bool> rank0_first{true};
  std::atomic<bool> rank1_first{true};
  RankWorker rank0(RankId{0}, completions, StateDigest{0}, [&] {
    if (rank0_first.exchange(false)) release_prepare.arrive_and_wait();
  });
  RankWorker rank1(RankId{1}, completions, StateDigest{0}, [&] {
    if (rank1_first.exchange(false)) release_prepare.arrive_and_wait();
  });
  auto admission = AdmissionController::Create(
      {{RankId{0}, 1}, {RankId{1}, 1}}).value();
  auto gate = std::make_shared<ReplicaGate>();
  auto made = Controller::Create(
      admission, gate, completions,
      {{RankId{0}, [&](RankCommand command) {
          return rank0.Enqueue(std::move(command));
        }},
       {RankId{1}, [&](RankCommand command) {
          return rank1.Enqueue(std::move(command));
        }}});
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink = std::make_shared<Sink>();
  ASSERT_TRUE(controller->Submit(
      {RequestId{30}, "model", {TokenId{1}}, 1, 1000, 1}, sink).ok());
  ASSERT_TRUE(controller->Cancel(RequestId{30},
                                 CancelReason::kClientDisconnected).ok());
  EXPECT_TRUE(controller->idle());

  release_prepare.arrive_and_wait();
  completions.WaitForAbort();
  ASSERT_TRUE(completions.aborted());
  EXPECT_TRUE(controller->PollOnce());
  EXPECT_EQ(gate->health(), ReplicaHealth::kFatal);
  const auto fatal = gate->fatal_error();
  ASSERT_TRUE(fatal.has_value());
  EXPECT_EQ(fatal->code(), ErrorCode::kReplicaFatal);
  EXPECT_FALSE(controller->PollOnce());
  EXPECT_EQ(gate->fatal_error()->message(), fatal->message());

  auto rejected_sink = std::make_shared<Sink>();
  const Status rejected = controller->Submit(
      {RequestId{31}, "model", {TokenId{1}}, 1, 1000, 1}, rejected_sink);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error()->code(), ErrorCode::kReplicaFatal);
  EXPECT_EQ(rejected_sink->accepted, 0);
  rank0.Shutdown();
  rank1.Shutdown();
}

TEST(ControllerFaults, CommitAcknowledgementErrorIsFatalAndNeverPublishes) {
  ManualRank rank0{RankId{0}, FaultCase::kCommitApplyThenAckLoss};
  ManualRank rank1{RankId{1}, FaultCase::kCommitApplyThenAckLoss};
  auto admission = AdmissionController::Create(
      {{RankId{0}, 1}, {RankId{1}, 1}}).value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(
      admission, gate, completions, {rank0.endpoint(), rank1.endpoint()},
      [](const GenerateRequest&, SequenceNumber) { return TokenId{9}; });
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink = std::make_shared<Sink>();
  ASSERT_TRUE(controller->Submit(
      {RequestId{40}, "model", {TokenId{1}}, 1, 1000, 1}, sink).ok());
  const auto prepare = std::get<Prepare>(rank0.commands.front());
  const auto push = [&](RankCompletion completion) {
    ASSERT_TRUE(completions.Push(std::move(completion)));
    controller->PollOnce();
  };
  push(PreparedAck{RankId{0}, prepare.epoch});
  push(PreparedAck{RankId{1}, prepare.epoch});
  push(ExecutedAck{RankId{0}, prepare.epoch, TokenId{9}, StateDigest{1}});
  push(ExecutedAck{RankId{1}, prepare.epoch, TokenId{9}, StateDigest{2}});
  push(RankError{RankId{1}, prepare.transaction_id, prepare.epoch,
                 Error::From(ErrorCode::kInternal, "commit ack failed")});

  EXPECT_EQ(controller->last_terminal_state_for_testing(),
            TransactionState::kFatal);
  EXPECT_EQ(rank0.canonical, 1);
  EXPECT_EQ(rank1.canonical, 1);
  EXPECT_EQ(controller->logical_publish_count(), 0U);
  EXPECT_EQ(controller->network_delivery_count(), 0U);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  EXPECT_EQ(gate->health(), ReplicaHealth::kFatal);
}
}  // namespace
