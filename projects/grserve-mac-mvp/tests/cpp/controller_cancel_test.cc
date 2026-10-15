#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "engine/admission.h"
#include "engine/controller.h"

namespace {
using namespace grserve;

class CancelSink : public RequestEventSink {
 public:
  bool TryAccepted(RequestId) noexcept override { ++accepted; return true; }
  bool TryToken(const TokenEvent&) noexcept override { ++tokens; return true; }
  bool TryFinished(const FinishedEvent& event) noexcept override {
    finish = event.finish_reason;
    return true;
  }
  bool TryError(const Error& error) noexcept override {
    ++errors;
    last_error = error.code();
    error_request = error.request_id();
    error_transaction = error.transaction_id();
    return true;
  }
  int accepted{0};
  int tokens{0};
  int errors{0};
  std::optional<FinishReason> finish;
  std::optional<ErrorCode> last_error;
  std::optional<RequestId> error_request;
  std::optional<TransactionId> error_transaction;
};

struct Rank {
  RankId id;
  std::vector<RankCommand> commands;
  int canonical{0};
  ControllerRank endpoint() {
    return {id, [this](RankCommand command) {
      commands.push_back(command);
      if (std::holds_alternative<Commit>(command)) ++canonical;
      return Status::Ok();
    }};
  }
};

struct Fixture {
  Rank rank0{RankId{0}};
  Rank rank1{RankId{1}};
  AdmissionController admission =
      AdmissionController::Create({{RankId{0}, 2}, {RankId{1}, 2}}).value();
  std::shared_ptr<ReplicaGate> gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  std::unique_ptr<Controller> controller;
  std::shared_ptr<CancelSink> sink = std::make_shared<CancelSink>();

  Fixture() {
    auto made = Controller::Create(
        admission, gate, completions, {rank0.endpoint(), rank1.endpoint()},
        [](const GenerateRequest&, SequenceNumber) { return TokenId{7}; });
    EXPECT_TRUE(made.ok());
    controller = std::move(made).value();
    EXPECT_TRUE(controller->Submit(
        {RequestId{30}, "model", {TokenId{1}}, 3, 1000, 1}, sink).ok());
  }
  Prepare prepare() const { return std::get<Prepare>(rank0.commands.front()); }
  void push(RankCompletion completion) {
    ASSERT_TRUE(completions.Push(std::move(completion)));
    controller->PollOnce();
  }
  void reach_decision() {
    const auto p = prepare();
    push(PreparedAck{RankId{0}, p.epoch});
    push(PreparedAck{RankId{1}, p.epoch});
    push(ExecutedAck{RankId{0}, p.epoch, TokenId{7}, StateDigest{1}});
    push(ExecutedAck{RankId{1}, p.epoch, TokenId{7}, StateDigest{2}});
  }
};

TEST(ControllerCancel, BeforeDecisionAbortsAndReleasesWithoutPublication) {
  Fixture fixture;
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kServerShutdown).ok());
  const auto command_count = fixture.rank0.commands.size();
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kServerShutdown).ok());
  EXPECT_EQ(fixture.controller->last_terminal_state_for_testing(),
            TransactionState::kAborted);
  EXPECT_EQ(fixture.controller->logical_publish_count(), 0U);
  EXPECT_EQ(fixture.sink->errors, 1);
  EXPECT_EQ(fixture.sink->last_error, ErrorCode::kInternal);
  EXPECT_EQ(fixture.sink->error_request, RequestId{30});
  EXPECT_FALSE(fixture.sink->finish.has_value());
  EXPECT_EQ(fixture.admission.active_reservation_count(), 0U);
  EXPECT_EQ(fixture.gate->health(), ReplicaHealth::kHealthy);
  EXPECT_TRUE(std::holds_alternative<Abort>(fixture.rank0.commands.back()));
  EXPECT_TRUE(std::holds_alternative<Abort>(fixture.rank1.commands.back()));
  EXPECT_EQ(fixture.rank0.commands.size(), command_count);
}

TEST(ControllerCancel, AfterDecisionCompletesCommitAndStartsNoNextToken) {
  Fixture fixture;
  fixture.reach_decision();
  const auto p = fixture.prepare();
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kServerShutdown).ok());
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kServerShutdown).ok());
  EXPECT_EQ(fixture.sink->errors, 1);
  fixture.push(CommittedAck{RankId{0}, p.epoch});
  fixture.push(CommittedAck{RankId{1}, p.epoch});
  EXPECT_EQ(fixture.controller->last_terminal_state_for_testing(),
            TransactionState::kPublished);
  EXPECT_EQ(fixture.rank0.canonical, 1);
  EXPECT_EQ(fixture.rank1.canonical, 1);
  EXPECT_EQ(fixture.controller->logical_publish_count(), 1U);
  EXPECT_EQ(fixture.controller->network_delivery_count(), 0U);
  EXPECT_EQ(fixture.controller->last_delivery_outcome(),
            DeliveryOutcome::kDroppedClientDisconnect);
  EXPECT_EQ(fixture.sink->tokens, 0);
  EXPECT_EQ(fixture.sink->errors, 1);
  EXPECT_FALSE(fixture.sink->finish.has_value());
  EXPECT_EQ(fixture.rank0.commands.size(), 3U);
  EXPECT_EQ(fixture.rank1.commands.size(), 3U);
  EXPECT_EQ(fixture.admission.active_reservation_count(), 0U);
  EXPECT_EQ(fixture.gate->health(), ReplicaHealth::kHealthy);
}

TEST(ControllerCancel, UnknownRequestIsRejectedWithoutSideEffects) {
  Fixture fixture;
  const auto count = fixture.rank0.commands.size();
  const Status status = fixture.controller->Cancel(
      RequestId{99}, CancelReason::kDeadline);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error()->code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(fixture.rank0.commands.size(), count);
  const Status detach = fixture.controller->DetachTransport(RequestId{99});
  ASSERT_FALSE(detach.ok());
  EXPECT_EQ(detach.error()->code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(fixture.rank0.commands.size(), count);
}

TEST(ControllerCancel, DetachBeforeDecisionAbortsWithoutNetworkEvents) {
  Fixture fixture;
  ASSERT_TRUE(fixture.controller->DetachTransport(RequestId{30}).ok());
  const auto command_count = fixture.rank0.commands.size();
  ASSERT_TRUE(fixture.controller->DetachTransport(RequestId{30}).ok());
  EXPECT_EQ(fixture.controller->last_terminal_state_for_testing(),
            TransactionState::kAborted);
  EXPECT_EQ(fixture.controller->logical_publish_count(), 0U);
  EXPECT_EQ(fixture.sink->tokens, 0);
  EXPECT_FALSE(fixture.sink->finish.has_value());
  EXPECT_FALSE(fixture.sink->last_error.has_value());
  EXPECT_EQ(fixture.admission.active_reservation_count(), 0U);
  EXPECT_TRUE(std::holds_alternative<Abort>(fixture.rank0.commands.back()));
  EXPECT_TRUE(std::holds_alternative<Abort>(fixture.rank1.commands.back()));
  EXPECT_EQ(fixture.rank0.commands.size(), command_count);
}

TEST(ControllerCancel, DeadlineCancellationAfterDecisionIsImmediatelyFatal) {
  Fixture fixture;
  fixture.reach_decision();
  const auto command_count = fixture.rank0.commands.size();
  ASSERT_TRUE(fixture.controller->Cancel(RequestId{30},
                                         CancelReason::kDeadline).ok());
  ASSERT_TRUE(fixture.controller->Cancel(RequestId{30},
                                         CancelReason::kDeadline).ok());
  EXPECT_EQ(fixture.controller->last_terminal_state_for_testing(),
            TransactionState::kFatal);
  EXPECT_EQ(fixture.controller->logical_publish_count(), 0U);
  EXPECT_EQ(fixture.controller->network_delivery_count(), 0U);
  EXPECT_EQ(fixture.rank0.canonical, 1);
  EXPECT_EQ(fixture.rank1.canonical, 1);
  EXPECT_EQ(fixture.admission.active_reservation_count(), 0U);
  EXPECT_EQ(fixture.gate->health(), ReplicaHealth::kFatal);
  EXPECT_EQ(fixture.sink->errors, 1);
  EXPECT_EQ(fixture.sink->last_error, ErrorCode::kDeadlineExceeded);
  EXPECT_EQ(fixture.sink->error_transaction,
            fixture.prepare().transaction_id);
  EXPECT_EQ(fixture.rank0.commands.size(), command_count);
}

class PostDecisionDeadlineUpgrade
    : public testing::TestWithParam<CancelReason> {};

TEST_P(PostDecisionDeadlineUpgrade, NonDeadlineCancelAlwaysUpgradesToFatal) {
  Fixture fixture;
  fixture.reach_decision();
  const CancelReason first = GetParam();
  ASSERT_NE(first, CancelReason::kDeadline);
  ASSERT_TRUE(fixture.controller->Cancel(RequestId{30}, first).ok());
  ASSERT_TRUE(fixture.controller->Cancel(RequestId{30},
                                         CancelReason::kDeadline).ok());
  EXPECT_EQ(fixture.controller->last_terminal_state_for_testing(),
            TransactionState::kFatal);
  EXPECT_EQ(fixture.controller->logical_publish_count(), 0U);
  EXPECT_EQ(fixture.controller->network_delivery_count(), 0U);
  EXPECT_EQ(fixture.gate->health(), ReplicaHealth::kFatal);
  EXPECT_EQ(fixture.admission.active_reservation_count(), 0U);
}

INSTANTIATE_TEST_SUITE_P(
    CancelReasons, PostDecisionDeadlineUpgrade,
    testing::Values(CancelReason::kServerShutdown,
                    CancelReason::kClientDisconnected));

TEST(ControllerCancel, DetachAfterDecisionStillUpgradesToDeadlineFatal) {
  Fixture fixture;
  fixture.reach_decision();
  ASSERT_TRUE(fixture.controller->DetachTransport(RequestId{30}).ok());
  ASSERT_TRUE(fixture.controller->Cancel(RequestId{30},
                                         CancelReason::kDeadline).ok());
  EXPECT_EQ(fixture.controller->last_terminal_state_for_testing(),
            TransactionState::kFatal);
  EXPECT_EQ(fixture.controller->logical_publish_count(), 0U);
  EXPECT_EQ(fixture.controller->network_delivery_count(), 0U);
  EXPECT_EQ(fixture.gate->health(), ReplicaHealth::kFatal);
  EXPECT_EQ(fixture.sink->errors, 0);
}

TEST(ControllerCancel, DeadlineFatalFreezesLaterControlOperations) {
  Fixture fixture;
  fixture.reach_decision();
  ASSERT_TRUE(fixture.controller->Cancel(RequestId{30},
                                         CancelReason::kDeadline).ok());
  const auto rank0_count = fixture.rank0.commands.size();
  const auto errors = fixture.sink->errors;
  EXPECT_FALSE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kServerShutdown).ok());
  EXPECT_FALSE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kClientDisconnected).ok());
  EXPECT_FALSE(fixture.controller->DetachTransport(RequestId{30}).ok());
  EXPECT_EQ(fixture.rank0.commands.size(), rank0_count);
  EXPECT_EQ(fixture.sink->errors, errors);
  EXPECT_EQ(fixture.controller->logical_publish_count(), 0U);
  EXPECT_EQ(fixture.gate->health(), ReplicaHealth::kFatal);
}

TEST(ControllerCancel, DifferentCancelReasonIsNotFoldedAsTheSameReplay) {
  Fixture fixture;
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kServerShutdown).ok());
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kServerShutdown).ok());
  const Status different = fixture.controller->Cancel(
      RequestId{30}, CancelReason::kDeadline);
  ASSERT_FALSE(different.ok());
  EXPECT_EQ(different.error()->code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(fixture.sink->errors, 1);
}

TEST(ControllerCancel, ClientDisconnectedReasonMatchesDetachSemantics) {
  Fixture fixture;
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kClientDisconnected).ok());
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kClientDisconnected).ok());
  EXPECT_EQ(fixture.controller->last_terminal_state_for_testing(),
            TransactionState::kAborted);
  EXPECT_EQ(fixture.sink->errors, 0);
  EXPECT_FALSE(fixture.sink->finish.has_value());
  EXPECT_EQ(fixture.admission.active_reservation_count(), 0U);
}

TEST(ControllerCancel, ClientDisconnectedReasonAfterDecisionCommitsSilently) {
  Fixture fixture;
  fixture.reach_decision();
  const auto p = fixture.prepare();
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kClientDisconnected).ok());
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kClientDisconnected).ok());
  fixture.push(CommittedAck{RankId{0}, p.epoch});
  fixture.push(CommittedAck{RankId{1}, p.epoch});
  EXPECT_EQ(fixture.controller->last_terminal_state_for_testing(),
            TransactionState::kPublished);
  EXPECT_EQ(fixture.controller->logical_publish_count(), 1U);
  EXPECT_EQ(fixture.controller->network_delivery_count(), 0U);
  EXPECT_EQ(fixture.sink->errors, 0);
  EXPECT_EQ(fixture.gate->health(), ReplicaHealth::kHealthy);
}

TEST(ControllerCancel, DeadlineReasonBeforeDecisionAbortsWithoutFatal) {
  Fixture fixture;
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kDeadline).ok());
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{30}, CancelReason::kDeadline).ok());
  EXPECT_EQ(fixture.controller->last_terminal_state_for_testing(),
            TransactionState::kAborted);
  EXPECT_EQ(fixture.sink->errors, 1);
  EXPECT_EQ(fixture.sink->last_error, ErrorCode::kDeadlineExceeded);
  EXPECT_EQ(fixture.admission.active_reservation_count(), 0U);
  EXPECT_EQ(fixture.gate->health(), ReplicaHealth::kHealthy);
}

TEST(ControllerCancel, RepeatedDetachAfterDecisionKeepsTheFrozenOutcome) {
  Fixture fixture;
  fixture.reach_decision();
  const auto p = fixture.prepare();
  ASSERT_TRUE(fixture.controller->DetachTransport(RequestId{30}).ok());
  ASSERT_TRUE(fixture.controller->DetachTransport(RequestId{30}).ok());
  fixture.push(CommittedAck{RankId{0}, p.epoch});
  fixture.push(CommittedAck{RankId{1}, p.epoch});
  ASSERT_TRUE(fixture.controller->DetachTransport(RequestId{30}).ok());
  EXPECT_EQ(fixture.controller->last_terminal_state_for_testing(),
            TransactionState::kPublished);
  EXPECT_EQ(fixture.controller->logical_publish_count(), 1U);
  EXPECT_EQ(fixture.controller->network_delivery_count(), 0U);
  EXPECT_EQ(fixture.rank0.canonical, 1);
  EXPECT_EQ(fixture.rank1.canonical, 1);
  EXPECT_EQ(fixture.sink->errors, 0);
  EXPECT_EQ(fixture.rank0.commands.size(), 3U);
  EXPECT_EQ(fixture.rank1.commands.size(), 3U);
}

TEST(ControllerCancel, PendingCancelReleasesOnceWithoutRankCommands) {
  Fixture fixture;
  auto pending_sink = std::make_shared<CancelSink>();
  ASSERT_TRUE(fixture.controller->Submit(
      {RequestId{31}, "model", {TokenId{1}}, 1, 1000, 1}, pending_sink).ok());
  const auto rank0_count = fixture.rank0.commands.size();
  const auto rank1_count = fixture.rank1.commands.size();
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{31}, CancelReason::kServerShutdown).ok());
  ASSERT_TRUE(fixture.controller->Cancel(
      RequestId{31}, CancelReason::kServerShutdown).ok());
  EXPECT_EQ(pending_sink->accepted, 1);
  EXPECT_EQ(pending_sink->errors, 1);
  EXPECT_EQ(pending_sink->error_request, RequestId{31});
  EXPECT_EQ(fixture.rank0.commands.size(), rank0_count);
  EXPECT_EQ(fixture.rank1.commands.size(), rank1_count);
  EXPECT_EQ(fixture.admission.active_reservation_count(), 1U);
}

TEST(ControllerCancel, PendingDetachReleasesOnceWithoutDeliveryOrRankCommands) {
  Fixture fixture;
  auto pending_sink = std::make_shared<CancelSink>();
  ASSERT_TRUE(fixture.controller->Submit(
      {RequestId{32}, "model", {TokenId{1}}, 1, 1000, 1}, pending_sink).ok());
  const auto rank0_count = fixture.rank0.commands.size();
  const auto rank1_count = fixture.rank1.commands.size();
  ASSERT_TRUE(fixture.controller->DetachTransport(RequestId{32}).ok());
  ASSERT_TRUE(fixture.controller->DetachTransport(RequestId{32}).ok());
  EXPECT_EQ(pending_sink->accepted, 1);
  EXPECT_EQ(pending_sink->errors, 0);
  EXPECT_EQ(pending_sink->tokens, 0);
  EXPECT_FALSE(pending_sink->finish.has_value());
  EXPECT_EQ(fixture.rank0.commands.size(), rank0_count);
  EXPECT_EQ(fixture.rank1.commands.size(), rank1_count);
  EXPECT_EQ(fixture.admission.active_reservation_count(), 1U);
}

TEST(ControllerCancel, ProcessedControlReplayIsBounded) {
  Rank rank0{RankId{0}};
  auto admission = AdmissionController::Create({{RankId{0}, 2}}).value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions,
                                 {rank0.endpoint()});
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto active_sink = std::make_shared<CancelSink>();
  ASSERT_TRUE(controller->Submit(
      {RequestId{50}, "model", {TokenId{1}}, 1, 1000, 1}, active_sink).ok());
  for (std::uint64_t value = 1000; value < 1257; ++value) {
    auto sink = std::make_shared<CancelSink>();
    ASSERT_TRUE(controller->Submit(
        {RequestId{value}, "model", {TokenId{1}}, 1, 1000, 1}, sink).ok());
    ASSERT_TRUE(controller->Cancel(RequestId{value},
                                   CancelReason::kServerShutdown).ok());
  }
  EXPECT_EQ(controller->control_replay_size_for_testing(), 256U);
  EXPECT_FALSE(controller->Cancel(RequestId{1000},
                                  CancelReason::kServerShutdown).ok());
  EXPECT_TRUE(controller->Cancel(RequestId{1256},
                                 CancelReason::kServerShutdown).ok());
  auto reused_sink = std::make_shared<CancelSink>();
  const Status reused = controller->Submit(
      {RequestId{1000}, "model", {TokenId{1}}, 1, 1000, 1}, reused_sink);
  ASSERT_FALSE(reused.ok());
  EXPECT_EQ(reused.error()->code(), ErrorCode::kInvalidArgument);
  auto newer_sink = std::make_shared<CancelSink>();
  EXPECT_TRUE(controller->Submit(
      {RequestId{2000}, "model", {TokenId{1}}, 1, 1000, 1}, newer_sink).ok());
}

class RejectAcceptedSink final : public CancelSink {
 public:
  bool TryAccepted(RequestId) noexcept override { return false; }
};

TEST(ControllerRequestIds, OnlySuccessfulAcceptanceAdvancesTheWatermark) {
  Rank rank{RankId{0}};
  auto admission = AdmissionController::Create({{RankId{0}, 2}}).value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions, {rank.endpoint()});
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();

  auto invalid_sink = std::make_shared<CancelSink>();
  EXPECT_FALSE(controller->Submit(
      {RequestId{100}, "", {TokenId{1}}, 1, 1000, 1}, invalid_sink).ok());
  auto rejected_sink = std::make_shared<RejectAcceptedSink>();
  EXPECT_FALSE(controller->Submit(
      {RequestId{100}, "model", {TokenId{1}}, 1, 1000, 1}, rejected_sink).ok());
  auto accepted_sink = std::make_shared<CancelSink>();
  EXPECT_TRUE(controller->Submit(
      {RequestId{100}, "model", {TokenId{1}}, 1, 1000, 1}, accepted_sink).ok());

  auto exhausted_sink = std::make_shared<CancelSink>();
  EXPECT_FALSE(controller->Submit(
      {RequestId{101}, "model", {TokenId{1}}, 1, 1000, 2}, exhausted_sink).ok());
  auto next_sink = std::make_shared<CancelSink>();
  EXPECT_TRUE(controller->Submit(
      {RequestId{101}, "model", {TokenId{1}}, 1, 1000, 1}, next_sink).ok());
}

TEST(ControllerRequestIds, AcceptedIdsAreStrictlyIncreasingAndNeverReusable) {
  struct CountingClock {
    std::size_t calls{0};
    Controller::TimePoint operator()() noexcept {
      ++calls;
      return {};
    }
  } clock;
  Rank rank{RankId{0}};
  auto admission = AdmissionController::Create({{RankId{0}, 2}}).value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions, {rank.endpoint()},
                                 {}, [&clock] { return clock(); });
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink = std::make_shared<CancelSink>();
  ASSERT_TRUE(controller->Submit(
      {RequestId{500}, "model", {TokenId{1}}, 1, 1000, 1}, sink).ok());
  ASSERT_TRUE(controller->Cancel(
      RequestId{500}, CancelReason::kServerShutdown).ok());
  const auto calls = clock.calls;
  const auto commands = rank.commands.size();

  for (const RequestId id : {RequestId{500}, RequestId{499}, RequestId{1}}) {
    auto rejected = std::make_shared<CancelSink>();
    const Status status = controller->Submit(
        {id, "model", {TokenId{1}}, 1, 1000, 1}, rejected);
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.error()->code(), ErrorCode::kInvalidArgument);
    EXPECT_EQ(rejected->accepted, 0);
  }
  EXPECT_EQ(clock.calls, calls);
  EXPECT_EQ(rank.commands.size(), commands);
  EXPECT_EQ(admission.active_reservation_count(), 0U);

  auto newer = std::make_shared<CancelSink>();
  EXPECT_TRUE(controller->Submit(
      {RequestId{501}, "model", {TokenId{1}}, 1, 1000, 1}, newer).ok());
  EXPECT_EQ(newer->accepted, 1);
}
}  // namespace
