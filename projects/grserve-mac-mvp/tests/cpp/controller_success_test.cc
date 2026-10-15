#include <algorithm>
#include <array>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "engine/admission.h"
#include "engine/controller.h"
#include "engine/replica_gate.h"

namespace {

using namespace grserve;

class RecordingSink final : public RequestEventSink {
 public:
  bool TryAccepted(RequestId request) noexcept override {
    accepted.push_back(request);
    return true;
  }
  bool TryToken(const TokenEvent& token) noexcept override {
    tokens.push_back(token);
    return true;
  }
  bool TryFinished(const FinishedEvent& finished) noexcept override {
    finishes.push_back(finished);
    return true;
  }
  bool TryError(const Error& error) noexcept override {
    errors.push_back(error.code());
    full_errors.push_back(error);
    return true;
  }

  std::vector<RequestId> accepted;
  std::vector<TokenEvent> tokens;
  std::vector<FinishedEvent> finishes;
  std::vector<ErrorCode> errors;
  std::vector<Error> full_errors;
};

class DroppingSink final : public RequestEventSink {
 public:
  explicit DroppingSink(bool drop_token = true, bool drop_finish = true)
      : drop_token_(drop_token), drop_finish_(drop_finish) {}
  bool TryAccepted(RequestId) noexcept override {
    ++accepted;
    return true;
  }
  bool TryToken(const TokenEvent&) noexcept override {
    ++token_attempts;
    return !drop_token_;
  }
  bool TryFinished(const FinishedEvent&) noexcept override {
    ++finish_attempts;
    return !drop_finish_;
  }
  bool TryError(const Error&) noexcept override {
    ++errors;
    return true;
  }
  std::size_t accepted{0};
  std::size_t token_attempts{0};
  std::size_t finish_attempts{0};
  std::size_t errors{0};

 private:
  bool drop_token_;
  bool drop_finish_;
};

struct ManualRank {
  RankId id;
  std::vector<RankCommand> commands;

  ControllerRank endpoint() {
    return ControllerRank{id, [this](RankCommand command) {
                            commands.push_back(std::move(command));
                            return Status::Ok();
                          }};
  }
};

struct RunResult {
  TokenEvent token;
  FinishedEvent finished;
};

class StepGate {
 public:
  void Enter() {
    std::unique_lock lock(mutex_);
    const std::size_t step = entered_++;
    cv_.notify_all();
    cv_.wait(lock, [this, step] { return allowed_ > step; });
  }

  void WaitEntered(std::size_t count) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this, count] { return entered_ >= count; });
  }

  void AllowNext() {
    {
      std::lock_guard lock(mutex_);
      ++allowed_;
    }
    cv_.notify_all();
  }

  [[nodiscard]] std::size_t entered() const {
    std::lock_guard lock(mutex_);
    return entered_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::size_t entered_{0};
  std::size_t allowed_{0};
};

void PumpUntil(Controller& controller, const std::function<bool()>& predicate) {
  while (!predicate()) {
    controller.PollOnce();
    std::this_thread::yield();
  }
}

void RunWithOrder(const std::array<std::size_t, 2>& prepare_order,
                  const std::array<std::size_t, 2>& execute_order,
                  const std::array<std::size_t, 2>& commit_order,
                  std::optional<RunResult>& result) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  ManualRank rank0{kRanks[0]};
  ManualRank rank1{kRanks[1]};
  std::array<ManualRank*, 2> manual{&rank0, &rank1};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 2}, RankCapacity{kRanks[1], 2}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto controller_result = Controller::Create(
      admission, gate, completions, {rank0.endpoint(), rank1.endpoint()},
      [](const GenerateRequest&, SequenceNumber) { return TokenId{17}; });
  EXPECT_TRUE(controller_result.ok());
  auto controller = std::move(controller_result).value();
  auto sink = std::make_shared<RecordingSink>();

  GenerateRequest request{RequestId{41}, "mini-qwen38-hybrid", {TokenId{3}, TokenId{5}},
                          1, 1'000, 1};
  EXPECT_TRUE(controller->Submit(request, sink).ok());
  EXPECT_EQ(sink->accepted, std::vector{RequestId{41}});
  EXPECT_TRUE(sink->tokens.empty());
  ASSERT_EQ(rank0.commands.size(), 1U);
  ASSERT_EQ(rank1.commands.size(), 1U);
  const auto transaction = std::get<Prepare>(rank0.commands[0]).transaction_id;
  const auto epoch = std::get<Prepare>(rank0.commands[0]).epoch;
  EXPECT_EQ(std::get<Prepare>(rank1.commands[0]).transaction_id, transaction);

  for (std::size_t index = 0; index < prepare_order.size(); ++index) {
    const auto rank_index = prepare_order[index];
    ASSERT_TRUE(completions.Push(PreparedAck{kRanks[rank_index], epoch}));
    controller->PollOnce();
    EXPECT_TRUE(sink->tokens.empty());
    if (index == 0) {
      EXPECT_EQ(rank0.commands.size(), 1U);
      EXPECT_EQ(rank1.commands.size(), 1U);
    }
  }
  ASSERT_EQ(rank0.commands.size(), 2U);
  ASSERT_EQ(rank1.commands.size(), 2U);
  EXPECT_TRUE(std::holds_alternative<Execute>(rank0.commands[1]));
  EXPECT_TRUE(std::holds_alternative<Execute>(rank1.commands[1]));

  for (std::size_t index = 0; index < execute_order.size(); ++index) {
    const auto rank_index = execute_order[index];
    ASSERT_TRUE(completions.Push(
        ExecutedAck{kRanks[rank_index], epoch, TokenId{17},
                    StateDigest{999 + kRanks[rank_index].value()}}));
    controller->PollOnce();
    EXPECT_TRUE(sink->tokens.empty());
    if (index == 0) {
      EXPECT_EQ(rank0.commands.size(), 2U);
      EXPECT_EQ(rank1.commands.size(), 2U);
    }
  }
  ASSERT_EQ(rank0.commands.size(), 3U);
  ASSERT_EQ(rank1.commands.size(), 3U);
  EXPECT_TRUE(std::holds_alternative<Commit>(rank0.commands[2]));
  EXPECT_TRUE(std::holds_alternative<Commit>(rank1.commands[2]));

  for (std::size_t index = 0; index < commit_order.size(); ++index) {
    const auto rank_index = commit_order[index];
    ASSERT_TRUE(completions.Push(CommittedAck{kRanks[rank_index], epoch}));
    controller->PollOnce();
    EXPECT_EQ(sink->tokens.size(), index == 0 ? 0U : 1U);
  }

  ASSERT_TRUE(sink->errors.empty());
  ASSERT_EQ(sink->tokens.size(), 1U);
  ASSERT_EQ(sink->finishes.size(), 1U);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  EXPECT_EQ(sink->tokens[0].transaction_id, transaction);
  EXPECT_EQ(sink->tokens[0].sequence_number, SequenceNumber{1});
  EXPECT_EQ(sink->tokens[0].token_id, TokenId{17});
  result.emplace(RunResult{sink->tokens[0], sink->finishes[0]});
}

TEST(ControllerSuccess, PublishesOnlyAfterEveryRankCommits) {
  std::optional<RunResult> forward;
  RunWithOrder({0, 1}, {0, 1}, {0, 1}, forward);
  ASSERT_TRUE(forward.has_value());
  EXPECT_EQ(forward->finished.last_sequence_number, SequenceNumber{1});
  EXPECT_EQ(forward->finished.completion_tokens, 1U);
}

TEST(ControllerSuccess, CompletionOrderDoesNotChangeNormalizedEvents) {
  std::optional<RunResult> forward;
  std::optional<RunResult> reverse;
  RunWithOrder({0, 1}, {0, 1}, {0, 1}, forward);
  RunWithOrder({1, 0}, {1, 0}, {1, 0}, reverse);
  ASSERT_TRUE(forward.has_value());
  ASSERT_TRUE(reverse.has_value());
  EXPECT_EQ(forward->token, reverse->token);
  EXPECT_EQ(forward->finished, reverse->finished);
}

TEST(ControllerSuccess, UsesANewTransactionAndIncreasingSequenceForEveryToken) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  ManualRank rank0{kRanks[0]};
  ManualRank rank1{kRanks[1]};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto result = Controller::Create(
      admission, gate, completions, {rank0.endpoint(), rank1.endpoint()},
      [](const GenerateRequest&, SequenceNumber sequence) {
        return TokenId{static_cast<std::uint32_t>(20 + sequence.value())};
      });
  ASSERT_TRUE(result.ok());
  auto controller = std::move(result).value();
  auto sink = std::make_shared<RecordingSink>();
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{77}, "mini-qwen38-hybrid",
                                           {TokenId{2}}, 2, 1000, 1},
                           sink)
                  .ok());

  const auto drive_token = [&](std::size_t command_index, StateDigest digest) {
    const auto prepare = std::get<Prepare>(rank0.commands.at(command_index));
    for (const RankId rank : kRanks) {
      ASSERT_TRUE(completions.Push(PreparedAck{rank, prepare.epoch}));
      controller->PollOnce();
    }
    for (const RankId rank : kRanks) {
      ASSERT_TRUE(completions.Push(ExecutedAck{
          rank, prepare.epoch,
          TokenId{static_cast<std::uint32_t>(21 + command_index / 3)}, digest}));
      controller->PollOnce();
    }
    for (const RankId rank : kRanks) {
      ASSERT_TRUE(completions.Push(CommittedAck{rank, prepare.epoch}));
      controller->PollOnce();
    }
  };

  drive_token(0, StateDigest{100});
  ASSERT_EQ(sink->tokens.size(), 1U);
  ASSERT_EQ(rank0.commands.size(), 4U);
  drive_token(3, StateDigest{101});
  ASSERT_EQ(sink->tokens.size(), 2U);
  EXPECT_NE(sink->tokens[0].transaction_id, sink->tokens[1].transaction_id);
  EXPECT_NE(std::get<Prepare>(rank0.commands[0]).epoch,
            std::get<Prepare>(rank0.commands[3]).epoch);
  EXPECT_EQ(sink->tokens[0].sequence_number, SequenceNumber{1});
  EXPECT_EQ(sink->tokens[1].sequence_number, SequenceNumber{2});
  EXPECT_EQ(sink->tokens[0].token_id, TokenId{21});
  EXPECT_EQ(sink->tokens[1].token_id, TokenId{22});
  EXPECT_EQ(sink->finishes.size(), 1U);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
}

TEST(ControllerSuccess, RejectsInvalidPublicInputsBeforeDispatch) {
  ManualRank rank{RankId{0}};
  auto admission = AdmissionController::Create({RankCapacity{RankId{0}, 1}}).value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  EXPECT_FALSE(Controller::Create(admission, {}, completions, {rank.endpoint()}).ok());
  EXPECT_FALSE(Controller::Create(admission, gate, completions, {}).ok());

  auto result = Controller::Create(admission, gate, completions, {rank.endpoint()});
  ASSERT_TRUE(result.ok());
  auto controller = std::move(result).value();
  auto sink = std::make_shared<RecordingSink>();
  EXPECT_FALSE(controller
                   ->Submit(GenerateRequest{RequestId{0}, "mini-qwen38-hybrid",
                                            {TokenId{1}}, 1, 1000, 1},
                            sink)
                   .ok());
  EXPECT_FALSE(controller
                   ->Submit(GenerateRequest{RequestId{9}, "", {TokenId{1}}, 1,
                                            1000, 1},
                            sink)
                   .ok());
  EXPECT_FALSE(controller->Cancel(RequestId{0}, CancelReason::kDeadline).ok());
  EXPECT_TRUE(rank.commands.empty());
  EXPECT_TRUE(sink->accepted.empty());
}

TEST(ControllerSuccess, IgnoresCompletionsOutsideTheCurrentRoutingContext) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  ManualRank rank0{kRanks[0]};
  ManualRank rank1{kRanks[1]};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto result = Controller::Create(admission, gate, completions,
                                   {rank0.endpoint(), rank1.endpoint()},
                                   [](const GenerateRequest&, SequenceNumber) {
                                     return TokenId{31};
                                   });
  ASSERT_TRUE(result.ok());
  auto controller = std::move(result).value();
  auto sink = std::make_shared<RecordingSink>();
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{88}, "mini-qwen38-hybrid",
                                           {TokenId{1}}, 2, 1000, 1},
                           sink)
                  .ok());
  const auto first = std::get<Prepare>(rank0.commands[0]);

  const auto ignored = [&](RankCompletion completion) {
    ASSERT_TRUE(completions.Push(std::move(completion)));
    controller->PollOnce();
    EXPECT_TRUE(sink->errors.empty());
    EXPECT_TRUE(sink->tokens.empty());
  };
  ignored(PreparedAck{RankId{9}, first.epoch});
  ignored(PreparedAck{RankId{0}, Epoch{first.epoch.value() + 1}});
  ignored(RankError{RankId{9}, first.transaction_id, first.epoch,
                    Error::From(ErrorCode::kRankFailure, "unknown rank")});
  ignored(RankError{RankId{0}, TransactionId{first.transaction_id.value() + 99},
                    first.epoch,
                    Error::From(ErrorCode::kRankFailure, "wrong transaction")});
  ignored(RankError{RankId{0}, first.transaction_id,
                    Epoch{first.epoch.value() + 1},
                    Error::From(ErrorCode::kRankFailure, "future epoch")});
  ASSERT_EQ(rank0.commands.size(), 1U);
  ASSERT_EQ(rank1.commands.size(), 1U);

  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(PreparedAck{rank, first.epoch}));
    controller->PollOnce();
  }
  ASSERT_EQ(rank0.commands.size(), 2U);
  ignored(ExecutedAck{RankId{9}, first.epoch, TokenId{31}, StateDigest{1}});
  ignored(ExecutedAck{RankId{0}, Epoch{first.epoch.value() + 1}, TokenId{31},
                      StateDigest{1}});
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(
        ExecutedAck{rank, first.epoch, TokenId{31}, StateDigest{200 + rank.value()}}));
    controller->PollOnce();
  }
  ASSERT_EQ(rank0.commands.size(), 3U);
  ignored(CommittedAck{RankId{9}, first.epoch});
  ignored(CommittedAck{RankId{0}, Epoch{first.epoch.value() + 1}});
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(CommittedAck{rank, first.epoch}));
    controller->PollOnce();
  }
  ASSERT_EQ(sink->tokens.size(), 1U);
  ASSERT_EQ(rank0.commands.size(), 4U);
  const auto second = std::get<Prepare>(rank0.commands[3]);
  ASSERT_GT(second.epoch.value(), first.epoch.value());

  ASSERT_TRUE(completions.Push(PreparedAck{RankId{0}, first.epoch}));
  controller->PollOnce();
  ASSERT_TRUE(completions.Push(RankError{
      RankId{0}, first.transaction_id, first.epoch,
      Error::From(ErrorCode::kRankFailure, "late previous transaction")}));
  controller->PollOnce();
  EXPECT_TRUE(sink->errors.empty());
  EXPECT_EQ(rank0.commands.size(), 4U);
  EXPECT_EQ(rank1.commands.size(), 4U);
}

void RunRealWorkers(const std::array<std::size_t, 2>& order,
                    std::optional<TokenEvent>& result) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  CompletionQueue<RankCompletion> completions;
  StepGate gate0;
  StepGate gate1;
  const auto transition = [](StateDigest previous, RankId rank, TransactionId,
                             Epoch, TokenId token) {
    return StateDigest{previous.value() * 131 + token.value() * 17 +
                       rank.value() + 1};
  };
  RankWorker rank0(kRanks[0], completions, StateDigest{0},
                   [&gate0] { gate0.Enter(); }, transition);
  RankWorker rank1(kRanks[1], completions, StateDigest{0},
                   [&gate1] { gate1.Enter(); }, transition);
  std::array<RankWorker*, 2> workers{&rank0, &rank1};
  std::array<StepGate*, 2> gates{&gate0, &gate1};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                       .value();
  auto replica_gate = std::make_shared<ReplicaGate>();
  auto controller_result = Controller::Create(
      admission, replica_gate, completions,
      {ControllerRank{kRanks[0], [&rank0](RankCommand command) {
                        return rank0.Enqueue(std::move(command));
                      }},
       ControllerRank{kRanks[1], [&rank1](RankCommand command) {
                        return rank1.Enqueue(std::move(command));
                      }}},
      [](const GenerateRequest&, SequenceNumber) { return TokenId{13}; });
  ASSERT_TRUE(controller_result.ok());
  auto controller = std::move(controller_result).value();
  auto sink = std::make_shared<RecordingSink>();
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{91}, "mini-qwen38-hybrid",
                                           {TokenId{4}}, 1, 1000, 1},
                           sink)
                  .ok());

  gate0.WaitEntered(1);
  gate1.WaitEntered(1);
  gates[order[0]]->AllowNext();
  PumpUntil(*controller, [&] {
    return workers[order[0]]->transaction_count_for_testing() == 1;
  });
  controller->PollOnce();
  EXPECT_TRUE(sink->tokens.empty());
  gates[order[1]]->AllowNext();
  PumpUntil(*controller, [&] {
    return workers[order[1]]->transaction_count_for_testing() == 1;
  });
  PumpUntil(*controller, [&] {
    return gate0.entered() >= 2 && gate1.entered() >= 2;
  });

  gates[order[0]]->AllowNext();
  gates[order[1]]->AllowNext();
  PumpUntil(*controller, [&] {
    return gate0.entered() >= 3 && gate1.entered() >= 3;
  });
  EXPECT_EQ(rank0.canonical_digest(), StateDigest{0});
  EXPECT_EQ(rank1.canonical_digest(), StateDigest{0});

  gates[order[0]]->AllowNext();
  PumpUntil(*controller, [&] {
    return workers[order[0]]->canonical_digest() != StateDigest{0};
  });
  controller->PollOnce();
  EXPECT_TRUE(sink->tokens.empty());
  EXPECT_EQ(workers[order[1]]->canonical_digest(), StateDigest{0});

  gates[order[1]]->AllowNext();
  PumpUntil(*controller, [&] {
    return workers[order[1]]->canonical_digest() != StateDigest{0};
  });
  controller->PollOnce();
  ASSERT_EQ(sink->tokens.size(), 1U);
  ASSERT_EQ(sink->finishes.size(), 1U);
  EXPECT_TRUE(sink->errors.empty());
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  result.emplace(sink->tokens[0]);
}

TEST(ControllerSuccess, RealRankWorkersRespectBarrierAndStableDigest) {
  std::optional<TokenEvent> forward;
  std::optional<TokenEvent> reverse;
  RunRealWorkers({0, 1}, forward);
  RunRealWorkers({1, 0}, reverse);
  ASSERT_TRUE(forward.has_value());
  ASSERT_TRUE(reverse.has_value());
  EXPECT_EQ(*forward, *reverse);
}

TEST(ControllerSuccess, DuplicateAcksNeverRedispatchOrRepublish) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  ManualRank rank0{kRanks[0]};
  ManualRank rank1{kRanks[1]};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto created = Controller::Create(admission, gate, completions,
                                    {rank0.endpoint(), rank1.endpoint()});
  ASSERT_TRUE(created.ok());
  auto controller = std::move(created).value();
  auto sink = std::make_shared<RecordingSink>();
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{101}, "model", {TokenId{8}},
                                           1, 1000, 1},
                           sink)
                  .ok());
  const auto prepare = std::get<Prepare>(rank0.commands[0]);
  const auto push = [&](RankCompletion completion) {
    ASSERT_TRUE(completions.Push(std::move(completion)));
    controller->PollOnce();
  };
  push(PreparedAck{RankId{0}, prepare.epoch});
  push(PreparedAck{RankId{1}, prepare.epoch});
  ASSERT_EQ(rank0.commands.size(), 2U);
  push(PreparedAck{RankId{1}, prepare.epoch});
  EXPECT_EQ(rank0.commands.size(), 2U);
  EXPECT_EQ(rank1.commands.size(), 2U);

  const TokenId candidate = std::get<Execute>(rank0.commands[1]).candidate_token;
  push(ExecutedAck{RankId{0}, prepare.epoch, candidate, StateDigest{10}});
  push(ExecutedAck{RankId{1}, prepare.epoch, candidate, StateDigest{11}});
  ASSERT_EQ(rank0.commands.size(), 3U);
  push(ExecutedAck{RankId{1}, prepare.epoch, candidate, StateDigest{11}});
  EXPECT_EQ(rank0.commands.size(), 3U);
  EXPECT_EQ(rank1.commands.size(), 3U);

  push(CommittedAck{RankId{0}, prepare.epoch});
  push(CommittedAck{RankId{0}, prepare.epoch});
  EXPECT_TRUE(sink->tokens.empty());
  push(CommittedAck{RankId{1}, prepare.epoch});
  ASSERT_EQ(sink->tokens.size(), 1U);
  push(CommittedAck{RankId{1}, prepare.epoch});
  EXPECT_EQ(sink->tokens.size(), 1U);
  EXPECT_EQ(sink->finishes.size(), 1U);
}

TEST(ControllerSuccess, PartialCommitDispatchFailureMakesReplicaStickyFatal) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  std::vector<RankCommand> rank0_commands;
  std::vector<RankCommand> rank1_commands;
  std::size_t rank0_canonical_commits = 0;
  ControllerRank rank0{kRanks[0], [&](RankCommand command) {
                         if (std::holds_alternative<Commit>(command)) {
                           ++rank0_canonical_commits;
                         }
                         rank0_commands.push_back(std::move(command));
                         return Status::Ok();
                       }};
  ControllerRank rank1{kRanks[1], [&](RankCommand command) {
                         rank1_commands.push_back(command);
                         if (std::holds_alternative<Commit>(command)) {
                           return Status{Error::From(ErrorCode::kRankFailure,
                                                     "rank 1 commit rejected")};
                         }
                         return Status::Ok();
                       }};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions,
                                 {std::move(rank0), std::move(rank1)},
                                 [](const GenerateRequest&, SequenceNumber) {
                                   return TokenId{7};
                                 });
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink = std::make_shared<RecordingSink>();
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{110}, "model", {TokenId{1}},
                                           1, 1000, 1},
                           sink)
                  .ok());
  const auto prepare = std::get<Prepare>(rank0_commands[0]);
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(PreparedAck{rank, prepare.epoch}));
    controller->PollOnce();
  }
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(
        ExecutedAck{rank, prepare.epoch, TokenId{7}, StateDigest{50 + rank.value()}}));
    controller->PollOnce();
  }

  EXPECT_EQ(rank0_canonical_commits, 1U);
  EXPECT_EQ(gate->health(), ReplicaHealth::kFatal);
  EXPECT_TRUE(sink->tokens.empty());
  ASSERT_EQ(sink->errors.size(), 1U);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  const auto before0 = rank0_commands.size();
  const auto before1 = rank1_commands.size();
  auto next_sink = std::make_shared<RecordingSink>();
  const Status rejected = controller->Submit(
      GenerateRequest{RequestId{111}, "model", {TokenId{1}}, 1, 1000, 1},
      next_sink);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error()->code(), ErrorCode::kReplicaFatal);
  EXPECT_EQ(rank0_commands.size(), before0);
  EXPECT_EQ(rank1_commands.size(), before1);
  EXPECT_TRUE(next_sink->accepted.empty());
}

TEST(ControllerSuccess, PreDecisionDispatchFailureAbortsAndReleasesReservation) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  std::vector<RankCommand> rank0_commands;
  ControllerRank rank0{kRanks[0], [&](RankCommand command) {
                         rank0_commands.push_back(std::move(command));
                         return Status::Ok();
                       }};
  ControllerRank rank1{kRanks[1], [](RankCommand command) {
                         if (std::holds_alternative<Prepare>(command)) {
                           return Status{Error::From(ErrorCode::kRankFailure,
                                                     "prepare dispatch failed")};
                         }
                         return Status::Ok();
                       }};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions,
                                 {std::move(rank0), std::move(rank1)});
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink = std::make_shared<RecordingSink>();
  EXPECT_NO_THROW(EXPECT_TRUE(
      controller
          ->Submit(GenerateRequest{RequestId{120}, "model", {TokenId{1}}, 1,
                                   1000, 1},
                   sink)
          .ok()));
  ASSERT_EQ(sink->accepted.size(), 1U);
  ASSERT_EQ(sink->errors.size(), 1U);
  ASSERT_EQ(rank0_commands.size(), 2U);
  EXPECT_TRUE(std::holds_alternative<Prepare>(rank0_commands[0]));
  EXPECT_TRUE(std::holds_alternative<Abort>(rank0_commands[1]));
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  EXPECT_EQ(gate->health(), ReplicaHealth::kHealthy);
}

TEST(ControllerSuccess, ThrowingPlannerAndEnqueueBecomeInternalFailures) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  for (const bool throw_planner : {true, false}) {
    std::vector<RankCommand> rank0_commands;
    std::vector<RankCommand> rank1_commands;
    ControllerRank rank0{kRanks[0], [&](RankCommand command) {
                           rank0_commands.push_back(std::move(command));
                           return Status::Ok();
                         }};
    ControllerRank rank1{kRanks[1], [&](RankCommand command) -> Status {
                           rank1_commands.push_back(command);
                           if (!throw_planner &&
                               std::holds_alternative<Execute>(command)) {
                             throw std::runtime_error("execute enqueue threw");
                           }
                           return Status::Ok();
                         }};
    auto admission = AdmissionController::Create(
                         {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                         .value();
    auto gate = std::make_shared<ReplicaGate>();
    CompletionQueue<RankCompletion> completions;
    auto planner = [throw_planner](const GenerateRequest&, SequenceNumber) {
      if (throw_planner) throw std::runtime_error("planner threw");
      return TokenId{9};
    };
    auto made = Controller::Create(admission, gate, completions,
                                   {std::move(rank0), std::move(rank1)}, planner);
    ASSERT_TRUE(made.ok());
    auto controller = std::move(made).value();
    auto sink = std::make_shared<RecordingSink>();
    EXPECT_NO_THROW(EXPECT_TRUE(
        controller
            ->Submit(GenerateRequest{RequestId{static_cast<std::uint64_t>(
                                         130 + throw_planner)},
                                     "model",
                                     {TokenId{1}}, 1, 1000, 1},
                     sink)
            .ok()));
    const auto prepare = std::get<Prepare>(rank0_commands[0]);
    for (const RankId rank : kRanks) {
      ASSERT_TRUE(completions.Push(PreparedAck{rank, prepare.epoch}));
      EXPECT_NO_THROW(controller->PollOnce());
    }
    if (!throw_planner) {
      EXPECT_TRUE(std::any_of(rank0_commands.begin(), rank0_commands.end(),
                              [](const auto& command) {
                                return std::holds_alternative<Abort>(command);
                              }));
    }
    ASSERT_EQ(sink->accepted.size(), 1U);
    ASSERT_EQ(sink->errors.size(), 1U);
    EXPECT_EQ(sink->errors[0], ErrorCode::kInternal);
    EXPECT_EQ(admission.active_reservation_count(), 0U);
    EXPECT_EQ(gate->health(), ReplicaHealth::kHealthy);
  }
}

TEST(ControllerSuccess, EveryThrowingRankEnqueueBoundaryIsContained) {
  enum class ThrowStage { kPrepare, kExecute, kCommit };
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  for (const ThrowStage stage : {ThrowStage::kPrepare, ThrowStage::kExecute,
                                 ThrowStage::kCommit}) {
    std::vector<RankCommand> rank0_commands;
    std::vector<RankCommand> rank1_commands;
    const auto should_throw = [stage](const RankCommand& command) {
      return (stage == ThrowStage::kPrepare &&
              std::holds_alternative<Prepare>(command)) ||
             (stage == ThrowStage::kExecute &&
              std::holds_alternative<Execute>(command)) ||
             (stage == ThrowStage::kCommit &&
              std::holds_alternative<Commit>(command));
    };
    ControllerRank rank0{kRanks[0], [&](RankCommand command) {
                           rank0_commands.push_back(std::move(command));
                           return Status::Ok();
                         }};
    ControllerRank rank1{kRanks[1], [&](RankCommand command) -> Status {
                           rank1_commands.push_back(command);
                           if (should_throw(command)) {
                             throw std::runtime_error("rank enqueue threw");
                           }
                           return Status::Ok();
                         }};
    auto admission = AdmissionController::Create(
                         {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                         .value();
    auto gate = std::make_shared<ReplicaGate>();
    CompletionQueue<RankCompletion> completions;
    auto made = Controller::Create(admission, gate, completions,
                                   {std::move(rank0), std::move(rank1)},
                                   [](const GenerateRequest&, SequenceNumber) {
                                     return TokenId{12};
                                   });
    ASSERT_TRUE(made.ok());
    auto controller = std::move(made).value();
    auto sink = std::make_shared<RecordingSink>();
    EXPECT_NO_THROW(controller->Submit(
        GenerateRequest{RequestId{160 + static_cast<std::uint64_t>(stage)},
                        "model", {TokenId{1}}, 1, 1000, 1},
        sink));
    if (stage != ThrowStage::kPrepare) {
      const auto prepare = std::get<Prepare>(rank0_commands[0]);
      for (const RankId rank : kRanks) {
        ASSERT_TRUE(completions.Push(PreparedAck{rank, prepare.epoch}));
        EXPECT_NO_THROW(controller->PollOnce());
      }
      if (stage == ThrowStage::kCommit) {
        for (const RankId rank : kRanks) {
          ASSERT_TRUE(completions.Push(ExecutedAck{
              rank, prepare.epoch, TokenId{12}, StateDigest{80 + rank.value()}}));
          EXPECT_NO_THROW(controller->PollOnce());
        }
      }
    }
    ASSERT_EQ(sink->accepted.size(), 1U);
    ASSERT_EQ(sink->errors.size(), 1U);
    EXPECT_EQ(sink->errors[0], ErrorCode::kInternal);
    EXPECT_TRUE(sink->tokens.empty());
    EXPECT_EQ(admission.active_reservation_count(), 0U);
    EXPECT_EQ(gate->health(), stage == ThrowStage::kCommit
                                  ? ReplicaHealth::kFatal
                                  : ReplicaHealth::kHealthy);
  }
}

TEST(ControllerSuccess, NonOwnerCallsNeverMutateControllerState) {
  ManualRank rank0{RankId{0}};
  ManualRank rank1{RankId{1}};
  auto admission = AdmissionController::Create(
                       {RankCapacity{RankId{0}, 1}, RankCapacity{RankId{1}, 1}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions,
                                 {rank0.endpoint(), rank1.endpoint()});
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto foreign_sink = std::make_shared<RecordingSink>();
  std::optional<Status> foreign_submit;
  std::thread submitter([&] {
    foreign_submit.emplace(controller->Submit(
        GenerateRequest{RequestId{140}, "model", {TokenId{1}}, 1, 1000, 1},
        foreign_sink));
  });
  submitter.join();
  ASSERT_TRUE(foreign_submit.has_value());
  ASSERT_FALSE(foreign_submit->ok());
  EXPECT_EQ(foreign_submit->error()->code(), ErrorCode::kInternal);
  EXPECT_TRUE(rank0.commands.empty());
  EXPECT_TRUE(foreign_sink->accepted.empty());
  EXPECT_EQ(admission.active_reservation_count(), 0U);

  auto sink = std::make_shared<RecordingSink>();
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{141}, "model", {TokenId{1}},
                                           1, 1000, 1},
                           sink)
                  .ok());
  const auto prepare = std::get<Prepare>(rank0.commands[0]);
  ASSERT_TRUE(completions.Push(PreparedAck{RankId{0}, prepare.epoch}));
  std::optional<Status> foreign_cancel;
  std::optional<Status> foreign_detach;
  std::optional<Status> foreign_shutdown;
  std::thread foreign([&] {
    controller->PollOnce();
    foreign_cancel.emplace(
        controller->Cancel(RequestId{141}, CancelReason::kDeadline));
    foreign_detach.emplace(controller->DetachTransport(RequestId{141}));
    foreign_shutdown.emplace(controller->Shutdown());
  });
  foreign.join();
  ASSERT_FALSE(foreign_cancel->ok());
  ASSERT_FALSE(foreign_detach->ok());
  ASSERT_FALSE(foreign_shutdown->ok());
  EXPECT_EQ(foreign_cancel->error()->code(), ErrorCode::kInternal);
  EXPECT_EQ(foreign_detach->error()->code(), ErrorCode::kInternal);
  EXPECT_EQ(foreign_shutdown->error()->code(), ErrorCode::kInternal);
  EXPECT_EQ(rank0.commands.size(), 1U);
  controller->PollOnce();
  ASSERT_TRUE(completions.Push(PreparedAck{RankId{1}, prepare.epoch}));
  controller->PollOnce();
  EXPECT_EQ(rank0.commands.size(), 2U);
}

TEST(ControllerSuccess, TransportBackpressureAfterCommitDropsDeliveryOnly) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  ManualRank rank0{kRanks[0]};
  ManualRank rank1{kRanks[1]};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions,
                                 {rank0.endpoint(), rank1.endpoint()},
                                 [](const GenerateRequest&, SequenceNumber) {
                                   return TokenId{6};
                                 });
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink = std::make_shared<DroppingSink>();
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{150}, "model", {TokenId{1}},
                                           2, 1000, 1},
                           sink)
                  .ok());
  const auto prepare = std::get<Prepare>(rank0.commands[0]);
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(PreparedAck{rank, prepare.epoch}));
    controller->PollOnce();
  }
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(
        ExecutedAck{rank, prepare.epoch, TokenId{6}, StateDigest{70 + rank.value()}}));
    controller->PollOnce();
  }
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(CommittedAck{rank, prepare.epoch}));
    controller->PollOnce();
  }
  EXPECT_EQ(sink->token_attempts, 1U);
  EXPECT_EQ(sink->finish_attempts, 0U);
  EXPECT_EQ(sink->errors, 0U);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  EXPECT_EQ(gate->health(), ReplicaHealth::kHealthy);
  EXPECT_EQ(rank0.commands.size(), 3U);
  EXPECT_EQ(rank1.commands.size(), 3U);
}

TEST(ControllerSuccess, FinishedBackpressureDoesNotUndoPublishedToken) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  ManualRank rank0{kRanks[0]};
  ManualRank rank1{kRanks[1]};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions,
                                 {rank0.endpoint(), rank1.endpoint()},
                                 [](const GenerateRequest&, SequenceNumber) {
                                   return TokenId{16};
                                 });
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink = std::make_shared<DroppingSink>(false, true);
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{170}, "model", {TokenId{1}},
                                           1, 1000, 1},
                           sink)
                  .ok());
  const auto prepare = std::get<Prepare>(rank0.commands[0]);
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(PreparedAck{rank, prepare.epoch}));
    controller->PollOnce();
  }
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(ExecutedAck{
        rank, prepare.epoch, TokenId{16}, StateDigest{90 + rank.value()}}));
    controller->PollOnce();
  }
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(CommittedAck{rank, prepare.epoch}));
    controller->PollOnce();
  }
  EXPECT_EQ(sink->token_attempts, 1U);
  EXPECT_EQ(sink->finish_attempts, 1U);
  EXPECT_EQ(sink->errors, 0U);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  EXPECT_EQ(gate->health(), ReplicaHealth::kHealthy);
}

TEST(ControllerSuccess, StartupFailuresDrainUntilTheNextRunnablePendingRequest) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  std::vector<RankCommand> rank0_commands;
  std::vector<RankCommand> rank1_commands;
  ControllerRank rank0{kRanks[0], [&](RankCommand command) {
                         rank0_commands.push_back(std::move(command));
                         return Status::Ok();
                       }};
  ControllerRank rank1{kRanks[1], [&](RankCommand command) {
                         rank1_commands.push_back(command);
                         if (const auto* prepare = std::get_if<Prepare>(&command);
                             prepare != nullptr &&
                             prepare->transaction_id == TransactionId{2}) {
                           return Status{Error::From(ErrorCode::kRankFailure,
                                                     "request two rejected")};
                         }
                         return Status::Ok();
                       }};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 3}, RankCapacity{kRanks[1], 3}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions,
                                 {std::move(rank0), std::move(rank1)},
                                 [](const GenerateRequest&, SequenceNumber) {
                                   return TokenId{23};
                                 });
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink1 = std::make_shared<RecordingSink>();
  auto sink2 = std::make_shared<RecordingSink>();
  auto sink3 = std::make_shared<RecordingSink>();
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{201}, "model", {TokenId{1}},
                                           1, 1000, 1},
                           sink1)
                  .ok());
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{202}, "model", {TokenId{1}},
                                           1, 1000, 1},
                           sink2)
                  .ok());
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{203}, "model", {TokenId{1}},
                                           1, 1000, 1},
                           sink3)
                  .ok());
  const auto first = std::get<Prepare>(rank0_commands[0]);
  ASSERT_TRUE(completions.Push(RankError{
      RankId{0}, first.transaction_id, first.epoch,
      Error::From(ErrorCode::kRankFailure, "request one failed")}));
  controller->PollOnce();

  ASSERT_EQ(sink1->errors.size(), 1U);
  ASSERT_EQ(sink2->errors.size(), 1U);
  EXPECT_TRUE(sink3->errors.empty());
  const auto was_abort_for_two = [](const RankCommand& command) {
    const auto* abort = std::get_if<Abort>(&command);
    return abort != nullptr && abort->transaction_id == TransactionId{2};
  };
  EXPECT_TRUE(std::any_of(rank0_commands.begin(), rank0_commands.end(),
                          was_abort_for_two));
  EXPECT_TRUE(std::any_of(rank1_commands.begin(), rank1_commands.end(),
                          was_abort_for_two));
  const auto prepare3 = std::find_if(
      rank0_commands.begin(), rank0_commands.end(), [](const RankCommand& command) {
        const auto* prepare = std::get_if<Prepare>(&command);
        return prepare != nullptr && prepare->transaction_id == TransactionId{3};
      });
  ASSERT_NE(prepare3, rank0_commands.end());
  const auto third = std::get<Prepare>(*prepare3);
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(PreparedAck{rank, third.epoch}));
    controller->PollOnce();
  }
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(ExecutedAck{
        rank, third.epoch, TokenId{23}, StateDigest{300 + rank.value()}}));
    controller->PollOnce();
  }
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(CommittedAck{rank, third.epoch}));
    controller->PollOnce();
  }
  EXPECT_EQ(sink3->tokens.size(), 1U);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  EXPECT_EQ(gate->health(), ReplicaHealth::kHealthy);
}

TEST(ControllerSuccess, AcceptedStartupAllocationFailureIsContainedAndDrained) {
  ManualRank rank0{RankId{0}};
  ManualRank rank1{RankId{1}};
  auto admission = AdmissionController::Create(
                       {RankCapacity{RankId{0}, 2}, RankCapacity{RankId{1}, 2}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  std::size_t starts = 0;
  Controller::StartupHook startup = [&](const GenerateRequest&) {
    if (starts++ == 0) throw std::bad_alloc{};
  };
  auto made = Controller::Create(admission, gate, completions,
                                 {rank0.endpoint(), rank1.endpoint()}, {}, {},
                                 std::move(startup));
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto sink1 = std::make_shared<RecordingSink>();
  auto sink2 = std::make_shared<RecordingSink>();
  EXPECT_NO_THROW(controller->Submit(
      GenerateRequest{RequestId{210}, "model", {TokenId{1}}, 1, 1000, 1},
      sink1));
  EXPECT_EQ(sink1->accepted.size(), 1U);
  EXPECT_EQ(sink1->errors.size(), 1U);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{211}, "model", {TokenId{1}},
                                           1, 1000, 1},
                           sink2)
                  .ok());
  EXPECT_EQ(rank0.commands.size(), 1U);
  EXPECT_EQ(sink2->accepted.size(), 1U);
}

TEST(ControllerSuccess, ShutdownIsIdempotentAndTerminatesActiveAndPending) {
  ManualRank rank0{RankId{0}};
  ManualRank rank1{RankId{1}};
  auto admission = AdmissionController::Create(
                       {RankCapacity{RankId{0}, 2}, RankCapacity{RankId{1}, 2}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions,
                                 {rank0.endpoint(), rank1.endpoint()});
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto active_sink = std::make_shared<RecordingSink>();
  auto pending_sink = std::make_shared<RecordingSink>();
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{220}, "model", {TokenId{1}},
                                           1, 1000, 1},
                           active_sink)
                  .ok());
  ASSERT_TRUE(controller
                  ->Submit(GenerateRequest{RequestId{221}, "model", {TokenId{1}},
                                           1, 1000, 1},
                           pending_sink)
                  .ok());
  ASSERT_TRUE(controller->Shutdown().ok());
  const auto command_count = rank0.commands.size() + rank1.commands.size();
  EXPECT_EQ(active_sink->errors.size(), 1U);
  EXPECT_EQ(pending_sink->errors.size(), 1U);
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  EXPECT_EQ(gate->health(), ReplicaHealth::kHealthy);
  ASSERT_TRUE(controller->Shutdown().ok());
  EXPECT_EQ(rank0.commands.size() + rank1.commands.size(), command_count);
}

TEST(ControllerSuccess, ShutdownAndDestructorMakeUnconfirmedCommitFatal) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  for (const bool explicit_shutdown : {true, false}) {
    ManualRank rank0{kRanks[0]};
    ManualRank rank1{kRanks[1]};
    auto admission = AdmissionController::Create(
                         {RankCapacity{kRanks[0], 1}, RankCapacity{kRanks[1], 1}})
                         .value();
    auto gate = std::make_shared<ReplicaGate>();
    CompletionQueue<RankCompletion> completions;
    auto made = Controller::Create(admission, gate, completions,
                                   {rank0.endpoint(), rank1.endpoint()},
                                   [](const GenerateRequest&, SequenceNumber) {
                                     return TokenId{27};
                                   });
    ASSERT_TRUE(made.ok());
    auto controller = std::move(made).value();
    auto sink = std::make_shared<RecordingSink>();
    ASSERT_TRUE(controller
                    ->Submit(GenerateRequest{RequestId{230}, "model", {TokenId{1}},
                                             1, 1000, 1},
                             sink)
                    .ok());
    const auto prepare = std::get<Prepare>(rank0.commands[0]);
    for (const RankId rank : kRanks) {
      ASSERT_TRUE(completions.Push(PreparedAck{rank, prepare.epoch}));
      controller->PollOnce();
    }
    for (const RankId rank : kRanks) {
      ASSERT_TRUE(completions.Push(ExecutedAck{
          rank, prepare.epoch, TokenId{27}, StateDigest{400 + rank.value()}}));
      controller->PollOnce();
    }
    ASSERT_EQ(rank0.commands.size(), 3U);
    if (explicit_shutdown) {
      EXPECT_TRUE(controller->Shutdown().ok());
    } else {
      std::thread foreign_destructor(
          [owned = std::move(controller)]() mutable { owned.reset(); });
      foreign_destructor.join();
    }
    EXPECT_EQ(gate->health(), ReplicaHealth::kFatal);
    EXPECT_TRUE(sink->tokens.empty());
    EXPECT_EQ(sink->errors.size(), 1U);
    EXPECT_EQ(admission.active_reservation_count(), 0U);
  }
}

TEST(ControllerSuccess, PreDecisionDestructorBestEffortAborts) {
  ManualRank rank0{RankId{0}};
  ManualRank rank1{RankId{1}};
  auto admission = AdmissionController::Create(
                       {RankCapacity{RankId{0}, 1}, RankCapacity{RankId{1}, 1}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  {
    auto made = Controller::Create(admission, gate, completions,
                                   {rank0.endpoint(), rank1.endpoint()});
    ASSERT_TRUE(made.ok());
    auto controller = std::move(made).value();
    auto sink = std::make_shared<RecordingSink>();
    ASSERT_TRUE(controller
                    ->Submit(GenerateRequest{RequestId{240}, "model", {TokenId{1}},
                                             1, 1000, 1},
                             sink)
                    .ok());
  }
  EXPECT_TRUE(std::holds_alternative<Abort>(rank0.commands.back()));
  EXPECT_TRUE(std::holds_alternative<Abort>(rank1.commands.back()));
  EXPECT_EQ(admission.active_reservation_count(), 0U);
  EXPECT_EQ(gate->health(), ReplicaHealth::kHealthy);
}

TEST(ControllerSuccess, RuntimeFatalKeepsRootCauseAndScopesEveryPendingError) {
  constexpr std::array kRanks{RankId{0}, RankId{1}};
  std::vector<RankCommand> rank0_commands;
  std::vector<RankCommand> rank1_commands;
  ControllerRank rank0{kRanks[0], [&](RankCommand command) {
                         rank0_commands.push_back(std::move(command));
                         return Status::Ok();
                       }};
  ControllerRank rank1{kRanks[1], [&](RankCommand command) {
                         rank1_commands.push_back(command);
                         if (std::holds_alternative<Commit>(command)) {
                           return Status{Error::From(
                               ErrorCode::kRankFailure,
                               "rank 1 lost after commit decision")};
                         }
                         return Status::Ok();
                       }};
  auto admission = AdmissionController::Create(
                       {RankCapacity{kRanks[0], 3}, RankCapacity{kRanks[1], 3}})
                       .value();
  auto gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  auto made = Controller::Create(admission, gate, completions,
                                 {std::move(rank0), std::move(rank1)},
                                 [](const GenerateRequest&, SequenceNumber) {
                                   return TokenId{29};
                                 });
  ASSERT_TRUE(made.ok());
  auto controller = std::move(made).value();
  auto active = std::make_shared<RecordingSink>();
  auto pending1 = std::make_shared<RecordingSink>();
  auto pending2 = std::make_shared<RecordingSink>();
  for (const auto& [id, sink] :
       std::array<std::pair<RequestId, std::shared_ptr<RecordingSink>>, 3>{
           std::pair{RequestId{301}, active},
           std::pair{RequestId{302}, pending1},
           std::pair{RequestId{303}, pending2}}) {
    ASSERT_TRUE(controller
                    ->Submit(GenerateRequest{id, "model", {TokenId{1}}, 1,
                                             1000, 1},
                             sink)
                    .ok());
  }
  const auto prepare = std::get<Prepare>(rank0_commands[0]);
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(PreparedAck{rank, prepare.epoch}));
    controller->PollOnce();
  }
  for (const RankId rank : kRanks) {
    ASSERT_TRUE(completions.Push(ExecutedAck{
        rank, prepare.epoch, TokenId{29}, StateDigest{500 + rank.value()}}));
    controller->PollOnce();
  }
  ASSERT_EQ(gate->health(), ReplicaHealth::kFatal);
  ASSERT_EQ(pending1->full_errors.size(), 1U);
  ASSERT_EQ(pending2->full_errors.size(), 1U);
  for (const auto& [expected_request, error] :
       std::array<std::pair<RequestId, const Error*>, 2>{
           std::pair{RequestId{302}, &pending1->full_errors[0]},
           std::pair{RequestId{303}, &pending2->full_errors[0]}}) {
    EXPECT_EQ(error->code(), ErrorCode::kReplicaFatal);
    EXPECT_EQ(error->request_id(), std::optional{expected_request});
    EXPECT_EQ(error->message(), "rank 1 lost after commit decision");
    EXPECT_EQ(error->message().find("shutdown"), std::string::npos);
  }
  EXPECT_EQ(admission.active_reservation_count(), 0U);
}

}  // namespace
