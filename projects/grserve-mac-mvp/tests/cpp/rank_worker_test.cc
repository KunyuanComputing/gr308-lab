#include <barrier>
#include <future>
#include <memory>
#include <new>
#include <thread>
#include <variant>

#include <gtest/gtest.h>

#include "engine/completion_queue.h"
#include "engine/rank_worker.h"

namespace {

using grserve::Commit;
using grserve::Abort;
using grserve::CompletionQueue;
using grserve::Epoch;
using grserve::Execute;
using grserve::Prepare;
using grserve::RankCompletion;
using grserve::RankId;
using grserve::RankWorker;
using grserve::StateDigest;
using grserve::TokenId;
using grserve::TransactionId;
using grserve::PreparedAck;
using grserve::ExecutedAck;
using grserve::CommittedAck;
using grserve::AbortedAck;
using grserve::RankError;
using grserve::ErrorCode;

RankCompletion Next(CompletionQueue<RankCompletion>& queue) {
  auto value = queue.WaitPop();
  EXPECT_TRUE(value.has_value());
  return std::move(*value);
}

TEST(RankWorker, KeepsCanonicalStateHiddenUntilCommit) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{0}, completions);
  const auto before = worker.canonical_digest();

  ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{7}, RankId{0}, Epoch{3}}).ok());
  const auto prepared = std::get<PreparedAck>(Next(completions));
  EXPECT_EQ(prepared.rank_id, RankId{0});
  EXPECT_EQ(prepared.epoch, Epoch{3});
  ASSERT_TRUE(worker.Enqueue(
      Execute{TransactionId{7}, RankId{0}, Epoch{3}, TokenId{19}}).ok());
  const auto executed = Next(completions);
  ASSERT_TRUE(std::holds_alternative<ExecutedAck>(executed));
  EXPECT_EQ(std::get<ExecutedAck>(executed).candidate_token, TokenId{19});
  const auto candidate_digest = std::get<ExecutedAck>(executed).state_digest;
  EXPECT_EQ(worker.canonical_digest(), before);

  ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{7}, RankId{0}, Epoch{3}}).ok());
  const auto committed = std::get<CommittedAck>(Next(completions));
  EXPECT_EQ(committed.rank_id, RankId{0});
  EXPECT_EQ(committed.epoch, Epoch{3});
  EXPECT_NE(worker.canonical_digest(), before);
  EXPECT_EQ(worker.canonical_digest(), candidate_digest);
}

TEST(RankWorker, SequentialCommitsAdvanceCanonicalDigestPerTransaction) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{1}, completions);
  StateDigest previous = worker.canonical_digest();
  for (const auto transaction : {TransactionId{20}, TransactionId{21}}) {
    ASSERT_TRUE(worker.Enqueue(Prepare{transaction, RankId{1}, Epoch{1}}).ok());
    Next(completions);
    ASSERT_TRUE(worker.Enqueue(Execute{transaction, RankId{1}, Epoch{1}, TokenId{7}}).ok());
    const auto executed = std::get<ExecutedAck>(Next(completions));
    ASSERT_TRUE(worker.Enqueue(Commit{transaction, RankId{1}, Epoch{1}}).ok());
    Next(completions);
    EXPECT_NE(worker.canonical_digest(), previous);
    EXPECT_EQ(worker.canonical_digest(), executed.state_digest);
    previous = worker.canonical_digest();
  }
}

TEST(RankWorker, StaleExecutedShadowCannotOverwriteNewerCanonicalDigest) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{1}, completions);
  const auto prepare_execute = [&](TransactionId transaction, TokenId token) -> ExecutedAck {
    EXPECT_TRUE(worker.Enqueue(Prepare{transaction, RankId{1}, Epoch{1}}).ok());
    Next(completions);
    EXPECT_TRUE(worker.Enqueue(Execute{transaction, RankId{1}, Epoch{1}, token}).ok());
    return std::get<ExecutedAck>(Next(completions));
  };
  const auto first = prepare_execute(TransactionId{100}, TokenId{1});
  const auto stale = prepare_execute(TransactionId{101}, TokenId{2});
  ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{100}, RankId{1}, Epoch{1}}).ok());
  EXPECT_EQ(std::get<CommittedAck>(Next(completions)).rank_id, RankId{1});
  EXPECT_EQ(worker.canonical_digest(), first.state_digest);
  ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{101}, RankId{1}, Epoch{1}}).ok());
  const auto rejected = Next(completions);
  ASSERT_TRUE(std::holds_alternative<RankError>(rejected));
  EXPECT_EQ(std::get<RankError>(rejected).error.code(), ErrorCode::kInternal);
  EXPECT_EQ(worker.canonical_digest(), first.state_digest);
  EXPECT_NE(first.state_digest, stale.state_digest);
}

TEST(RankWorker, EvictedTerminalTransactionCannotBeResurrected) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{2}, completions);
  for (std::uint64_t id = 1; id <= 257; ++id) {
    ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{id}, RankId{2}, Epoch{1}}).ok());
    Next(completions);
    ASSERT_TRUE(worker.Enqueue(Execute{TransactionId{id}, RankId{2}, Epoch{1}, TokenId{1}}).ok());
    Next(completions);
    ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{id}, RankId{2}, Epoch{1}}).ok());
    Next(completions);
  }
  const auto before = worker.canonical_digest();
  ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{1}, RankId{2}, Epoch{1}}).ok());
  const auto late = Next(completions);
  ASSERT_TRUE(std::holds_alternative<RankError>(late));
  EXPECT_EQ(std::get<RankError>(late).error.code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(worker.canonical_digest(), before);
  ASSERT_TRUE(worker.Enqueue(Execute{TransactionId{1}, RankId{2}, Epoch{1}, TokenId{1}}).ok());
  EXPECT_TRUE(std::holds_alternative<RankError>(Next(completions)));
  EXPECT_EQ(worker.canonical_digest(), before);
  ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{1}, RankId{2}, Epoch{1}}).ok());
  EXPECT_TRUE(std::holds_alternative<RankError>(Next(completions)));
  EXPECT_EQ(worker.canonical_digest(), before);
}

TEST(RankWorker, FatalPlanningFailureClosesOutputAndReleasesProducer) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{8}, completions, StateDigest{0}, [] {
    throw std::bad_alloc{};
  });
  ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{80}, RankId{8}, Epoch{1}}).ok());
  EXPECT_FALSE(completions.WaitPop().has_value());
  worker.Shutdown();
  EXPECT_FALSE(worker.Enqueue(Prepare{TransactionId{81}, RankId{8}, Epoch{1}}).ok());
}

TEST(RankWorker, FatalAbortUnblocksSharedQueueWhileAnotherRankIsIdle) {
  CompletionQueue<RankCompletion> completions;
  RankWorker failing(RankId{9}, completions, StateDigest{0}, [] {
    throw std::bad_alloc{};
  });
  RankWorker idle(RankId{10}, completions);
  ASSERT_TRUE(failing.Enqueue(Prepare{TransactionId{90}, RankId{9}, Epoch{1}}).ok());
  EXPECT_FALSE(completions.WaitPop().has_value());
  EXPECT_TRUE(completions.aborted());
  failing.Shutdown();
  idle.Shutdown();
}

TEST(RankWorker, FirstTransactionIdsAreMonotonicPerEpochAndReuseOnRollover) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{11}, completions);
  ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{2}, RankId{11}, Epoch{1}}).ok());
  Next(completions);
  ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{1}, RankId{11}, Epoch{1}}).ok());
  const auto nonmonotonic = Next(completions);
  ASSERT_TRUE(std::holds_alternative<RankError>(nonmonotonic));
  EXPECT_EQ(std::get<RankError>(nonmonotonic).error.code(), ErrorCode::kInvalidArgument);
  ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{1}, RankId{11}, Epoch{2}}).ok());
  EXPECT_TRUE(std::holds_alternative<PreparedAck>(Next(completions)));
  ASSERT_TRUE(worker.Enqueue(Execute{TransactionId{2}, RankId{11}, Epoch{1}, TokenId{1}}).ok());
  const auto old_epoch = Next(completions);
  ASSERT_TRUE(std::holds_alternative<RankError>(old_epoch));
  EXPECT_EQ(std::get<RankError>(old_epoch).error.code(), ErrorCode::kIncompatibleVersion);
}

TEST(RankWorker, CanonicalVersionRejectsStaleShadowEvenWhenDigestReturnsToBase) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(
      RankId{12}, completions, StateDigest{7}, {},
      [](StateDigest, RankId, TransactionId, Epoch, TokenId) { return StateDigest{7}; });
  const auto prepare_execute = [&](TransactionId transaction) -> ExecutedAck {
    EXPECT_TRUE(worker.Enqueue(Prepare{transaction, RankId{12}, Epoch{1}}).ok());
    Next(completions);
    EXPECT_TRUE(worker.Enqueue(Execute{transaction, RankId{12}, Epoch{1}, TokenId{1}}).ok());
    return std::get<ExecutedAck>(Next(completions));
  };
  const auto first = prepare_execute(TransactionId{1});
  const auto stale = prepare_execute(TransactionId{2});
  EXPECT_EQ(first.state_digest, StateDigest{7});
  EXPECT_EQ(stale.state_digest, StateDigest{7});
  ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{1}, RankId{12}, Epoch{1}}).ok());
  Next(completions);
  EXPECT_EQ(worker.canonical_digest(), StateDigest{7});
  ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{2}, RankId{12}, Epoch{1}}).ok());
  const auto rejected = Next(completions);
  ASSERT_TRUE(std::holds_alternative<RankError>(rejected));
  EXPECT_EQ(std::get<RankError>(rejected).error.code(), ErrorCode::kInternal);
}

TEST(RankWorker, RolloverClearsOldGenerationStateAndReusesTransactionIds) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{13}, completions);
  for (std::uint64_t epoch = 1; epoch <= 5; ++epoch) {
    ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{1}, RankId{13}, Epoch{epoch}}).ok());
    EXPECT_TRUE(std::holds_alternative<PreparedAck>(Next(completions)));
    EXPECT_EQ(worker.transaction_count_for_testing(), 1U);
  }
  const auto before = worker.canonical_digest();
  ASSERT_TRUE(worker.Enqueue(Abort{TransactionId{1}, RankId{13}, Epoch{4}}).ok());
  const auto old_epoch = Next(completions);
  ASSERT_TRUE(std::holds_alternative<RankError>(old_epoch));
  EXPECT_EQ(worker.canonical_digest(), before);
  ASSERT_TRUE(worker.Enqueue(Execute{TransactionId{1}, RankId{13}, Epoch{5}, TokenId{1}}).ok());
  const auto executed = std::get<ExecutedAck>(Next(completions));
  ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{1}, RankId{13}, Epoch{5}}).ok());
  EXPECT_TRUE(std::holds_alternative<CommittedAck>(Next(completions)));
  EXPECT_EQ(worker.canonical_digest(), executed.state_digest);
  EXPECT_EQ(worker.transaction_count_for_testing(), 1U);
}

TEST(CompletionQueue, DeliversFifoThenClosesAndWakesWaiter) {
  CompletionQueue<int> queue;
  ASSERT_TRUE(queue.Push(1));
  ASSERT_TRUE(queue.Emplace(2));
  EXPECT_EQ(queue.WaitPop(), 1);
  EXPECT_EQ(queue.WaitPop(), 2);
  std::promise<std::optional<int>> result;
  auto ready = result.get_future();
  std::thread waiter([&] { result.set_value(queue.WaitPop()); });
  queue.Close();
  EXPECT_FALSE(ready.get().has_value());
  waiter.join();
  EXPECT_FALSE(queue.Push(3));
  queue.Close();
}

TEST(CompletionQueue, RejectsAtHardCapacityWithoutGrowingAndDrainsToZero) {
  CompletionQueue<int> queue(3);
  EXPECT_EQ(queue.capacity(), 3U);
  ASSERT_TRUE(queue.Push(1));
  ASSERT_TRUE(queue.Push(2));
  ASSERT_TRUE(queue.Push(3));
  EXPECT_EQ(queue.size(), 3U);
  EXPECT_FALSE(queue.Push(4));
  EXPECT_EQ(queue.size(), 3U);
  EXPECT_EQ(queue.TryPop(), 1);
  EXPECT_EQ(queue.TryPop(), 2);
  EXPECT_EQ(queue.TryPop(), 3);
  EXPECT_FALSE(queue.TryPop().has_value());
  EXPECT_EQ(queue.size(), 0U);
}

TEST(CompletionQueue, LongProducerSequenceNeverExceedsBoundAndFullyDrains) {
  CompletionQueue<int> queue(17);
  for (int value = 0; value < 10000; ++value) {
    if (!queue.Push(value)) {
      ASSERT_TRUE(queue.TryPop().has_value());
      ASSERT_TRUE(queue.Push(value));
    }
    ASSERT_LE(queue.size(), queue.capacity());
  }
  while (queue.TryPop().has_value()) {}
  EXPECT_EQ(queue.size(), 0U);
}

TEST(CompletionQueue, RegisteredProducerDrainsThroughCloseWithMoveOnlyValues) {
  CompletionQueue<std::unique_ptr<int>> queue;
  auto producer = queue.RegisterProducer();
  ASSERT_TRUE(producer.active());
  queue.Close();
  EXPECT_FALSE(queue.Push(std::make_unique<int>(9)));
  ASSERT_TRUE(producer.Push(std::make_unique<int>(1)));
  ASSERT_TRUE(producer.Push(std::make_unique<int>(2)));
  EXPECT_EQ(*queue.WaitPop().value(), 1);
  EXPECT_EQ(*queue.WaitPop().value(), 2);
  producer.Release();
  EXPECT_FALSE(queue.WaitPop().has_value());
}

TEST(CompletionQueue, ClosedEmptyQueueWaitsForRegisteredProducerDrain) {
  CompletionQueue<int> queue;
  auto producer = queue.RegisterProducer();
  ASSERT_TRUE(producer.active());
  queue.Close();

  const auto wait_generation = queue.wait_generation_for_testing();
  std::promise<std::optional<int>> result;
  auto result_ready = result.get_future();
  std::thread consumer([&] {
    result.set_value(queue.WaitPop());
  });
  queue.WaitForConsumerWaitForTesting(wait_generation);
  ASSERT_TRUE(producer.Push(42));
  const auto received = result_ready.get();
  consumer.join();
  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(*received, 42);
  producer.Release();
  EXPECT_FALSE(queue.WaitPop().has_value());
}

TEST(RankWorker, ClosedOutputStillDrainsAcceptedCommandsBeforeProducerRelease) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{6}, completions);
  ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{60}, RankId{6}, Epoch{1}}).ok());
  ASSERT_TRUE(worker.Enqueue(Execute{TransactionId{60}, RankId{6}, Epoch{1}, TokenId{3}}).ok());
  ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{60}, RankId{6}, Epoch{1}}).ok());
  std::barrier close_barrier(2);
  std::thread closer([&] {
    close_barrier.arrive_and_wait();
    completions.Close();
  });
  close_barrier.arrive_and_wait();
  closer.join();
  EXPECT_FALSE(worker.Enqueue(Prepare{TransactionId{61}, RankId{6}, Epoch{1}}).ok());
  worker.Shutdown();
  const auto prepared = std::get<PreparedAck>(Next(completions));
  const auto executed = std::get<ExecutedAck>(Next(completions));
  const auto committed = std::get<CommittedAck>(Next(completions));
  EXPECT_EQ(prepared.rank_id, RankId{6});
  EXPECT_EQ(prepared.epoch, Epoch{1});
  EXPECT_EQ(executed.rank_id, RankId{6});
  EXPECT_EQ(executed.epoch, Epoch{1});
  EXPECT_EQ(executed.candidate_token, TokenId{3});
  EXPECT_EQ(committed.rank_id, RankId{6});
  EXPECT_EQ(committed.epoch, Epoch{1});
  EXPECT_EQ(worker.canonical_digest(), executed.state_digest);
  EXPECT_FALSE(completions.WaitPop().has_value());
}

TEST(RankWorker, ReplaysIdenticalCommandsAndRejectsConflictingExecute) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{2}, completions);
  const Prepare prepare{TransactionId{8}, RankId{2}, Epoch{4}};
  const Execute execute{TransactionId{8}, RankId{2}, Epoch{4}, TokenId{21}};
  const Commit commit{TransactionId{8}, RankId{2}, Epoch{4}};
  ASSERT_TRUE(worker.Enqueue(prepare).ok());
  EXPECT_TRUE(std::holds_alternative<PreparedAck>(Next(completions)));
  ASSERT_TRUE(worker.Enqueue(prepare).ok());
  EXPECT_TRUE(std::holds_alternative<PreparedAck>(Next(completions)));
  ASSERT_TRUE(worker.Enqueue(execute).ok());
  const auto first_execute = std::get<ExecutedAck>(Next(completions));
  ASSERT_TRUE(worker.Enqueue(execute).ok());
  EXPECT_EQ(std::get<ExecutedAck>(Next(completions)).state_digest, first_execute.state_digest);
  ASSERT_TRUE(worker.Enqueue(Execute{TransactionId{8}, RankId{2}, Epoch{4}, TokenId{22}}).ok());
  const auto conflict = Next(completions);
  ASSERT_TRUE(std::holds_alternative<RankError>(conflict));
  const auto& conflict_error = std::get<RankError>(conflict);
  EXPECT_EQ(conflict_error.rank_id, RankId{2});
  EXPECT_EQ(conflict_error.transaction_id, TransactionId{8});
  EXPECT_EQ(conflict_error.epoch, Epoch{4});
  EXPECT_EQ(conflict_error.error.code(), ErrorCode::kInternal);
  const auto before_commit = worker.canonical_digest();
  ASSERT_TRUE(worker.Enqueue(commit).ok());
  EXPECT_TRUE(std::holds_alternative<CommittedAck>(Next(completions)));
  const auto committed = worker.canonical_digest();
  EXPECT_NE(committed, before_commit);
  ASSERT_TRUE(worker.Enqueue(commit).ok());
  EXPECT_TRUE(std::holds_alternative<CommittedAck>(Next(completions)));
  EXPECT_EQ(worker.canonical_digest(), committed);
}

TEST(RankWorker, PlannerFailureIsRecoverableAndDoesNotReportRankLoss) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(
      RankId{2}, completions, StateDigest{0}, {}, {},
      [](const Execute& execute) -> grserve::StatusOr<TokenId> {
        return grserve::StatusOr<TokenId>(grserve::Error::From(
            ErrorCode::kRankFailure, "one execute failed", std::nullopt,
            execute.transaction_id));
      });
  ASSERT_TRUE(worker.Enqueue(
      Prepare{TransactionId{80}, RankId{2}, Epoch{4}}).ok());
  EXPECT_TRUE(std::holds_alternative<PreparedAck>(Next(completions)));
  ASSERT_TRUE(worker.Enqueue(Execute{TransactionId{80}, RankId{2}, Epoch{4},
                                    TokenId{0}, {TokenId{1}},
                                    grserve::SequenceNumber{1}}).ok());
  const auto completion = Next(completions);
  ASSERT_TRUE(std::holds_alternative<RankError>(completion));
  EXPECT_EQ(std::get<RankError>(completion).error.code(),
            ErrorCode::kRankFailure);
  EXPECT_FALSE(std::get<RankError>(completion).rank_lost);
}

TEST(RankWorker, PlannerCanExplicitlyReportLostRank) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(
      RankId{2}, completions, StateDigest{0}, {}, {},
      [](const Execute& execute) -> RankWorker::ExecuteResult {
        return RankWorker::ExecuteResult{
            grserve::StatusOr<TokenId>(grserve::Error::From(
                ErrorCode::kRankFailure, "rank transport lost", std::nullopt,
                execute.transaction_id)),
            true};
      });
  ASSERT_TRUE(worker.Enqueue(
      Prepare{TransactionId{82}, RankId{2}, Epoch{4}}).ok());
  Next(completions);
  ASSERT_TRUE(worker.Enqueue(Execute{TransactionId{82}, RankId{2}, Epoch{4},
                                    TokenId{0}, {TokenId{1}},
                                    grserve::SequenceNumber{1}}).ok());
  const auto completion = Next(completions);
  ASSERT_TRUE(std::holds_alternative<RankError>(completion));
  EXPECT_TRUE(std::get<RankError>(completion).rank_lost);
}

TEST(RankWorker, PlannerReplayRequiresIdenticalExecutePayload) {
  CompletionQueue<RankCompletion> completions;
  int calls = 0;
  RankWorker worker(
      RankId{2}, completions, StateDigest{0}, {}, {},
      [&calls](const Execute&) -> grserve::StatusOr<TokenId> {
        ++calls;
        return grserve::StatusOr<TokenId>(TokenId{9});
      });
  const Prepare prepare{TransactionId{81}, RankId{2}, Epoch{4}};
  const Execute execute{TransactionId{81}, RankId{2}, Epoch{4}, TokenId{0},
                        {TokenId{1}, TokenId{4}},
                        grserve::SequenceNumber{1}};
  ASSERT_TRUE(worker.Enqueue(prepare).ok());
  Next(completions);
  ASSERT_TRUE(worker.Enqueue(execute).ok());
  EXPECT_TRUE(std::holds_alternative<ExecutedAck>(Next(completions)));
  ASSERT_TRUE(worker.Enqueue(execute).ok());
  EXPECT_TRUE(std::holds_alternative<ExecutedAck>(Next(completions)));
  EXPECT_EQ(calls, 1);

  auto conflicting = execute;
  conflicting.context_token_ids.push_back(TokenId{7});
  conflicting.sequence_number = grserve::SequenceNumber{2};
  ASSERT_TRUE(worker.Enqueue(std::move(conflicting)).ok());
  const auto conflict = Next(completions);
  ASSERT_TRUE(std::holds_alternative<RankError>(conflict));
  EXPECT_EQ(std::get<RankError>(conflict).error.code(), ErrorCode::kInternal);
  EXPECT_EQ(calls, 1);
}

TEST(RankWorker, AbortAndEpochRolloverNeverMutateCanonicalState) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{3}, completions);
  const auto before = worker.canonical_digest();
  ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{9}, RankId{3}, Epoch{1}}).ok());
  Next(completions);
  ASSERT_TRUE(worker.Enqueue(Abort{TransactionId{9}, RankId{3}, Epoch{1}}).ok());
  const auto aborted = std::get<AbortedAck>(Next(completions));
  EXPECT_EQ(aborted.rank_id, RankId{3});
  EXPECT_EQ(aborted.epoch, Epoch{1});
  ASSERT_TRUE(worker.Enqueue(Abort{TransactionId{9}, RankId{3}, Epoch{1}}).ok());
  const auto replayed_abort = std::get<AbortedAck>(Next(completions));
  EXPECT_EQ(replayed_abort.rank_id, RankId{3});
  EXPECT_EQ(replayed_abort.epoch, Epoch{1});
  ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{9}, RankId{3}, Epoch{1}}).ok());
  EXPECT_TRUE(std::holds_alternative<RankError>(Next(completions)));
  ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{9}, RankId{3}, Epoch{2}}).ok());
  const auto rollover = Next(completions);
  EXPECT_TRUE(std::holds_alternative<PreparedAck>(rollover));
  EXPECT_EQ(worker.canonical_digest(), before);
}

TEST(RankWorker, ShutdownDrainsAcceptedCommandsAndRejectsNewOnes) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{4}, completions);
  ASSERT_TRUE(worker.Enqueue(Prepare{TransactionId{10}, RankId{4}, Epoch{1}}).ok());
  ASSERT_TRUE(worker.Enqueue(Execute{TransactionId{10}, RankId{4}, Epoch{1}, TokenId{1}}).ok());
  ASSERT_TRUE(worker.Enqueue(Commit{TransactionId{10}, RankId{4}, Epoch{1}}).ok());
  worker.Shutdown();
  EXPECT_TRUE(std::holds_alternative<PreparedAck>(Next(completions)));
  EXPECT_TRUE(std::holds_alternative<ExecutedAck>(Next(completions)));
  EXPECT_TRUE(std::holds_alternative<CommittedAck>(Next(completions)));
  EXPECT_FALSE(worker.Enqueue(Prepare{TransactionId{11}, RankId{4}, Epoch{1}}).ok());
}

TEST(RankWorker, RejectsInvalidCommandsSynchronously) {
  CompletionQueue<RankCompletion> completions;
  RankWorker worker(RankId{5}, completions);
  const auto bad_rank = worker.Enqueue(Prepare{TransactionId{1}, RankId{6}, Epoch{1}});
  ASSERT_FALSE(bad_rank.ok());
  EXPECT_EQ(bad_rank.error()->code(), ErrorCode::kInvalidArgument);
  const auto bad_id = worker.Enqueue(Prepare{TransactionId{0}, RankId{5}, Epoch{1}});
  ASSERT_FALSE(bad_id.ok());
  EXPECT_EQ(bad_id.error()->code(), ErrorCode::kInvalidArgument);
}

}  // namespace
