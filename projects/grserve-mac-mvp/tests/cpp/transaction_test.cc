#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "grserve/error.h"
#include "engine/transaction.h"

namespace {

using grserve::Epoch;
using grserve::RankId;
using grserve::Transaction;
using grserve::TransactionId;
using grserve::TransactionState;
using grserve::PreparedAck;
using grserve::ExecutedAck;
using grserve::CommittedAck;
using grserve::StateDigest;
using grserve::TokenId;
using grserve::SequenceNumber;
using grserve::Error;
using grserve::ErrorCode;

static_assert(!std::is_copy_constructible_v<Transaction>);
static_assert(!std::is_copy_assignable_v<Transaction>);
static_assert(std::is_nothrow_move_constructible_v<Transaction>);
static_assert(std::is_nothrow_move_assignable_v<Transaction>);

constexpr std::array kRanks{RankId{0}, RankId{1}};

Error StopReason() {
  return Error::From(ErrorCode::kRankFailure, "rank stopped");
}

Error OtherStopReason() {
  return Error::From(ErrorCode::kRankFailure, "other rank stopped");
}

Transaction NewTransaction() {
  return Transaction::Begin(TransactionId{7}, Epoch{3}, kRanks).value();
}

void Prepare(Transaction& tx) {
  ASSERT_TRUE(tx.ReserveAll().ok());
  ASSERT_TRUE(tx.AcknowledgePrepared({RankId{1}, Epoch{3}}).ok());
  ASSERT_TRUE(tx.AcknowledgePrepared({RankId{0}, Epoch{3}}).ok());
  ASSERT_EQ(tx.state(), TransactionState::kPrepared);
}

void Execute(Transaction& tx) {
  Prepare(tx);
  ASSERT_TRUE(
      tx.AcknowledgeExecuted({RankId{1}, Epoch{3}, TokenId{19}, StateDigest{8}})
          .ok());
  ASSERT_TRUE(
      tx.AcknowledgeExecuted({RankId{0}, Epoch{3}, TokenId{19}, StateDigest{5}})
          .ok());
  ASSERT_EQ(tx.state(), TransactionState::kExecuted);
}

void Decide(Transaction& tx) {
  Execute(tx);
  ASSERT_TRUE(tx.DecideCommit().ok());
  ASSERT_EQ(tx.state(), TransactionState::kCommitDecided);
}

void Commit(Transaction& tx) {
  Decide(tx);
  ASSERT_TRUE(tx.AcknowledgeCommitted({RankId{1}, Epoch{3}}).ok());
  ASSERT_TRUE(tx.AcknowledgeCommitted({RankId{0}, Epoch{3}}).ok());
  ASSERT_EQ(tx.state(), TransactionState::kLocalCommitted);
}

void ExpectErrorCode(const grserve::Status& status, ErrorCode code) {
  ASSERT_FALSE(status.ok());
  ASSERT_NE(status.error(), nullptr);
  EXPECT_EQ(status.error()->code(), code);
}

TEST(Transaction, DirectPrepareAcksImplicitlyReserveAndAggregate) {
  std::array ranks{RankId{0}, RankId{1}};
  auto tx = Transaction::Begin(TransactionId{7}, Epoch{3}, ranks).value();
  EXPECT_TRUE(tx.AcknowledgePrepared(PreparedAck{RankId{1}, Epoch{3}}).ok());
  EXPECT_TRUE(tx.AcknowledgePrepared(PreparedAck{RankId{0}, Epoch{3}}).ok());
  EXPECT_EQ(tx.state(), TransactionState::kPrepared);
  EXPECT_FALSE(tx.DecideCommit().ok());
  EXPECT_EQ(tx.state(), TransactionState::kPrepared);
}

TEST(Transaction, HappyPathPreservesOrderAndRecordsPerRankDigests) {
  auto tx = NewTransaction();
  ASSERT_EQ(tx.ranks().size(), kRanks.size());
  EXPECT_EQ(tx.ranks()[0], kRanks[0]);
  EXPECT_EQ(tx.ranks()[1], kRanks[1]);
  EXPECT_EQ(tx.mutation_count(), 0U);
  EXPECT_TRUE(tx.ReserveAll().ok());
  EXPECT_EQ(tx.state(), TransactionState::kReserved);
  EXPECT_TRUE(tx.AcknowledgePrepared({RankId{1}, Epoch{3}}).ok());
  EXPECT_TRUE(tx.AcknowledgePrepared({RankId{0}, Epoch{3}}).ok());
  EXPECT_EQ(tx.state(), TransactionState::kPrepared);

  EXPECT_TRUE(
      tx.AcknowledgeExecuted({RankId{1}, Epoch{3}, TokenId{19}, StateDigest{8}})
          .ok());
  EXPECT_TRUE(
      tx.AcknowledgeExecuted({RankId{0}, Epoch{3}, TokenId{19}, StateDigest{5}})
          .ok());
  EXPECT_EQ(tx.state(), TransactionState::kExecuted);
  EXPECT_EQ(tx.candidate_token(), std::optional{TokenId{19}});
  EXPECT_EQ(tx.state_digest(RankId{0}), std::optional{StateDigest{5}});
  EXPECT_EQ(tx.state_digest(RankId{1}), std::optional{StateDigest{8}});

  EXPECT_TRUE(tx.DecideCommit().ok());
  EXPECT_EQ(tx.state(), TransactionState::kCommitDecided);
  EXPECT_FALSE(tx.may_abort());
  EXPECT_TRUE(tx.AcknowledgeCommitted({RankId{1}, Epoch{3}}).ok());
  EXPECT_TRUE(tx.AcknowledgeCommitted({RankId{0}, Epoch{3}}).ok());
  EXPECT_EQ(tx.state(), TransactionState::kLocalCommitted);
  EXPECT_TRUE(tx.MarkPublished(SequenceNumber{42}).ok());
  EXPECT_EQ(tx.state(), TransactionState::kPublished);
  EXPECT_EQ(tx.sequence_number(), std::optional{SequenceNumber{42}});
  EXPECT_EQ(tx.mutation_count(), 9U);
}

TEST(Transaction, MovingAdvancedTransactionPreservesDestinationInvariants) {
  auto source = NewTransaction();
  Commit(source);
  const auto source_mutations = source.mutation_count();

  auto destination = NewTransaction();
  destination = std::move(source);

  EXPECT_EQ(destination.id(), TransactionId{7});
  EXPECT_EQ(destination.epoch(), Epoch{3});
  EXPECT_EQ(destination.state(), TransactionState::kLocalCommitted);
  EXPECT_EQ(destination.candidate_token(), std::optional{TokenId{19}});
  EXPECT_EQ(destination.state_digest(RankId{0}), std::optional{StateDigest{5}});
  EXPECT_EQ(destination.state_digest(RankId{1}), std::optional{StateDigest{8}});
  EXPECT_EQ(destination.mutation_count(), source_mutations);
  EXPECT_TRUE(destination.MarkPublished(SequenceNumber{42}).ok());
  EXPECT_EQ(destination.state(), TransactionState::kPublished);
}

TEST(Transaction, BeginRejectsInvalidInputWithoutAValue) {
  const std::array<RankId, 0> no_ranks{};
  auto empty = Transaction::Begin(TransactionId{7}, Epoch{3}, no_ranks);
  ASSERT_FALSE(empty.ok());
  EXPECT_EQ(empty.error()->code(), ErrorCode::kInvalidArgument);

  const std::array duplicate_ranks{RankId{0}, RankId{0}};
  auto duplicate = Transaction::Begin(TransactionId{7}, Epoch{3}, duplicate_ranks);
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.error()->code(), ErrorCode::kInvalidArgument);

  auto no_id = Transaction::Begin(TransactionId{0}, Epoch{3}, kRanks);
  ASSERT_FALSE(no_id.ok());
  EXPECT_EQ(no_id.error()->code(), ErrorCode::kInvalidArgument);
  auto no_epoch = Transaction::Begin(TransactionId{7}, Epoch{0}, kRanks);
  ASSERT_FALSE(no_epoch.ok());
  EXPECT_EQ(no_epoch.error()->code(), ErrorCode::kInvalidArgument);
}

TEST(Transaction, DuplicateAcksAreIdempotentAndConflictsDoNotMutate) {
  auto tx = NewTransaction();
  ASSERT_TRUE(tx.AcknowledgePrepared({RankId{0}, Epoch{3}}).ok());
  const auto after_prepare = tx.mutation_count();
  EXPECT_TRUE(tx.AcknowledgePrepared({RankId{0}, Epoch{3}}).ok());
  EXPECT_EQ(tx.mutation_count(), after_prepare);
  ASSERT_TRUE(tx.AcknowledgePrepared({RankId{1}, Epoch{3}}).ok());
  EXPECT_TRUE(tx.AcknowledgePrepared({RankId{0}, Epoch{3}}).ok());

  ASSERT_TRUE(
      tx.AcknowledgeExecuted({RankId{0}, Epoch{3}, TokenId{19}, StateDigest{5}})
          .ok());
  const auto after_execute = tx.mutation_count();
  EXPECT_TRUE(
      tx.AcknowledgeExecuted({RankId{0}, Epoch{3}, TokenId{19}, StateDigest{5}})
          .ok());
  EXPECT_EQ(tx.mutation_count(), after_execute);
  const auto state_before_conflict = tx.state();
  ExpectErrorCode(
      tx.AcknowledgeExecuted({RankId{0}, Epoch{3}, TokenId{20}, StateDigest{5}}),
      ErrorCode::kInternal);
  EXPECT_EQ(tx.state(), state_before_conflict);
  EXPECT_EQ(tx.mutation_count(), after_execute);
  ExpectErrorCode(
      tx.AcknowledgeExecuted({RankId{1}, Epoch{3}, TokenId{20}, StateDigest{8}}),
      ErrorCode::kInternal);
  EXPECT_EQ(tx.mutation_count(), after_execute);
}

TEST(Transaction, AcksValidateRankEpochAndPhaseBeforeMutating) {
  auto tx = NewTransaction();
  const auto mutations = tx.mutation_count();
  ExpectErrorCode(tx.AcknowledgePrepared({RankId{9}, Epoch{3}}),
                  ErrorCode::kInvalidArgument);
  ExpectErrorCode(tx.AcknowledgePrepared({RankId{0}, Epoch{2}}),
                  ErrorCode::kIncompatibleVersion);
  ExpectErrorCode(tx.AcknowledgePrepared({RankId{0}, Epoch{4}}),
                  ErrorCode::kIncompatibleVersion);
  ExpectErrorCode(
      tx.AcknowledgeExecuted({RankId{0}, Epoch{2}, TokenId{19}, StateDigest{5}}),
      ErrorCode::kIncompatibleVersion);
  ExpectErrorCode(tx.AcknowledgeCommitted({RankId{0}, Epoch{4}}),
                  ErrorCode::kIncompatibleVersion);
  ExpectErrorCode(
      tx.AcknowledgeExecuted({RankId{9}, Epoch{3}, TokenId{19}, StateDigest{5}}),
      ErrorCode::kInvalidArgument);
  ExpectErrorCode(tx.AcknowledgeCommitted({RankId{9}, Epoch{3}}),
                  ErrorCode::kInvalidArgument);
  ExpectErrorCode(
      tx.AcknowledgeExecuted({RankId{0}, Epoch{3}, TokenId{19}, StateDigest{5}}),
      ErrorCode::kInvalidArgument);
  ExpectErrorCode(tx.AcknowledgeCommitted({RankId{0}, Epoch{3}}),
                  ErrorCode::kInvalidArgument);
  EXPECT_EQ(tx.state(), TransactionState::kCreated);
  EXPECT_EQ(tx.mutation_count(), mutations);
}

TEST(Transaction, AbortAndFatalAreTerminalWithDefinedIdempotence) {
  for (const auto target : {TransactionState::kCreated, TransactionState::kReserved,
                            TransactionState::kPrepared, TransactionState::kExecuted}) {
    auto tx = NewTransaction();
    if (target == TransactionState::kReserved) {
      ASSERT_TRUE(tx.ReserveAll().ok());
    } else if (target == TransactionState::kPrepared) {
      Prepare(tx);
    } else if (target == TransactionState::kExecuted) {
      Execute(tx);
    }
    ASSERT_EQ(tx.state(), target);
    EXPECT_TRUE(tx.may_abort());
    EXPECT_TRUE(tx.Abort(StopReason()).ok());
    EXPECT_EQ(tx.state(), TransactionState::kAborted);
    EXPECT_FALSE(tx.may_abort());
    const auto mutations = tx.mutation_count();
    EXPECT_TRUE(tx.Abort(StopReason()).ok());
    EXPECT_EQ(tx.mutation_count(), mutations);
    ExpectErrorCode(tx.Abort(OtherStopReason()), ErrorCode::kInternal);
    EXPECT_EQ(tx.mutation_count(), mutations);
  }

  auto decided = NewTransaction();
  Decide(decided);
  const auto mutations = decided.mutation_count();
  ExpectErrorCode(decided.Abort(StopReason()), ErrorCode::kInvalidArgument);
  EXPECT_EQ(decided.mutation_count(), mutations);
  EXPECT_TRUE(decided.MarkFatal(StopReason()).ok());
  EXPECT_EQ(decided.state(), TransactionState::kFatal);
  EXPECT_EQ(decided.terminal_error()->message(), "rank stopped");
  const auto fatal_mutations = decided.mutation_count();
  EXPECT_TRUE(decided.MarkFatal(StopReason()).ok());
  EXPECT_EQ(decided.mutation_count(), fatal_mutations);
  ExpectErrorCode(decided.MarkFatal(OtherStopReason()), ErrorCode::kInternal);
  ExpectErrorCode(decided.MarkPublished(SequenceNumber{9}), ErrorCode::kInvalidArgument);
}

TEST(Transaction, PublishIsOneWayAndOnlyAfterLocalCommit) {
  auto tx = NewTransaction();
  ExpectErrorCode(tx.MarkPublished(SequenceNumber{42}), ErrorCode::kInvalidArgument);
  Commit(tx);
  ASSERT_TRUE(tx.MarkPublished(SequenceNumber{42}).ok());
  const auto mutations = tx.mutation_count();
  ExpectErrorCode(tx.MarkPublished(SequenceNumber{42}), ErrorCode::kInvalidArgument);
  EXPECT_EQ(tx.state(), TransactionState::kPublished);
  EXPECT_EQ(tx.mutation_count(), mutations);
  ExpectErrorCode(tx.MarkFatal(StopReason()), ErrorCode::kInvalidArgument);
  ExpectErrorCode(tx.Abort(StopReason()), ErrorCode::kInvalidArgument);
}

enum class EventKind {
  kReserveAll,
  kPrepared,
  kExecuted,
  kDecideCommit,
  kCommitted,
  kPublished,
  kAbort,
  kFatal,
  kCount,
};

constexpr std::array kAllStates{
    TransactionState::kCreated,       TransactionState::kReserved,
    TransactionState::kPrepared,      TransactionState::kExecuted,
    TransactionState::kCommitDecided, TransactionState::kLocalCommitted,
    TransactionState::kPublished,     TransactionState::kAborted,
    TransactionState::kFatal,
};
constexpr std::array kAllEvents{
    EventKind::kReserveAll, EventKind::kPrepared, EventKind::kExecuted,
    EventKind::kDecideCommit, EventKind::kCommitted, EventKind::kPublished,
    EventKind::kAbort, EventKind::kFatal,
};
static_assert(kAllStates.size() ==
              static_cast<std::size_t>(TransactionState::kCount));
static_assert(kAllEvents.size() == static_cast<std::size_t>(EventKind::kCount));

Transaction Reach(TransactionState state) {
  auto tx = NewTransaction();
  switch (state) {
    case TransactionState::kCreated:
      break;
    case TransactionState::kReserved:
      EXPECT_TRUE(tx.ReserveAll().ok());
      break;
    case TransactionState::kPrepared:
      Prepare(tx);
      break;
    case TransactionState::kExecuted:
      Execute(tx);
      break;
    case TransactionState::kCommitDecided:
      Decide(tx);
      break;
    case TransactionState::kLocalCommitted:
      Commit(tx);
      break;
    case TransactionState::kPublished:
      Commit(tx);
      EXPECT_TRUE(tx.MarkPublished(SequenceNumber{42}).ok());
      break;
    case TransactionState::kAborted:
      EXPECT_TRUE(tx.Abort(StopReason()).ok());
      break;
    case TransactionState::kFatal:
      Decide(tx);
      EXPECT_TRUE(tx.MarkFatal(StopReason()).ok());
      break;
    case TransactionState::kCount:
      ADD_FAILURE() << "kCount is not a transaction state";
      break;
  }
  EXPECT_EQ(tx.state(), state);
  return tx;
}

struct EventExpectation {
  bool ok;
  TransactionState state;
  std::size_t mutation_delta;
};

EventExpectation Expected(TransactionState state, EventKind event) {
  const auto unchanged = EventExpectation{false, state, 0};
  switch (state) {
    case TransactionState::kCreated:
      switch (event) {
        case EventKind::kReserveAll: return {true, TransactionState::kReserved, 1};
        case EventKind::kPrepared: return {true, TransactionState::kReserved, 1};
        case EventKind::kAbort: return {true, TransactionState::kAborted, 1};
        default: return unchanged;
      }
    case TransactionState::kReserved:
      switch (event) {
        case EventKind::kReserveAll: return {true, state, 0};
        case EventKind::kPrepared: return {true, state, 1};
        case EventKind::kAbort: return {true, TransactionState::kAborted, 1};
        default: return unchanged;
      }
    case TransactionState::kPrepared:
      switch (event) {
        case EventKind::kPrepared: return {true, state, 0};
        case EventKind::kExecuted: return {true, state, 1};
        case EventKind::kAbort: return {true, TransactionState::kAborted, 1};
        default: return unchanged;
      }
    case TransactionState::kExecuted:
      switch (event) {
        case EventKind::kPrepared:
        case EventKind::kExecuted: return {true, state, 0};
        case EventKind::kDecideCommit: return {true, TransactionState::kCommitDecided, 1};
        case EventKind::kAbort: return {true, TransactionState::kAborted, 1};
        default: return unchanged;
      }
    case TransactionState::kCommitDecided:
      switch (event) {
        case EventKind::kPrepared:
        case EventKind::kExecuted: return {true, state, 0};
        case EventKind::kCommitted: return {true, state, 1};
        case EventKind::kFatal: return {true, TransactionState::kFatal, 1};
        default: return unchanged;
      }
    case TransactionState::kLocalCommitted:
      switch (event) {
        case EventKind::kPrepared:
        case EventKind::kExecuted:
        case EventKind::kCommitted: return {true, state, 0};
        case EventKind::kPublished: return {true, TransactionState::kPublished, 1};
        case EventKind::kFatal: return {true, TransactionState::kFatal, 1};
        default: return unchanged;
      }
    case TransactionState::kPublished:
      return unchanged;
    case TransactionState::kAborted:
      return event == EventKind::kAbort ? EventExpectation{true, state, 0} : unchanged;
    case TransactionState::kFatal:
      return event == EventKind::kFatal ? EventExpectation{true, state, 0} : unchanged;
    case TransactionState::kCount:
      return unchanged;
  }
  return unchanged;
}

grserve::Status Attempt(Transaction& tx, EventKind event) {
  switch (event) {
    case EventKind::kReserveAll: return tx.ReserveAll();
    case EventKind::kPrepared: return tx.AcknowledgePrepared({RankId{0}, Epoch{3}});
    case EventKind::kExecuted:
      return tx.AcknowledgeExecuted({RankId{0}, Epoch{3}, TokenId{19}, StateDigest{5}});
    case EventKind::kDecideCommit: return tx.DecideCommit();
    case EventKind::kCommitted: return tx.AcknowledgeCommitted({RankId{0}, Epoch{3}});
    case EventKind::kPublished: return tx.MarkPublished(SequenceNumber{42});
    case EventKind::kAbort: return tx.Abort(StopReason());
    case EventKind::kFatal: return tx.MarkFatal(StopReason());
    case EventKind::kCount: return grserve::Status::Ok();
  }
  return grserve::Status::Ok();
}

TEST(Transaction, EveryPublicEventHasAStateTableEntry) {
  static_assert(kAllStates.size() * kAllEvents.size() == 72);
  for (const auto state : kAllStates) {
    for (const auto event : kAllEvents) {
      auto tx = Reach(state);
      const auto before = tx.mutation_count();
      const auto expected = Expected(state, event);
      const auto result = Attempt(tx, event);
      EXPECT_EQ(result.ok(), expected.ok);
      if (!expected.ok) {
        ASSERT_NE(result.error(), nullptr);
        EXPECT_EQ(result.error()->code(), ErrorCode::kInvalidArgument);
      }
      EXPECT_EQ(tx.state(), expected.state);
      EXPECT_EQ(tx.mutation_count(), before + expected.mutation_delta);
    }
  }
}

}  // namespace
