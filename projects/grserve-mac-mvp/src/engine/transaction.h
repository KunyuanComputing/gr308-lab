#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "engine/rank_protocol.h"
#include "grserve/error.h"

namespace grserve {

enum class TransactionState {
  kCreated,
  kReserved,
  kPrepared,
  kExecuted,
  kCommitDecided,
  kLocalCommitted,
  kPublished,
  kAborted,
  kFatal,
  kCount,
};

class Transaction {
 public:
  Transaction(const Transaction&) = delete;
  Transaction& operator=(const Transaction&) = delete;
  Transaction(Transaction&&) noexcept = default;
  Transaction& operator=(Transaction&&) noexcept = default;

  [[nodiscard]] static StatusOr<Transaction> Begin(
      TransactionId id, Epoch epoch, std::span<const RankId> ranks);

  Status ReserveAll();
  Status AcknowledgePrepared(const PreparedAck& ack);
  Status AcknowledgeExecuted(const ExecutedAck& ack);
  Status DecideCommit();
  Status AcknowledgeCommitted(const CommittedAck& ack);
  Status MarkPublished(SequenceNumber sequence_number);
  Status Abort(const Error& reason);
  Status MarkFatal(const Error& reason);

  [[nodiscard]] TransactionState state() const noexcept { return state_; }
  [[nodiscard]] bool may_abort() const noexcept;

  [[nodiscard]] TransactionId id() const noexcept { return id_; }
  [[nodiscard]] Epoch epoch() const noexcept { return epoch_; }
  [[nodiscard]] std::span<const RankId> ranks() const noexcept { return ranks_; }
  [[nodiscard]] std::optional<TokenId> candidate_token() const noexcept {
    return candidate_token_;
  }
  [[nodiscard]] std::optional<StateDigest> state_digest(RankId rank) const noexcept;
  [[nodiscard]] std::optional<SequenceNumber> sequence_number() const noexcept {
    return sequence_number_;
  }
  [[nodiscard]] const Error* terminal_error() const noexcept {
    return terminal_error_ ? &*terminal_error_ : nullptr;
  }
  [[nodiscard]] std::size_t mutation_count() const noexcept {
    return mutation_count_;
  }

 private:
  Transaction(TransactionId id, Epoch epoch, std::vector<RankId> ranks);

  [[nodiscard]] Status Failure(ErrorCode code, const char* message) const;
  [[nodiscard]] Status ValidateRankAndEpoch(RankId rank, Epoch epoch) const;
  [[nodiscard]] std::size_t RankIndex(RankId rank) const noexcept;
  [[nodiscard]] bool AllPrepared() const noexcept;
  [[nodiscard]] bool AllExecuted() const noexcept;
  [[nodiscard]] bool AllCommitted() const noexcept;
  [[nodiscard]] static bool SameError(const Error& left, const Error& right) noexcept;
  [[nodiscard]] bool IsTerminal() const noexcept;

  static constexpr std::size_t kNoRank = static_cast<std::size_t>(-1);

  TransactionId id_;
  Epoch epoch_;
  std::vector<RankId> ranks_;
  std::vector<bool> prepared_;
  std::vector<std::optional<ExecutedAck>> executed_;
  std::vector<bool> committed_;
  TransactionState state_{TransactionState::kCreated};
  std::optional<TokenId> candidate_token_;
  std::optional<SequenceNumber> sequence_number_;
  std::optional<Error> terminal_error_;
  std::size_t mutation_count_{0};
};

}  // namespace grserve
