#include "engine/transaction.h"

#include <algorithm>
#include <utility>

namespace grserve {

StatusOr<Transaction> Transaction::Begin(TransactionId id, Epoch epoch,
                                          std::span<const RankId> ranks) {
  if (id.value() == 0 || epoch.value() == 0) {
    return StatusOr<Transaction>{
        Error::From(ErrorCode::kInvalidArgument, "transaction id and epoch must be nonzero")};
  }
  if (ranks.empty()) {
    return StatusOr<Transaction>{
        Error::From(ErrorCode::kInvalidArgument, "transaction requires ranks")};
  }
  for (std::size_t index = 0; index < ranks.size(); ++index) {
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (ranks[index] == ranks[prior]) {
        return StatusOr<Transaction>{
            Error::From(ErrorCode::kInvalidArgument, "transaction ranks must be unique")};
      }
    }
  }
  return StatusOr<Transaction>{Transaction{id, epoch,
                                             std::vector<RankId>(ranks.begin(), ranks.end())}};
}

Transaction::Transaction(TransactionId id, Epoch epoch, std::vector<RankId> ranks)
    : id_(id),
      epoch_(epoch),
      ranks_(std::move(ranks)),
      prepared_(ranks_.size(), false),
      executed_(ranks_.size()),
      committed_(ranks_.size(), false) {}

Status Transaction::ReserveAll() {
  if (state_ == TransactionState::kReserved) {
    return Status::Ok();
  }
  if (state_ != TransactionState::kCreated) {
    return Failure(ErrorCode::kInvalidArgument, "reserve requires created transaction");
  }
  state_ = TransactionState::kReserved;
  ++mutation_count_;
  return Status::Ok();
}

Status Transaction::AcknowledgePrepared(const PreparedAck& ack) {
  const Status validation = ValidateRankAndEpoch(ack.rank_id, ack.epoch);
  if (!validation.ok()) {
    return validation;
  }
  if (IsTerminal()) {
    return Failure(ErrorCode::kInvalidArgument, "prepare ack rejected by terminal transaction");
  }
  const std::size_t index = RankIndex(ack.rank_id);
  if (prepared_[index]) {
    return Status::Ok();
  }
  if (state_ != TransactionState::kCreated && state_ != TransactionState::kReserved) {
    return Failure(ErrorCode::kInvalidArgument, "prepare ack is out of phase");
  }

  prepared_[index] = true;
  if (state_ == TransactionState::kCreated) {
    state_ = TransactionState::kReserved;
  }
  if (AllPrepared()) {
    state_ = TransactionState::kPrepared;
  }
  ++mutation_count_;
  return Status::Ok();
}

Status Transaction::AcknowledgeExecuted(const ExecutedAck& ack) {
  const Status validation = ValidateRankAndEpoch(ack.rank_id, ack.epoch);
  if (!validation.ok()) {
    return validation;
  }
  if (IsTerminal()) {
    return Failure(ErrorCode::kInvalidArgument, "execute ack rejected by terminal transaction");
  }
  const std::size_t index = RankIndex(ack.rank_id);
  if (executed_[index]) {
    const ExecutedAck& recorded = *executed_[index];
    if (recorded.candidate_token == ack.candidate_token &&
        recorded.state_digest == ack.state_digest) {
      return Status::Ok();
    }
    return Failure(ErrorCode::kInternal, "conflicting execute acknowledgement");
  }
  if (state_ != TransactionState::kPrepared) {
    return Failure(ErrorCode::kInvalidArgument, "execute ack is out of phase");
  }
  if (candidate_token_ && *candidate_token_ != ack.candidate_token) {
    return Failure(ErrorCode::kInternal, "candidate token mismatch");
  }

  executed_[index] = ack;
  if (!candidate_token_) {
    candidate_token_ = ack.candidate_token;
  }
  if (AllExecuted()) {
    state_ = TransactionState::kExecuted;
  }
  ++mutation_count_;
  return Status::Ok();
}

Status Transaction::DecideCommit() {
  if (state_ != TransactionState::kExecuted) {
    return Failure(ErrorCode::kInvalidArgument, "commit decision requires executed transaction");
  }
  state_ = TransactionState::kCommitDecided;
  ++mutation_count_;
  return Status::Ok();
}

Status Transaction::AcknowledgeCommitted(const CommittedAck& ack) {
  const Status validation = ValidateRankAndEpoch(ack.rank_id, ack.epoch);
  if (!validation.ok()) {
    return validation;
  }
  if (IsTerminal()) {
    return Failure(ErrorCode::kInvalidArgument, "commit ack rejected by terminal transaction");
  }
  const std::size_t index = RankIndex(ack.rank_id);
  if (committed_[index]) {
    return Status::Ok();
  }
  if (state_ != TransactionState::kCommitDecided) {
    return Failure(ErrorCode::kInvalidArgument, "commit ack is out of phase");
  }

  committed_[index] = true;
  if (AllCommitted()) {
    state_ = TransactionState::kLocalCommitted;
  }
  ++mutation_count_;
  return Status::Ok();
}

Status Transaction::MarkPublished(SequenceNumber sequence_number) {
  if (state_ != TransactionState::kLocalCommitted) {
    return Failure(ErrorCode::kInvalidArgument, "publish requires local commit");
  }
  sequence_number_ = sequence_number;
  state_ = TransactionState::kPublished;
  ++mutation_count_;
  return Status::Ok();
}

Status Transaction::Abort(const Error& reason) {
  if (state_ == TransactionState::kAborted) {
    if (terminal_error_ && SameError(*terminal_error_, reason)) {
      return Status::Ok();
    }
    return Failure(ErrorCode::kInternal, "conflicting abort reason");
  }
  if (!may_abort()) {
    return Failure(ErrorCode::kInvalidArgument, "abort is not allowed after decision");
  }

  std::optional<Error> reason_copy{reason};
  terminal_error_ = std::move(reason_copy);
  state_ = TransactionState::kAborted;
  ++mutation_count_;
  return Status::Ok();
}

Status Transaction::MarkFatal(const Error& reason) {
  if (state_ == TransactionState::kFatal) {
    if (terminal_error_ && SameError(*terminal_error_, reason)) {
      return Status::Ok();
    }
    return Failure(ErrorCode::kInternal, "conflicting fatal reason");
  }
  if (state_ != TransactionState::kCommitDecided &&
      state_ != TransactionState::kLocalCommitted) {
    return Failure(ErrorCode::kInvalidArgument, "fatal requires post-decision transaction");
  }

  std::optional<Error> reason_copy{reason};
  terminal_error_ = std::move(reason_copy);
  state_ = TransactionState::kFatal;
  ++mutation_count_;
  return Status::Ok();
}

bool Transaction::may_abort() const noexcept {
  return state_ == TransactionState::kCreated || state_ == TransactionState::kReserved ||
         state_ == TransactionState::kPrepared || state_ == TransactionState::kExecuted;
}

std::optional<StateDigest> Transaction::state_digest(RankId rank) const noexcept {
  const std::size_t index = RankIndex(rank);
  if (index == kNoRank || !executed_[index]) {
    return std::nullopt;
  }
  return executed_[index]->state_digest;
}

Status Transaction::Failure(ErrorCode code, const char* message) const {
  return Status{Error::From(code, message, std::nullopt, id_)};
}

Status Transaction::ValidateRankAndEpoch(RankId rank, Epoch epoch) const {
  if (RankIndex(rank) == kNoRank) {
    return Failure(ErrorCode::kInvalidArgument, "acknowledgement is from an unknown rank");
  }
  if (epoch != epoch_) {
    return Failure(ErrorCode::kIncompatibleVersion, "acknowledgement epoch does not match transaction");
  }
  return Status::Ok();
}

std::size_t Transaction::RankIndex(RankId rank) const noexcept {
  const auto found = std::find(ranks_.begin(), ranks_.end(), rank);
  if (found == ranks_.end()) {
    return kNoRank;
  }
  return static_cast<std::size_t>(found - ranks_.begin());
}

bool Transaction::AllPrepared() const noexcept {
  return std::all_of(prepared_.begin(), prepared_.end(), [](bool value) { return value; });
}

bool Transaction::AllExecuted() const noexcept {
  return std::all_of(executed_.begin(), executed_.end(),
                     [](const auto& value) { return value.has_value(); });
}

bool Transaction::AllCommitted() const noexcept {
  return std::all_of(committed_.begin(), committed_.end(), [](bool value) { return value; });
}

bool Transaction::SameError(const Error& left, const Error& right) noexcept {
  return left.code() == right.code() && left.message() == right.message() &&
         left.request_id() == right.request_id() &&
         left.transaction_id() == right.transaction_id();
}

bool Transaction::IsTerminal() const noexcept {
  return state_ == TransactionState::kPublished || state_ == TransactionState::kAborted ||
         state_ == TransactionState::kFatal;
}

}  // namespace grserve
