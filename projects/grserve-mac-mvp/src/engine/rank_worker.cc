#include "engine/rank_worker.h"

#include <type_traits>

namespace grserve {
namespace {

Error MakeError(ErrorCode code, const char* message, TransactionId transaction) {
  return Error::From(code, message, std::nullopt, transaction);
}

}  // namespace

RankWorker::RankWorker(RankId rank_id, CompletionQueue<RankCompletion>& completions,
                       StateDigest initial_digest, std::function<void()> before_plan,
                       DigestTransition digest_transition,
                       ExecutePlanner execute_planner)
    : rank_id_(rank_id),
      completions_(completions),
      producer_(completions_.RegisterProducer()),
      canonical_(initial_digest.value()),
      before_plan_(std::move(before_plan)),
      digest_transition_(std::move(digest_transition)),
      execute_planner_(std::move(execute_planner)),
      thread_(&RankWorker::Run, this) {}

RankWorker::~RankWorker() { Shutdown(); }

Status RankWorker::Enqueue(RankCommand command) {
  const TransactionId transaction = TransactionOf(command);
  if (RankOf(command) != rank_id_) {
    return Status{MakeError(ErrorCode::kInvalidArgument,
                            "command rank does not match worker", transaction)};
  }
  if (transaction.value() == 0 || EpochOf(command).value() == 0) {
    return Status{MakeError(ErrorCode::kInvalidArgument,
                            "transaction and epoch must be nonzero", transaction)};
  }
  std::lock_guard lock(enqueue_mutex_);
  if (!accepting_.load(std::memory_order_acquire) || !producer_.CanAccept()) {
    return Status{MakeError(ErrorCode::kInternal, "rank worker is stopped", transaction)};
  }
  outstanding_commands_.fetch_add(1, std::memory_order_acq_rel);
  if (!commands_.Push(std::move(command))) {
    outstanding_commands_.fetch_sub(1, std::memory_order_acq_rel);
    return Status{MakeError(ErrorCode::kInternal, "rank worker is stopped", transaction)};
  }
  return Status::Ok();
}

void RankWorker::Shutdown() noexcept {
  {
    std::lock_guard lock(enqueue_mutex_);
    accepting_.store(false, std::memory_order_release);
    commands_.Close();
  }
  if (thread_.joinable()) thread_.join();
  producer_.Release();
}

StateDigest RankWorker::canonical_digest() const noexcept {
  return StateDigest{canonical_.load(std::memory_order_acquire)};
}

void RankWorker::Run() noexcept {
  try {
    while (auto command = commands_.WaitPop()) {
      if (before_plan_) before_plan_();
      Transition transition = Plan(*command);
      if (!producer_.Publish(std::move(transition.completion),
                             [this, &transition]() noexcept { Apply(transition); })) {
        Fatal();
        return;
      }
      outstanding_commands_.fetch_sub(1, std::memory_order_acq_rel);
    }
  } catch (...) {
    // Planning and publication allocation happen before Apply. A fatal abort
    // gives every consumer a terminal result even if other ranks stay idle.
    Fatal();
  }
}

void RankWorker::Fatal() noexcept {
  {
    std::lock_guard lock(enqueue_mutex_);
    accepting_.store(false, std::memory_order_release);
    commands_.Close();
  }
  completions_.Abort();
  producer_.Release();
}

RankWorker::Transition RankWorker::Plan(const RankCommand& command) {
  const TransactionId transaction = TransactionOf(command);
  const Epoch epoch = EpochOf(command);
  const TransactionKey key = KeyFor(transaction, epoch);

  if (epoch.value() < highest_epoch_) {
    return Transition{Failure(command, ErrorCode::kIncompatibleVersion,
                              "message epoch is older than current generation")};
  }
  const auto existing = transactions_.find(key);

  return std::visit(
      [this, transaction, epoch, key, existing](const auto& value) -> Transition {
        using Command = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Command, Prepare>) {
          if (existing != transactions_.end()) {
            if (existing->second.phase == Phase::kAborted) {
              return Transition{Failure(RankCommand{value}, ErrorCode::kIncompatibleVersion,
                                        "transaction was aborted")};
            }
            return Transition{PreparedAck{rank_id_, epoch}};
          }
          if (epoch.value() == highest_epoch_ &&
              transaction.value() <= last_first_transaction_) {
            return Transition{Failure(RankCommand{value}, ErrorCode::kInvalidArgument,
                                      "first transaction id must increase within epoch")};
          }
          std::map<TransactionKey, TransactionRecord> staging;
          staging.emplace(key, TransactionRecord{epoch});
          Transition transition{PreparedAck{rank_id_, epoch}};
          transition.mutation = Mutation::kInsert;
          transition.key = key;
          transition.inserted = staging.extract(staging.begin());
          transition.sets_generation = true;
          transition.clears_old_generation = epoch.value() > highest_epoch_;
          transition.new_highest_epoch = epoch.value();
          transition.new_last_first_transaction = transaction.value();
          return transition;
        } else if constexpr (std::is_same_v<Command, Execute>) {
          if (epoch.value() != highest_epoch_ || existing == transactions_.end()) {
            return Transition{Failure(RankCommand{value}, ErrorCode::kIncompatibleVersion,
                                      "execute has no matching current transaction")};
          }
          const TransactionRecord& record = existing->second;
          if (record.phase == Phase::kAborted) {
            return Transition{Failure(RankCommand{value}, ErrorCode::kInvalidArgument,
                                      "transaction was aborted")};
          }
          if (record.phase == Phase::kExecuted || record.phase == Phase::kCommitted) {
            if (!record.execute ||
                record.execute->candidate_token != value.candidate_token ||
                record.execute->context_token_ids != value.context_token_ids ||
                record.execute->sequence_number != value.sequence_number) {
              return Transition{Failure(RankCommand{value}, ErrorCode::kInternal,
                                        "conflicting execute payload")};
            }
            return Transition{ExecutedAck{rank_id_, epoch, record.token,
                                           record.next_digest}};
          }
          TokenId candidate = value.candidate_token;
          if (execute_planner_) {
            try {
              auto planned = execute_planner_(value);
              if (!planned.result.ok()) {
                return Transition{RankError{
                    rank_id_, transaction, epoch,
                    Error::From(planned.result.error()->code(),
                                planned.result.error()->message(),
                                std::nullopt, transaction),
                    planned.rank_lost}};
              }
              candidate = planned.result.value();
            } catch (...) {
              return Transition{RankError{
                  rank_id_, transaction, epoch,
                  MakeError(ErrorCode::kRankFailure,
                            "rank execute planner threw", transaction),
                  false}};
            }
          }
          const StateDigest next =
              ComputeNextDigest(canonical_digest(), transaction, epoch,
                                candidate);
          Transition transition{ExecutedAck{rank_id_, epoch, candidate, next}};
          transition.mutation = Mutation::kUpdate;
          transition.key = key;
          transition.record = record;
          transition.record.token = candidate;
          transition.record.base_version = canonical_version_.load(std::memory_order_acquire);
          transition.record.next_digest = next;
          transition.record.execute = value;
          transition.record.phase = Phase::kExecuted;
          return transition;
        } else if constexpr (std::is_same_v<Command, Commit>) {
          if (epoch.value() != highest_epoch_ || existing == transactions_.end()) {
            return Transition{Failure(RankCommand{value}, ErrorCode::kIncompatibleVersion,
                                      "commit has no matching current transaction")};
          }
          const TransactionRecord& record = existing->second;
          if (record.phase == Phase::kAborted || record.phase == Phase::kPrepared) {
            return Transition{Failure(RankCommand{value}, ErrorCode::kInvalidArgument,
                                      "commit requires execute")};
          }
          if (record.phase == Phase::kCommitted) return Transition{CommittedAck{rank_id_, epoch}};
          if (canonical_version_.load(std::memory_order_acquire) != record.base_version) {
            return Transition{Failure(RankCommand{value}, ErrorCode::kInternal,
                                      "execute shadow is stale")};
          }
          Transition transition{CommittedAck{rank_id_, epoch}};
          transition.mutation = Mutation::kCommit;
          transition.key = key;
          transition.record = record;
          transition.record.phase = Phase::kCommitted;
          return transition;
        } else {
          if (epoch.value() != highest_epoch_ || existing == transactions_.end()) {
            return Transition{Failure(RankCommand{value}, ErrorCode::kIncompatibleVersion,
                                      "abort has no matching current transaction")};
          }
          const TransactionRecord& record = existing->second;
          if (record.phase == Phase::kCommitted) {
            return Transition{Failure(RankCommand{value}, ErrorCode::kInvalidArgument,
                                      "committed transaction cannot abort")};
          }
          if (record.phase == Phase::kAborted) return Transition{AbortedAck{rank_id_, epoch}};
          Transition transition{AbortedAck{rank_id_, epoch}};
          transition.mutation = Mutation::kAbort;
          transition.key = key;
          transition.record = record;
          transition.record.phase = Phase::kAborted;
          return transition;
        }
      },
      command);
}

void RankWorker::Apply(Transition& transition) noexcept {
  switch (transition.mutation) {
    case Mutation::kNone:
      return;
    case Mutation::kInsert:
      if (transition.clears_old_generation) {
        transactions_.clear();
        terminal_count_ = 0;
        terminal_cursor_ = 0;
        transaction_count_.store(0, std::memory_order_release);
      }
      transactions_.insert(std::move(transition.inserted));
      transaction_count_.fetch_add(1, std::memory_order_release);
      if (transition.sets_generation) {
        highest_epoch_ = transition.new_highest_epoch;
        last_first_transaction_ = transition.new_last_first_transaction;
      }
      return;
    case Mutation::kUpdate:
      transactions_.at(transition.key) = transition.record;
      return;
    case Mutation::kCommit:
      transactions_.at(transition.key) = transition.record;
      RememberTerminal(transition.key);
      canonical_.store(transition.record.next_digest.value(), std::memory_order_release);
      canonical_version_.fetch_add(1, std::memory_order_release);
      return;
    case Mutation::kAbort:
      transactions_.at(transition.key) = transition.record;
      RememberTerminal(transition.key);
      return;
  }
}

RankCompletion RankWorker::Failure(const RankCommand& command, ErrorCode code,
                                   const char* message) const {
  return RankError{rank_id_, TransactionOf(command), EpochOf(command),
                   MakeError(code, message, TransactionOf(command))};
}

TransactionId RankWorker::TransactionOf(const RankCommand& command) noexcept {
  return std::visit([](const auto& value) { return value.transaction_id; }, command);
}

RankId RankWorker::RankOf(const RankCommand& command) noexcept {
  return std::visit([](const auto& value) { return value.rank_id; }, command);
}

Epoch RankWorker::EpochOf(const RankCommand& command) noexcept {
  return std::visit([](const auto& value) { return value.epoch; }, command);
}

StateDigest RankWorker::NextDigest(StateDigest previous, RankId rank,
                                   TransactionId transaction, Epoch epoch,
                                   TokenId token) noexcept {
  std::uint64_t value = previous.value() ^ 0x9e3779b97f4a7c15ULL;
  const auto mix = [&value](std::uint64_t input) {
    value ^= input + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
  };
  mix(rank.value());
  mix(transaction.value());
  mix(epoch.value());
  mix(token.value());
  return StateDigest{value};
}

StateDigest RankWorker::ComputeNextDigest(StateDigest previous,
                                          TransactionId transaction, Epoch epoch,
                                          TokenId token) const {
  if (digest_transition_) {
    return digest_transition_(previous, rank_id_, transaction, epoch, token);
  }
  return NextDigest(previous, rank_id_, transaction, epoch, token);
}

RankWorker::TransactionKey RankWorker::KeyFor(TransactionId transaction,
                                              Epoch epoch) noexcept {
  return {epoch.value(), transaction.value()};
}

void RankWorker::RememberTerminal(TransactionKey key) noexcept {
  if (terminal_count_ < kTerminalReplayLimit) {
    terminal_keys_[terminal_count_++] = key;
    return;
  }
  transactions_.erase(terminal_keys_[terminal_cursor_]);
  transaction_count_.fetch_sub(1, std::memory_order_release);
  terminal_keys_[terminal_cursor_] = key;
  terminal_cursor_ = (terminal_cursor_ + 1) % kTerminalReplayLimit;
}

}  // namespace grserve
