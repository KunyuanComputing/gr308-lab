#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <variant>

#include "engine/completion_queue.h"
#include "engine/rank_protocol.h"
#include "grserve/error.h"

namespace grserve {

using RankCommand = std::variant<Prepare, Execute, Commit, Abort>;

struct RankError {
  RankId rank_id;
  TransactionId transaction_id;
  Epoch epoch;
  Error error;
  // True only when the worker/transport is no longer serviceable. A normal
  // transaction-scoped rank error remains recoverable before commit decision.
  bool rank_lost{false};
};

using RankCompletion =
    std::variant<PreparedAck, ExecutedAck, CommittedAck, AbortedAck, RankError>;

class RankWorker {
 public:
  using DigestTransition = std::function<StateDigest(
      StateDigest, RankId, TransactionId, Epoch, TokenId)>;
  struct ExecuteResult {
    // Conversion from StatusOr keeps ordinary planner failures source
    // compatible and recoverable. Backends that know their rank resource is
    // no longer serviceable must set rank_lost explicitly.
    ExecuteResult(StatusOr<TokenId> value, bool lost = false)
        : result(std::move(value)), rank_lost(lost) {}
    StatusOr<TokenId> result;
    bool rank_lost{false};
  };
  using ExecutePlanner = std::function<ExecuteResult(const Execute&)>;

  explicit RankWorker(RankId rank_id, CompletionQueue<RankCompletion>& completions,
                      StateDigest initial_digest = StateDigest{0},
                      std::function<void()> before_plan = {},
                      DigestTransition digest_transition = {},
                      ExecutePlanner execute_planner = {});
  RankWorker(const RankWorker&) = delete;
  RankWorker& operator=(const RankWorker&) = delete;
  ~RankWorker();

  [[nodiscard]] Status Enqueue(RankCommand command);
  void Shutdown() noexcept;
  [[nodiscard]] StateDigest canonical_digest() const noexcept;
  [[nodiscard]] bool quiescent() const noexcept {
    return outstanding_commands_.load(std::memory_order_acquire) == 0;
  }
  [[nodiscard]] std::size_t transaction_count_for_testing() const noexcept {
    return transaction_count_.load(std::memory_order_acquire);
  }

 private:
  using TransactionKey = std::pair<std::uint64_t, std::uint64_t>;
  enum class Phase { kPrepared, kExecuted, kCommitted, kAborted };
  struct TransactionRecord {
    Epoch epoch;
    Phase phase{Phase::kPrepared};
    TokenId token{0};
    std::uint64_t base_version{0};
    StateDigest next_digest{0};
    std::optional<Execute> execute;
  };
  enum class Mutation { kNone, kInsert, kUpdate, kCommit, kAbort };
  struct Transition {
    RankCompletion completion;
    Mutation mutation{Mutation::kNone};
    TransactionKey key{};
    TransactionRecord record{Epoch{0}};
    std::map<TransactionKey, TransactionRecord>::node_type inserted;
    bool sets_generation{false};
    bool clears_old_generation{false};
    std::uint64_t new_highest_epoch{0};
    std::uint64_t new_last_first_transaction{0};
  };

  void Run() noexcept;
  [[nodiscard]] Transition Plan(const RankCommand& command);
  void Apply(Transition& transition) noexcept;
  void Fatal() noexcept;
  [[nodiscard]] RankCompletion Failure(const RankCommand& command, ErrorCode code,
                                       const char* message) const;
  [[nodiscard]] static TransactionId TransactionOf(const RankCommand& command) noexcept;
  [[nodiscard]] static RankId RankOf(const RankCommand& command) noexcept;
  [[nodiscard]] static Epoch EpochOf(const RankCommand& command) noexcept;
  [[nodiscard]] static StateDigest NextDigest(StateDigest previous, RankId rank,
                                              TransactionId transaction, Epoch epoch,
                                              TokenId token) noexcept;
  [[nodiscard]] StateDigest ComputeNextDigest(StateDigest previous,
                                               TransactionId transaction,
                                               Epoch epoch, TokenId token) const;
  [[nodiscard]] static TransactionKey KeyFor(TransactionId transaction,
                                             Epoch epoch) noexcept;
  void RememberTerminal(TransactionKey key) noexcept;

  static constexpr std::size_t kTerminalReplayLimit = 256;
  RankId rank_id_;
  CompletionQueue<RankCompletion>& completions_;
  CompletionQueue<RankCompletion>::Producer producer_;
  CompletionQueue<RankCommand> commands_;
  mutable std::mutex enqueue_mutex_;
  std::atomic<std::uint64_t> canonical_;
  std::atomic<std::uint64_t> canonical_version_{0};
  std::atomic<bool> accepting_{true};
  std::atomic<std::size_t> outstanding_commands_{0};
  std::function<void()> before_plan_;
  DigestTransition digest_transition_;
  ExecutePlanner execute_planner_;
  std::map<TransactionKey, TransactionRecord> transactions_;
  std::atomic<std::size_t> transaction_count_{0};
  std::array<TransactionKey, kTerminalReplayLimit> terminal_keys_{};
  std::size_t terminal_count_{0};
  std::size_t terminal_cursor_{0};
  // Generation invariant: a first Prepare establishes a strictly increasing
  // transaction id within the highest accepted epoch. A larger epoch begins a
  // new generation (and may reuse ids); messages from older epochs are late.
  std::uint64_t highest_epoch_{0};
  std::uint64_t last_first_transaction_{0};
  // Last: it must not run until every state field is initialized.
  std::thread thread_;
};

}  // namespace grserve
