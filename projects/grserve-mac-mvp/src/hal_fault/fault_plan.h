#pragma once

#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "grserve/error.h"
#include "grserve/types.h"

namespace grserve {

enum class FaultAction { kFail, kBlock, kTimeout, kDropAck, kRankLoss };
enum class FaultPhase { kReserve, kPrepare, kExecute, kCommit, kAbort, kHalLaunch };

struct FaultRule {
  RankId rank;
  FaultPhase phase;
  std::uint64_t occurrence;
  FaultAction action;
  std::uint64_t delay_ticks;
};

struct FaultEffect {
  FaultAction action;
  std::uint64_t delay_ticks;
  RankId rank;
  FaultPhase phase;
  std::uint64_t occurrence;
};

[[nodiscard]] const char* FaultActionText(FaultAction action) noexcept;
[[nodiscard]] const char* FaultPhaseText(FaultPhase phase) noexcept;

class FaultPlan {
 public:
  // The constructor rejects malformed rules with std::invalid_argument. Prefer
  // Create() at boundaries where an Error status is more useful to callers.
  FaultPlan(std::uint64_t seed, std::vector<FaultRule> rules);
  FaultPlan(std::uint64_t seed, std::initializer_list<FaultRule> rules)
      : FaultPlan(seed, std::vector<FaultRule>(rules)) {}
  FaultPlan(const FaultPlan&) = delete;
  FaultPlan& operator=(const FaultPlan&) = delete;
  FaultPlan(FaultPlan&& other) noexcept;
  FaultPlan& operator=(FaultPlan&& other) noexcept;

  [[nodiscard]] static StatusOr<FaultPlan> Create(
      std::uint64_t seed, std::vector<FaultRule> rules);

  // Match advances a per-(rank, phase) counter. It is thread-safe and does
  // not wrap: calls after UINT64_MAX return no effect for that key.
  [[nodiscard]] std::optional<FaultEffect> Match(RankId rank, FaultPhase phase);
  // Pure lookup for controller-supplied ordinals (used by retry-idempotent hooks).
  [[nodiscard]] std::optional<FaultEffect> EffectAt(
      RankId rank, FaultPhase phase, std::uint64_t occurrence) const noexcept;
  [[nodiscard]] std::string CanonicalDigest() const;
  [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }

 private:
  struct Key {
    RankId rank;
    FaultPhase phase;
    bool operator==(const Key&) const = default;
  };
  struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept;
  };

  static void ValidateOrThrow(const std::vector<FaultRule>& rules);

  std::uint64_t seed_;
  std::vector<FaultRule> rules_;
  mutable std::mutex mutex_;
  std::unordered_map<Key, std::uint64_t, KeyHash> observed_occurrences_;
};

}  // namespace grserve
