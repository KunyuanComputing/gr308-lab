#include "hal_fault/fault_plan.h"

#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace grserve {
namespace {

struct RuleKey {
  std::uint32_t rank;
  FaultPhase phase;
  std::uint64_t occurrence;
  bool operator==(const RuleKey&) const = default;
};

struct RuleKeyHash {
  std::size_t operator()(const RuleKey& key) const noexcept {
    std::size_t value = static_cast<std::size_t>(key.rank);
    value = value * 131U + static_cast<std::size_t>(key.phase);
    return value * 131U + static_cast<std::size_t>(key.occurrence);
  }
};

bool IsKnownAction(FaultAction action) noexcept {
  return action == FaultAction::kFail || action == FaultAction::kBlock ||
         action == FaultAction::kTimeout || action == FaultAction::kDropAck ||
         action == FaultAction::kRankLoss;
}

bool IsKnownPhase(FaultPhase phase) noexcept {
  return phase == FaultPhase::kReserve || phase == FaultPhase::kPrepare ||
         phase == FaultPhase::kExecute || phase == FaultPhase::kCommit ||
         phase == FaultPhase::kAbort || phase == FaultPhase::kHalLaunch;
}

}  // namespace

const char* FaultActionText(FaultAction action) noexcept {
  switch (action) {
    case FaultAction::kFail: return "fail";
    case FaultAction::kBlock: return "block";
    case FaultAction::kTimeout: return "timeout";
    case FaultAction::kDropAck: return "drop_ack";
    case FaultAction::kRankLoss: return "rank_loss";
  }
  return "invalid";
}

const char* FaultPhaseText(FaultPhase phase) noexcept {
  switch (phase) {
    case FaultPhase::kReserve: return "reserve";
    case FaultPhase::kPrepare: return "prepare";
    case FaultPhase::kExecute: return "execute";
    case FaultPhase::kCommit: return "commit";
    case FaultPhase::kAbort: return "abort";
    case FaultPhase::kHalLaunch: return "hal_launch";
  }
  return "invalid";
}

FaultPlan::FaultPlan(std::uint64_t seed, std::vector<FaultRule> rules)
    : seed_(seed), rules_(std::move(rules)) {
  ValidateOrThrow(rules_);
}

FaultPlan::FaultPlan(FaultPlan&& other) noexcept {
  std::lock_guard<std::mutex> lock(other.mutex_);
  seed_ = other.seed_;
  rules_ = std::move(other.rules_);
  observed_occurrences_ = std::move(other.observed_occurrences_);
}

FaultPlan& FaultPlan::operator=(FaultPlan&& other) noexcept {
  if (this != &other) {
    std::scoped_lock lock(mutex_, other.mutex_);
    seed_ = other.seed_;
    rules_ = std::move(other.rules_);
    observed_occurrences_ = std::move(other.observed_occurrences_);
  }
  return *this;
}

StatusOr<FaultPlan> FaultPlan::Create(std::uint64_t seed,
                                      std::vector<FaultRule> rules) {
  try {
    return StatusOr<FaultPlan>(FaultPlan(seed, std::move(rules)));
  } catch (const std::invalid_argument& error) {
    return StatusOr<FaultPlan>(
        Error::From(ErrorCode::kInvalidArgument, error.what()));
  }
}

std::size_t FaultPlan::KeyHash::operator()(const Key& key) const noexcept {
  return (static_cast<std::size_t>(key.rank.value()) << 3U) ^
         static_cast<std::size_t>(key.phase);
}

void FaultPlan::ValidateOrThrow(const std::vector<FaultRule>& rules) {
  std::unordered_set<RuleKey, RuleKeyHash> seen;
  for (const FaultRule& rule : rules) {
    if (rule.occurrence == 0U) {
      throw std::invalid_argument("fault rule occurrence must be positive");
    }
    if (!IsKnownPhase(rule.phase) || !IsKnownAction(rule.action)) {
      throw std::invalid_argument("fault rule action or phase is invalid");
    }
    const RuleKey key{rule.rank.value(), rule.phase, rule.occurrence};
    if (!seen.insert(key).second) {
      throw std::invalid_argument("duplicate fault rule rank/phase/occurrence");
    }
  }
}

std::optional<FaultEffect> FaultPlan::EffectAt(
    RankId rank, FaultPhase phase, std::uint64_t occurrence) const noexcept {
  if (occurrence == 0U) {
    return std::nullopt;
  }
  for (const FaultRule& rule : rules_) {
    if (rule.rank == rank && rule.phase == phase && rule.occurrence == occurrence) {
      return FaultEffect{rule.action, rule.delay_ticks, rank, phase, occurrence};
    }
  }
  return std::nullopt;
}

std::optional<FaultEffect> FaultPlan::Match(RankId rank, FaultPhase phase) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::uint64_t& ordinal = observed_occurrences_[Key{rank, phase}];
  if (ordinal == std::numeric_limits<std::uint64_t>::max()) {
    return std::nullopt;
  }
  ++ordinal;
  return EffectAt(rank, phase, ordinal);
}

std::string FaultPlan::CanonicalDigest() const {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "seed=" << seed_;
  for (const FaultRule& rule : rules_) {
    output << ";rank=" << rule.rank.value()
           << ",phase=" << FaultPhaseText(rule.phase)
           << ",occurrence=" << rule.occurrence
           << ",action=" << FaultActionText(rule.action)
           << ",delay=" << rule.delay_ticks;
  }
  return output.str();
}

}  // namespace grserve
