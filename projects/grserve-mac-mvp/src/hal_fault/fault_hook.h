#pragma once

#include <cstdint>
#include <cstddef>
#include <locale>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "hal_fault/fault_plan.h"

namespace grserve {

class IFaultHook {
 public:
  virtual ~IFaultHook() = default;
  virtual std::optional<FaultEffect> Before(RankId rank, FaultPhase phase,
                                            std::uint64_t ordinal) = 0;
  virtual std::optional<FaultEffect> After(RankId rank, FaultPhase phase,
                                           std::uint64_t ordinal) = 0;
};

// Maps pre-operation actions (fail/block/timeout/rank_loss) to Before and the
// acknowledgement-only action (drop_ack) to After. Calls are idempotent per
// rank/phase/ordinal/point, making controller retries reproducible.
class PlanFaultHook final : public IFaultHook {
 public:
  using CommitProbe = void (*)();

  // commit_probe is an in-process test seam; it is not part of the C HAL ABI.
  explicit PlanFaultHook(const FaultPlan& plan,
                         CommitProbe commit_probe = nullptr)
      : plan_(plan), commit_probe_(commit_probe) {}

  std::optional<FaultEffect> Before(RankId rank, FaultPhase phase,
                                    std::uint64_t ordinal) override {
    return At(rank, phase, ordinal, Point::kBefore);
  }
  std::optional<FaultEffect> After(RankId rank, FaultPhase phase,
                                   std::uint64_t ordinal) override {
    return At(rank, phase, ordinal, Point::kAfter);
  }

  [[nodiscard]] std::string ObservedDigest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "seed=" << plan_.seed();
    for (const Event& event : events_) {
      output << ";rank=" << event.rank.value()
             << ",phase=" << FaultPhaseText(event.phase)
             << ",ordinal=" << event.ordinal
             << ",point=" << (event.point == Point::kBefore ? "before" : "after")
             << ",action=" << FaultActionText(event.effect.action)
             << ",delay=" << event.effect.delay_ticks;
    }
    return output.str();
  }

  [[nodiscard]] std::size_t CachedResponseCountForTesting() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return responses_.size();
  }

  [[nodiscard]] std::size_t ObservedEventCountForTesting() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_.size();
  }

 private:
  enum class Point { kBefore, kAfter };
  struct EventKey {
    RankId rank;
    FaultPhase phase;
    std::uint64_t ordinal;
    Point point;
    bool operator==(const EventKey&) const = default;
  };
  struct EventKeyHash {
    std::size_t operator()(const EventKey& key) const noexcept {
      std::size_t value = static_cast<std::size_t>(key.rank.value());
      value = value * 131U + static_cast<std::size_t>(key.phase);
      value = value * 131U + static_cast<std::size_t>(key.ordinal);
      return value * 2U + static_cast<std::size_t>(key.point);
    }
  };
  struct Event {
    RankId rank;
    FaultPhase phase;
    std::uint64_t ordinal;
    Point point;
    FaultEffect effect;
  };
  std::optional<FaultEffect> At(RankId rank, FaultPhase phase,
                                std::uint64_t ordinal, Point point) {
    const EventKey key{rank, phase, ordinal, point};
    std::lock_guard<std::mutex> lock(mutex_);
    const auto previous = responses_.find(key);
    if (previous != responses_.end()) {
      return previous->second;
    }
    std::optional<FaultEffect> effect = plan_.EffectAt(rank, phase, ordinal);
    if (effect.has_value() &&
        ((point == Point::kBefore && effect->action == FaultAction::kDropAck) ||
         (point == Point::kAfter && effect->action != FaultAction::kDropAck))) {
      effect.reset();
    }
    if (!effect.has_value()) {
      return std::nullopt;
    }

    // Reserve before publishing the cache entry, so a failed allocation cannot
    // leave a retry-idempotent response without its corresponding event.
    events_.reserve(events_.size() + 1U);
    if (commit_probe_ != nullptr) {
      commit_probe_();
    }
    const auto [stored, inserted] = responses_.emplace(key, *effect);
    if (!inserted) {
      return stored->second;
    }
    try {
      events_.push_back(Event{rank, phase, ordinal, point, *effect});
    } catch (...) {
      responses_.erase(stored);
      throw;
    }
    return stored->second;
  }

  const FaultPlan& plan_;
  CommitProbe commit_probe_;
  mutable std::mutex mutex_;
  std::unordered_map<EventKey, FaultEffect, EventKeyHash> responses_;
  std::vector<Event> events_;
};

}  // namespace grserve
