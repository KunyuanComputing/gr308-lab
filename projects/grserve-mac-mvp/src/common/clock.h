#pragma once

#include <chrono>
#include <limits>
#include <mutex>

namespace grserve {

class Clock {
 public:
  using clock = std::chrono::steady_clock;
  using time_point = clock::time_point;
  using duration = clock::duration;

  virtual ~Clock() = default;
  virtual time_point Now() const noexcept = 0;
};

// A deterministic clock for tests and mock control.  Negative advances are
// ignored, so callers cannot move virtual time backwards in release builds.
class ManualClock final : public Clock {
 public:
  ManualClock() noexcept = default;

  time_point Now() const noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    return now_;
  }

  void Advance(duration delta) {
    if (delta <= duration::zero()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (delta > time_point::max() - now_) {
      now_ = time_point::max();
      return;
    }
    now_ += delta;
  }

 private:
  mutable std::mutex mutex_;
  time_point now_{};
};

}  // namespace grserve
