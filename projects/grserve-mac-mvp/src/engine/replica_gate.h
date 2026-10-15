#pragma once

#include <mutex>
#include <optional>

#include "grserve/error.h"

namespace grserve {

enum class ReplicaHealth { kHealthy, kFatal };

class ReplicaGate {
 public:
  [[nodiscard]] Status CheckAdmission() const;
  void MarkFatal(Error error);
  [[nodiscard]] ReplicaHealth health() const noexcept;
  [[nodiscard]] std::optional<Error> fatal_error() const;

 private:
  mutable std::mutex mutex_;
  std::optional<Error> fatal_error_;
};

}  // namespace grserve
