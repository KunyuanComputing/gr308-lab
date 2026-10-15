#include "engine/replica_gate.h"

#include <utility>

namespace grserve {

Status ReplicaGate::CheckAdmission() const {
  std::lock_guard lock(mutex_);
  if (!fatal_error_) return Status::Ok();
  return Status{Error::From(ErrorCode::kReplicaFatal,
                            "replica is unavailable after a fatal transaction")};
}

void ReplicaGate::MarkFatal(Error error) {
  std::lock_guard lock(mutex_);
  if (!fatal_error_) fatal_error_.emplace(std::move(error));
}

ReplicaHealth ReplicaGate::health() const noexcept {
  std::lock_guard lock(mutex_);
  return fatal_error_ ? ReplicaHealth::kFatal : ReplicaHealth::kHealthy;
}

std::optional<Error> ReplicaGate::fatal_error() const {
  std::lock_guard lock(mutex_);
  return fatal_error_;
}

}  // namespace grserve
