#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "grserve/error.h"

namespace grserve {
struct RankCapacity { RankId rank_id; std::uint32_t max_slots; };
class ReservationLease;
class AdmissionController {
 public:
  [[nodiscard]] static StatusOr<AdmissionController> Create(std::vector<RankCapacity> capacities);
  [[nodiscard]] StatusOr<ReservationLease> ReserveAll(RequestId request, std::uint32_t state_slots);
  [[nodiscard]] StatusOr<std::uint32_t> used_slots(RankId rank) const;
  [[nodiscard]] StatusOr<std::uint32_t> capacity(RankId rank) const;
 [[nodiscard]] std::size_t active_reservation_count() const noexcept;
 private:
  friend class ReservationLease;
  struct SharedState;
  explicit AdmissionController(std::shared_ptr<SharedState> state) noexcept : state_(std::move(state)) {}
  std::shared_ptr<SharedState> state_;
};
class ReservationLease {
 public:
  ReservationLease() = delete;
  ReservationLease(const ReservationLease&) = delete;
  ReservationLease& operator=(const ReservationLease&) = delete;
  ReservationLease(ReservationLease&& other) noexcept;
  ReservationLease& operator=(ReservationLease&& other) noexcept;
  ~ReservationLease();
  void reset() noexcept;
  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] RequestId request_id() const noexcept { return request_; }
 private:
  friend class AdmissionController;
  ReservationLease(std::shared_ptr<AdmissionController::SharedState> state, RequestId request, std::uint32_t slots) noexcept;
  std::shared_ptr<AdmissionController::SharedState> state_;
  RequestId request_{0}; std::uint32_t slots_{0}; bool active_{false};
};
}  // namespace grserve
