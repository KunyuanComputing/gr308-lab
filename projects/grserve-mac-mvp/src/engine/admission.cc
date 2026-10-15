#include "engine/admission.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace grserve {
struct AdmissionController::SharedState {
  struct Counter { RankId rank_id; std::uint32_t capacity; std::uint32_t used{0}; };
  mutable std::mutex mutex;
  std::vector<Counter> counters;
  std::unordered_map<std::uint64_t, std::uint32_t> active_requests;
};
namespace {
Error AdmissionError(ErrorCode code, const char* message, std::optional<RequestId> request = std::nullopt) { return Error::From(code, message, request); }
}  // namespace
StatusOr<AdmissionController> AdmissionController::Create(std::vector<RankCapacity> capacities) {
  if (capacities.empty()) return StatusOr<AdmissionController>{AdmissionError(ErrorCode::kInvalidArgument, "at least one rank is required")};
  for (std::size_t index = 0; index < capacities.size(); ++index) {
    bool duplicate = false;
    for (std::size_t previous = 0; previous < index; ++previous) {
      duplicate = duplicate || capacities[previous].rank_id == capacities[index].rank_id;
    }
    if (capacities[index].max_slots == 0 || duplicate)
      return StatusOr<AdmissionController>{AdmissionError(ErrorCode::kInvalidArgument, "rank capacities must be unique and nonzero")};
  }
  auto state = std::make_shared<SharedState>(); state->counters.reserve(capacities.size());
  for (const auto capacity : capacities) state->counters.push_back({capacity.rank_id, capacity.max_slots, 0});
  return StatusOr<AdmissionController>{AdmissionController{std::move(state)}};
}
StatusOr<ReservationLease> AdmissionController::ReserveAll(RequestId request, std::uint32_t state_slots) {
  if (request.value() == 0 || state_slots == 0) return StatusOr<ReservationLease>{AdmissionError(ErrorCode::kInvalidArgument, "request and reservation slots must be nonzero", request)};
  std::lock_guard lock(state_->mutex);
  if (state_->active_requests.contains(request.value())) return StatusOr<ReservationLease>{AdmissionError(ErrorCode::kInvalidArgument, "request already has a reservation", request)};
  for (const auto& counter : state_->counters)
    if (counter.used > counter.capacity || state_slots > counter.capacity - counter.used)
      return StatusOr<ReservationLease>{AdmissionError(ErrorCode::kResourceExhausted, "rank has insufficient reservation capacity", request)};
  state_->active_requests.emplace(request.value(), state_slots);
  for (auto& counter : state_->counters) counter.used += state_slots;
  return StatusOr<ReservationLease>{ReservationLease{state_, request, state_slots}};
}
StatusOr<std::uint32_t> AdmissionController::used_slots(RankId rank) const {
  std::lock_guard lock(state_->mutex);
  const auto found = std::find_if(state_->counters.begin(), state_->counters.end(), [rank](const auto& counter) { return counter.rank_id == rank; });
  if (found == state_->counters.end()) return StatusOr<std::uint32_t>{AdmissionError(ErrorCode::kInvalidArgument, "unknown rank")};
  return StatusOr<std::uint32_t>{found->used};
}
StatusOr<std::uint32_t> AdmissionController::capacity(RankId rank) const {
  std::lock_guard lock(state_->mutex);
  const auto found = std::find_if(state_->counters.begin(), state_->counters.end(), [rank](const auto& counter) { return counter.rank_id == rank; });
  if (found == state_->counters.end()) return StatusOr<std::uint32_t>{AdmissionError(ErrorCode::kInvalidArgument, "unknown rank")};
  return StatusOr<std::uint32_t>{found->capacity};
}
std::size_t AdmissionController::active_reservation_count() const noexcept { std::lock_guard lock(state_->mutex); return state_->active_requests.size(); }
ReservationLease::ReservationLease(std::shared_ptr<AdmissionController::SharedState> state, RequestId request, std::uint32_t slots) noexcept : state_(std::move(state)), request_(request), slots_(slots), active_(true) {}
ReservationLease::ReservationLease(ReservationLease&& other) noexcept : state_(std::move(other.state_)), request_(other.request_), slots_(other.slots_), active_(other.active_) { other.active_ = false; other.request_ = RequestId{0}; other.slots_ = 0; }
ReservationLease& ReservationLease::operator=(ReservationLease&& other) noexcept {
  if (this != &other) { reset(); state_ = std::move(other.state_); request_ = other.request_; slots_ = other.slots_; active_ = other.active_; other.active_ = false; other.request_ = RequestId{0}; other.slots_ = 0; }
  return *this;
}
ReservationLease::~ReservationLease() { reset(); }
void ReservationLease::reset() noexcept {
  if (!active_) return;
  auto state = std::move(state_);
  {
    std::lock_guard lock(state->mutex);
    const auto found = state->active_requests.find(request_.value());
    if (found != state->active_requests.end()) {
      for (auto& counter : state->counters) counter.used -= slots_;
      state->active_requests.erase(found);
    }
  }
  active_ = false;
  request_ = RequestId{0};
  slots_ = 0;
}
}  // namespace grserve
