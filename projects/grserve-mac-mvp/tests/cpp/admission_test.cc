#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "engine/admission.h"

namespace {

using grserve::AdmissionController;
using grserve::RankCapacity;
using grserve::RankId;
using grserve::RequestId;
using grserve::ReservationLease;
using grserve::ErrorCode;

static_assert(!std::is_copy_constructible_v<ReservationLease>);
static_assert(std::is_nothrow_move_constructible_v<ReservationLease>);

TEST(AdmissionController, FailureDoesNotPartiallyReserveAnyRank) {
  auto controller = AdmissionController::Create(
      {RankCapacity{RankId{0}, 4}, RankCapacity{RankId{1}, 2}}).value();
  auto first = controller.ReserveAll(RequestId{1}, 2);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(controller.used_slots(RankId{0}).value(), 2U);
  EXPECT_EQ(controller.used_slots(RankId{1}).value(), 2U);

  auto second = controller.ReserveAll(RequestId{2}, 1);
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(controller.used_slots(RankId{0}).value(), 2U);
  EXPECT_EQ(controller.used_slots(RankId{1}).value(), 2U);
}

TEST(AdmissionController, ResetMoveAndDuplicateRequestReleaseExactlyOnce) {
  auto controller = AdmissionController::Create(
      {RankCapacity{RankId{0}, 3}, RankCapacity{RankId{1}, 3}}).value();
  auto first = controller.ReserveAll(RequestId{3}, 2).value();
  EXPECT_EQ(controller.active_reservation_count(), 1U);
  EXPECT_FALSE(controller.ReserveAll(RequestId{3}, 1).ok());
  ReservationLease moved = std::move(first);
  EXPECT_FALSE(first.active());
  moved.reset();
  moved.reset();
  EXPECT_EQ(controller.used_slots(RankId{0}).value(), 0U);
  EXPECT_EQ(controller.used_slots(RankId{1}).value(), 0U);
  EXPECT_EQ(controller.active_reservation_count(), 0U);
}

TEST(AdmissionController, MoveAssignmentReleasesDestinationAndSelfMoveIsInert) {
  auto controller = AdmissionController::Create(
      {RankCapacity{RankId{1}, 4}, RankCapacity{RankId{0}, 4}}).value();
  auto source = controller.ReserveAll(RequestId{70}, 1).value();
  auto destination = controller.ReserveAll(RequestId{71}, 2).value();
  EXPECT_EQ(controller.active_reservation_count(), 2U);
  destination = std::move(source);
  EXPECT_FALSE(source.active());
  EXPECT_TRUE(destination.active());
  EXPECT_EQ(destination.request_id(), RequestId{70});
  EXPECT_EQ(controller.active_reservation_count(), 1U);
  EXPECT_EQ(controller.used_slots(RankId{1}).value(), 1U);
  destination = std::move(destination);
  EXPECT_TRUE(destination.active());
  EXPECT_EQ(controller.used_slots(RankId{0}).value(), 1U);
  destination.reset();
  EXPECT_EQ(controller.active_reservation_count(), 0U);
  EXPECT_EQ(controller.used_slots(RankId{0}).value(), 0U);
}

TEST(AdmissionController, ValidatesConfigurationRequestsAndUnknownRanks) {
  EXPECT_FALSE(AdmissionController::Create({}).ok());
  EXPECT_FALSE(AdmissionController::Create(
      {RankCapacity{RankId{0}, 0}}).ok());
  EXPECT_FALSE(AdmissionController::Create(
      {RankCapacity{RankId{0}, 1}, RankCapacity{RankId{0}, 2}}).ok());
  auto controller = AdmissionController::Create({RankCapacity{RankId{0}, 1}}).value();
  EXPECT_EQ(controller.ReserveAll(RequestId{0}, 1).error()->code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(controller.ReserveAll(RequestId{1}, 0).error()->code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(controller.ReserveAll(RequestId{2}, std::numeric_limits<std::uint32_t>::max())
                .error()->code(),
            ErrorCode::kResourceExhausted);
  EXPECT_FALSE(controller.used_slots(RankId{9}).ok());
  EXPECT_FALSE(controller.capacity(RankId{9}).ok());
}

TEST(AdmissionController, ConcurrentFailuresDoNotChangeHeldReservation) {
  auto controller = AdmissionController::Create(
      {RankCapacity{RankId{0}, 1}, RankCapacity{RankId{1}, 1}}).value();
  auto held = controller.ReserveAll(RequestId{30}, 1).value();
  std::vector<std::thread> threads;
  for (std::uint64_t index = 1; index <= 8; ++index) {
    threads.emplace_back([&controller, index] {
      EXPECT_FALSE(controller.ReserveAll(RequestId{40 + index}, 1).ok());
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(controller.used_slots(RankId{0}).value(), 1U);
  EXPECT_EQ(controller.used_slots(RankId{1}).value(), 1U);
  held.reset();
  EXPECT_EQ(controller.used_slots(RankId{0}).value(), 0U);
  EXPECT_EQ(controller.used_slots(RankId{1}).value(), 0U);
}

TEST(AdmissionController, SupportsMultipleLeasesAndOutlivesItsController) {
  ReservationLease escaped = [] {
    auto controller = AdmissionController::Create(
        {RankCapacity{RankId{0}, 4}, RankCapacity{RankId{1}, 4}}).value();
    auto first = controller.ReserveAll(RequestId{11}, 1).value();
    auto second = controller.ReserveAll(RequestId{12}, 2).value();
    EXPECT_EQ(controller.used_slots(RankId{0}).value(), 3U);
    first.reset();
    return second;
  }();
  EXPECT_TRUE(escaped.active());
  escaped.reset();
  EXPECT_FALSE(escaped.active());
}

TEST(AdmissionController, ConcurrentReserveAndReleaseLeavesNoSlotsUsed) {
  auto controller = AdmissionController::Create(
      {RankCapacity{RankId{0}, 32}, RankCapacity{RankId{1}, 32}}).value();
  std::vector<std::thread> threads;
  for (std::uint64_t index = 1; index <= 16; ++index) {
    threads.emplace_back([&controller, index] {
      auto lease = controller.ReserveAll(RequestId{100 + index}, 1);
      ASSERT_TRUE(lease.ok());
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(controller.used_slots(RankId{0}).value(), 0U);
  EXPECT_EQ(controller.used_slots(RankId{1}).value(), 0U);
  EXPECT_EQ(controller.active_reservation_count(), 0U);
}

}  // namespace
