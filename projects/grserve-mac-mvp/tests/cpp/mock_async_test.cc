#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/clock.h"
#include "hal/hal_library.h"
#include "hal/hal_raii.h"
#include "hal_mock/mock_scheduler.h"

namespace {

using grserve::Error;
using grserve::ErrorCode;
using grserve::ManualClock;
using grserve::RankId;
using grserve::TokenId;
using grserve::hal_mock::Completion;
using grserve::hal_mock::MockScheduler;
using grserve::hal_mock::ScriptedResult;

ScriptedResult Result(std::uint32_t token, std::uint64_t digest,
                      ManualClock::duration duration =
                          std::chrono::nanoseconds::zero()) {
  return ScriptedResult{TokenId{token},
                        grserve::hal_mock::CanonicalStateDigest{digest},
                        duration};
}

std::string SequenceDigest(const std::vector<Completion>& completions) {
  std::ostringstream out;
  for (const Completion& completion : completions) {
    out << completion.ordinal().value() << ':' << completion.operation_id().value()
        << ':' << completion.rank_id().value() << ':';
    if (completion.success()) {
      out << completion.result().candidate_token().value() << ':'
          << completion.result().state_digest().value() << ';';
    } else {
      out << static_cast<int>(completion.error().code()) << ':'
          << completion.error().message() << ';';
    }
  }
  return out.str();
}

TEST(MockScheduler, CompletesInControllerTransitionOrder) {
  ManualClock clock;
  MockScheduler scheduler(clock);
  const auto first = scheduler.Submit(RankId{0}, Result(7, 0xabc));
  const auto second = scheduler.Submit(RankId{1}, Result(9, 0xdef));

  EXPECT_EQ(first.value(), 1U);
  EXPECT_EQ(second.value(), 2U);
  EXPECT_TRUE(scheduler.IsPending(first));
  EXPECT_TRUE(scheduler.IsPending(second));
  ASSERT_TRUE(scheduler.Complete(first).ok());
  ASSERT_TRUE(scheduler.Complete(second).ok());

  const std::vector<Completion> completions = scheduler.DrainReady();
  ASSERT_EQ(completions.size(), 2U);
  EXPECT_EQ(SequenceDigest(completions), "1:1:0:7:2748;2:2:1:9:3567;");
  EXPECT_FALSE(scheduler.IsPending(first));
  EXPECT_FALSE(scheduler.IsPending(second));
  EXPECT_TRUE(scheduler.DrainReady().empty());
}

TEST(MockScheduler, CompletionSequenceDoesNotFollowRankOrSubmissionOrder) {
  ManualClock clock;
  MockScheduler scheduler(clock);
  const auto first = scheduler.Submit(RankId{0}, Result(7, 1));
  const auto second = scheduler.Submit(RankId{1}, Result(8, 2));

  ASSERT_TRUE(scheduler.Complete(second).ok());
  ASSERT_TRUE(scheduler.Complete(first).ok());

  const std::vector<Completion> completions = scheduler.DrainReady();
  ASSERT_EQ(completions.size(), 2U);
  EXPECT_EQ(SequenceDigest(completions), "1:2:1:8:2;2:1:0:7:1;");
}

TEST(MockScheduler, VirtualDeadlineKeepsEarlyOperationPending) {
  ManualClock clock;
  MockScheduler scheduler(clock);
  const auto operation = scheduler.Submit(
      RankId{0}, Result(11, 42, std::chrono::nanoseconds{5}));

  const auto early = scheduler.Complete(operation);
  ASSERT_FALSE(early.ok());
  EXPECT_EQ(early.error()->code(), ErrorCode::kDeadlineExceeded);
  EXPECT_TRUE(scheduler.IsPending(operation));
  EXPECT_TRUE(scheduler.DrainReady().empty());

  clock.Advance(std::chrono::nanoseconds{5});
  ASSERT_TRUE(scheduler.Complete(operation).ok());
  const auto completions = scheduler.DrainReady();
  ASSERT_EQ(completions.size(), 1U);
  EXPECT_TRUE(completions.front().success());
  EXPECT_EQ(completions.front().result().candidate_token(), TokenId{11});
  EXPECT_EQ(completions.front().result().state_digest().value(), 42U);
}

TEST(MockScheduler, FailureRetainsExactErrorAndDrainsOnce) {
  ManualClock clock;
  MockScheduler scheduler(clock);
  const auto first = scheduler.Submit(RankId{0}, Result(7, 1));
  const auto second = scheduler.Submit(RankId{1}, Result(8, 2));
  const Error failure = Error::From(ErrorCode::kRankFailure, "rank one failed");

  ASSERT_TRUE(scheduler.Fail(second, failure).ok());
  ASSERT_TRUE(scheduler.Complete(first).ok());
  const auto completions = scheduler.DrainReady();
  ASSERT_EQ(completions.size(), 2U);
  EXPECT_EQ(SequenceDigest(completions),
            "1:2:1:5:rank one failed;2:1:0:7:1;");
  EXPECT_FALSE(completions[0].success());
  EXPECT_EQ(completions[0].error().code(), ErrorCode::kRankFailure);
  EXPECT_EQ(completions[0].error().message(), "rank one failed");
  EXPECT_TRUE(scheduler.DrainReady().empty());

  const auto duplicate = scheduler.Fail(second, failure);
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.error()->code(), ErrorCode::kInvalidArgument);
  const auto duplicate_complete = scheduler.Complete(first);
  ASSERT_FALSE(duplicate_complete.ok());
  EXPECT_EQ(duplicate_complete.error()->code(), ErrorCode::kInvalidArgument);
  const auto unknown = scheduler.Complete(grserve::hal_mock::OperationId{999});
  ASSERT_FALSE(unknown.ok());
  EXPECT_EQ(unknown.error()->code(), ErrorCode::kInvalidArgument);
  const auto unknown_fail =
      scheduler.Fail(grserve::hal_mock::OperationId{998}, failure);
  ASSERT_FALSE(unknown_fail.ok());
  EXPECT_EQ(unknown_fail.error()->code(), ErrorCode::kInvalidArgument);
}

TEST(MockScheduler, NegativeClockAdvanceAndResultDurationAreReleaseSafe) {
  ManualClock clock;
  const auto epoch = clock.Now();
  clock.Advance(std::chrono::nanoseconds{-1});
  EXPECT_EQ(clock.Now(), epoch);

  MockScheduler scheduler(clock);
  const auto operation = scheduler.Submit(
      RankId{0}, Result(3, 4, std::chrono::nanoseconds{-9}));
  ASSERT_TRUE(scheduler.Complete(operation).ok());
  const auto completions = scheduler.DrainReady();
  ASSERT_EQ(completions.size(), 1U);
  EXPECT_EQ(completions.front().ordinal().value(), 1U);
}

TEST(MockScheduler, ReadyQueueFailureLeavesOperationPendingAndOrdinalReusable) {
  ManualClock clock;
  MockScheduler scheduler(clock);
  const auto operation = scheduler.Submit(RankId{0}, Result(17, 18));

  grserve::hal_mock::testing::FailNextReadyQueueAppend();
  EXPECT_THROW({ (void)scheduler.Complete(operation); }, std::bad_alloc);
  EXPECT_TRUE(scheduler.IsPending(operation));
  EXPECT_TRUE(scheduler.DrainReady().empty());

  ASSERT_TRUE(scheduler.Complete(operation).ok());
  const auto completions = scheduler.DrainReady();
  ASSERT_EQ(completions.size(), 1U);
  EXPECT_EQ(completions[0].ordinal().value(), 1U);
}

TEST(MockScheduler, FailedTransitionAlsoRetainsPendingStateWhenReadyAppendThrows) {
  ManualClock clock;
  MockScheduler scheduler(clock);
  const auto operation = scheduler.Submit(RankId{1}, Result(19, 20));
  const Error failure = Error::From(ErrorCode::kRankFailure, "injected failure");

  grserve::hal_mock::testing::FailNextReadyQueueAppend();
  EXPECT_THROW({ (void)scheduler.Fail(operation, failure); }, std::bad_alloc);
  EXPECT_TRUE(scheduler.IsPending(operation));
  EXPECT_TRUE(scheduler.DrainReady().empty());

  ASSERT_TRUE(scheduler.Fail(operation, failure).ok());
  const auto completions = scheduler.DrainReady();
  ASSERT_EQ(completions.size(), 1U);
  EXPECT_EQ(completions[0].ordinal().value(), 1U);
  EXPECT_FALSE(completions[0].success());
}

TEST(MockScheduler, RepeatedOrderScenarioHasNoSharedState) {
  for (int iteration = 0; iteration != 20; ++iteration) {
    ManualClock clock;
    MockScheduler scheduler(clock);
    const auto first = scheduler.Submit(RankId{0}, Result(1, 1));
    const auto second = scheduler.Submit(RankId{1}, Result(2, 2));
    ASSERT_TRUE(scheduler.Complete(second).ok());
    ASSERT_TRUE(scheduler.Complete(first).ok());
    EXPECT_EQ(SequenceDigest(scheduler.DrainReady()), "1:2:1:2:2;2:1:0:1:1;");
  }
}

TEST(MockHal, LoadsAndAdvancesEventOnlyThroughVirtualTimeout) {
  const auto opened = grserve::HalLibrary::Open(
      MOCK_HAL_PATH, GRSERVE_HAL_CAPABILITY_STREAMS | GRSERVE_HAL_CAPABILITY_EVENTS);
  ASSERT_TRUE(opened.ok()) << opened.error()->message();
  const std::shared_ptr<grserve::HalLibrary> library = opened.value();
  const auto runtime = grserve::HalRuntime::Create(library);
  ASSERT_TRUE(runtime.ok());
  const auto device = runtime.value().CreateDevice(0U);
  ASSERT_TRUE(device.ok());
  const auto stream = device.value().CreateStream(0U);
  ASSERT_TRUE(stream.ok());
  const auto event = device.value().CreateEvent(0U);
  ASSERT_TRUE(event.ok());
  const auto buffer = device.value().CreateBuffer(16U, 8U, 0U);
  ASSERT_TRUE(buffer.ok());
  const auto executable = device.value().CreateExecutable();
  ASSERT_TRUE(executable.ok());

  grserve_hal_event_state_t state = GRSERVE_HAL_EVENT_STATE_COMPLETE;
  ASSERT_EQ(library->api().event_query(event.value().get(), &state),
            GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(state, GRSERVE_HAL_EVENT_STATE_PENDING);
  EXPECT_EQ(library->api().event_wait(event.value().get(), 0U),
            GRSERVE_HAL_STATUS_TIMEOUT);
  ASSERT_EQ(library->api().event_query(event.value().get(), &state),
            GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(state, GRSERVE_HAL_EVENT_STATE_PENDING);
  EXPECT_EQ(library->api().event_wait(event.value().get(), 1U),
            GRSERVE_HAL_STATUS_OK);
  ASSERT_EQ(library->api().event_query(event.value().get(), &state),
            GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(state, GRSERVE_HAL_EVENT_STATE_COMPLETE);

  ASSERT_EQ(library->api().event_record(event.value().get(), stream.value().get()),
            GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(library->api().event_wait(event.value().get(), 0U),
            GRSERVE_HAL_STATUS_TIMEOUT);
  EXPECT_EQ(library->api().event_wait(event.value().get(), 1U),
            GRSERVE_HAL_STATUS_OK);
}

TEST(MockHal, RejectsOverflowSafeOutOfBoundsBufferCopies) {
  const auto opened = grserve::HalLibrary::Open(MOCK_HAL_PATH, 0U);
  ASSERT_TRUE(opened.ok()) << opened.error()->message();
  const auto runtime = grserve::HalRuntime::Create(opened.value());
  ASSERT_TRUE(runtime.ok());
  const auto device = runtime.value().CreateDevice(0U);
  ASSERT_TRUE(device.ok());
  const auto stream = device.value().CreateStream(0U);
  ASSERT_TRUE(stream.ok());
  const auto buffer = device.value().CreateBuffer(16U, 1U, 0U);
  ASSERT_TRUE(buffer.ok());

  const auto copy = opened.value()->api().stream_copy;
  EXPECT_EQ(copy(stream.value().get(), buffer.value().get(), 0U,
                 buffer.value().get(), 0U, 17U),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(copy(stream.value().get(), buffer.value().get(), UINT64_MAX,
                 buffer.value().get(), 0U, 0U),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(copy(stream.value().get(), buffer.value().get(), 0U,
                 buffer.value().get(), UINT64_MAX, 0U),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(copy(stream.value().get(), buffer.value().get(), 16U,
                 buffer.value().get(), 16U, 1U),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(copy(stream.value().get(), buffer.value().get(), 16U,
                 buffer.value().get(), 16U, 0U), GRSERVE_HAL_STATUS_OK);
}

TEST(MockHal, ValidatesNestedDescriptorPointersBeforeStateMutation) {
  const auto opened = grserve::HalLibrary::Open(MOCK_HAL_PATH, 0U);
  ASSERT_TRUE(opened.ok());
  const auto runtime = grserve::HalRuntime::Create(opened.value());
  ASSERT_TRUE(runtime.ok());
  const auto device = runtime.value().CreateDevice(0U);
  ASSERT_TRUE(device.ok());
  const auto stream = device.value().CreateStream(0U);
  ASSERT_TRUE(stream.ok());
  const auto event = device.value().CreateEvent(0U);
  ASSERT_TRUE(event.ok());
  const auto executable = device.value().CreateExecutable();
  ASSERT_TRUE(executable.ok());

  grserve_hal_submit_info_t submit = {};
  submit.struct_size = static_cast<uint32_t>(offsetof(grserve_hal_submit_info_t,
                                                       signal_event_count));
  EXPECT_EQ(opened.value()->api().stream_submit(stream.value().get(), &submit),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  submit.struct_size = static_cast<uint32_t>(sizeof(submit));
  submit.wait_event_count = 1U;
  EXPECT_EQ(opened.value()->api().stream_submit(stream.value().get(), &submit),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  submit.wait_event_count = 0U;
  submit.signal_event_count = 1U;
  EXPECT_EQ(opened.value()->api().stream_submit(stream.value().get(), &submit),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  const grserve_hal_event_t null_event = nullptr;
  submit.signal_events = &null_event;
  EXPECT_EQ(opened.value()->api().stream_submit(stream.value().get(), &submit),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  const grserve_hal_event_t waits[] = {event.value().get()};
  submit.signal_events = nullptr;
  submit.signal_event_count = 0U;
  submit.wait_events = waits;
  submit.wait_event_count = 1U;
  EXPECT_EQ(opened.value()->api().stream_submit(stream.value().get(), &submit),
            GRSERVE_HAL_STATUS_TIMEOUT);
  ASSERT_EQ(opened.value()->api().event_wait(event.value().get(), 1U),
            GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(opened.value()->api().stream_submit(stream.value().get(), &submit),
            GRSERVE_HAL_STATUS_OK);

  grserve_hal_launch_info_t launch = {};
  launch.struct_size = static_cast<uint32_t>(sizeof(launch));
  launch.argument_bytes = 1U;
  EXPECT_EQ(opened.value()->api().executable_launch(
                executable.value().get(), stream.value().get(), nullptr, 0U,
                &launch), GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  launch.argument_bytes = 0U;
  EXPECT_EQ(opened.value()->api().executable_launch(
                executable.value().get(), stream.value().get(), nullptr, 1U,
                &launch), GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  launch.struct_size = static_cast<uint32_t>(offsetof(grserve_hal_launch_info_t,
                                                       argument_bytes));
  EXPECT_EQ(opened.value()->api().executable_launch(
                executable.value().get(), stream.value().get(), nullptr, 0U,
                &launch), GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
}

TEST(MockHal, SynchronizeCompletesRecordedAndSignaledEventsWithoutDanglingLinks) {
  const auto opened = grserve::HalLibrary::Open(MOCK_HAL_PATH, 0U);
  ASSERT_TRUE(opened.ok());
  const auto runtime = grserve::HalRuntime::Create(opened.value());
  ASSERT_TRUE(runtime.ok());
  const auto device = runtime.value().CreateDevice(0U);
  ASSERT_TRUE(device.ok());
  auto stream = device.value().CreateStream(0U);
  ASSERT_TRUE(stream.ok());
  const auto recorded = device.value().CreateEvent(0U);
  const auto signaled = device.value().CreateEvent(0U);
  ASSERT_TRUE(recorded.ok());
  ASSERT_TRUE(signaled.ok());
  const auto& api = opened.value()->api();
  grserve_hal_event_state_t state = GRSERVE_HAL_EVENT_STATE_COMPLETE;

  EXPECT_EQ(api.stream_synchronize(stream.value().get(), 0U),
            GRSERVE_HAL_STATUS_OK);
  ASSERT_EQ(api.event_record(recorded.value().get(), stream.value().get()),
            GRSERVE_HAL_STATUS_OK);
  ASSERT_EQ(api.stream_synchronize(stream.value().get(), 0U),
            GRSERVE_HAL_STATUS_TIMEOUT);
  ASSERT_EQ(api.event_query(recorded.value().get(), &state), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(state, GRSERVE_HAL_EVENT_STATE_PENDING);
  ASSERT_EQ(api.stream_synchronize(stream.value().get(), 1U),
            GRSERVE_HAL_STATUS_OK);
  ASSERT_EQ(api.event_query(recorded.value().get(), &state), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(state, GRSERVE_HAL_EVENT_STATE_COMPLETE);

  ASSERT_EQ(api.event_record(recorded.value().get(), stream.value().get()),
            GRSERVE_HAL_STATUS_OK);
  ASSERT_EQ(api.event_query(recorded.value().get(), &state), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(state, GRSERVE_HAL_EVENT_STATE_PENDING);

  const grserve_hal_event_t signals[] = {signaled.value().get()};
  grserve_hal_submit_info_t submit = {};
  submit.struct_size = static_cast<uint32_t>(sizeof(submit));
  submit.signal_events = signals;
  submit.signal_event_count = 1U;
  ASSERT_EQ(api.stream_submit(stream.value().get(), &submit), GRSERVE_HAL_STATUS_OK);
  ASSERT_EQ(api.event_query(signaled.value().get(), &state), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(state, GRSERVE_HAL_EVENT_STATE_PENDING);
  ASSERT_EQ(api.stream_synchronize(stream.value().get(), 0U),
            GRSERVE_HAL_STATUS_TIMEOUT);
  ASSERT_EQ(api.event_query(recorded.value().get(), &state), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(state, GRSERVE_HAL_EVENT_STATE_PENDING);
  ASSERT_EQ(api.event_query(signaled.value().get(), &state), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(state, GRSERVE_HAL_EVENT_STATE_PENDING);
  ASSERT_EQ(api.stream_synchronize(stream.value().get(), UINT64_MAX),
            GRSERVE_HAL_STATUS_OK);
  ASSERT_EQ(api.event_query(signaled.value().get(), &state), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(state, GRSERVE_HAL_EVENT_STATE_COMPLETE);
  EXPECT_EQ(api.stream_synchronize(stream.value().get(), 0U),
            GRSERVE_HAL_STATUS_OK);

  grserve_hal_event_t destroyed = nullptr;
  ASSERT_EQ(api.event_create(device.value().get(), 0U, &destroyed),
            GRSERVE_HAL_STATUS_OK);
  ASSERT_EQ(api.event_record(destroyed, stream.value().get()), GRSERVE_HAL_STATUS_OK);
  ASSERT_EQ(api.event_destroy(&destroyed), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(destroyed, nullptr);
  EXPECT_EQ(api.stream_synchronize(stream.value().get(), 0U),
            GRSERVE_HAL_STATUS_OK);

  ASSERT_EQ(api.event_record(recorded.value().get(), stream.value().get()),
            GRSERVE_HAL_STATUS_OK);
  ASSERT_TRUE(stream.value().reset().ok());
  EXPECT_EQ(api.event_wait(recorded.value().get(), 0U), GRSERVE_HAL_STATUS_TIMEOUT);
  EXPECT_EQ(api.event_wait(recorded.value().get(), 1U), GRSERVE_HAL_STATUS_OK);
}

}  // namespace
