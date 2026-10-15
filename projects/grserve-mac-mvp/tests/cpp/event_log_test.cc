#include "common/event_log.h"

#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace grserve {
namespace {

EventRecord CompleteEvent() {
  return EventRecord{
      .monotonic_tick = 912,
      .run_id = "20260907T101112Z-a1b2c3d4",
      .request_id = RequestId{41},
      .transaction_id = TransactionId{52},
      .rank = RankId{1},
      .epoch = Epoch{63},
      .state = event::v1::EVENT_STATE_PUBLISHED,
      .error = event::v1::EVENT_ERROR_CODE_NONE,
      .sequence = SequenceNumber{7},
      .delivery = event::v1::DELIVERY_OUTCOME_DELIVERED,
  };
}

TEST(EventLogTest, SerializesTypedEventAsOneRedactedJsonLine) {
  std::ostringstream output;
  EventLog log(output);

  ASSERT_TRUE(log.Write(CompleteEvent()).ok());
  const std::string line = output.str();
  EXPECT_EQ(line.find('\n'), line.size() - 1);
  EXPECT_NE(line.find("\"schemaVersion\":1"), std::string::npos);
  EXPECT_NE(line.find("\"monotonicTick\":\"912\""), std::string::npos);
  EXPECT_NE(line.find("\"runId\":\"20260907T101112Z-a1b2c3d4\""),
            std::string::npos);
  EXPECT_NE(line.find("\"requestId\":\"41\""), std::string::npos);
  EXPECT_NE(line.find("\"transactionId\":\"52\""), std::string::npos);
  EXPECT_NE(line.find("\"rank\":1"), std::string::npos);
  EXPECT_NE(line.find("\"epoch\":\"63\""), std::string::npos);
  EXPECT_NE(line.find("\"state\":\"EVENT_STATE_PUBLISHED\""), std::string::npos);
  EXPECT_NE(line.find("\"error\":\"EVENT_ERROR_CODE_NONE\""), std::string::npos);
  EXPECT_NE(line.find("\"sequenceNumber\":\"7\""), std::string::npos);
  EXPECT_NE(line.find("\"delivery\":\"DELIVERY_OUTCOME_DELIVERED\""),
            std::string::npos);

  for (const std::string secret : {"101,102,103", "/tmp/private.sock",
                                   "/opt/private/backend.dylib",
                                   "backend private message",
                                   "fault rank=1 after=prepare"}) {
    EXPECT_EQ(line.find(secret), std::string::npos);
  }
}

TEST(EventLogTest, RejectsUnspecifiedEnumsAndZeroIdentityFields) {
  auto event = CompleteEvent();
  event.state = event::v1::EVENT_STATE_UNSPECIFIED;
  EXPECT_FALSE(EventLog::ToJsonLine(event).ok());

  event = CompleteEvent();
  event.request_id = RequestId{0};
  EXPECT_FALSE(EventLog::ToJsonLine(event).ok());

  event = CompleteEvent();
  event.delivery = static_cast<event::v1::DeliveryOutcome>(999);
  EXPECT_FALSE(EventLog::ToJsonLine(event).ok());
}

TEST(EventLogTest, DoesNotAppendPartialLineWhenOutputFails) {
  std::ostringstream output;
  output.setstate(std::ios::badbit);
  EventLog log(output);
  EXPECT_FALSE(log.Write(CompleteEvent()).ok());
}

class DelayedFailureBuffer final : public std::stringbuf {
 public:
  int sync() override { return -1; }
};

class ThrowingFlushBuffer final : public std::stringbuf {
 public:
  int sync() override { throw std::ios_base::failure("injected flush failure"); }
};

TEST(EventLogTest, ReportsBufferedFailureBeforeReturningSuccess) {
  DelayedFailureBuffer buffer;
  std::ostream output(&buffer);
  EventLog log(output);
  EXPECT_FALSE(log.Write(CompleteEvent()).ok());
}

TEST(EventLogTest, ConvertsThrowingStreamFailureToStableStatus) {
  ThrowingFlushBuffer buffer;
  std::ostream output(&buffer);
  output.exceptions(std::ios::badbit | std::ios::failbit);
  EventLog log(output);
  EXPECT_NO_THROW({ EXPECT_FALSE(log.Write(CompleteEvent()).ok()); });
}

TEST(EventLogTest, ConcurrentWritesAreWholeWithUniqueMonotonicTicks) {
  constexpr int kThreadCount = 8;
  constexpr int kEventsPerThread = 1000;
  std::ostringstream output;
  EventLog log(output);
  std::vector<std::thread> threads;
  for (int thread = 0; thread < kThreadCount; ++thread) {
    threads.emplace_back([&, thread] {
      for (int ordinal = 0; ordinal < kEventsPerThread; ++ordinal) {
        auto record = CompleteEvent();
        record.monotonic_tick = 1;
        record.request_id = RequestId{
            static_cast<std::uint64_t>(thread * kEventsPerThread + ordinal + 1)};
        ASSERT_TRUE(log.Write(record).ok());
      }
    });
  }
  for (auto& thread : threads) thread.join();

  std::istringstream lines(output.str());
  std::string line;
  std::uint64_t expected_tick = 1;
  int count = 0;
  while (std::getline(lines, line)) {
    ASSERT_FALSE(line.empty());
    EXPECT_NE(line.find("\"monotonicTick\":\"" +
                        std::to_string(expected_tick) + "\""),
              std::string::npos);
    EXPECT_EQ(line.front(), '{');
    EXPECT_EQ(line.back(), '}');
    ++expected_tick;
    ++count;
  }
  EXPECT_EQ(count, kThreadCount * kEventsPerThread);
}

}  // namespace
}  // namespace grserve
