#pragma once

#include <cstdint>
#include <mutex>
#include <ostream>
#include <string>

#include "event.pb.h"
#include "grserve/error.h"
#include "grserve/types.h"

namespace grserve {

struct EventRecord {
  std::uint64_t monotonic_tick;
  std::string run_id;
  RequestId request_id;
  TransactionId transaction_id;
  RankId rank;
  Epoch epoch;
  event::v1::EventState state;
  event::v1::EventErrorCode error;
  SequenceNumber sequence;
  event::v1::DeliveryOutcome delivery;
};

// Serializes only the frozen public Event protobuf schema. It intentionally has
// no prompt, filesystem path, private backend message, or fault-plan input.
class EventLog {
 public:
  explicit EventLog(std::ostream& output) noexcept : output_(&output) {}

  [[nodiscard]] Status Write(const EventRecord& record);
  [[nodiscard]] static StatusOr<std::string> ToJsonLine(
      const EventRecord& record);

 private:
  std::ostream* output_;
  std::mutex mutex_;
  std::uint64_t last_tick_ = 0;
  bool has_tick_ = false;
};

}  // namespace grserve
