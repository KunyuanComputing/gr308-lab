#include "common/event_log.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <string_view>

#include <google/protobuf/util/json_util.h>

namespace grserve {
namespace {

Error Invalid(std::string message) {
  return Error::From(ErrorCode::kInvalidArgument, std::move(message));
}

bool ValidRunId(std::string_view value) noexcept {
  return !value.empty() && value.size() <= 128 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') || character == '-' ||
                  character == '_';
         });
}

bool ValidState(event::v1::EventState value) noexcept {
  return event::v1::EventState_IsValid(value) &&
         value != event::v1::EVENT_STATE_UNSPECIFIED;
}

bool ValidError(event::v1::EventErrorCode value) noexcept {
  return event::v1::EventErrorCode_IsValid(value) &&
         value != event::v1::EVENT_ERROR_CODE_UNSPECIFIED;
}

bool ValidDelivery(event::v1::DeliveryOutcome value) noexcept {
  return event::v1::DeliveryOutcome_IsValid(value) &&
         value != event::v1::DELIVERY_OUTCOME_UNSPECIFIED;
}

}  // namespace

StatusOr<std::string> EventLog::ToJsonLine(const EventRecord& record) {
  if (!ValidRunId(record.run_id) || record.request_id.value() == 0 ||
      record.transaction_id.value() == 0 || record.epoch.value() == 0 ||
      record.sequence.value() == 0 || !ValidState(record.state) ||
      !ValidError(record.error) || !ValidDelivery(record.delivery)) {
    return StatusOr<std::string>(Invalid("invalid structured event"));
  }

  event::v1::Event message;
  message.set_schema_version(1);
  message.set_monotonic_tick(record.monotonic_tick);
  message.set_run_id(record.run_id);
  message.set_request_id(record.request_id.value());
  message.set_transaction_id(record.transaction_id.value());
  message.set_rank(record.rank.value());
  message.set_epoch(record.epoch.value());
  message.set_state(record.state);
  message.set_error(record.error);
  message.set_sequence_number(record.sequence.value());
  message.set_delivery(record.delivery);

  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = false;
  options.always_print_fields_with_no_presence = true;
  options.preserve_proto_field_names = false;
  std::string json;
  const auto converted =
      google::protobuf::util::MessageToJsonString(message, &json, options);
  if (!converted.ok() || json.find_first_of("\r\n") != std::string::npos) {
    return StatusOr<std::string>(
        Error::From(ErrorCode::kInternal, "failed to serialize structured event"));
  }
  json.push_back('\n');
  return StatusOr<std::string>(std::move(json));
}

Status EventLog::Write(const EventRecord& record) {
  // Tick assignment, protobuf serialization, and the complete stream operation
  // form one transaction. This also protects non-thread-safe ostream buffers.
  std::lock_guard<std::mutex> lock(mutex_);
  if (has_tick_ && last_tick_ == std::numeric_limits<std::uint64_t>::max()) {
    return Status{Error::From(ErrorCode::kInternal,
                              "structured event tick exhausted")};
  }
  EventRecord assigned = record;
  if (has_tick_) {
    assigned.monotonic_tick =
        std::max(record.monotonic_tick, last_tick_ + std::uint64_t{1});
  }
  auto line = ToJsonLine(assigned);
  if (!line.ok()) return line.status();
  try {
    output_->write(line.value().data(),
                   static_cast<std::streamsize>(line.value().size()));
    output_->flush();
    if (!*output_) {
      return Status{Error::From(ErrorCode::kInternal,
                                "failed to write structured event")};
    }
  } catch (...) {
    return Status{Error::From(ErrorCode::kInternal,
                              "failed to write structured event")};
  }
  last_tick_ = assigned.monotonic_tick;
  has_tick_ = true;
  return Status::Ok();
}

}  // namespace grserve
