#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "engine/rank_protocol.h"
#include "grserve/types.h"

namespace grserve {

enum class CancelReason {
  kClientDisconnected,
  kDeadline,
  kServerShutdown,
};

enum class CancelDisposition {
  kAbortedBeforeDecision,
  kCommittingDetached,
  kAlreadyTerminal,
  kNotFound,
};

enum class FinishReason {
  kLength,
  kCancelled,
};

enum class DeliveryOutcome {
  kNone,
  kDelivered,
  kDroppedClientDisconnect,
};

struct GenerateRequest {
  RequestId request_id;
  std::string model_id;
  std::vector<TokenId> prompt_token_ids;
  std::uint32_t max_tokens;
  std::uint64_t timeout_ms;
  std::uint32_t state_slots{1};

  bool operator==(const GenerateRequest&) const = default;
};

struct TokenEvent {
  RequestId request_id;
  TransactionId transaction_id;
  SequenceNumber sequence_number;
  TokenId token_id;
  StateDigest state_digest;

  bool operator==(const TokenEvent&) const = default;
};

struct FinishedEvent {
  RequestId request_id;
  FinishReason finish_reason;
  std::uint32_t prompt_tokens;
  std::uint32_t completion_tokens;
  SequenceNumber last_sequence_number;

  bool operator==(const FinishedEvent&) const = default;
};

}  // namespace grserve
