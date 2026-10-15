#include "grserve/error.h"

namespace grserve {

Error Error::From(ErrorCode code, std::string message,
                  std::optional<RequestId> request_id,
                  std::optional<TransactionId> transaction_id) {
  return Error{code,
               std::make_shared<const std::string>(std::move(message)),
               request_id, transaction_id};
}

bool Error::retryable() const noexcept {
  switch (code_) {
    case ErrorCode::kResourceExhausted:
    case ErrorCode::kDeadlineExceeded:
    case ErrorCode::kRankFailure:
      return true;
    case ErrorCode::kInvalidArgument:
    case ErrorCode::kIncompatibleVersion:
    case ErrorCode::kUnsupportedCapability:
    case ErrorCode::kReplicaFatal:
    case ErrorCode::kInternal:
      return false;
  }
  return false;
}

}  // namespace grserve
