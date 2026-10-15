#pragma once

#include "engine/request.h"
#include "grserve/error.h"

namespace grserve {

// Implementations must only attempt a bounded, nonblocking enqueue. In
// particular, these callbacks must never perform socket or file I/O.
class RequestEventSink {
 public:
  virtual ~RequestEventSink() = default;
  virtual bool TryAccepted(RequestId request) noexcept = 0;
  virtual bool TryToken(const TokenEvent& token) noexcept = 0;
  virtual bool TryFinished(const FinishedEvent& finished) noexcept = 0;
  virtual bool TryError(const Error& error) noexcept = 0;
};

}  // namespace grserve
