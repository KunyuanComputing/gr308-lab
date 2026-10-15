#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

#include <sys/types.h>

#include "engine.pb.h"
#include "grserve/error.h"

namespace grserve::rpc {

inline constexpr std::size_t kFrameHeaderBytes = 4;
inline constexpr std::uint32_t kMaxFrameBytes = 4U * 1024U * 1024U;
inline constexpr int kFrameProtoRecursionLimit = 64;

// The framing API is intentionally blocking. A clean EOF before the first
// header byte is represented by a successful empty optional. EOF after any
// frame byte is a protocol/transport error.
class FrameReader {
 public:
  using ReadFunction = std::function<ssize_t(int, void*, std::size_t)>;

  FrameReader();
  explicit FrameReader(ReadFunction read_function);

  [[nodiscard]] StatusOr<std::optional<v1::ClientEnvelope>> ReadOne(
      int fd) const;

 private:
  ReadFunction read_function_;
};

class FrameWriter {
 public:
  using WriteFunction =
      std::function<ssize_t(int, const void*, std::size_t)>;

  FrameWriter();
  explicit FrameWriter(WriteFunction write_function);

  [[nodiscard]] Status WriteOne(int fd,
                                const v1::ServerEnvelope& envelope) const;

 private:
  WriteFunction write_function_;
};

}  // namespace grserve::rpc
