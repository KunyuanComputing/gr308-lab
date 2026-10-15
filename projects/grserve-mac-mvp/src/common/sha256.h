#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "grserve/error.h"

namespace grserve {

class Sha256TestPeer;

class Sha256 {
 public:
  Sha256() noexcept;

  [[nodiscard]] Status Update(std::span<const std::byte> bytes);
  [[nodiscard]] Status Update(std::string_view bytes);
  [[nodiscard]] StatusOr<std::array<std::byte, 32>> Final();
  [[nodiscard]] StatusOr<std::string> FinalHex();

 private:
  void Transform(const std::byte* block) noexcept;
  [[nodiscard]] Status Fail(std::string message);

  friend class Sha256TestPeer;

  std::array<std::uint32_t, 8> state_;
  std::array<std::byte, 64> buffer_{};
  std::uint64_t total_bytes_ = 0;
  std::size_t buffered_ = 0;
  bool finalized_ = false;
  std::optional<Error> failure_;
  std::array<std::byte, 32> digest_{};
};

[[nodiscard]] StatusOr<std::string> Sha256Hex(std::string_view bytes);

}  // namespace grserve
