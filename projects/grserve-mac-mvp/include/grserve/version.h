#pragma once

#include <compare>
#include <cstdint>

namespace grserve {

struct SemanticVersion {
  std::uint32_t major;
  std::uint32_t minor;
  std::uint32_t patch;

  auto operator<=>(const SemanticVersion&) const = default;
};

inline constexpr SemanticVersion kEngineVersion{0, 1, 0};
inline constexpr SemanticVersion kRpcVersion{1, 0, 0};
inline constexpr SemanticVersion kHalAbiVersion{1, 0, 0};

}  // namespace grserve
