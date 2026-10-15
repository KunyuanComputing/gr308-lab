#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "grserve/error.h"

namespace grserve {

inline constexpr std::size_t kMiniQwenHiddenSize = 8U;
inline constexpr std::size_t kMiniQwenGdnLayers = 3U;
inline constexpr std::uint64_t kMiniQwenInitialRng =
    UINT64_C(0x9e3779b97f4a7c15);

using HiddenVector = std::array<std::int32_t, kMiniQwenHiddenSize>;

struct CanonicalState {
  std::vector<HiddenVector> full_keys;
  std::vector<HiddenVector> full_values;
  std::array<HiddenVector, kMiniQwenGdnLayers> gdn_states{};
  std::uint64_t rng_state = kMiniQwenInitialRng;

  bool operator==(const CanonicalState&) const = default;
  void Swap(CanonicalState& other) noexcept;
};

[[nodiscard]] CanonicalState InitialCanonicalState();
[[nodiscard]] StatusOr<std::vector<std::byte>> EncodeCanonicalState(
    std::uint32_t model_format_version, const CanonicalState& state);
[[nodiscard]] StatusOr<CanonicalState> DecodeCanonicalState(
    std::span<const std::byte> bytes, std::uint32_t model_format_version,
    std::uint32_t max_context_tokens);
[[nodiscard]] StatusOr<std::string> DigestCanonicalState(
    std::uint32_t model_format_version, const CanonicalState& state);
[[nodiscard]] bool CanonicalStateMatchesStep(const CanonicalState& state,
                                             std::uint32_t step_index) noexcept;

}  // namespace grserve
