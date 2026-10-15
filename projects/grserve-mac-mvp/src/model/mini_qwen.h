#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "grserve/error.h"
#include "grserve/hal.h"
#include "grserve/types.h"
#include "model/manifest.h"
#include "model/state_codec.h"

namespace grserve {

// Entry-specific payload carried through the stable C HAL launch descriptor.
// All buffers are caller-owned and no pointer is retained after launch.
struct MiniQwenHalLaunchArguments {
  std::uint32_t struct_size;
  std::uint32_t step_index;
  const std::uint32_t* context_token_ids;
  std::uint32_t context_token_count;
  const std::uint8_t* canonical_state_bytes;
  std::uint64_t canonical_state_byte_count;
  std::uint32_t* out_token_id;
  char* out_canonical_digest;
  std::uint32_t canonical_digest_capacity;
  std::uint8_t* out_next_state_bytes;
  std::uint64_t next_state_capacity;
  std::uint64_t* out_next_state_byte_count;
};

inline constexpr char kMiniQwenEntryPoint[] = "mini_qwen_execute_next";

[[nodiscard]] inline grserve_hal_status_t ValidateMiniQwenHalArguments(
    const void* argument_data, std::uint64_t argument_bytes,
    const MiniQwenHalLaunchArguments** output) noexcept {
  constexpr std::uint32_t required_size = static_cast<std::uint32_t>(
      offsetof(MiniQwenHalLaunchArguments, out_next_state_byte_count) +
      sizeof(((MiniQwenHalLaunchArguments*)nullptr)->out_next_state_byte_count));
  if (output == nullptr || argument_data == nullptr ||
      argument_bytes < required_size) {
    return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
  }
  const auto* arguments =
      static_cast<const MiniQwenHalLaunchArguments*>(argument_data);
  if (arguments->struct_size < required_size || arguments->step_index >= 8U ||
      arguments->context_token_ids == nullptr ||
      arguments->context_token_count == 0U ||
      arguments->context_token_count > 64U ||
      arguments->canonical_state_byte_count > 8192U ||
      (arguments->canonical_state_bytes == nullptr) !=
          (arguments->canonical_state_byte_count == 0U) ||
      arguments->out_token_id == nullptr ||
      arguments->out_canonical_digest == nullptr ||
      arguments->canonical_digest_capacity < 65U ||
      arguments->out_next_state_bytes == nullptr ||
      arguments->next_state_capacity == 0U ||
      arguments->out_next_state_byte_count == nullptr) {
    return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
  }
  for (std::uint32_t index = 0U; index < arguments->context_token_count; ++index) {
    if (arguments->context_token_ids[index] >= 16U) {
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
  }
  *output = arguments;
  return GRSERVE_HAL_STATUS_OK;
}

[[nodiscard]] inline grserve_hal_status_t ValidateMiniQwenHalLaunch(
    const std::uint8_t* entry_name, std::uint64_t entry_name_bytes,
    const grserve_hal_launch_info_t* launch,
    const MiniQwenHalLaunchArguments** output) noexcept {
  constexpr std::uint32_t required_launch_size = static_cast<std::uint32_t>(
      offsetof(grserve_hal_launch_info_t, argument_bytes) +
      sizeof(((grserve_hal_launch_info_t*)nullptr)->argument_bytes));
  if (entry_name == nullptr ||
      entry_name_bytes != std::strlen(kMiniQwenEntryPoint) ||
      std::memcmp(entry_name, kMiniQwenEntryPoint,
                  static_cast<std::size_t>(entry_name_bytes)) != 0 ||
      launch == nullptr || launch->struct_size < required_launch_size ||
      launch->grid_x != 1U || launch->grid_y != 1U || launch->grid_z != 1U ||
      launch->block_x != 1U || launch->block_y != 1U || launch->block_z != 1U) {
    return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
  }
  return ValidateMiniQwenHalArguments(launch->argument_data,
                                      launch->argument_bytes, output);
}

struct CandidateStep {
  CandidateStep(TokenId candidate_token, CanonicalState candidate_state,
                std::string digest)
      : token(candidate_token),
        next_state(std::move(candidate_state)),
        canonical_digest(std::move(digest)) {}
  CandidateStep(const CandidateStep&) = delete;
  CandidateStep& operator=(const CandidateStep&) = delete;
  CandidateStep(CandidateStep&& other) noexcept
      : token(other.token),
        next_state(std::move(other.next_state)),
        canonical_digest(std::move(other.canonical_digest)) {
    other.next_state.reset();
  }
  CandidateStep& operator=(CandidateStep&& other) noexcept {
    if (this != &other) {
      token = other.token;
      next_state = std::move(other.next_state);
      canonical_digest = std::move(other.canonical_digest);
      other.next_state.reset();
    }
    return *this;
  }

  TokenId token;
  std::optional<CanonicalState> next_state;
  std::string canonical_digest;

  [[nodiscard]] bool consumed() const noexcept { return !next_state.has_value(); }
};

[[nodiscard]] Status CommitCandidate(CanonicalState& canonical,
                                     CandidateStep& candidate);

class MiniQwen {
 public:
  [[nodiscard]] static StatusOr<MiniQwen> Load(const ModelAssets& assets);
  [[nodiscard]] StatusOr<CandidateStep> ExecuteNext(
      std::span<const TokenId> context, const CanonicalState& state) const;

 private:
  MiniQwen(ModelManifest manifest, std::vector<std::int32_t> weights)
      : manifest_(std::move(manifest)), weights_(std::move(weights)) {}

  ModelManifest manifest_;
  std::vector<std::int32_t> weights_;
};

}  // namespace grserve
