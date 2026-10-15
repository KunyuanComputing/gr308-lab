#include "model/state_codec.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <utility>

#include "common/sha256.h"

namespace grserve {
namespace {

Error InvalidState() {
  return Error::From(ErrorCode::kInvalidArgument, "invalid canonical state");
}

bool StateShapeIsValid(const CanonicalState& state,
                       std::uint32_t max_context_tokens) {
  return state.full_keys.size() == state.full_values.size() &&
         state.full_keys.size() <= max_context_tokens;
}

void AppendU32(std::vector<std::byte>& output, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32U; shift += 8U) {
    output.push_back(std::byte((value >> shift) & 0xffU));
  }
}

void AppendU64(std::vector<std::byte>& output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64U; shift += 8U) {
    output.push_back(std::byte((value >> shift) & 0xffU));
  }
}

void AppendI32(std::vector<std::byte>& output, std::int32_t value) {
  AppendU32(output, static_cast<std::uint32_t>(value));
}

void AppendHidden(std::vector<std::byte>& output, const HiddenVector& vector) {
  AppendU32(output, static_cast<std::uint32_t>(vector.size()));
  for (const std::int32_t value : vector) {
    AppendI32(output, value);
  }
}

class Reader {
 public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  bool U32(std::uint32_t& result) {
    if (remaining() < sizeof(result)) {
      return false;
    }
    result = 0U;
    for (unsigned index = 0; index < 4U; ++index) {
      result |= static_cast<std::uint32_t>(std::to_integer<unsigned char>(
                    bytes_[offset_ + index]))
                << (index * 8U);
    }
    offset_ += sizeof(result);
    return true;
  }

  bool U64(std::uint64_t& result) {
    if (remaining() < sizeof(result)) {
      return false;
    }
    result = 0U;
    for (unsigned index = 0; index < 8U; ++index) {
      result |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(
                    bytes_[offset_ + index]))
                << (index * 8U);
    }
    offset_ += sizeof(result);
    return true;
  }

  bool Hidden(HiddenVector& result) {
    std::uint32_t length = 0U;
    if (!U32(length) || length != kMiniQwenHiddenSize ||
        remaining() < kMiniQwenHiddenSize * sizeof(std::int32_t)) {
      return false;
    }
    for (std::int32_t& value : result) {
      std::uint32_t bits = 0U;
      if (!U32(bits)) {
        return false;
      }
      value = std::bit_cast<std::int32_t>(bits);
    }
    return true;
  }

  [[nodiscard]] bool done() const noexcept { return offset_ == bytes_.size(); }

 private:
  [[nodiscard]] std::size_t remaining() const noexcept {
    return bytes_.size() - offset_;
  }

  std::span<const std::byte> bytes_;
  std::size_t offset_ = 0U;
};

}  // namespace

void CanonicalState::Swap(CanonicalState& other) noexcept {
  full_keys.swap(other.full_keys);
  full_values.swap(other.full_values);
  gdn_states.swap(other.gdn_states);
  std::swap(rng_state, other.rng_state);
}

CanonicalState InitialCanonicalState() { return CanonicalState{}; }

StatusOr<std::vector<std::byte>> EncodeCanonicalState(
    std::uint32_t model_format_version, const CanonicalState& state) {
  if (model_format_version == 0U || !StateShapeIsValid(state, UINT32_MAX) ||
      state.full_keys.size() > UINT32_MAX) {
    return StatusOr<std::vector<std::byte>>(InvalidState());
  }
  std::vector<std::byte> encoded;
  if (state.full_keys.size() >
      (std::numeric_limits<std::size_t>::max() - state.gdn_states.size()) / 2U) {
    return StatusOr<std::vector<std::byte>>(InvalidState());
  }
  const std::size_t vector_count = state.full_keys.size() +
                                   state.full_values.size() +
                                   state.gdn_states.size();
  if (vector_count >
      (std::numeric_limits<std::size_t>::max() - 24U) /
          (4U + kMiniQwenHiddenSize * 4U)) {
    return StatusOr<std::vector<std::byte>>(InvalidState());
  }
  encoded.reserve(24U + vector_count * (4U + kMiniQwenHiddenSize * 4U));
  AppendU32(encoded, model_format_version);
  AppendU32(encoded, static_cast<std::uint32_t>(state.full_keys.size()));
  for (const HiddenVector& vector : state.full_keys) {
    AppendHidden(encoded, vector);
  }
  AppendU32(encoded, static_cast<std::uint32_t>(state.full_values.size()));
  for (const HiddenVector& vector : state.full_values) {
    AppendHidden(encoded, vector);
  }
  AppendU32(encoded, static_cast<std::uint32_t>(state.gdn_states.size()));
  for (const HiddenVector& vector : state.gdn_states) {
    AppendHidden(encoded, vector);
  }
  AppendU64(encoded, state.rng_state);
  return StatusOr<std::vector<std::byte>>(std::move(encoded));
}

StatusOr<CanonicalState> DecodeCanonicalState(
    std::span<const std::byte> bytes, std::uint32_t model_format_version,
    std::uint32_t max_context_tokens) {
  Reader reader(bytes);
  std::uint32_t encoded_version = 0U;
  std::uint32_t count = 0U;
  CanonicalState state;
  if (!reader.U32(encoded_version) || encoded_version != model_format_version ||
      !reader.U32(count) || count > max_context_tokens) {
    return StatusOr<CanonicalState>(InvalidState());
  }
  state.full_keys.resize(count);
  for (HiddenVector& vector : state.full_keys) {
    if (!reader.Hidden(vector)) {
      return StatusOr<CanonicalState>(InvalidState());
    }
  }
  if (!reader.U32(count) || count != state.full_keys.size()) {
    return StatusOr<CanonicalState>(InvalidState());
  }
  state.full_values.resize(count);
  for (HiddenVector& vector : state.full_values) {
    if (!reader.Hidden(vector)) {
      return StatusOr<CanonicalState>(InvalidState());
    }
  }
  if (!reader.U32(count) || count != kMiniQwenGdnLayers) {
    return StatusOr<CanonicalState>(InvalidState());
  }
  for (HiddenVector& vector : state.gdn_states) {
    if (!reader.Hidden(vector)) {
      return StatusOr<CanonicalState>(InvalidState());
    }
  }
  if (!reader.U64(state.rng_state) || !reader.done() ||
      !StateShapeIsValid(state, max_context_tokens)) {
    return StatusOr<CanonicalState>(InvalidState());
  }
  return StatusOr<CanonicalState>(std::move(state));
}

StatusOr<std::string> DigestCanonicalState(
    std::uint32_t model_format_version, const CanonicalState& state) {
  auto encoded = EncodeCanonicalState(model_format_version, state);
  if (!encoded.ok()) {
    return StatusOr<std::string>(*encoded.error());
  }
  Sha256 sha;
  const Status update = sha.Update(encoded.value());
  if (!update.ok()) {
    return StatusOr<std::string>(*update.error());
  }
  return sha.FinalHex();
}

bool CanonicalStateMatchesStep(const CanonicalState& state,
                               std::uint32_t step_index) noexcept {
  if (step_index > 8U ||
      state.full_keys.size() != state.full_values.size() ||
      (step_index == 0U && state != InitialCanonicalState())) {
    return false;
  }
  std::uint64_t expected_rng = kMiniQwenInitialRng;
  for (std::uint32_t index = 0U; index < step_index; ++index) {
    expected_rng ^= expected_rng << 13U;
    expected_rng ^= expected_rng >> 7U;
    expected_rng ^= expected_rng << 17U;
  }
  return state.rng_state == expected_rng;
}

}  // namespace grserve
