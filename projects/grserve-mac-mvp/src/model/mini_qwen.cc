#include "model/mini_qwen.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

namespace grserve {
namespace {

constexpr std::size_t kVocabSize = 16U;
constexpr std::size_t kEmbeddingWeights = kVocabSize * kMiniQwenHiddenSize;
constexpr std::size_t kMatrixWeights =
    kMiniQwenHiddenSize * kMiniQwenHiddenSize;

Error InvalidInput() {
  return Error::From(ErrorCode::kInvalidArgument, "invalid MiniQwen input");
}

bool CheckedAdd(std::int64_t left, std::int64_t right,
                std::int64_t& result) noexcept {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
    return false;
  }
  result = left + right;
  return true;
}

std::int32_t RoundShiftSaturate(std::int64_t value) noexcept {
  const bool negative = value < 0;
  std::uint64_t magnitude = negative
                                ? static_cast<std::uint64_t>(-(value + 1)) + 1U
                                : static_cast<std::uint64_t>(value);
  magnitude = (magnitude + (UINT64_C(1) << 15U)) >> 16U;
  if (!negative) {
    return magnitude > static_cast<std::uint64_t>(INT32_MAX)
               ? INT32_MAX
               : static_cast<std::int32_t>(magnitude);
  }
  if (magnitude >= (UINT64_C(1) << 31U)) {
    return INT32_MIN;
  }
  return -static_cast<std::int32_t>(magnitude);
}

bool Dot(std::span<const std::int32_t> row, const HiddenVector& input,
         std::int32_t& output) noexcept {
  std::int64_t accumulator = 0;
  for (std::size_t dimension = 0; dimension < kMiniQwenHiddenSize; ++dimension) {
    const std::int64_t product = static_cast<std::int64_t>(row[dimension]) *
                                 static_cast<std::int64_t>(input[dimension]);
    if (!CheckedAdd(accumulator, product, accumulator)) {
      return false;
    }
  }
  output = RoundShiftSaturate(accumulator);
  return true;
}

bool MatrixVector(std::span<const std::int32_t> matrix,
                  const HiddenVector& input, HiddenVector& output) noexcept {
  for (std::size_t row = 0; row < kMiniQwenHiddenSize; ++row) {
    if (!Dot(matrix.subspan(row * kMiniQwenHiddenSize, kMiniQwenHiddenSize),
             input, output[row])) {
      return false;
    }
  }
  return true;
}

std::uint64_t AdvanceRng(std::uint64_t value) noexcept {
  value ^= value << 13U;
  value ^= value >> 7U;
  value ^= value << 17U;
  return value;
}

struct TokenResult {
  CanonicalState state;
  std::array<std::int32_t, kVocabSize> logits{};
};

StatusOr<TokenResult> ProcessToken(const std::vector<std::int32_t>& weights,
                                   TokenId token,
                                   const CanonicalState& previous) {
  if (token.value() >= kVocabSize) {
    return StatusOr<TokenResult>(InvalidInput());
  }
  TokenResult result{.state = previous, .logits = {}};
  std::size_t cursor = 0U;
  HiddenVector hidden{};
  std::copy_n(weights.begin() + token.value() * kMiniQwenHiddenSize,
              kMiniQwenHiddenSize, hidden.begin());
  cursor += kEmbeddingWeights;
  for (std::size_t layer = 0; layer < kMiniQwenGdnLayers; ++layer) {
    const auto gate = std::span(weights).subspan(cursor, kMiniQwenHiddenSize);
    const auto input =
        std::span(weights).subspan(cursor + kMiniQwenHiddenSize,
                                   kMiniQwenHiddenSize);
    const auto recurrent =
        std::span(weights).subspan(cursor + 2U * kMiniQwenHiddenSize,
                                   kMiniQwenHiddenSize);
    cursor += 3U * kMiniQwenHiddenSize;
    HiddenVector next_hidden{};
    for (std::size_t dimension = 0; dimension < kMiniQwenHiddenSize; ++dimension) {
      const std::int64_t input_term =
          static_cast<std::int64_t>(hidden[dimension]) * input[dimension];
      const std::int64_t recurrent_term =
          static_cast<std::int64_t>(previous.gdn_states[layer][dimension]) *
          recurrent[dimension];
      std::int64_t accumulator = 0;
      if (!CheckedAdd(input_term, recurrent_term, accumulator)) {
        return StatusOr<TokenResult>(InvalidInput());
      }
      const std::int32_t next_state = RoundShiftSaturate(accumulator);
      result.state.gdn_states[layer][dimension] = next_state;
      next_hidden[dimension] = RoundShiftSaturate(
          static_cast<std::int64_t>(next_state) * gate[dimension]);
    }
    hidden = next_hidden;
  }
  HiddenVector key{};
  HiddenVector value{};
  if (!MatrixVector(std::span(weights).subspan(cursor, kMatrixWeights), hidden,
                    key)) {
    return StatusOr<TokenResult>(InvalidInput());
  }
  cursor += kMatrixWeights;
  if (!MatrixVector(std::span(weights).subspan(cursor, kMatrixWeights), hidden,
                    value)) {
    return StatusOr<TokenResult>(InvalidInput());
  }
  cursor += kMatrixWeights;
  result.state.full_keys.push_back(key);
  result.state.full_values.push_back(value);
  for (std::size_t dimension = 0; dimension < kMiniQwenHiddenSize; ++dimension) {
    std::int64_t sum = 0;
    for (const HiddenVector& entry : result.state.full_values) {
      if (!CheckedAdd(sum, entry[dimension], sum)) {
        return StatusOr<TokenResult>(InvalidInput());
      }
    }
    const std::int64_t divisor =
        static_cast<std::int64_t>(result.state.full_values.size());
    hidden[dimension] = static_cast<std::int32_t>(sum / divisor);
  }
  for (std::size_t token_index = 0; token_index < kVocabSize; ++token_index) {
    if (!Dot(std::span(weights).subspan(
                 cursor + token_index * kMiniQwenHiddenSize,
                 kMiniQwenHiddenSize),
             hidden, result.logits[token_index])) {
      return StatusOr<TokenResult>(InvalidInput());
    }
  }
  return StatusOr<TokenResult>(std::move(result));
}

}  // namespace

Status CommitCandidate(CanonicalState& canonical, CandidateStep& candidate) {
  if (!candidate.next_state.has_value() ||
      &canonical == &*candidate.next_state) {
    return Status(InvalidInput());
  }
  canonical = std::move(*candidate.next_state);
  candidate.next_state.reset();
  return Status::Ok();
}

StatusOr<MiniQwen> MiniQwen::Load(const ModelAssets& assets) {
  const auto& manifest = assets.manifest();
  if (manifest.model_format_version != 1U || manifest.vocab_size != kVocabSize ||
      manifest.hidden_size != kMiniQwenHiddenSize ||
      manifest.max_context_tokens != 64U ||
      manifest.topology != "GDN,GDN,GDN,FULL" || manifest.dtype != "Q16_16" ||
      assets.weights().size() != kMiniQwenWeightBytes) {
    return StatusOr<MiniQwen>(InvalidInput());
  }
  std::vector<std::int32_t> weights;
  weights.reserve(kMiniQwenWeightCount);
  for (std::size_t offset = 0; offset < assets.weights().size(); offset += 4U) {
    std::uint32_t bits = 0U;
    for (unsigned byte = 0; byte < 4U; ++byte) {
      bits |= static_cast<std::uint32_t>(std::to_integer<unsigned char>(
                  assets.weights()[offset + byte]))
              << (byte * 8U);
    }
    weights.push_back(std::bit_cast<std::int32_t>(bits));
  }
  return StatusOr<MiniQwen>(MiniQwen(manifest, std::move(weights)));
}

StatusOr<CandidateStep> MiniQwen::ExecuteNext(
    std::span<const TokenId> context, const CanonicalState& state) const {
  if (context.empty() || context.size() > manifest_.max_context_tokens ||
      state.full_keys.size() != state.full_values.size() ||
      state.full_values.size() >= context.size() ||
      state.full_values.size() > manifest_.max_context_tokens) {
    return StatusOr<CandidateStep>(InvalidInput());
  }
  for (TokenId token : context) {
    if (token.value() >= manifest_.vocab_size) {
      return StatusOr<CandidateStep>(InvalidInput());
    }
  }
  CanonicalState next = state;
  std::array<std::int32_t, kVocabSize> logits{};
  for (std::size_t index = state.full_values.size(); index < context.size(); ++index) {
    auto result = ProcessToken(weights_, context[index], next);
    if (!result.ok()) {
      return StatusOr<CandidateStep>(*result.error());
    }
    next = std::move(result.value().state);
    logits = result.value().logits;
  }
  const auto best = std::max_element(logits.begin(), logits.end());
  next.rng_state = AdvanceRng(next.rng_state);
  auto digest = DigestCanonicalState(manifest_.model_format_version, next);
  if (!digest.ok()) {
    return StatusOr<CandidateStep>(*digest.error());
  }
  return StatusOr<CandidateStep>(CandidateStep(
      TokenId{static_cast<std::uint32_t>(best - logits.begin())},
      std::move(next), std::move(digest.value())));
}

}  // namespace grserve
