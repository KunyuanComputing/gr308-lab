#include "hal_mock/scripted_result.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

#include "model/state_codec.h"

namespace grserve::hal_mock {
namespace {

using google::protobuf::ListValue;
using google::protobuf::Struct;
using google::protobuf::Value;

constexpr std::uintmax_t kMaximumGoldenBytes = 64U * 1024U;
constexpr std::array<std::uint32_t, 3> kPrompt = {1U, 4U, 7U};
constexpr std::array<std::string_view, 8> kFrozenDigests = {
    "2378a37ce02cb0d7018a1ff56d0157f9ea7dde7ca5803341cc4da9c485345579",
    "22dd73a35b891b12b96734a1ff99771d33661b5dd028586e4939d539577866cc",
    "d17fe3d922302c6149f9cc4f38a127aaff205a0b18ed6f9692f29e69480d81cc",
    "bc881992c06ab43f7085d6fbae064ec2a77e10bd54ee9cebe34218d09636a7f2",
    "fcbc53ece9834cd85ed9f62d4dab547af53486929926a82f49b270bfdfdca249",
    "9e4c50aaa52d5b8e040f0654df0c3b58436d5c3373ec97170e373dd5cbea0482",
    "c5add18e69a449a00ed0b53a8bbba64b20c9a6e3a3a00d3972492fcf8d0bba5f",
    "51736d0ed73c32082b99f32fc5a32b0c273f7fe49c4d3ea0630a4a3a3ad63f44",
};

Error InvalidGolden() {
  return Error::From(ErrorCode::kInvalidArgument, "invalid scripted golden");
}

const Value* Field(const Struct& object, std::string_view name) {
  const auto found = object.fields().find(std::string(name));
  return found == object.fields().end() ? nullptr : &found->second;
}

bool ExactFields(const Struct& object,
                 std::initializer_list<std::string_view> names) {
  if (object.fields_size() != static_cast<int>(names.size())) return false;
  return std::all_of(names.begin(), names.end(), [&](std::string_view name) {
    return Field(object, name) != nullptr;
  });
}

bool StringIs(const Value* value, std::string_view expected) {
  return value != nullptr && value->kind_case() == Value::kStringValue &&
         value->string_value() == expected;
}

bool NumberIs(const Value* value, std::int64_t expected) {
  return value != nullptr && value->kind_case() == Value::kNumberValue &&
         std::isfinite(value->number_value()) &&
         value->number_value() == static_cast<double>(expected);
}

bool BoolIs(const Value* value, bool expected) {
  return value != nullptr && value->kind_case() == Value::kBoolValue &&
         value->bool_value() == expected;
}

const Struct* Object(const Value* value) {
  return value != nullptr && value->kind_case() == Value::kStructValue
             ? &value->struct_value()
             : nullptr;
}

const ListValue* List(const Value* value) {
  return value != nullptr && value->kind_case() == Value::kListValue
             ? &value->list_value()
             : nullptr;
}

bool StringListIs(const Value* value,
                  std::initializer_list<std::string_view> expected) {
  const ListValue* list = List(value);
  if (list == nullptr || list->values_size() != static_cast<int>(expected.size())) {
    return false;
  }
  int index = 0;
  for (std::string_view item : expected) {
    if (!StringIs(&list->values(index++), item)) return false;
  }
  return true;
}

bool NumberListIs(const Value* value,
                  std::initializer_list<std::int64_t> expected) {
  const ListValue* list = List(value);
  if (list == nullptr || list->values_size() != static_cast<int>(expected.size())) {
    return false;
  }
  int index = 0;
  for (std::int64_t item : expected) {
    if (!NumberIs(&list->values(index++), item)) return false;
  }
  return true;
}

bool FrozenArithmeticIsValid(const Struct& root) {
  const Struct* value = Object(Field(root, "arithmetic"));
  return value != nullptr &&
         ExactFields(*value, {"accumulator", "format", "full", "gdn",
                              "output", "round_shift", "saturation"}) &&
         StringIs(Field(*value, "accumulator"), "signed int64") &&
         StringIs(Field(*value, "format"), "Q16.16") &&
         StringIs(Field(*value, "full"),
                  "append projected key/value; hidden=per-dimension integer mean of all values") &&
         StringIs(Field(*value, "gdn"),
                  "state=round(input*input_diag+state*state_diag); hidden=round(state*gate_diag)") &&
         StringIs(Field(*value, "output"),
                  "frozen matrix logits; first token wins argmax ties") &&
         StringIs(Field(*value, "round_shift"),
                  "sign(value) * ((abs(value) + 2^(bits-1)) >> bits)") &&
         StringIs(Field(*value, "saturation"), "signed int32");
}

bool FrozenRngIsValid(const Struct& root) {
  const Struct* value = Object(Field(root, "rng"));
  return value != nullptr &&
         ExactFields(*value, {"advance", "algorithm", "greedy_argmax_uses_rng",
                              "initial_state", "operations"}) &&
         StringIs(Field(*value, "advance"),
                  "once before each committed token") &&
         StringIs(Field(*value, "algorithm"), "xorshift64") &&
         BoolIs(Field(*value, "greedy_argmax_uses_rng"), false) &&
         StringIs(Field(*value, "initial_state"), "11400714819323198485") &&
         StringListIs(Field(*value, "operations"),
                      {"x ^= (x << 13) & 0xffffffffffffffff", "x ^= x >> 7",
                       "x ^= (x << 17) & 0xffffffffffffffff",
                       "x &= 0xffffffffffffffff"});
}

bool FrozenEncodingIsValid(const Struct& root) {
  const Struct* value = Object(Field(root, "state_digest_encoding"));
  return value != nullptr &&
         ExactFields(*value,
                     {"algorithm", "byte_order", "fields", "vector_length"}) &&
         StringIs(Field(*value, "algorithm"), "sha256") &&
         StringIs(Field(*value, "byte_order"), "little-endian") &&
         StringIs(Field(*value, "vector_length"),
                  "uint32 before every vector") &&
         StringListIs(Field(*value, "fields"),
                      {"uint32 model_format_version",
                       "vector<vector<int32>> full_keys",
                       "vector<vector<int32>> full_values",
                       "vector<vector<int32>> gdn_states", "uint64 rng_state"});
}

bool FrozenLayoutIsValid(const Struct& root) {
  const ListValue* layout = List(Field(root, "weight_layout"));
  constexpr std::array<std::string_view, 4> names = {
      "token_embedding", "gdn_diagonal_gate_input_state",
      "full_key_value_matrices", "output_logits_matrix"};
  constexpr std::array<std::int64_t, 4> counts = {128, 72, 128, 128};
  if (layout == nullptr || layout->values_size() != 4) return false;
  for (int index = 0; index < 4; ++index) {
    const Struct* item = Object(&layout->values(index));
    if (item == nullptr || !ExactFields(*item, {"count", "name"}) ||
        !NumberIs(Field(*item, "count"), counts[static_cast<std::size_t>(index)]) ||
        !StringIs(Field(*item, "name"), names[static_cast<std::size_t>(index)])) {
      return false;
    }
  }
  return true;
}

bool FrozenMetadataIsValid(const Struct& root) {
  return ExactFields(root, {"arithmetic", "golden_version",
                            "model_format_version", "model_id",
                            "prompt_token_ids", "rng", "state_digest_encoding",
                            "steps", "topology", "weight_layout",
                            "weight_sha256"}) &&
         NumberIs(Field(root, "golden_version"), 1) &&
         NumberIs(Field(root, "model_format_version"), 1) &&
         StringIs(Field(root, "model_id"), "mini-qwen38-hybrid") &&
         NumberListIs(Field(root, "prompt_token_ids"), {1, 4, 7}) &&
         StringListIs(Field(root, "topology"), {"GDN", "GDN", "GDN", "FULL"}) &&
         StringIs(Field(root, "weight_sha256"),
                  "e4834d97653c2831980dd79ee4c7c2769de26bfe5dcd28145f4f2bf1ea767398") &&
         FrozenArithmeticIsValid(root) && FrozenRngIsValid(root) &&
         FrozenEncodingIsValid(root) && FrozenLayoutIsValid(root);
}

bool DecodeHex(std::string_view text, std::vector<std::byte>& output) {
  if (text.empty() || text.size() % 2U != 0U ||
      text.size() > 8192U * 2U) {
    return false;
  }
  auto nibble = [](char value) -> int {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
  };
  output.clear();
  output.reserve(text.size() / 2U);
  for (std::size_t index = 0; index < text.size(); index += 2U) {
    const int high = nibble(text[index]);
    const int low = nibble(text[index + 1U]);
    if (high < 0 || low < 0) return false;
    output.push_back(std::byte(static_cast<unsigned>(high * 16 + low)));
  }
  return true;
}

bool ParseI32(const Value& value, std::int32_t& output) {
  if (value.kind_case() != Value::kNumberValue ||
      !std::isfinite(value.number_value()) ||
      std::trunc(value.number_value()) != value.number_value() ||
      value.number_value() < static_cast<double>(INT32_MIN) ||
      value.number_value() > static_cast<double>(INT32_MAX)) {
    return false;
  }
  output = static_cast<std::int32_t>(value.number_value());
  return true;
}

bool ParseVectors(const Value* value, std::vector<HiddenVector>& output) {
  const ListValue* vectors = List(value);
  if (vectors == nullptr || vectors->values_size() > 64) return false;
  output.clear();
  output.reserve(static_cast<std::size_t>(vectors->values_size()));
  for (const Value& vector_value : vectors->values()) {
    const ListValue* vector = List(&vector_value);
    if (vector == nullptr || vector->values_size() != 8) return false;
    HiddenVector parsed{};
    for (int index = 0; index < 8; ++index) {
      if (!ParseI32(vector->values(index), parsed[static_cast<std::size_t>(index)])) {
        return false;
      }
    }
    output.push_back(parsed);
  }
  return true;
}

bool ParseU64(std::string_view text, std::uint64_t& output) {
  if (text.empty() || text.size() > 20U) return false;
  output = 0U;
  for (char digit : text) {
    if (digit < '0' || digit > '9') return false;
    const unsigned value = static_cast<unsigned>(digit - '0');
    if (output > (UINT64_MAX - value) / 10U) return false;
    output = output * 10U + value;
  }
  return true;
}

bool StateFieldsMatch(const Struct& step, const CanonicalState& decoded) {
  std::vector<HiddenVector> keys;
  std::vector<HiddenVector> values;
  std::vector<HiddenVector> gdn;
  std::uint64_t rng = 0U;
  const Value* rng_value = Field(step, "rng_state");
  if (!ParseVectors(Field(step, "full_keys"), keys) ||
      !ParseVectors(Field(step, "full_values"), values) ||
      !ParseVectors(Field(step, "gdn_states"), gdn) || gdn.size() != 3U ||
      rng_value == nullptr || rng_value->kind_case() != Value::kStringValue ||
      !ParseU64(rng_value->string_value(), rng)) {
    return false;
  }
  return keys == decoded.full_keys && values == decoded.full_values &&
         std::equal(gdn.begin(), gdn.end(), decoded.gdn_states.begin()) &&
         rng == decoded.rng_state;
}

bool StateExtends(const CanonicalState& previous, const CanonicalState& next) {
  const std::size_t appended = previous.full_keys.empty() ? 3U : 1U;
  return next.full_keys.size() == previous.full_keys.size() + appended &&
         next.full_values.size() == previous.full_values.size() + appended &&
         std::equal(previous.full_keys.begin(), previous.full_keys.end(),
                    next.full_keys.begin()) &&
         std::equal(previous.full_values.begin(), previous.full_values.end(),
                    next.full_values.begin());
}

}  // namespace

StatusOr<std::vector<ScriptedResult>> LoadFrozenGolden(
    const std::filesystem::path& golden_path) {
  std::error_code file_error;
  const std::uintmax_t file_size =
      std::filesystem::file_size(golden_path, file_error);
  if (file_error || file_size == 0U || file_size > kMaximumGoldenBytes) {
    return StatusOr<std::vector<ScriptedResult>>(InvalidGolden());
  }
  std::ifstream input(golden_path, std::ios::binary);
  std::string text(static_cast<std::size_t>(file_size), '\0');
  if (!input.read(text.data(), static_cast<std::streamsize>(text.size())) ||
      input.peek() != std::char_traits<char>::eof()) {
    return StatusOr<std::vector<ScriptedResult>>(InvalidGolden());
  }
  Struct root;
  if (!google::protobuf::util::JsonStringToMessage(text, &root).ok() ||
      !FrozenMetadataIsValid(root)) {
    return StatusOr<std::vector<ScriptedResult>>(InvalidGolden());
  }
  const ListValue* steps = List(Field(root, "steps"));
  if (steps == nullptr || steps->values_size() != 8) {
    return StatusOr<std::vector<ScriptedResult>>(InvalidGolden());
  }
  auto initial = EncodeCanonicalState(1U, InitialCanonicalState());
  if (!initial.ok()) {
    return StatusOr<std::vector<ScriptedResult>>(InvalidGolden());
  }
  CanonicalState previous = InitialCanonicalState();
  std::vector<std::byte> predecessor = std::move(initial.value());
  std::vector<std::uint32_t> context(kPrompt.begin(), kPrompt.end());
  std::vector<ScriptedResult> results;
  results.reserve(8U);
  for (int index = 0; index < steps->values_size(); ++index) {
    const Struct* step = Object(&steps->values(index));
    if (step == nullptr ||
        !ExactFields(*step, {"canonical_digest", "canonical_state_hex",
                             "full_keys", "full_values", "gdn_states",
                             "rng_state", "token_id"}) ||
        !NumberIs(Field(*step, "token_id"), 0) ||
        !StringIs(Field(*step, "canonical_digest"),
                  kFrozenDigests[static_cast<std::size_t>(index)])) {
      return StatusOr<std::vector<ScriptedResult>>(InvalidGolden());
    }
    const Value* state_hex = Field(*step, "canonical_state_hex");
    std::vector<std::byte> encoded;
    if (state_hex == nullptr || state_hex->kind_case() != Value::kStringValue ||
        !DecodeHex(state_hex->string_value(), encoded)) {
      return StatusOr<std::vector<ScriptedResult>>(InvalidGolden());
    }
    auto decoded = DecodeCanonicalState(encoded, 1U, 64U);
    if (!decoded.ok()) {
      return StatusOr<std::vector<ScriptedResult>>(InvalidGolden());
    }
    auto digest = DigestCanonicalState(1U, decoded.value());
    if (!digest.ok() ||
        digest.value() != kFrozenDigests[static_cast<std::size_t>(index)] ||
        !CanonicalStateMatchesStep(decoded.value(),
                                   static_cast<std::uint32_t>(index + 1)) ||
        decoded.value().full_keys.size() !=
            static_cast<std::size_t>(index + 3) ||
        !StateFieldsMatch(*step, decoded.value()) ||
        !StateExtends(previous, decoded.value())) {
      return StatusOr<std::vector<ScriptedResult>>(InvalidGolden());
    }
    std::vector<std::byte> next_predecessor = encoded;
    results.emplace_back(TokenId{0U}, digest.value(), std::move(encoded), context,
                         predecessor);
    context.push_back(0U);
    predecessor = std::move(next_predecessor);
    previous = std::move(decoded.value());
  }
  return StatusOr<std::vector<ScriptedResult>>(std::move(results));
}

}  // namespace grserve::hal_mock
