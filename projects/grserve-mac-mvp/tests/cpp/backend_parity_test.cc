#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "grserve/hal.h"
#include "hal/hal_library.h"
#include "hal/hal_raii.h"
#include "model/mini_qwen.h"
#include "model/state_codec.h"

namespace {

struct BackendRun {
  std::vector<std::string> protocol;
  std::vector<std::uint32_t> tokens;
  std::vector<std::string> digests;
  std::vector<std::vector<std::uint8_t>> states;
};

enum class InitialRepresentation { kNullAndZero, kEncoded };

enum class ScriptViolation {
  kWrongPrompt,
  kContextPrefixMutation,
  kCanonicalValueMutation,
};

enum class CorruptGolden {
  kInvalidJson,
  kDuplicateField,
  kUnknownField,
  kMissingField,
  kInvalidUtf8,
  kOversize,
  kCrossStepState,
  kBadHex,
  kBadDigest,
  kBadRng,
  kBadContext,
  kTokenOutOfRange,
  kWrongStepCount,
};

class UniqueTempDirectory {
 public:
  UniqueTempDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "grserve-script-test-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }
  ~UniqueTempDirectory() { std::filesystem::remove_all(path_); }
  UniqueTempDirectory(const UniqueTempDirectory&) = delete;
  UniqueTempDirectory& operator=(const UniqueTempDirectory&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

std::string FrozenGoldenText() {
  std::ifstream input(std::filesystem::path(GRSERVE_MODEL_DIR) / "golden.json",
                      std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(input)), {});
}

void ReplaceOnce(std::string& text, std::string_view from,
                 std::string_view to) {
  const auto position = text.find(from);
  ASSERT_NE(position, std::string::npos);
  text.replace(position, from.size(), to);
}

std::string CorruptGoldenText(CorruptGolden corruption) {
  std::string text = FrozenGoldenText();
  switch (corruption) {
    case CorruptGolden::kInvalidJson:
      return "not-json\n" + text;
    case CorruptGolden::kDuplicateField:
      text.insert(1U, "\n  \"model_id\": \"duplicate\",");
      break;
    case CorruptGolden::kUnknownField:
      text.insert(1U, "\n  \"unknown\": 0,");
      break;
    case CorruptGolden::kMissingField:
      ReplaceOnce(text, "  \"golden_version\": 1,\n", "");
      break;
    case CorruptGolden::kInvalidUtf8:
      text.insert(text.begin(), static_cast<char>(0xff));
      break;
    case CorruptGolden::kOversize:
      return std::string(70U * 1024U, ' ') + text;
    case CorruptGolden::kCrossStepState: {
      constexpr std::string_view marker = "\"canonical_state_hex\": \"";
      const auto first = text.find(marker);
      const auto first_begin = first + marker.size();
      const auto first_end = text.find('"', first_begin);
      const auto second = text.find(marker, first_end);
      const auto second_begin = second + marker.size();
      const auto second_end = text.find('"', second_begin);
      if (first == std::string::npos || second == std::string::npos) {
        throw std::runtime_error("golden state marker missing");
      }
      const std::string first_state = text.substr(first_begin, first_end - first_begin);
      const std::string second_state =
          text.substr(second_begin, second_end - second_begin);
      text.replace(second_begin, second_end - second_begin, first_state);
      text.replace(first_begin, first_end - first_begin, second_state);
      break;
    }
    case CorruptGolden::kBadHex:
      ReplaceOnce(text, "\"canonical_state_hex\": \"01",
                  "\"canonical_state_hex\": \"zz");
      break;
    case CorruptGolden::kBadDigest:
      ReplaceOnce(text, "\"canonical_digest\": \"2",
                  "\"canonical_digest\": \"3");
      break;
    case CorruptGolden::kBadRng:
      ReplaceOnce(text, "\"rng_state\": \"15860402102123842989\"",
                  "\"rng_state\": \"15860402102123842988\"");
      break;
    case CorruptGolden::kBadContext:
      ReplaceOnce(text, "\"prompt_token_ids\": [\n    1,",
                  "\"prompt_token_ids\": [\n    2,");
      break;
    case CorruptGolden::kTokenOutOfRange:
      ReplaceOnce(text, "\"token_id\": 0", "\"token_id\": 16");
      break;
    case CorruptGolden::kWrongStepCount: {
      const auto step = text.rfind("    {\n      \"canonical_digest\"");
      const auto end = text.find("\n    }\n  ]", step);
      if (step == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("golden step marker missing");
      }
      text.erase(step - 2U, end + 7U - (step - 2U));
      break;
    }
  }
  return text;
}

grserve_hal_status_t LoadGoldenThroughPlugin(const char* path,
                                             std::string_view golden) {
  UniqueTempDirectory directory;
  {
    std::ofstream output(directory.path() / "golden.json", std::ios::binary);
    output.write(golden.data(), static_cast<std::streamsize>(golden.size()));
  }
  auto library = grserve::HalLibrary::Open(path, GRSERVE_HAL_CAPABILITY_EXECUTABLES);
  EXPECT_TRUE(library.ok());
  auto runtime = grserve::HalRuntime::Create(library.value());
  EXPECT_TRUE(runtime.ok());
  auto device = runtime.value().CreateDevice(0U);
  EXPECT_TRUE(device.ok());
  auto executable = device.value().CreateExecutable();
  EXPECT_TRUE(executable.ok());
  const std::string model_directory = directory.path().string();
  return library.value()->api().executable_load(
      executable.value().get(),
      reinterpret_cast<const std::uint8_t*>(model_directory.data()),
      model_directory.size());
}

enum class Malformed {
  kStepOutOfRange,
  kStepAhead,
  kRepeatedStep,
  kCanonicalNullWithBytes,
  kCanonicalPointerWithZeroBytes,
  kInitialStateWrongSize,
  kInitialStateValueMutation,
  kCanonicalByteCountOverflow,
  kOutputStateNull,
  kOutputStateNullWithCapacity,
  kOutputStateZeroCapacity,
  kContextNull,
  kContextEmpty,
  kContextTooLong,
  kTokenOutOfRange,
  kOutputTokenNull,
  kDigestNull,
  kDigestTooSmall,
  kStateCapacityTooSmall,
  kMalformedState,
  kOutputStateByteCountNull,
  kArgumentsStructTooSmall,
  kArgumentBytesTooSmall,
  kEntryNull,
  kEntryWrong,
  kLaunchNull,
  kLaunchStructTooSmall,
  kGridZero,
};

struct ErrorRun {
  grserve_hal_status_t status;
  std::vector<std::string> protocol;
  std::uint32_t token;
  std::array<char, 65> digest;
  std::array<std::uint8_t, 4096> state;
  std::uint64_t required_state_bytes;
  std::string diagnostic;
};

std::string LastDiagnostic(const grserve_hal_api_t& api) {
  std::uint32_t required = 0;
  if (api.last_error_copy(nullptr, 0, &required) != GRSERVE_HAL_STATUS_OK ||
      required == 0) {
    return {};
  }
  std::string result(required, '\0');
  if (api.last_error_copy(result.data(), required, &required) !=
      GRSERVE_HAL_STATUS_OK) {
    return {};
  }
  result.resize(std::char_traits<char>::length(result.c_str()));
  return result;
}

std::string NormalizedError(grserve_hal_status_t status) {
  if (status == GRSERVE_HAL_STATUS_INVALID_ARGUMENT) {
    return "invalid_argument";
  }
  if (status == GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL) {
    return "buffer_too_small";
  }
  return "unexpected";
}

ErrorRun RunMalformed(const char* path, Malformed malformed) {
  auto library = grserve::HalLibrary::Open(path, GRSERVE_HAL_CAPABILITY_EXECUTABLES);
  EXPECT_TRUE(library.ok()) << library.error()->message();
  auto runtime = grserve::HalRuntime::Create(library.value());
  EXPECT_TRUE(runtime.ok());
  auto device = runtime.value().CreateDevice(0U);
  EXPECT_TRUE(device.ok());
  auto stream = device.value().CreateStream(0U);
  EXPECT_TRUE(stream.ok());
  auto executable = device.value().CreateExecutable();
  EXPECT_TRUE(executable.ok());
  const std::string model_directory = GRSERVE_MODEL_DIR;
  EXPECT_EQ(library.value()->api().executable_load(
                executable.value().get(),
                reinterpret_cast<const std::uint8_t*>(model_directory.data()),
                model_directory.size()),
            GRSERVE_HAL_STATUS_OK);

  std::vector<std::uint32_t> context = {1U, 4U, 7U};
  auto initial = grserve::EncodeCanonicalState(
      1U, grserve::InitialCanonicalState());
  EXPECT_TRUE(initial.ok());
  std::vector<std::uint8_t> initial_state;
  for (std::byte value : initial.value()) {
    initial_state.push_back(std::to_integer<std::uint8_t>(value));
  }
  std::array<std::uint8_t, 1> malformed_state{0xffU};
  std::array<std::uint8_t, 4096> first_state{};
  std::array<char, 65> first_digest{};
  std::uint64_t first_state_bytes = 0U;
  std::uint32_t first_token = 0U;
  ErrorRun result{.status = GRSERVE_HAL_STATUS_INTERNAL,
                  .protocol = {},
                  .token = UINT32_C(0xdeadbeef),
                  .digest = {},
                  .state = {},
                  .required_state_bytes = UINT64_C(0xfeedface)};
  result.digest.fill('D');
  result.state.fill(0xa5U);
  grserve::MiniQwenHalLaunchArguments arguments = {
      .struct_size = static_cast<std::uint32_t>(sizeof(arguments)),
      .step_index = 0U,
      .context_token_ids = context.data(),
      .context_token_count = static_cast<std::uint32_t>(context.size()),
      .canonical_state_bytes = initial_state.data(),
      .canonical_state_byte_count = initial_state.size(),
      .out_token_id = &result.token,
      .out_canonical_digest = result.digest.data(),
      .canonical_digest_capacity = result.digest.size(),
      .out_next_state_bytes = result.state.data(),
      .next_state_capacity = result.state.size(),
      .out_next_state_byte_count = &result.required_state_bytes,
  };
  grserve_hal_launch_info_t launch = {
      .struct_size = static_cast<std::uint32_t>(sizeof(launch)),
      .grid_x = 1U,
      .grid_y = 1U,
      .grid_z = 1U,
      .block_x = 1U,
      .block_y = 1U,
      .block_z = 1U,
      .argument_data = &arguments,
      .argument_bytes = sizeof(arguments),
  };
  const std::uint8_t* entry = reinterpret_cast<const std::uint8_t*>(
      grserve::kMiniQwenEntryPoint);
  std::uint64_t entry_size = std::strlen(grserve::kMiniQwenEntryPoint);
  const grserve_hal_launch_info_t* launch_pointer = &launch;
  switch (malformed) {
    case Malformed::kStepOutOfRange:
      arguments.step_index = 8U;
      break;
    case Malformed::kStepAhead:
      arguments.step_index = 1U;
      break;
    case Malformed::kRepeatedStep: {
      auto first_arguments = arguments;
      first_arguments.out_token_id = &first_token;
      first_arguments.out_canonical_digest = first_digest.data();
      first_arguments.out_next_state_bytes = first_state.data();
      first_arguments.next_state_capacity = first_state.size();
      first_arguments.out_next_state_byte_count = &first_state_bytes;
      auto first_launch = launch;
      first_launch.argument_data = &first_arguments;
      EXPECT_EQ(library.value()->api().executable_launch(
                    executable.value().get(), stream.value().get(),
                    reinterpret_cast<const std::uint8_t*>(
                        grserve::kMiniQwenEntryPoint),
                    std::strlen(grserve::kMiniQwenEntryPoint), &first_launch),
                GRSERVE_HAL_STATUS_OK);
      context.push_back(first_token);
      arguments.context_token_ids = context.data();
      arguments.context_token_count = context.size();
      arguments.canonical_state_bytes = first_state.data();
      arguments.canonical_state_byte_count = first_state_bytes;
      break;
    }
    case Malformed::kCanonicalNullWithBytes:
      arguments.canonical_state_bytes = nullptr;
      arguments.canonical_state_byte_count = 1U;
      break;
    case Malformed::kCanonicalPointerWithZeroBytes:
      arguments.canonical_state_bytes = malformed_state.data();
      arguments.canonical_state_byte_count = 0U;
      break;
    case Malformed::kInitialStateWrongSize:
      --arguments.canonical_state_byte_count;
      break;
    case Malformed::kInitialStateValueMutation:
      if (initial_state.size() <= 20U) {
        throw std::runtime_error("encoded initial state is unexpectedly short");
      }
      initial_state[20] = 1U;
      break;
    case Malformed::kCanonicalByteCountOverflow:
      arguments.canonical_state_bytes = malformed_state.data();
      arguments.canonical_state_byte_count = UINT64_MAX;
      break;
    case Malformed::kOutputStateNull:
      arguments.out_next_state_bytes = nullptr;
      arguments.next_state_capacity = 0U;
      break;
    case Malformed::kOutputStateNullWithCapacity:
      arguments.out_next_state_bytes = nullptr;
      break;
    case Malformed::kOutputStateZeroCapacity:
      arguments.next_state_capacity = 0U;
      break;
    case Malformed::kContextNull:
      arguments.context_token_ids = nullptr;
      break;
    case Malformed::kContextEmpty:
      arguments.context_token_count = 0U;
      break;
    case Malformed::kContextTooLong:
      arguments.context_token_count = 65U;
      break;
    case Malformed::kTokenOutOfRange:
      context[2] = 16U;
      break;
    case Malformed::kOutputTokenNull:
      arguments.out_token_id = nullptr;
      break;
    case Malformed::kDigestNull:
      arguments.out_canonical_digest = nullptr;
      break;
    case Malformed::kDigestTooSmall:
      arguments.canonical_digest_capacity = 64U;
      break;
    case Malformed::kStateCapacityTooSmall:
      arguments.next_state_capacity = 1U;
      break;
    case Malformed::kMalformedState:
      arguments.canonical_state_bytes = malformed_state.data();
      arguments.canonical_state_byte_count = malformed_state.size();
      arguments.step_index = 1U;
      break;
    case Malformed::kOutputStateByteCountNull:
      arguments.out_next_state_byte_count = nullptr;
      break;
    case Malformed::kArgumentsStructTooSmall:
      arguments.struct_size -= 1U;
      break;
    case Malformed::kArgumentBytesTooSmall:
      launch.argument_bytes -= 1U;
      break;
    case Malformed::kEntryNull:
      entry = nullptr;
      entry_size = 0U;
      break;
    case Malformed::kEntryWrong:
      entry = reinterpret_cast<const std::uint8_t*>("wrong");
      entry_size = 5U;
      break;
    case Malformed::kLaunchNull:
      launch_pointer = nullptr;
      break;
    case Malformed::kLaunchStructTooSmall:
      launch.struct_size -= 1U;
      break;
    case Malformed::kGridZero:
      launch.grid_x = 0U;
      break;
  }
  result.status = library.value()->api().executable_launch(
      executable.value().get(), stream.value().get(), entry, entry_size,
      launch_pointer);
  result.diagnostic = LastDiagnostic(library.value()->api());
  result.protocol.push_back("error:" + NormalizedError(result.status));
  return result;
}

BackendRun RunBackend(const char* path, InitialRepresentation representation) {
  BackendRun run;
  auto library = grserve::HalLibrary::Open(
      path, GRSERVE_HAL_CAPABILITY_EXECUTABLES);
  EXPECT_TRUE(library.ok()) << library.error()->message();
  auto runtime = grserve::HalRuntime::Create(library.value());
  EXPECT_TRUE(runtime.ok());
  auto device = runtime.value().CreateDevice(0U);
  EXPECT_TRUE(device.ok());
  auto stream = device.value().CreateStream(0U);
  EXPECT_TRUE(stream.ok());
  auto executable = device.value().CreateExecutable();
  EXPECT_TRUE(executable.ok());
  const std::string model_directory = GRSERVE_MODEL_DIR;
  EXPECT_EQ(library.value()->api().executable_load(
                executable.value().get(),
                reinterpret_cast<const std::uint8_t*>(model_directory.data()),
                model_directory.size()),
            GRSERVE_HAL_STATUS_OK);

  std::vector<std::uint32_t> context = {1U, 4U, 7U};
  auto initial = grserve::EncodeCanonicalState(
      1U, grserve::InitialCanonicalState());
  EXPECT_TRUE(initial.ok());
  std::vector<std::uint8_t> canonical;
  if (representation == InitialRepresentation::kEncoded) {
    canonical.reserve(initial.value().size());
    for (std::byte value : initial.value()) {
      canonical.push_back(std::to_integer<std::uint8_t>(value));
    }
  }
  for (std::uint32_t step = 0U; step < 8U; ++step) {
    std::array<char, 65> digest{};
    std::array<std::uint8_t, 4096> next_state{};
    std::uint64_t next_state_bytes = 0U;
    std::uint32_t token = UINT32_MAX;
    grserve::MiniQwenHalLaunchArguments arguments = {
        .struct_size = static_cast<std::uint32_t>(sizeof(arguments)),
        .step_index = step,
        .context_token_ids = context.data(),
        .context_token_count = static_cast<std::uint32_t>(context.size()),
        .canonical_state_bytes = canonical.empty() ? nullptr : canonical.data(),
        .canonical_state_byte_count = canonical.size(),
        .out_token_id = &token,
        .out_canonical_digest = digest.data(),
        .canonical_digest_capacity = digest.size(),
        .out_next_state_bytes = next_state.data(),
        .next_state_capacity = next_state.size(),
        .out_next_state_byte_count = &next_state_bytes,
    };
    const grserve_hal_launch_info_t launch = {
        .struct_size = static_cast<std::uint32_t>(sizeof(launch)),
        .grid_x = 1U,
        .grid_y = 1U,
        .grid_z = 1U,
        .block_x = 1U,
        .block_y = 1U,
        .block_z = 1U,
        .argument_data = &arguments,
        .argument_bytes = sizeof(arguments),
    };
    run.protocol.push_back("accepted");
    const auto status = library.value()->api().executable_launch(
        executable.value().get(), stream.value().get(),
        reinterpret_cast<const std::uint8_t*>(grserve::kMiniQwenEntryPoint),
        std::strlen(grserve::kMiniQwenEntryPoint), &launch);
    EXPECT_EQ(status, GRSERVE_HAL_STATUS_OK);
    if (status != GRSERVE_HAL_STATUS_OK) {
      run.protocol.push_back("error");
      return run;
    }
    run.protocol.push_back("token:" + std::to_string(token));
    run.protocol.push_back("finished");
    run.tokens.push_back(token);
    run.digests.emplace_back(digest.data());
    context.push_back(token);
    if (next_state_bytes != 0U) {
      EXPECT_LE(next_state_bytes, next_state.size());
      if (next_state_bytes > next_state.size()) {
        return run;
      }
      canonical.assign(next_state.begin(),
                       next_state.begin() + static_cast<std::ptrdiff_t>(next_state_bytes));
    }
    run.states.push_back(canonical);
  }
  return run;
}

ErrorRun RunScriptViolation(const char* path, ScriptViolation violation) {
  auto library = grserve::HalLibrary::Open(path, GRSERVE_HAL_CAPABILITY_EXECUTABLES);
  EXPECT_TRUE(library.ok());
  auto runtime = grserve::HalRuntime::Create(library.value());
  EXPECT_TRUE(runtime.ok());
  auto device = runtime.value().CreateDevice(0U);
  EXPECT_TRUE(device.ok());
  auto stream = device.value().CreateStream(0U);
  EXPECT_TRUE(stream.ok());
  auto executable = device.value().CreateExecutable();
  EXPECT_TRUE(executable.ok());
  const std::string model_directory = GRSERVE_MODEL_DIR;
  EXPECT_EQ(library.value()->api().executable_load(
                executable.value().get(),
                reinterpret_cast<const std::uint8_t*>(model_directory.data()),
                model_directory.size()),
            GRSERVE_HAL_STATUS_OK);

  auto initial = grserve::EncodeCanonicalState(
      1U, grserve::InitialCanonicalState());
  EXPECT_TRUE(initial.ok());
  std::vector<std::uint8_t> canonical;
  for (std::byte value : initial.value()) {
    canonical.push_back(std::to_integer<std::uint8_t>(value));
  }
  std::vector<std::uint32_t> context = {1U, 4U, 7U};
  std::array<std::uint8_t, 4096> first_state{};
  std::array<char, 65> first_digest{};
  std::uint64_t first_state_bytes = 0U;
  std::uint32_t first_token = 0U;

  if (violation == ScriptViolation::kContextPrefixMutation ||
      violation == ScriptViolation::kCanonicalValueMutation) {
    grserve::MiniQwenHalLaunchArguments first_arguments = {
        .struct_size = static_cast<std::uint32_t>(sizeof(first_arguments)),
        .step_index = 0U,
        .context_token_ids = context.data(),
        .context_token_count = static_cast<std::uint32_t>(context.size()),
        .canonical_state_bytes = canonical.data(),
        .canonical_state_byte_count = canonical.size(),
        .out_token_id = &first_token,
        .out_canonical_digest = first_digest.data(),
        .canonical_digest_capacity = first_digest.size(),
        .out_next_state_bytes = first_state.data(),
        .next_state_capacity = first_state.size(),
        .out_next_state_byte_count = &first_state_bytes,
    };
    const grserve_hal_launch_info_t first_launch = {
        .struct_size = static_cast<std::uint32_t>(sizeof(first_launch)),
        .grid_x = 1U, .grid_y = 1U, .grid_z = 1U,
        .block_x = 1U, .block_y = 1U, .block_z = 1U,
        .argument_data = &first_arguments,
        .argument_bytes = sizeof(first_arguments),
    };
    EXPECT_EQ(library.value()->api().executable_launch(
                  executable.value().get(), stream.value().get(),
                  reinterpret_cast<const std::uint8_t*>(
                      grserve::kMiniQwenEntryPoint),
                  std::strlen(grserve::kMiniQwenEntryPoint), &first_launch),
              GRSERVE_HAL_STATUS_OK);
    canonical.assign(first_state.begin(), first_state.begin() +
                                              static_cast<std::ptrdiff_t>(
                                                  first_state_bytes));
    context.push_back(first_token);
    if (violation == ScriptViolation::kContextPrefixMutation) {
      context[1] ^= 1U;
    } else {
      auto decoded = grserve::DecodeCanonicalState(
          std::span<const std::byte>(
              reinterpret_cast<const std::byte*>(canonical.data()),
              canonical.size()),
          1U, 64U);
      EXPECT_TRUE(decoded.ok());
      ++decoded.value().gdn_states[0][0];
      auto changed = grserve::EncodeCanonicalState(1U, decoded.value());
      EXPECT_TRUE(changed.ok());
      canonical.clear();
      for (std::byte value : changed.value()) {
        canonical.push_back(std::to_integer<std::uint8_t>(value));
      }
    }
  } else if (violation == ScriptViolation::kWrongPrompt) {
    context[0] = 2U;
  }

  ErrorRun result{.status = GRSERVE_HAL_STATUS_INTERNAL,
                  .protocol = {},
                  .token = UINT32_C(0xdeadbeef),
                  .digest = {},
                  .state = {},
                  .required_state_bytes = UINT64_C(0xfeedface)};
  result.digest.fill('D');
  result.state.fill(0xa5U);
  const std::uint32_t step =
      violation == ScriptViolation::kContextPrefixMutation ||
              violation == ScriptViolation::kCanonicalValueMutation
          ? 1U
          : 0U;
  grserve::MiniQwenHalLaunchArguments arguments = {
      .struct_size = static_cast<std::uint32_t>(sizeof(arguments)),
      .step_index = step,
      .context_token_ids = context.data(),
      .context_token_count = static_cast<std::uint32_t>(context.size()),
      .canonical_state_bytes = canonical.empty() ? nullptr : canonical.data(),
      .canonical_state_byte_count = canonical.size(),
      .out_token_id = &result.token,
      .out_canonical_digest = result.digest.data(),
      .canonical_digest_capacity = result.digest.size(),
      .out_next_state_bytes = result.state.data(),
      .next_state_capacity = result.state.size(),
      .out_next_state_byte_count = &result.required_state_bytes,
  };
  const grserve_hal_launch_info_t launch = {
      .struct_size = static_cast<std::uint32_t>(sizeof(launch)),
      .grid_x = 1U, .grid_y = 1U, .grid_z = 1U,
      .block_x = 1U, .block_y = 1U, .block_z = 1U,
      .argument_data = &arguments,
      .argument_bytes = sizeof(arguments),
  };
  result.status = library.value()->api().executable_launch(
      executable.value().get(), stream.value().get(),
      reinterpret_cast<const std::uint8_t*>(grserve::kMiniQwenEntryPoint),
      std::strlen(grserve::kMiniQwenEntryPoint), &launch);
  result.protocol.push_back("error:" + NormalizedError(result.status));
  return result;
}

class BackendSuccessParity
    : public testing::TestWithParam<InitialRepresentation> {};

TEST_P(BackendSuccessParity,
       RealCpuMockAndFaultAcceptEquivalentInitialStateRepresentations) {
  const BackendRun cpu = RunBackend(CPU_HAL_PATH, GetParam());
  const BackendRun mock = RunBackend(MOCK_HAL_PATH, GetParam());
  const BackendRun fault = RunBackend(FAULT_HAL_PATH, GetParam());
  EXPECT_EQ(cpu.protocol, mock.protocol);
  EXPECT_EQ(cpu.protocol, fault.protocol);
  EXPECT_EQ(cpu.tokens, mock.tokens);
  EXPECT_EQ(cpu.tokens, fault.tokens);
  EXPECT_EQ(cpu.digests, mock.digests);
  EXPECT_EQ(cpu.digests, fault.digests);
  EXPECT_EQ(cpu.states, mock.states);
  EXPECT_EQ(cpu.states, fault.states);
}

INSTANTIATE_TEST_SUITE_P(
    FrozenInitialState, BackendSuccessParity,
    testing::Values(InitialRepresentation::kNullAndZero,
                    InitialRepresentation::kEncoded));

class ScriptDomainValidation : public testing::TestWithParam<ScriptViolation> {};

TEST_P(ScriptDomainValidation, MockAndFaultRejectRequestsOutsideFrozenScript) {
  const ErrorRun mock = RunScriptViolation(MOCK_HAL_PATH, GetParam());
  const ErrorRun fault = RunScriptViolation(FAULT_HAL_PATH, GetParam());
  EXPECT_EQ(mock.status, GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(mock.status, fault.status);
  EXPECT_EQ(mock.protocol, fault.protocol);
  for (const ErrorRun* run : {&mock, &fault}) {
    EXPECT_EQ(run->token, UINT32_C(0xdeadbeef));
    EXPECT_TRUE(std::all_of(run->digest.begin(), run->digest.end(),
                            [](char value) { return value == 'D'; }));
    EXPECT_TRUE(std::all_of(run->state.begin(), run->state.end(),
                            [](std::uint8_t value) { return value == 0xa5U; }));
    EXPECT_EQ(run->required_state_bytes, UINT64_C(0xfeedface));
  }
}

INSTANTIATE_TEST_SUITE_P(
    FrozenScriptOnly, ScriptDomainValidation,
    testing::Values(ScriptViolation::kWrongPrompt,
                    ScriptViolation::kContextPrefixMutation,
                    ScriptViolation::kCanonicalValueMutation));

class StrictGoldenLoading : public testing::TestWithParam<CorruptGolden> {};

TEST_P(StrictGoldenLoading, MockAndFaultRejectMalformedOrInconsistentGolden) {
  const std::string corrupted = CorruptGoldenText(GetParam());
  EXPECT_EQ(LoadGoldenThroughPlugin(MOCK_HAL_PATH, corrupted),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(LoadGoldenThroughPlugin(FAULT_HAL_PATH, corrupted),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
}

INSTANTIATE_TEST_SUITE_P(
    FrozenSchema, StrictGoldenLoading,
    testing::Values(CorruptGolden::kInvalidJson,
                    CorruptGolden::kDuplicateField,
                    CorruptGolden::kUnknownField, CorruptGolden::kMissingField,
                    CorruptGolden::kInvalidUtf8, CorruptGolden::kOversize,
                    CorruptGolden::kCrossStepState, CorruptGolden::kBadHex,
                    CorruptGolden::kBadDigest, CorruptGolden::kBadRng,
                    CorruptGolden::kBadContext,
                    CorruptGolden::kTokenOutOfRange,
                    CorruptGolden::kWrongStepCount));

class BackendErrorParity : public testing::TestWithParam<Malformed> {};

TEST_P(BackendErrorParity, RejectsWithoutPartialOutputMutation) {
  const ErrorRun cpu = RunMalformed(CPU_HAL_PATH, GetParam());
  const ErrorRun mock = RunMalformed(MOCK_HAL_PATH, GetParam());
  const ErrorRun fault = RunMalformed(FAULT_HAL_PATH, GetParam());
  EXPECT_EQ(cpu.status, mock.status);
  EXPECT_EQ(cpu.status, fault.status);
  EXPECT_EQ(cpu.protocol, mock.protocol);
  EXPECT_EQ(cpu.protocol, fault.protocol);
  const bool reports_required = GetParam() == Malformed::kStateCapacityTooSmall;
  EXPECT_EQ(cpu.status, reports_required ? GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL
                                         : GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  for (const ErrorRun* run : {&cpu, &mock, &fault}) {
    EXPECT_EQ(run->token, UINT32_C(0xdeadbeef));
    EXPECT_TRUE(std::all_of(run->digest.begin(), run->digest.end(),
                            [](char value) { return value == 'D'; }));
    EXPECT_TRUE(std::all_of(run->state.begin(), run->state.end(),
                            [](std::uint8_t value) { return value == 0xa5U; }));
    if (reports_required) {
      EXPECT_NE(run->required_state_bytes, UINT64_C(0xfeedface));
    } else {
      EXPECT_EQ(run->required_state_bytes, UINT64_C(0xfeedface));
    }
  }
}

TEST(BackendParity, MockRejectsStepPastFrozenScriptBeforeAccess) {
  const ErrorRun mock = RunMalformed(MOCK_HAL_PATH, Malformed::kStepOutOfRange);
  EXPECT_EQ(mock.status, GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(mock.diagnostic, "scripted step index is out of range");
}

INSTANTIATE_TEST_SUITE_P(
    FrozenLaunchContract, BackendErrorParity,
    testing::Values(Malformed::kStepOutOfRange, Malformed::kStepAhead,
                    Malformed::kRepeatedStep,
                    Malformed::kCanonicalNullWithBytes,
                    Malformed::kCanonicalPointerWithZeroBytes,
                    Malformed::kInitialStateWrongSize,
                    Malformed::kInitialStateValueMutation,
                    Malformed::kCanonicalByteCountOverflow,
                    Malformed::kOutputStateNull, Malformed::kContextNull,
                    Malformed::kOutputStateNullWithCapacity,
                    Malformed::kOutputStateZeroCapacity,
                    Malformed::kContextEmpty, Malformed::kContextTooLong,
                    Malformed::kTokenOutOfRange,
                    Malformed::kOutputTokenNull, Malformed::kDigestNull,
                    Malformed::kDigestTooSmall,
                    Malformed::kStateCapacityTooSmall,
                    Malformed::kMalformedState,
                    Malformed::kOutputStateByteCountNull,
                    Malformed::kArgumentsStructTooSmall,
                    Malformed::kArgumentBytesTooSmall, Malformed::kEntryNull,
                    Malformed::kEntryWrong, Malformed::kLaunchNull,
                    Malformed::kLaunchStructTooSmall, Malformed::kGridZero));

}  // namespace
