#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "grserve/hal.h"

namespace {

constexpr std::uint64_t kFakeCapabilities =
    GRSERVE_HAL_CAPABILITY_STREAMS | GRSERVE_HAL_CAPABILITY_EVENTS;

struct ApiWithCanary {
  grserve_hal_api_t api;
  std::array<std::uint8_t, 32U> canary;
};

struct RequestWithCanary {
  grserve_hal_api_request_t request;
  std::array<std::uint8_t, 24U> canary;
};

thread_local std::string g_fake_tls_diagnostic;

grserve_hal_status_t FakeLastErrorCopy(char* out_bytes, uint32_t out_capacity,
                                       uint32_t* out_required_bytes) {
  if (out_required_bytes == nullptr ||
      (out_bytes == nullptr && out_capacity != 0U) ||
      (out_bytes != nullptr && out_capacity == 0U)) {
    return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
  }
  const std::string& diagnostic = g_fake_tls_diagnostic;
  *out_required_bytes =
      static_cast<uint32_t>(diagnostic.size() + static_cast<std::size_t>(1U));
  if (out_bytes == nullptr) {
    return GRSERVE_HAL_STATUS_OK;
  }
  if (out_capacity >= *out_required_bytes) {
    std::memcpy(out_bytes, diagnostic.c_str(), *out_required_bytes);
    return GRSERVE_HAL_STATUS_OK;
  }
  const std::size_t prefix_bytes =
      static_cast<std::size_t>(out_capacity - 1U);
  std::memcpy(out_bytes, diagnostic.data(), prefix_bytes);
  out_bytes[prefix_bytes] = '\0';
  return GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL;
}

grserve_hal_status_t FakeGetApi(const grserve_hal_api_request_t* request,
                                grserve_hal_api_t* out_api) {
  if (request == nullptr || out_api == nullptr) {
    return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
  }
  if (request->struct_size < GRSERVE_HAL_API_REQUEST_REQUIRED_SIZE ||
      out_api->struct_size < GRSERVE_HAL_API_REQUIRED_SIZE) {
    return GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI;
  }
  if (request->abi_major != GRSERVE_HAL_ABI_MAJOR ||
      request->min_abi_minor > GRSERVE_HAL_ABI_MINOR ||
      (request->required_capabilities & ~kFakeCapabilities) != 0U) {
    return GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI;
  }

  grserve_hal_api_t response = {};
  response.struct_size = static_cast<std::uint32_t>(sizeof(response));
  response.abi_major = GRSERVE_HAL_ABI_MAJOR;
  response.abi_minor = GRSERVE_HAL_ABI_MINOR;
  response.capabilities = kFakeCapabilities;
  const std::size_t bytes_to_copy = std::min<std::size_t>(
      out_api->struct_size, sizeof(response));
  std::memcpy(out_api, &response, bytes_to_copy);
  return GRSERVE_HAL_STATUS_OK;
}

grserve_hal_api_request_t ValidRequest() {
  grserve_hal_api_request_t request = {};
  request.struct_size = static_cast<std::uint32_t>(sizeof(request));
  request.abi_major = GRSERVE_HAL_ABI_MAJOR;
  request.min_abi_minor = GRSERVE_HAL_ABI_MINOR;
  return request;
}

TEST(HalAbi, AcceptsFullRequestAndOutputStructs) {
  const auto request = ValidRequest();
  grserve_hal_api_t api = {};
  api.struct_size = static_cast<std::uint32_t>(sizeof(api));

  ASSERT_EQ(FakeGetApi(&request, &api), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(api.struct_size, sizeof(api));
  EXPECT_EQ(api.abi_major, GRSERVE_HAL_ABI_MAJOR);
  EXPECT_EQ(api.abi_minor, GRSERVE_HAL_ABI_MINOR);
  EXPECT_EQ(api.capabilities, kFakeCapabilities);
}

TEST(HalAbi, AcceptsOutputTruncatedToRequiredPrefixWithoutOverwritingCanary) {
  ApiWithCanary storage = {};
  storage.canary.fill(0xA5U);
  storage.api.struct_size = GRSERVE_HAL_API_REQUIRED_SIZE;
  const auto request = ValidRequest();

  ASSERT_EQ(FakeGetApi(&request, &storage.api), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(storage.api.abi_major, GRSERVE_HAL_ABI_MAJOR);
  EXPECT_EQ(storage.api.capabilities, kFakeCapabilities);
  for (const std::uint8_t byte : storage.canary) {
    EXPECT_EQ(byte, 0xA5U);
  }
}

TEST(HalAbi, PreservesAppendedOutputTail) {
  ApiWithCanary storage = {};
  storage.canary.fill(0x5AU);
  storage.api.struct_size = static_cast<std::uint32_t>(sizeof(storage));
  const auto request = ValidRequest();

  ASSERT_EQ(FakeGetApi(&request, &storage.api), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(storage.api.abi_major, GRSERVE_HAL_ABI_MAJOR);
  for (const std::uint8_t byte : storage.canary) {
    EXPECT_EQ(byte, 0x5AU);
  }
}

TEST(HalAbi, RejectsTruncatedRequestOrOutputWithoutUnsafeWrite) {
  auto short_request = ValidRequest();
  short_request.struct_size = GRSERVE_HAL_API_REQUEST_REQUIRED_SIZE - 1U;
  ApiWithCanary output = {};
  output.canary.fill(0xD3U);
  output.api.struct_size = static_cast<std::uint32_t>(sizeof(output.api));
  std::array<std::uint8_t, sizeof(output.api)> original_output;
  std::memcpy(original_output.data(), &output.api, sizeof(output.api));

  EXPECT_EQ(FakeGetApi(&short_request, &output.api),
            GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI);
  EXPECT_EQ(0, std::memcmp(&output.api, original_output.data(),
                           sizeof(output.api)));

  const auto request = ValidRequest();
  ApiWithCanary short_output = {};
  short_output.canary.fill(0x6CU);
  short_output.api.struct_size = GRSERVE_HAL_API_REQUIRED_SIZE - 1U;
  std::array<std::uint8_t, sizeof(short_output.api)> original_short_output;
  std::memcpy(original_short_output.data(), &short_output.api,
              sizeof(short_output.api));

  EXPECT_EQ(FakeGetApi(&request, &short_output.api),
            GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI);
  EXPECT_EQ(0, std::memcmp(&short_output.api, original_short_output.data(),
                           sizeof(short_output.api)));
}

TEST(HalAbi, RejectsMajorMinorAndCapabilityMismatches) {
  grserve_hal_api_t api = {};
  api.struct_size = static_cast<std::uint32_t>(sizeof(api));

  auto request = ValidRequest();
  request.abi_major += 1U;
  EXPECT_EQ(FakeGetApi(&request, &api), GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI);

  request = ValidRequest();
  request.min_abi_minor = GRSERVE_HAL_ABI_MINOR + 1U;
  EXPECT_EQ(FakeGetApi(&request, &api), GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI);

  request = ValidRequest();
  request.required_capabilities = GRSERVE_HAL_CAPABILITY_EXECUTABLES;
  EXPECT_EQ(FakeGetApi(&request, &api), GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI);
}

TEST(HalAbi, IgnoresCallerAppendedRequestTail) {
  RequestWithCanary storage = {};
  storage.canary.fill(0x9BU);
  storage.request = ValidRequest();
  storage.request.struct_size = static_cast<std::uint32_t>(sizeof(storage));
  const auto original_canary = storage.canary;
  grserve_hal_api_t api = {};
  api.struct_size = static_cast<std::uint32_t>(sizeof(api));

  ASSERT_EQ(FakeGetApi(&storage.request, &api), GRSERVE_HAL_STATUS_OK);
  EXPECT_EQ(storage.canary, original_canary);
}

TEST(HalAbi, LastErrorSizeQueryDoesNotConsumeThreadLocalDiagnostic) {
  const std::string expected = "device unavailable";
  g_fake_tls_diagnostic = expected;
  uint32_t required_bytes = 0U;

  ASSERT_EQ(FakeLastErrorCopy(nullptr, 0U, &required_bytes),
            GRSERVE_HAL_STATUS_OK);
  ASSERT_EQ(required_bytes, expected.size() + 1U);
  std::vector<char> output(required_bytes, '\0');
  ASSERT_EQ(FakeLastErrorCopy(output.data(), required_bytes, &required_bytes),
            GRSERVE_HAL_STATUS_OK);
  EXPECT_STREQ(output.data(), expected.c_str());
  EXPECT_EQ(g_fake_tls_diagnostic, expected);
}

TEST(HalAbi, LastErrorInsufficientCapacityPreservesThreadLocalDiagnostic) {
  const std::string expected = "copy protocol";
  g_fake_tls_diagnostic = expected;
  std::array<char, 5U> output = {'x', 'x', 'x', 'x', 'x'};
  uint32_t required_bytes = 0U;

  ASSERT_EQ(FakeLastErrorCopy(output.data(),
                              static_cast<uint32_t>(output.size()),
                              &required_bytes),
            GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL);
  EXPECT_EQ(required_bytes, expected.size() + 1U);
  EXPECT_STREQ(output.data(), "copy");
  EXPECT_EQ(g_fake_tls_diagnostic, expected);
}

TEST(HalAbi, LastErrorRejectsPointerCapacityMismatchWithoutMutation) {
  const std::string expected = "copy protocol";
  g_fake_tls_diagnostic = expected;
  uint32_t required_bytes = 0xA5A5A5A5U;
  std::array<char, 4U> output = {'x', 'x', 'x', 'x'};
  const auto original_output = output;

  EXPECT_EQ(FakeLastErrorCopy(nullptr, 1U, &required_bytes),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(required_bytes, 0xA5A5A5A5U);
  EXPECT_EQ(g_fake_tls_diagnostic, expected);
  EXPECT_EQ(output, original_output);

  EXPECT_EQ(FakeLastErrorCopy(output.data(), 0U, &required_bytes),
            GRSERVE_HAL_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(required_bytes, 0xA5A5A5A5U);
  EXPECT_EQ(g_fake_tls_diagnostic, expected);
  EXPECT_EQ(output, original_output);
}

}  // namespace
