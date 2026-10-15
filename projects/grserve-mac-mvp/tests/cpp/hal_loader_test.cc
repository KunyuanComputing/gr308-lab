#include <dlfcn.h>
#include <filesystem>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "grserve/error.h"
#include "hal/hal_library.h"

namespace {

class FixtureLibrary {
 public:
  explicit FixtureLibrary(const char* path) : handle_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) { EXPECT_NE(handle_, nullptr); }
  ~FixtureLibrary() { if (handle_) dlclose(handle_); }
  unsigned RuntimeCreates() const { return Symbol<unsigned (*)()>("grserve_test_runtime_create_calls")(); }
 private:
  template <class Fn> Fn Symbol(const char* name) const { return reinterpret_cast<Fn>(dlsym(handle_, name)); }
  void* handle_;
};

TEST(HalLoader, OpensNegotiatedPlugin) {
  const auto result = grserve::HalLibrary::Open(HAL_GOOD_PATH, GRSERVE_HAL_CAPABILITY_STREAMS);
  ASSERT_TRUE(result.ok()) << result.error()->message();
  EXPECT_EQ(result.value()->api().abi_major, GRSERVE_HAL_ABI_MAJOR);
  EXPECT_EQ(result.value()->api().capabilities & GRSERVE_HAL_CAPABILITY_STREAMS, GRSERVE_HAL_CAPABILITY_STREAMS);
}

TEST(HalLoader, RejectsMissingLibrary) {
  const auto result = grserve::HalLibrary::Open("definitely-not-a-hal-plugin", 0U);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), grserve::ErrorCode::kInvalidArgument);
}

TEST(HalLoader, RejectsLibraryWithoutAbiEntryBeforeRuntimeCreation) {
  FixtureLibrary fixture(HAL_MISSING_ENTRY_PATH);
  const auto result = grserve::HalLibrary::Open(HAL_MISSING_ENTRY_PATH, 0U);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), grserve::ErrorCode::kInternal);
  EXPECT_EQ(fixture.RuntimeCreates(), 0U);
}

TEST(HalLoader, RejectsMajorBeforeRuntimeCreation) {
  FixtureLibrary fixture(HAL_BAD_MAJOR_PATH);
  const auto result = grserve::HalLibrary::Open(HAL_BAD_MAJOR_PATH, 0U);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), grserve::ErrorCode::kIncompatibleVersion);
  EXPECT_NE(result.error()->message().find("incompatible major"), std::string::npos);
  EXPECT_EQ(fixture.RuntimeCreates(), 0U);
}

TEST(HalLoader, RejectsMissingCapabilityBeforeRuntimeCreation) {
  FixtureLibrary fixture(HAL_MISSING_CAPABILITY_PATH);
  const auto result = grserve::HalLibrary::Open(HAL_MISSING_CAPABILITY_PATH, GRSERVE_HAL_CAPABILITY_STREAMS);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), grserve::ErrorCode::kUnsupportedCapability);
  EXPECT_NE(result.error()->message().find("requested capability unavailable"), std::string::npos);
  EXPECT_EQ(fixture.RuntimeCreates(), 0U);
}

TEST(HalLoader, RejectsShortVtableBeforeRuntimeCreation) {
  FixtureLibrary fixture(HAL_GOOD_PATH);
  const auto result = grserve::HalLibrary::Open(HAL_GOOD_PATH, UINT64_C(1) << 63);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), grserve::ErrorCode::kIncompatibleVersion);
  EXPECT_EQ(fixture.RuntimeCreates(), 0U);
}

TEST(HalLoader, RejectsNullRequiredCallbackBeforeRuntimeCreation) {
  FixtureLibrary fixture(HAL_NULL_CALLBACK_PATH);
  const auto result = grserve::HalLibrary::Open(HAL_NULL_CALLBACK_PATH, 0U);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), grserve::ErrorCode::kInternal);
  EXPECT_EQ(fixture.RuntimeCreates(), 0U);
}

TEST(HalLoader, RejectsMalformedNegotiationDiagnosticWithoutOverread) {
  FixtureLibrary fixture(HAL_MALFORMED_NEGOTIATION_PATH);
  const auto result = grserve::HalLibrary::Open(HAL_MALFORMED_NEGOTIATION_PATH, 0U);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), grserve::ErrorCode::kIncompatibleVersion);
  EXPECT_NE(result.error()->message().find("backend diagnostic unavailable"),
            std::string::npos);
  EXPECT_EQ(fixture.RuntimeCreates(), 0U);
}

}  // namespace
