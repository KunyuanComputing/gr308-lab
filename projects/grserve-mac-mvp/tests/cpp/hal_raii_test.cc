#include <dlfcn.h>
#include <memory>

#include <gtest/gtest.h>

#include "hal/hal_library.h"
#include "hal/hal_raii.h"

namespace {

template <class Fn> Fn Symbol(void* handle, const char* name) { return reinterpret_cast<Fn>(dlsym(handle, name)); }

TEST(HalRaii, CreatesOwnsAndDestroysAllHandles) {
  const char* path = HAL_GOOD_PATH;
  void* fixture = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(fixture, nullptr);
  Symbol<void (*)()>(fixture, "grserve_test_reset_counts")();
  auto creates = Symbol<unsigned (*)()>(fixture, "grserve_test_runtime_create_calls");
  auto destroys = Symbol<unsigned (*)()>(fixture, "grserve_test_runtime_destroy_calls");
  auto device_destroys = Symbol<unsigned (*)()>(fixture, "grserve_test_device_destroy_calls");
  auto stream_destroys = Symbol<unsigned (*)()>(fixture, "grserve_test_stream_destroy_calls");
  auto event_destroys = Symbol<unsigned (*)()>(fixture, "grserve_test_event_destroy_calls");
  auto buffer_destroys = Symbol<unsigned (*)()>(fixture, "grserve_test_buffer_destroy_calls");
  auto executable_destroys = Symbol<unsigned (*)()>(fixture, "grserve_test_executable_destroy_calls");
  auto library = grserve::HalLibrary::Open(path, GRSERVE_HAL_CAPABILITY_STREAMS);
  ASSERT_TRUE(library.ok());
  auto owner = std::move(library).value();
  auto runtime = grserve::HalRuntime::Create(owner); ASSERT_TRUE(runtime.ok());
  auto device = runtime.value().CreateDevice(0U); ASSERT_TRUE(device.ok());
  auto stream = device.value().CreateStream(0U); ASSERT_TRUE(stream.ok());
  auto event = device.value().CreateEvent(0U); ASSERT_TRUE(event.ok());
  auto buffer = device.value().CreateBuffer(64U, 8U, 0U); ASSERT_TRUE(buffer.ok());
  auto executable = device.value().CreateExecutable(); ASSERT_TRUE(executable.ok());
  auto moved = std::move(runtime.value());
  EXPECT_FALSE(runtime.value());
  auto replacement = grserve::HalRuntime::Create(owner); ASSERT_TRUE(replacement.ok());
  replacement.value() = std::move(moved);
  EXPECT_FALSE(moved);
  ASSERT_TRUE(replacement.value().reset().ok());
  ASSERT_TRUE(replacement.value().reset().ok());
  EXPECT_EQ(creates(), 2U);
  owner.reset();
  EXPECT_TRUE(stream.value());
  ASSERT_TRUE(executable.value().reset().ok());
  ASSERT_TRUE(buffer.value().reset().ok());
  ASSERT_TRUE(event.value().reset().ok());
  ASSERT_TRUE(stream.value().reset().ok());
  ASSERT_TRUE(device.value().reset().ok());
  EXPECT_EQ(destroys(), 2U);
  EXPECT_EQ(device_destroys(), 1U);
  EXPECT_EQ(stream_destroys(), 1U);
  EXPECT_EQ(event_destroys(), 1U);
  EXPECT_EQ(buffer_destroys(), 1U);
  EXPECT_EQ(executable_destroys(), 1U);
  dlclose(fixture);
}

TEST(HalRaii, CapturesCreateFailureWithoutOwningHandle) {
  void* fixture = dlopen(HAL_GOOD_PATH, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(fixture, nullptr);
  Symbol<void (*)()>(fixture, "grserve_test_reset_counts")();
  auto destroys = Symbol<unsigned (*)()>(fixture, "grserve_test_runtime_destroy_calls");
  auto library = grserve::HalLibrary::Open(HAL_GOOD_PATH, 0U);
  ASSERT_TRUE(library.ok());
  auto runtime = grserve::HalRuntime::Create(library.value(), 0xDEADU);
  ASSERT_FALSE(runtime.ok());
  EXPECT_NE(runtime.error()->message().find("runtime create failed"), std::string::npos);
  EXPECT_EQ(destroys(), 0U);
  dlclose(fixture);
}

TEST(HalRaii, ScopeExitDestroysChildrenBeforeLibraryUnload) {
  void* fixture = dlopen(HAL_GOOD_PATH, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(fixture, nullptr);
  Symbol<void (*)()>(fixture, "grserve_test_reset_counts")();
  const auto runtime_destroys =
      Symbol<unsigned (*)()>(fixture, "grserve_test_runtime_destroy_calls");
  const auto device_destroys =
      Symbol<unsigned (*)()>(fixture, "grserve_test_device_destroy_calls");
  const auto stream_destroys =
      Symbol<unsigned (*)()>(fixture, "grserve_test_stream_destroy_calls");
  std::weak_ptr<grserve::HalLibrary> library_lifetime;
  {
    auto opened = grserve::HalLibrary::Open(HAL_GOOD_PATH, 0U);
    ASSERT_TRUE(opened.ok());
    auto owner = std::move(opened).value();
    library_lifetime = owner;
    auto runtime = grserve::HalRuntime::Create(owner);
    ASSERT_TRUE(runtime.ok());
    auto device = runtime.value().CreateDevice(0U);
    ASSERT_TRUE(device.ok());
    {
      auto stream = device.value().CreateStream(0U);
      ASSERT_TRUE(stream.ok());
      owner.reset();
      EXPECT_FALSE(library_lifetime.expired());
      EXPECT_TRUE(stream.value());
    }
    EXPECT_EQ(stream_destroys(), 1U);
    EXPECT_EQ(device_destroys(), 0U);
  }
  EXPECT_EQ(runtime_destroys(), 1U);
  EXPECT_EQ(device_destroys(), 1U);
  EXPECT_EQ(stream_destroys(), 1U);
  EXPECT_TRUE(library_lifetime.expired());
  dlclose(fixture);
}

TEST(HalRaii, SurvivesMalformedDestroyDiagnosticAndPreservesMoveAssignment) {
  void* fixture = dlopen(HAL_DESTROY_FAILURE_PATH, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(fixture, nullptr);
  Symbol<void (*)()>(fixture, "grserve_test_reset_counts")();
  const auto fail_next =
      Symbol<void (*)()>(fixture, "grserve_test_fail_next_runtime_destroy");
  const auto destroys =
      Symbol<unsigned (*)()>(fixture, "grserve_test_runtime_destroy_calls");
  auto opened = grserve::HalLibrary::Open(HAL_DESTROY_FAILURE_PATH, 0U);
  ASSERT_TRUE(opened.ok());
  auto owner = std::move(opened).value();

  const auto failed_create = grserve::HalRuntime::Create(owner, 0xDEADU);
  ASSERT_FALSE(failed_create.ok());
  EXPECT_NE(failed_create.error()->message().find("backend diagnostic unavailable"),
            std::string::npos);

  auto reporting = grserve::HalRuntime::Create(owner);
  ASSERT_TRUE(reporting.ok());
  const auto failed_reset = reporting.value().reset();
  EXPECT_FALSE(failed_reset.ok());
  EXPECT_NE(failed_reset.error()->message().find("backend diagnostic unavailable"),
            std::string::npos);
  EXPECT_TRUE(reporting.value());
  ASSERT_TRUE(reporting.value().reset().ok());

  auto destination = grserve::HalRuntime::Create(owner);
  auto source = grserve::HalRuntime::Create(owner);
  ASSERT_TRUE(destination.ok());
  ASSERT_TRUE(source.ok());
  fail_next();
  destination.value() = std::move(source.value());
  EXPECT_TRUE(destination.value());
  EXPECT_TRUE(source.value());
  EXPECT_EQ(destroys(), 1U);
  ASSERT_TRUE(destination.value().reset().ok());
  ASSERT_TRUE(source.value().reset().ok());
  EXPECT_EQ(destroys(), 3U);
  dlclose(fixture);
}

}  // namespace
