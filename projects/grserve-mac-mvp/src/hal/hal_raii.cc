#include "hal/hal_raii.h"

#include <string>
#include <utility>

namespace grserve {
namespace {

ErrorCode CodeFor(grserve_hal_status_t value) {
  switch (value) {
    case GRSERVE_HAL_STATUS_INVALID_ARGUMENT: return ErrorCode::kInvalidArgument;
    case GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI: return ErrorCode::kIncompatibleVersion;
    case GRSERVE_HAL_STATUS_UNSUPPORTED: return ErrorCode::kUnsupportedCapability;
    case GRSERVE_HAL_STATUS_OUT_OF_MEMORY: return ErrorCode::kResourceExhausted;
    case GRSERVE_HAL_STATUS_TIMEOUT: return ErrorCode::kDeadlineExceeded;
    default: return ErrorCode::kInternal;
  }
}

template <class Fn, class... Args>
grserve_hal_status_t InvokeNoexcept(Fn fn, Args... args) noexcept {
  try { return fn(args...); } catch (...) { return GRSERVE_HAL_STATUS_INTERNAL; }
}

Status CallStatus(const std::shared_ptr<HalLibrary>& library,
                  grserve_hal_status_t result, const char* operation) {
  if (result == GRSERVE_HAL_STATUS_OK) return Status::Ok();
  std::string message(operation);
  message += ": ";
  message += CopyHalDiagnostic(library->api());
  return Status{Error::From(CodeFor(result), std::move(message))};
}

Status NoLibrary(const char* operation) {
  return Status{Error::From(ErrorCode::kInvalidArgument,
                            std::string(operation) + ": null HAL library or parent")};
}

template <class Raw, class Destroy>
bool TryDestroyNoexcept(Raw* owned, Destroy destroy) noexcept {
  if (*owned == nullptr) return true;
  Raw callback_handle = *owned;
  if (InvokeNoexcept(destroy, &callback_handle) != GRSERVE_HAL_STATUS_OK) return false;
  *owned = nullptr;
  return true;
}

template <class Raw, class Destroy>
Status DestroyWithReport(const std::shared_ptr<HalLibrary>& library, Raw* owned,
                         Destroy destroy, const char* operation) {
  if (*owned == nullptr) return Status::Ok();
  Raw callback_handle = *owned;
  const auto result = InvokeNoexcept(destroy, &callback_handle);
  if (result == GRSERVE_HAL_STATUS_OK) *owned = nullptr;
  return CallStatus(library, result, operation);
}

}  // namespace

StatusOr<HalRuntime> HalRuntime::Create(std::shared_ptr<HalLibrary> library,
                                        uint32_t flags) {
  if (!library) return StatusOr<HalRuntime>(*NoLibrary("runtime create").error());
  grserve_hal_runtime_t handle = nullptr;
  const auto result = InvokeNoexcept(library->api().runtime_create, flags, &handle);
  const auto status = CallStatus(library, result, "runtime create");
  if (!status.ok()) return StatusOr<HalRuntime>(*status.error());
  if (!handle) return StatusOr<HalRuntime>(Error::From(ErrorCode::kInternal, "runtime create returned null handle"));
  return StatusOr<HalRuntime>(HalRuntime(std::move(library), handle));
}

HalRuntime::HalRuntime(HalRuntime&& other) noexcept
    : library_(std::move(other.library_)), handle_(std::exchange(other.handle_, nullptr)) {}
HalRuntime& HalRuntime::operator=(HalRuntime&& other) noexcept {
  if (this != &other && TryResetNoexcept()) {
    library_ = std::move(other.library_);
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}
HalRuntime::~HalRuntime() noexcept { (void)TryResetNoexcept(); }
bool HalRuntime::TryResetNoexcept() noexcept {
  if (handle_ == nullptr) return true;
  return TryDestroyNoexcept(&handle_, library_->api().runtime_destroy);
}
Status HalRuntime::reset() {
  if (handle_ == nullptr) return Status::Ok();
  return DestroyWithReport(library_, &handle_, library_->api().runtime_destroy, "runtime destroy");
}
StatusOr<HalDevice> HalRuntime::CreateDevice(uint32_t ordinal) const {
  if (!library_ || !handle_) return StatusOr<HalDevice>(*NoLibrary("device create").error());
  grserve_hal_device_t handle = nullptr;
  const auto result = InvokeNoexcept(library_->api().device_create, handle_, ordinal, &handle);
  const auto status = CallStatus(library_, result, "device create");
  if (!status.ok()) return StatusOr<HalDevice>(*status.error());
  if (!handle) return StatusOr<HalDevice>(Error::From(ErrorCode::kInternal, "device create returned null handle"));
  return StatusOr<HalDevice>(HalDevice(library_, handle));
}

#define GRSERVE_DEFINE_HANDLE(Type, Raw, destroy_member, label)               \
  Type::Type(Type&& other) noexcept                                             \
      : library_(std::move(other.library_)), handle_(std::exchange(other.handle_, nullptr)) {} \
  Type& Type::operator=(Type&& other) noexcept {                                \
    if (this != &other && TryResetNoexcept()) {                                 \
      library_ = std::move(other.library_);                                     \
      handle_ = std::exchange(other.handle_, nullptr);                          \
    }                                                                            \
    return *this;                                                                \
  }                                                                              \
  Type::~Type() noexcept { (void)TryResetNoexcept(); }                          \
  bool Type::TryResetNoexcept() noexcept {                                      \
    if (handle_ == nullptr) return true;                                        \
    return TryDestroyNoexcept(&handle_, library_->api().destroy_member);       \
  }                                                                              \
  Status Type::reset() {                                                        \
    if (handle_ == nullptr) return Status::Ok();                                \
    return DestroyWithReport(library_, &handle_, library_->api().destroy_member, label " destroy"); \
  }
GRSERVE_DEFINE_HANDLE(HalDevice, grserve_hal_device_t, device_destroy, "device")
GRSERVE_DEFINE_HANDLE(HalStream, grserve_hal_stream_t, stream_destroy, "stream")
GRSERVE_DEFINE_HANDLE(HalEvent, grserve_hal_event_t, event_destroy, "event")
GRSERVE_DEFINE_HANDLE(HalBuffer, grserve_hal_buffer_t, buffer_destroy, "buffer")
GRSERVE_DEFINE_HANDLE(HalExecutable, grserve_hal_executable_t, executable_destroy, "executable")
#undef GRSERVE_DEFINE_HANDLE

StatusOr<HalStream> HalDevice::CreateStream(uint32_t flags) const {
  if (!library_ || !handle_) return StatusOr<HalStream>(*NoLibrary("stream create").error());
  grserve_hal_stream_t handle = nullptr;
  const auto status = CallStatus(library_, InvokeNoexcept(library_->api().stream_create, handle_, flags, &handle), "stream create");
  if (!status.ok()) return StatusOr<HalStream>(*status.error());
  if (!handle) return StatusOr<HalStream>(Error::From(ErrorCode::kInternal, "stream create returned null handle"));
  return StatusOr<HalStream>(HalStream(library_, handle));
}
StatusOr<HalEvent> HalDevice::CreateEvent(uint32_t flags) const {
  if (!library_ || !handle_) return StatusOr<HalEvent>(*NoLibrary("event create").error());
  grserve_hal_event_t handle = nullptr;
  const auto status = CallStatus(library_, InvokeNoexcept(library_->api().event_create, handle_, flags, &handle), "event create");
  if (!status.ok()) return StatusOr<HalEvent>(*status.error());
  if (!handle) return StatusOr<HalEvent>(Error::From(ErrorCode::kInternal, "event create returned null handle"));
  return StatusOr<HalEvent>(HalEvent(library_, handle));
}
StatusOr<HalBuffer> HalDevice::CreateBuffer(uint64_t bytes, uint64_t alignment, uint32_t flags) const {
  if (!library_ || !handle_) return StatusOr<HalBuffer>(*NoLibrary("buffer create").error());
  grserve_hal_buffer_t handle = nullptr;
  const auto status = CallStatus(library_, InvokeNoexcept(library_->api().buffer_create, handle_, bytes, alignment, flags, &handle), "buffer create");
  if (!status.ok()) return StatusOr<HalBuffer>(*status.error());
  if (!handle) return StatusOr<HalBuffer>(Error::From(ErrorCode::kInternal, "buffer create returned null handle"));
  return StatusOr<HalBuffer>(HalBuffer(library_, handle));
}
StatusOr<HalExecutable> HalDevice::CreateExecutable() const {
  if (!library_ || !handle_) return StatusOr<HalExecutable>(*NoLibrary("executable create").error());
  grserve_hal_executable_t handle = nullptr;
  const auto status = CallStatus(library_, InvokeNoexcept(library_->api().executable_create, handle_, &handle), "executable create");
  if (!status.ok()) return StatusOr<HalExecutable>(*status.error());
  if (!handle) return StatusOr<HalExecutable>(Error::From(ErrorCode::kInternal, "executable create returned null handle"));
  return StatusOr<HalExecutable>(HalExecutable(library_, handle));
}

}  // namespace grserve
