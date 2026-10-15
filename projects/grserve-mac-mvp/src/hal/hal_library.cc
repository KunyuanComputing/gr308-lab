#include "hal/hal_library.h"

#include <dlfcn.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace grserve {
namespace {

using GetApiFn = grserve_hal_status_t (*)(const grserve_hal_api_request_t*,
                                          grserve_hal_api_t*);

constexpr uint32_t kMaxDiagnosticBytes = 4096U;

class NativeLibraryHandle {
 public:
  explicit NativeLibraryHandle(void* handle) noexcept : handle_(handle) {}
  NativeLibraryHandle(const NativeLibraryHandle&) = delete;
  NativeLibraryHandle& operator=(const NativeLibraryHandle&) = delete;
  ~NativeLibraryHandle() { Reset(); }
  void* get() const noexcept { return handle_; }
  void* release() noexcept { return std::exchange(handle_, nullptr); }

 private:
  void Reset() noexcept {
    if (handle_ != nullptr) {
      dlclose(handle_);
      handle_ = nullptr;
    }
  }
  void* handle_;
};

ErrorCode ErrorCodeFor(grserve_hal_status_t status) {
  switch (status) {
    case GRSERVE_HAL_STATUS_INVALID_ARGUMENT:
      return ErrorCode::kInvalidArgument;
    case GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI:
      return ErrorCode::kIncompatibleVersion;
    case GRSERVE_HAL_STATUS_UNSUPPORTED:
      return ErrorCode::kUnsupportedCapability;
    case GRSERVE_HAL_STATUS_OUT_OF_MEMORY:
      return ErrorCode::kResourceExhausted;
    case GRSERVE_HAL_STATUS_TIMEOUT:
      return ErrorCode::kDeadlineExceeded;
    case GRSERVE_HAL_STATUS_OK:
    case GRSERVE_HAL_STATUS_INTERNAL:
    case GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL:
      return ErrorCode::kInternal;
  }
  return ErrorCode::kInternal;
}

}  // namespace

std::string CopyHalDiagnostic(const grserve_hal_api_t& api) {
  if (api.struct_size < GRSERVE_HAL_API_REQUIRED_SIZE ||
      api.last_error_copy == nullptr) {
    return "backend diagnostic unavailable";
  }
  try {
    uint32_t first_required = 0U;
    if (api.last_error_copy(nullptr, 0U, &first_required) !=
            GRSERVE_HAL_STATUS_OK ||
        first_required == 0U || first_required > kMaxDiagnosticBytes) {
      return "backend diagnostic unavailable";
    }
    std::vector<char> text(first_required, '\0');
    uint32_t second_required = first_required;
    if (api.last_error_copy(text.data(), first_required, &second_required) !=
            GRSERVE_HAL_STATUS_OK ||
        second_required != first_required) {
      return "backend diagnostic unavailable";
    }
    const auto* nul = static_cast<const char*>(
        std::memchr(text.data(), '\0', text.size()));
    if (nul == nullptr) {
      return "backend diagnostic unavailable";
    }
    return std::string(text.data(), static_cast<std::size_t>(nul - text.data()));
  } catch (...) {
    return "backend diagnostic unavailable";
  }
}

namespace {

Status Failure(ErrorCode code, std::string message,
               const grserve_hal_api_t* api = nullptr) {
  if (api != nullptr) {
    const std::string diagnostic = CopyHalDiagnostic(*api);
    if (!diagnostic.empty()) {
      message.append(": ").append(diagnostic);
    }
  }
  return Status{Error::From(code, std::move(message))};
}

bool HasRequiredCallbacks(const grserve_hal_api_t& api) {
  return api.backend_name_copy != nullptr && api.build_id_copy != nullptr &&
         api.runtime_create != nullptr && api.runtime_destroy != nullptr &&
         api.device_create != nullptr && api.device_destroy != nullptr &&
         api.stream_create != nullptr && api.stream_destroy != nullptr &&
         api.event_create != nullptr && api.event_destroy != nullptr &&
         api.buffer_create != nullptr && api.buffer_destroy != nullptr &&
         api.executable_create != nullptr && api.executable_destroy != nullptr &&
         api.stream_copy != nullptr && api.stream_submit != nullptr &&
         api.event_record != nullptr && api.event_wait != nullptr &&
         api.event_query != nullptr && api.stream_synchronize != nullptr &&
         api.executable_load != nullptr && api.executable_launch != nullptr &&
         api.last_error_copy != nullptr;
}

}  // namespace

StatusOr<std::shared_ptr<HalLibrary>> HalLibrary::Open(
    const std::filesystem::path& path, std::uint64_t required_capabilities) {
  const std::string native_path = path.string();
  if (native_path.empty()) {
    return StatusOr<std::shared_ptr<HalLibrary>>(
        Error::From(ErrorCode::kInvalidArgument, "HAL library path is empty"));
  }
  NativeLibraryHandle native_library(
      dlopen(native_path.c_str(), RTLD_NOW | RTLD_LOCAL));
  if (native_library.get() == nullptr) {
    const char* diagnostic = dlerror();
    return StatusOr<std::shared_ptr<HalLibrary>>(Error::From(
        ErrorCode::kInvalidArgument,
        "unable to open HAL library '" + native_path + "': " +
            (diagnostic == nullptr ? "unknown dynamic-loader error" : diagnostic)));
  }
  const auto reject = [&](Status failure) -> StatusOr<std::shared_ptr<HalLibrary>> {
    return StatusOr<std::shared_ptr<HalLibrary>>(*failure.error());
  };

  dlerror();
  const auto get_api = reinterpret_cast<GetApiFn>(
      dlsym(native_library.get(), "grserve_hal_get_api"));
  const char* symbol_error = dlerror();
  if (get_api == nullptr || symbol_error != nullptr) {
    return reject(Failure(ErrorCode::kInternal,
                          "HAL library does not export grserve_hal_get_api"));
  }

  grserve_hal_api_request_t request = {};
  request.struct_size = static_cast<uint32_t>(sizeof(request));
  request.abi_major = GRSERVE_HAL_ABI_MAJOR;
  request.min_abi_minor = GRSERVE_HAL_ABI_MINOR;
  request.required_capabilities = required_capabilities;
  grserve_hal_api_t api = {};
  api.struct_size = static_cast<uint32_t>(sizeof(api));
  grserve_hal_status_t result = GRSERVE_HAL_STATUS_INTERNAL;
  try {
    result = get_api(&request, &api);
  } catch (...) {
    return reject(Failure(ErrorCode::kInternal,
                          "HAL API negotiation threw an exception", &api));
  }
  if (result != GRSERVE_HAL_STATUS_OK) {
    return reject(Failure(ErrorCodeFor(result), "HAL API negotiation failed", &api));
  }
  if (api.struct_size < GRSERVE_HAL_API_REQUIRED_SIZE) {
    return reject(Failure(ErrorCode::kIncompatibleVersion,
                          "HAL API table is shorter than the required ABI prefix", &api));
  }
  if (api.abi_major != GRSERVE_HAL_ABI_MAJOR ||
      api.abi_minor < GRSERVE_HAL_ABI_MINOR) {
    return reject(Failure(ErrorCode::kIncompatibleVersion,
                          "HAL API version is incompatible", &api));
  }
  if ((required_capabilities & ~api.capabilities) != 0U) {
    return reject(Failure(ErrorCode::kUnsupportedCapability,
                          "HAL API lacks a required capability", &api));
  }
  if (!HasRequiredCallbacks(api)) {
    return reject(Failure(ErrorCode::kInternal,
                          "HAL API table has a required null callback", &api));
  }
  return StatusOr<std::shared_ptr<HalLibrary>>(
      std::shared_ptr<HalLibrary>(new HalLibrary(native_library.release(), api)));
}

HalLibrary::~HalLibrary() {
  if (native_library_ != nullptr) {
    dlclose(native_library_);
    native_library_ = nullptr;
  }
}

}  // namespace grserve
