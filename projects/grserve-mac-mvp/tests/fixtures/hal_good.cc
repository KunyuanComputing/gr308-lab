#include <algorithm>
#include <cstring>
#include <new>
#include <string>

#include "grserve/hal.h"

struct grserve_hal_runtime_s {};
struct grserve_hal_device_s {};
struct grserve_hal_stream_s {};
struct grserve_hal_event_s {};
struct grserve_hal_buffer_s {};
struct grserve_hal_executable_s {};

#ifndef GRSERVE_FIXTURE_MODE
#define GRSERVE_FIXTURE_MODE 0
#endif

namespace {

thread_local std::string g_diagnostic;
unsigned g_runtime_creates = 0;
unsigned g_runtime_destroys = 0;
unsigned g_device_creates = 0;
unsigned g_device_destroys = 0;
unsigned g_stream_creates = 0;
unsigned g_stream_destroys = 0;
unsigned g_event_creates = 0;
unsigned g_event_destroys = 0;
unsigned g_buffer_creates = 0;
unsigned g_buffer_destroys = 0;
unsigned g_executable_creates = 0;
unsigned g_executable_destroys = 0;
bool g_fail_next_runtime_destroy = false;

constexpr grserve_hal_capabilities_t kCapabilities =
    GRSERVE_HAL_CAPABILITY_STREAMS | GRSERVE_HAL_CAPABILITY_EVENTS |
    GRSERVE_HAL_CAPABILITY_ASYNC_COPY | GRSERVE_HAL_CAPABILITY_EXECUTABLES;

void SetDiagnostic(const char* value) noexcept {
  try {
    g_diagnostic = value;
  } catch (...) {
  }
}

grserve_hal_status_t CopyString(const std::string& value, char* bytes,
                                uint32_t capacity, uint32_t* required) {
  if (required == nullptr || (bytes == nullptr && capacity != 0U) ||
      (bytes != nullptr && capacity == 0U)) {
    return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
  }
  *required = static_cast<uint32_t>(value.size() + 1U);
  if (bytes == nullptr) return GRSERVE_HAL_STATUS_OK;
  if (capacity >= *required) {
    std::memcpy(bytes, value.c_str(), *required);
    return GRSERVE_HAL_STATUS_OK;
  }
  const auto prefix = static_cast<std::size_t>(capacity - 1U);
  std::memcpy(bytes, value.data(), prefix);
  bytes[prefix] = '\0';
  return GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL;
}

grserve_hal_status_t BackendName(char* b, uint32_t c, uint32_t* r) {
  try { return CopyString("fixture-good", b, c, r); } catch (...) { return GRSERVE_HAL_STATUS_INTERNAL; }
}
grserve_hal_status_t BuildId(char* b, uint32_t c, uint32_t* r) {
  try { return CopyString("fixture-build-1", b, c, r); } catch (...) { return GRSERVE_HAL_STATUS_INTERNAL; }
}
grserve_hal_status_t LastError(char* b, uint32_t c, uint32_t* r) {
#if GRSERVE_FIXTURE_MODE == 5 || GRSERVE_FIXTURE_MODE == 6
  try {
    if (r == nullptr || (b == nullptr && c != 0U) ||
        (b != nullptr && c == 0U)) return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    *r = 32U;
    if (b != nullptr) std::memset(b, 'X', c);
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) { return GRSERVE_HAL_STATUS_INTERNAL; }
#else
  try { return CopyString(g_diagnostic, b, c, r); } catch (...) { return GRSERVE_HAL_STATUS_INTERNAL; }
#endif
}

template <class Handle> grserve_hal_status_t Destroy(Handle* handle, unsigned* count) {
  try {
    if (handle == nullptr || *handle == nullptr) return GRSERVE_HAL_STATUS_OK;
    delete *handle;
    *handle = nullptr;
    ++*count;
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) { SetDiagnostic("destroy failed"); return GRSERVE_HAL_STATUS_INTERNAL; }
}

grserve_hal_status_t RuntimeCreate(uint32_t flags, grserve_hal_runtime_t* out) {
  try {
    if (out == nullptr) { SetDiagnostic("runtime output missing"); return GRSERVE_HAL_STATUS_INVALID_ARGUMENT; }
    if (flags == 0xDEADU) { SetDiagnostic("runtime create failed"); return GRSERVE_HAL_STATUS_INTERNAL; }
    *out = new grserve_hal_runtime_s(); ++g_runtime_creates; return GRSERVE_HAL_STATUS_OK;
  } catch (...) { SetDiagnostic("runtime allocation failed"); return GRSERVE_HAL_STATUS_OUT_OF_MEMORY; }
}
grserve_hal_status_t RuntimeDestroy(grserve_hal_runtime_t* handle) {
  try {
    if (g_fail_next_runtime_destroy) {
      g_fail_next_runtime_destroy = false;
      SetDiagnostic("runtime destroy failed");
      return GRSERVE_HAL_STATUS_INTERNAL;
    }
    return Destroy(handle, &g_runtime_destroys);
  } catch (...) { return GRSERVE_HAL_STATUS_INTERNAL; }
}
grserve_hal_status_t DeviceCreate(grserve_hal_runtime_t runtime, uint32_t, grserve_hal_device_t* out) {
  try { if (!runtime || !out) { SetDiagnostic("device parent missing"); return GRSERVE_HAL_STATUS_INVALID_ARGUMENT; } *out = new grserve_hal_device_s(); ++g_device_creates; return GRSERVE_HAL_STATUS_OK; } catch (...) { SetDiagnostic("device allocation failed"); return GRSERVE_HAL_STATUS_OUT_OF_MEMORY; }
}
grserve_hal_status_t StreamCreate(grserve_hal_device_t device, uint32_t, grserve_hal_stream_t* out) {
  try { if (!device || !out) { SetDiagnostic("stream parent missing"); return GRSERVE_HAL_STATUS_INVALID_ARGUMENT; } *out = new grserve_hal_stream_s(); ++g_stream_creates; return GRSERVE_HAL_STATUS_OK; } catch (...) { SetDiagnostic("stream allocation failed"); return GRSERVE_HAL_STATUS_OUT_OF_MEMORY; }
}
grserve_hal_status_t EventCreate(grserve_hal_device_t device, uint32_t, grserve_hal_event_t* out) {
  try { if (!device || !out) { SetDiagnostic("event parent missing"); return GRSERVE_HAL_STATUS_INVALID_ARGUMENT; } *out = new grserve_hal_event_s(); ++g_event_creates; return GRSERVE_HAL_STATUS_OK; } catch (...) { SetDiagnostic("event allocation failed"); return GRSERVE_HAL_STATUS_OUT_OF_MEMORY; }
}
grserve_hal_status_t BufferCreate(grserve_hal_device_t device, uint64_t, uint64_t, uint32_t, grserve_hal_buffer_t* out) {
  try { if (!device || !out) { SetDiagnostic("buffer parent missing"); return GRSERVE_HAL_STATUS_INVALID_ARGUMENT; } *out = new grserve_hal_buffer_s(); ++g_buffer_creates; return GRSERVE_HAL_STATUS_OK; } catch (...) { SetDiagnostic("buffer allocation failed"); return GRSERVE_HAL_STATUS_OUT_OF_MEMORY; }
}
grserve_hal_status_t ExecutableCreate(grserve_hal_device_t device, grserve_hal_executable_t* out) {
  try { if (!device || !out) { SetDiagnostic("executable parent missing"); return GRSERVE_HAL_STATUS_INVALID_ARGUMENT; } *out = new grserve_hal_executable_s(); ++g_executable_creates; return GRSERVE_HAL_STATUS_OK; } catch (...) { SetDiagnostic("executable allocation failed"); return GRSERVE_HAL_STATUS_OUT_OF_MEMORY; }
}
grserve_hal_status_t Ok() { return GRSERVE_HAL_STATUS_OK; }
grserve_hal_status_t StreamCopy(grserve_hal_stream_t, grserve_hal_buffer_t, uint64_t, grserve_hal_buffer_t, uint64_t, uint64_t) { return Ok(); }
grserve_hal_status_t StreamSubmit(grserve_hal_stream_t, const grserve_hal_submit_info_t*) { return Ok(); }
grserve_hal_status_t EventRecord(grserve_hal_event_t, grserve_hal_stream_t) { return Ok(); }
grserve_hal_status_t EventWait(grserve_hal_event_t, uint64_t) { return Ok(); }
grserve_hal_status_t EventQuery(grserve_hal_event_t, grserve_hal_event_state_t* state) { if (state) *state = GRSERVE_HAL_EVENT_STATE_COMPLETE; return Ok(); }
grserve_hal_status_t StreamSync(grserve_hal_stream_t, uint64_t) { return Ok(); }
grserve_hal_status_t ExecutableLoad(grserve_hal_executable_t, const uint8_t*, uint64_t) { return Ok(); }
grserve_hal_status_t ExecutableLaunch(grserve_hal_executable_t, grserve_hal_stream_t, const uint8_t*, uint64_t, const grserve_hal_launch_info_t*) { return Ok(); }

}  // namespace

#define GRSERVE_TEST_COUNTER(symbol, counter)                                 \
  extern "C" GRSERVE_HAL_EXPORT unsigned symbol() {                           \
    try {                                                                       \
      return counter;                                                           \
    } catch (...) {                                                             \
      return 0U;                                                                \
    }                                                                           \
  }
GRSERVE_TEST_COUNTER(grserve_test_runtime_create_calls, g_runtime_creates)
GRSERVE_TEST_COUNTER(grserve_test_runtime_destroy_calls, g_runtime_destroys)
GRSERVE_TEST_COUNTER(grserve_test_device_create_calls, g_device_creates)
GRSERVE_TEST_COUNTER(grserve_test_device_destroy_calls, g_device_destroys)
GRSERVE_TEST_COUNTER(grserve_test_stream_create_calls, g_stream_creates)
GRSERVE_TEST_COUNTER(grserve_test_stream_destroy_calls, g_stream_destroys)
GRSERVE_TEST_COUNTER(grserve_test_event_create_calls, g_event_creates)
GRSERVE_TEST_COUNTER(grserve_test_event_destroy_calls, g_event_destroys)
GRSERVE_TEST_COUNTER(grserve_test_buffer_create_calls, g_buffer_creates)
GRSERVE_TEST_COUNTER(grserve_test_buffer_destroy_calls, g_buffer_destroys)
GRSERVE_TEST_COUNTER(grserve_test_executable_create_calls, g_executable_creates)
GRSERVE_TEST_COUNTER(grserve_test_executable_destroy_calls, g_executable_destroys)
#undef GRSERVE_TEST_COUNTER

extern "C" GRSERVE_HAL_EXPORT void grserve_test_reset_counts() {
  try {
    g_runtime_creates = g_runtime_destroys = 0U;
    g_device_creates = g_device_destroys = 0U;
    g_stream_creates = g_stream_destroys = 0U;
    g_event_creates = g_event_destroys = 0U;
    g_buffer_creates = g_buffer_destroys = 0U;
    g_executable_creates = g_executable_destroys = 0U;
    g_fail_next_runtime_destroy = GRSERVE_FIXTURE_MODE == 6;
  } catch (...) {
  }
}

extern "C" GRSERVE_HAL_EXPORT void grserve_test_fail_next_runtime_destroy() {
  try { g_fail_next_runtime_destroy = true; } catch (...) {}
}

#if GRSERVE_FIXTURE_MODE != 3
extern "C" GRSERVE_HAL_EXPORT grserve_hal_status_t grserve_hal_get_api(
    const grserve_hal_api_request_t* request, grserve_hal_api_t* out_api) {
  try {
    if (request == nullptr || out_api == nullptr ||
        request->struct_size < GRSERVE_HAL_API_REQUEST_REQUIRED_SIZE ||
        out_api->struct_size < GRSERVE_HAL_API_REQUIRED_SIZE) return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
#if GRSERVE_FIXTURE_MODE == 1
    grserve_hal_api_t api = {};
    api.struct_size = static_cast<uint32_t>(sizeof(api));
    api.last_error_copy = LastError;
    std::memcpy(out_api, &api, std::min<std::size_t>(out_api->struct_size, sizeof(api)));
    SetDiagnostic("incompatible major"); return GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI;
#elif GRSERVE_FIXTURE_MODE == 2
    grserve_hal_api_t api = {};
    api.struct_size = static_cast<uint32_t>(sizeof(api));
    api.last_error_copy = LastError;
    std::memcpy(out_api, &api, std::min<std::size_t>(out_api->struct_size, sizeof(api)));
    SetDiagnostic("requested capability unavailable"); return GRSERVE_HAL_STATUS_UNSUPPORTED;
#elif GRSERVE_FIXTURE_MODE == 5
    grserve_hal_api_t api = {};
    api.struct_size = static_cast<uint32_t>(sizeof(api));
    api.last_error_copy = LastError;
    std::memcpy(out_api, &api, std::min<std::size_t>(out_api->struct_size, sizeof(api)));
    SetDiagnostic("malformed diagnostic"); return GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI;
#else
    if (request->abi_major != GRSERVE_HAL_ABI_MAJOR || request->min_abi_minor > GRSERVE_HAL_ABI_MINOR) { SetDiagnostic("incompatible abi"); return GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI; }
    grserve_hal_api_t api = {};
    api.struct_size = static_cast<uint32_t>(sizeof(api));
    api.abi_major = GRSERVE_HAL_ABI_MAJOR; api.abi_minor = GRSERVE_HAL_ABI_MINOR; api.capabilities = kCapabilities;
    api.backend_name_copy = BackendName; api.build_id_copy = BuildId;
    api.runtime_create = RuntimeCreate; api.runtime_destroy = RuntimeDestroy;
    api.device_create = DeviceCreate; api.device_destroy = [](grserve_hal_device_t* h) { return Destroy(h, &g_device_destroys); };
    api.stream_create = StreamCreate; api.stream_destroy = [](grserve_hal_stream_t* h) { return Destroy(h, &g_stream_destroys); };
    api.event_create = EventCreate; api.event_destroy = [](grserve_hal_event_t* h) { return Destroy(h, &g_event_destroys); };
    api.buffer_create = BufferCreate; api.buffer_destroy = [](grserve_hal_buffer_t* h) { return Destroy(h, &g_buffer_destroys); };
    api.executable_create = ExecutableCreate; api.executable_destroy = [](grserve_hal_executable_t* h) { return Destroy(h, &g_executable_destroys); };
    api.stream_copy = StreamCopy; api.stream_submit = StreamSubmit; api.event_record = EventRecord; api.event_wait = EventWait; api.event_query = EventQuery; api.stream_synchronize = StreamSync; api.executable_load = ExecutableLoad; api.executable_launch = ExecutableLaunch; api.last_error_copy = LastError;
#if GRSERVE_FIXTURE_MODE == 4
    api.stream_submit = nullptr;
#endif
    if ((request->required_capabilities & (UINT64_C(1) << 63)) != 0U) { api.struct_size = GRSERVE_HAL_API_REQUIRED_SIZE - 1U; api.capabilities = request->required_capabilities; }
    std::memcpy(out_api, &api, std::min<std::size_t>(out_api->struct_size, sizeof(api)));
    return GRSERVE_HAL_STATUS_OK;
#endif
  } catch (...) { return GRSERVE_HAL_STATUS_INTERNAL; }
}
#endif
