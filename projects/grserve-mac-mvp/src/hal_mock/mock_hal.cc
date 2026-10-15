#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <vector>

#if defined(GRSERVE_HAL_BLOCKING_TEST)
#include <unistd.h>
#endif

#include "grserve/hal.h"
#include "hal_mock/scripted_result.h"
#include "model/mini_qwen.h"

struct grserve_hal_runtime_s {};
struct grserve_hal_device_s {
  grserve_hal_runtime_t runtime;
};
struct grserve_hal_event_s;
struct grserve_hal_stream_s {
  grserve_hal_device_t device;
  std::vector<grserve_hal_event_s*> events;
};
struct grserve_hal_event_s {
  grserve_hal_stream_s* stream = nullptr;
  bool complete = false;
};
struct grserve_hal_buffer_s {
  grserve_hal_device_t device;
  uint64_t byte_length;
};
struct grserve_hal_executable_s {
  grserve_hal_device_t device;
  std::vector<grserve::hal_mock::ScriptedResult> script;
};

namespace {

constexpr grserve_hal_capabilities_t kCapabilities =
    GRSERVE_HAL_CAPABILITY_STREAMS | GRSERVE_HAL_CAPABILITY_EVENTS |
    GRSERVE_HAL_CAPABILITY_ASYNC_COPY | GRSERVE_HAL_CAPABILITY_EXECUTABLES;
constexpr uint64_t kEventVirtualDurationNs = 1U;
constexpr std::size_t kMaxDiagnosticBytes = 512U;
constexpr uint32_t kSubmitInfoRequiredSize = static_cast<uint32_t>(
    offsetof(grserve_hal_submit_info_t, signal_event_count) +
    sizeof(((grserve_hal_submit_info_t*)nullptr)->signal_event_count));
constexpr uint32_t kLaunchInfoRequiredSize = static_cast<uint32_t>(
    offsetof(grserve_hal_launch_info_t, argument_bytes) +
    sizeof(((grserve_hal_launch_info_t*)nullptr)->argument_bytes));

thread_local std::string g_diagnostic;
std::mutex g_state_mutex;

void SetDiagnostic(std::string_view text) noexcept {
  try {
    g_diagnostic.assign(text.data(), std::min(text.size(), kMaxDiagnosticBytes - 1U));
  } catch (...) {
  }
}

grserve_hal_status_t CopyString(std::string_view text, char* out_bytes,
                                uint32_t out_capacity,
                                uint32_t* out_required_bytes) noexcept {
  if (out_required_bytes == nullptr ||
      (out_bytes == nullptr && out_capacity != 0U) ||
      (out_bytes != nullptr && out_capacity == 0U)) {
    return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
  }
  const uint32_t required = static_cast<uint32_t>(text.size() + 1U);
  *out_required_bytes = required;
  if (out_bytes == nullptr) {
    return GRSERVE_HAL_STATUS_OK;
  }
  if (out_capacity >= required) {
    std::memcpy(out_bytes, text.data(), text.size());
    out_bytes[text.size()] = '\0';
    return GRSERVE_HAL_STATUS_OK;
  }
  const std::size_t prefix = static_cast<std::size_t>(out_capacity - 1U);
  std::memcpy(out_bytes, text.data(), prefix);
  out_bytes[prefix] = '\0';
  return GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL;
}

grserve_hal_status_t BackendName(char* bytes, uint32_t capacity,
                                 uint32_t* required) noexcept {
  try {
    return CopyString("mock-hal", bytes, capacity, required);
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t BuildId(char* bytes, uint32_t capacity,
                             uint32_t* required) noexcept {
  try {
    return CopyString("mock-hal-virtual-time-v1", bytes, capacity, required);
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t LastError(char* bytes, uint32_t capacity,
                               uint32_t* required) noexcept {
  try {
    return CopyString(g_diagnostic, bytes, capacity, required);
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

template <class Handle>
grserve_hal_status_t Destroy(Handle* handle) noexcept {
  if (handle == nullptr || *handle == nullptr) {
    return GRSERVE_HAL_STATUS_OK;
  }
  delete *handle;
  *handle = nullptr;
  return GRSERVE_HAL_STATUS_OK;
}

bool RangeFits(const grserve_hal_buffer_s* buffer, uint64_t offset,
               uint64_t byte_count) noexcept {
  return offset <= buffer->byte_length &&
         byte_count <= buffer->byte_length - offset;
}

void RemoveFromStream(grserve_hal_event_s* event) noexcept {
  if (event->stream == nullptr) {
    return;
  }
  std::vector<grserve_hal_event_s*>& events = event->stream->events;
  const auto found = std::find(events.begin(), events.end(), event);
  if (found != events.end()) {
    events.erase(found);
  }
  event->stream = nullptr;
}

// Caller holds g_state_mutex. The caller must reserve the destination vector
// before a batch of associations if allocation failure must be atomic.
void AssociatePending(grserve_hal_event_s* event,
                      grserve_hal_stream_s* stream) {
  if (event->stream != stream) {
    stream->events.push_back(event);
    RemoveFromStream(event);
    event->stream = stream;
  }
  event->complete = false;
}

grserve_hal_status_t RuntimeCreate(uint32_t, grserve_hal_runtime_t* out) noexcept {
  try {
    if (out == nullptr) {
      SetDiagnostic("runtime output missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    *out = new grserve_hal_runtime_s{};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    SetDiagnostic("runtime allocation failed");
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    SetDiagnostic("runtime create failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t RuntimeDestroy(grserve_hal_runtime_t* runtime) noexcept {
  try {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return Destroy(runtime);
  } catch (...) {
    SetDiagnostic("runtime destroy failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t DeviceCreate(grserve_hal_runtime_t runtime, uint32_t,
                                   grserve_hal_device_t* out) noexcept {
  try {
    if (runtime == nullptr || out == nullptr) {
      SetDiagnostic("device parent or output missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    *out = new grserve_hal_device_s{runtime};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    SetDiagnostic("device allocation failed");
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    SetDiagnostic("device create failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t DeviceDestroy(grserve_hal_device_t* device) noexcept {
  try {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return Destroy(device);
  } catch (...) {
    SetDiagnostic("device destroy failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t StreamCreate(grserve_hal_device_t device, uint32_t,
                                   grserve_hal_stream_t* out) noexcept {
  try {
    if (device == nullptr || out == nullptr) {
      SetDiagnostic("stream parent or output missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    *out = new grserve_hal_stream_s{device};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    SetDiagnostic("stream allocation failed");
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    SetDiagnostic("stream create failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t StreamDestroy(grserve_hal_stream_t* stream) noexcept {
  try {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (stream == nullptr || *stream == nullptr) {
      return GRSERVE_HAL_STATUS_OK;
    }
    for (grserve_hal_event_s* event : (*stream)->events) {
      if (event->stream == *stream) {
        event->stream = nullptr;
      }
    }
    (*stream)->events.clear();
    return Destroy(stream);
  } catch (...) {
    SetDiagnostic("stream destroy failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t EventCreate(grserve_hal_device_t device, uint32_t,
                                  grserve_hal_event_t* out) noexcept {
  try {
    if (device == nullptr || out == nullptr) {
      SetDiagnostic("event parent or output missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    *out = new grserve_hal_event_s{};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    SetDiagnostic("event allocation failed");
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    SetDiagnostic("event create failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t EventDestroy(grserve_hal_event_t* event) noexcept {
  try {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (event == nullptr || *event == nullptr) {
      return GRSERVE_HAL_STATUS_OK;
    }
    RemoveFromStream(*event);
    return Destroy(event);
  } catch (...) {
    SetDiagnostic("event destroy failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t BufferCreate(grserve_hal_device_t device, uint64_t byte_length,
                                   uint64_t, uint32_t,
                                   grserve_hal_buffer_t* out) noexcept {
  try {
    if (device == nullptr || out == nullptr) {
      SetDiagnostic("buffer parent or output missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    *out = new grserve_hal_buffer_s{device, byte_length};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    SetDiagnostic("buffer allocation failed");
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    SetDiagnostic("buffer create failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t BufferDestroy(grserve_hal_buffer_t* buffer) noexcept {
  try {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return Destroy(buffer);
  } catch (...) {
    SetDiagnostic("buffer destroy failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t ExecutableCreate(grserve_hal_device_t device,
                                       grserve_hal_executable_t* out) noexcept {
  try {
    if (device == nullptr || out == nullptr) {
      SetDiagnostic("executable parent or output missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    *out = new grserve_hal_executable_s{device, {}};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    SetDiagnostic("executable allocation failed");
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    SetDiagnostic("executable create failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t ExecutableDestroy(grserve_hal_executable_t* executable) noexcept {
  try {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return Destroy(executable);
  } catch (...) {
    SetDiagnostic("executable destroy failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t StreamCopy(grserve_hal_stream_t stream,
                                grserve_hal_buffer_t destination,
                                uint64_t destination_offset,
                                grserve_hal_buffer_t source,
                                uint64_t source_offset,
                                uint64_t byte_count) noexcept {
  try {
    if (stream == nullptr || destination == nullptr || source == nullptr) {
      SetDiagnostic("stream copy handle missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (!RangeFits(destination, destination_offset, byte_count) ||
        !RangeFits(source, source_offset, byte_count)) {
      SetDiagnostic("stream copy range out of bounds");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    SetDiagnostic("stream copy failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t StreamSubmit(grserve_hal_stream_t stream,
                                  const grserve_hal_submit_info_t* submit) noexcept {
  try {
    if (stream == nullptr || submit == nullptr ||
        submit->struct_size < kSubmitInfoRequiredSize ||
        (submit->wait_event_count != 0U && submit->wait_events == nullptr) ||
        (submit->signal_event_count != 0U && submit->signal_events == nullptr)) {
      SetDiagnostic("stream submit input missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    for (uint32_t index = 0U; index < submit->wait_event_count; ++index) {
      const grserve_hal_event_t event = submit->wait_events[index];
      if (event == nullptr) {
        SetDiagnostic("stream submit wait event missing");
        return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
      }
      if (!event->complete) {
        SetDiagnostic("stream submit wait event pending");
        return GRSERVE_HAL_STATUS_TIMEOUT;
      }
    }
    for (uint32_t index = 0U; index < submit->signal_event_count; ++index) {
      if (submit->signal_events[index] == nullptr) {
        SetDiagnostic("stream submit signal event missing");
        return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
      }
    }
    stream->events.reserve(stream->events.size() + submit->signal_event_count);
    for (uint32_t index = 0U; index < submit->signal_event_count; ++index) {
      AssociatePending(submit->signal_events[index], stream);
    }
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    SetDiagnostic("stream submit failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t EventRecord(grserve_hal_event_t event,
                                 grserve_hal_stream_t stream) noexcept {
  try {
    if (event == nullptr || stream == nullptr) {
      SetDiagnostic("event record handle missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    stream->events.reserve(stream->events.size() + 1U);
    AssociatePending(event, stream);
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    SetDiagnostic("event record failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t EventWait(grserve_hal_event_t event, uint64_t timeout_ns) noexcept {
  try {
    if (event == nullptr) {
      SetDiagnostic("event wait handle missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (event->complete) {
      return GRSERVE_HAL_STATUS_OK;
    }
    if (timeout_ns < kEventVirtualDurationNs) {
      SetDiagnostic("event virtual deadline exceeded");
      return GRSERVE_HAL_STATUS_TIMEOUT;
    }
    event->complete = true;
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    SetDiagnostic("event wait failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t EventQuery(grserve_hal_event_t event,
                                grserve_hal_event_state_t* out_state) noexcept {
  try {
    if (event == nullptr || out_state == nullptr) {
      SetDiagnostic("event query input missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    *out_state = event->complete ? GRSERVE_HAL_EVENT_STATE_COMPLETE
                                 : GRSERVE_HAL_EVENT_STATE_PENDING;
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    SetDiagnostic("event query failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t StreamSynchronize(grserve_hal_stream_t stream,
                                       uint64_t timeout_ns) noexcept {
  try {
    if (stream == nullptr) {
      SetDiagnostic("stream synchronize handle missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    const bool has_pending = std::any_of(
        stream->events.begin(), stream->events.end(),
        [](const grserve_hal_event_s* event) { return !event->complete; });
    if (has_pending && timeout_ns < kEventVirtualDurationNs) {
      SetDiagnostic("stream virtual deadline exceeded");
      return GRSERVE_HAL_STATUS_TIMEOUT;
    }
    for (grserve_hal_event_s* event : stream->events) {
      event->complete = true;
    }
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    SetDiagnostic("stream synchronize failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t ExecutableLoad(grserve_hal_executable_t executable,
                                    const uint8_t* image, uint64_t image_size) noexcept {
  try {
    if (executable == nullptr || image == nullptr || image_size == 0U ||
        image_size > 4096U) {
      SetDiagnostic("executable load input missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    executable->script.clear();
    if (std::memchr(image, '\0', static_cast<std::size_t>(image_size)) != nullptr) {
      return GRSERVE_HAL_STATUS_OK;
    }
    const std::string directory(reinterpret_cast<const char*>(image),
                                static_cast<std::size_t>(image_size));
    if (!std::filesystem::is_directory(directory)) {
      return GRSERVE_HAL_STATUS_OK;
    }
    auto loaded = grserve::hal_mock::LoadFrozenGolden(
        std::filesystem::path(directory) / "golden.json");
    if (!loaded.ok()) {
      SetDiagnostic("scripted golden load failed");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    executable->script = std::move(loaded.value());
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    SetDiagnostic("executable load failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t ExecutableLaunch(grserve_hal_executable_t executable,
                                      grserve_hal_stream_t stream,
                                      const uint8_t* entry_name,
                                      uint64_t entry_name_size,
                                      const grserve_hal_launch_info_t* launch) noexcept {
  try {
    if (executable == nullptr || stream == nullptr ||
        (entry_name == nullptr && entry_name_size != 0U) || launch == nullptr ||
        launch->struct_size < kLaunchInfoRequiredSize ||
        (launch->argument_data == nullptr && launch->argument_bytes != 0U)) {
      SetDiagnostic("executable launch input missing");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
#if defined(GRSERVE_HAL_BLOCKING_TEST)
    const char* entered_text = std::getenv("GRSERVE_BLOCKING_HAL_ENTERED_FD");
    const char* release_text = std::getenv("GRSERVE_BLOCKING_HAL_RELEASE_FD");
    if (entered_text != nullptr && release_text != nullptr) {
      const int entered_fd = std::atoi(entered_text);
      const int release_fd = std::atoi(release_text);
      const char entered = 1;
      ssize_t written;
      do { written = write(entered_fd, &entered, sizeof(entered)); }
      while (written < 0 && errno == EINTR);
      char release = 0;
      ssize_t read_result;
      do { read_result = read(release_fd, &release, sizeof(release)); }
      while (read_result < 0 && errno == EINTR);
    }
#endif
    if (executable->script.empty()) {
      return GRSERVE_HAL_STATUS_OK;
    }
    // Check the backend-owned table boundary before the common entry-point
    // validator so no future relaxation of the shared model contract can turn
    // a network-controlled step into an unchecked vector access.
    if (launch->argument_data != nullptr &&
        launch->argument_bytes >= sizeof(std::uint32_t) * 2U) {
      std::uint32_t header[2]{};
      std::memcpy(header, launch->argument_data, sizeof(header));
      if (header[0] >= sizeof(header) &&
          static_cast<std::size_t>(header[1]) >= executable->script.size()) {
        SetDiagnostic("scripted step index is out of range");
        return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
      }
    }
    const grserve::MiniQwenHalLaunchArguments* arguments = nullptr;
    if (grserve::ValidateMiniQwenHalLaunch(entry_name, entry_name_size, launch,
                                          &arguments) !=
        GRSERVE_HAL_STATUS_OK) {
      SetDiagnostic("invalid scripted launch arguments");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    const auto& scripted = executable->script[arguments->step_index];
    const bool context_matches =
        arguments->context_token_count == scripted.expected_context().size() &&
        std::equal(scripted.expected_context().begin(),
                   scripted.expected_context().end(),
                   arguments->context_token_ids);
    const bool explicit_predecessor_matches =
        arguments->canonical_state_byte_count ==
            scripted.expected_predecessor().size() &&
        std::memcmp(arguments->canonical_state_bytes,
                    scripted.expected_predecessor().data(),
                    scripted.expected_predecessor().size()) == 0;
    const bool implicit_initial =
        arguments->step_index == 0U &&
        arguments->canonical_state_bytes == nullptr &&
        arguments->canonical_state_byte_count == 0U;
    const bool predecessor_matches =
        explicit_predecessor_matches || implicit_initial;
    if (!context_matches || !predecessor_matches) {
      SetDiagnostic("request is outside scripted domain");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    if (arguments->next_state_capacity < scripted.encoded_state().size()) {
      *arguments->out_next_state_byte_count = scripted.encoded_state().size();
      SetDiagnostic("next-state buffer too small");
      return GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL;
    }
    *arguments->out_token_id = scripted.candidate_token().value();
    std::memcpy(arguments->out_canonical_digest, scripted.digest_hex().data(), 64U);
    arguments->out_canonical_digest[64] = '\0';
    std::memcpy(arguments->out_next_state_bytes,
                scripted.encoded_state().data(), scripted.encoded_state().size());
    *arguments->out_next_state_byte_count = scripted.encoded_state().size();
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    SetDiagnostic("executable launch failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

}  // namespace

extern "C" GRSERVE_HAL_EXPORT grserve_hal_status_t grserve_hal_get_api(
    const grserve_hal_api_request_t* request, grserve_hal_api_t* out_api) {
  try {
    if (request == nullptr || out_api == nullptr ||
        request->struct_size < GRSERVE_HAL_API_REQUEST_REQUIRED_SIZE ||
        out_api->struct_size < GRSERVE_HAL_API_REQUIRED_SIZE) {
      SetDiagnostic("invalid API negotiation input");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    if (request->abi_major != GRSERVE_HAL_ABI_MAJOR ||
        request->min_abi_minor > GRSERVE_HAL_ABI_MINOR) {
      SetDiagnostic("incompatible ABI request");
      return GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI;
    }
    if ((request->required_capabilities & ~kCapabilities) != 0U) {
      SetDiagnostic("requested capability unavailable");
      return GRSERVE_HAL_STATUS_UNSUPPORTED;
    }

    grserve_hal_api_t api = {};
    api.struct_size = static_cast<uint32_t>(sizeof(api));
    api.abi_major = GRSERVE_HAL_ABI_MAJOR;
    api.abi_minor = GRSERVE_HAL_ABI_MINOR;
    api.capabilities = kCapabilities;
    api.backend_name_copy = BackendName;
    api.build_id_copy = BuildId;
    api.runtime_create = RuntimeCreate;
    api.runtime_destroy = RuntimeDestroy;
    api.device_create = DeviceCreate;
    api.device_destroy = DeviceDestroy;
    api.stream_create = StreamCreate;
    api.stream_destroy = StreamDestroy;
    api.event_create = EventCreate;
    api.event_destroy = EventDestroy;
    api.buffer_create = BufferCreate;
    api.buffer_destroy = BufferDestroy;
    api.executable_create = ExecutableCreate;
    api.executable_destroy = ExecutableDestroy;
    api.stream_copy = StreamCopy;
    api.stream_submit = StreamSubmit;
    api.event_record = EventRecord;
    api.event_wait = EventWait;
    api.event_query = EventQuery;
    api.stream_synchronize = StreamSynchronize;
    api.executable_load = ExecutableLoad;
    api.executable_launch = ExecutableLaunch;
    api.last_error_copy = LastError;
    std::memcpy(out_api, &api,
                std::min<std::size_t>(out_api->struct_size, sizeof(api)));
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    SetDiagnostic("API negotiation failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
