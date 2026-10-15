#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>

#include "grserve/hal.h"
#include "model/manifest.h"
#include "model/mini_qwen.h"
#include "model/state_codec.h"

struct grserve_hal_runtime_s {};
struct grserve_hal_device_s {
  grserve_hal_runtime_t runtime;
};
struct grserve_hal_stream_s {
  grserve_hal_device_t device;
};
struct grserve_hal_event_s {
  grserve_hal_device_t device;
  std::atomic<bool> complete{false};
};
struct grserve_hal_buffer_s {
  grserve_hal_device_t device;
  std::vector<std::uint8_t> bytes;
};
struct grserve_hal_executable_s {
  grserve_hal_device_t device;
  std::unique_ptr<grserve::MiniQwen> model;
};

namespace {

constexpr grserve_hal_capabilities_t kCapabilities =
    GRSERVE_HAL_CAPABILITY_STREAMS | GRSERVE_HAL_CAPABILITY_EVENTS |
    GRSERVE_HAL_CAPABILITY_ASYNC_COPY | GRSERVE_HAL_CAPABILITY_EXECUTABLES;
constexpr std::uint32_t kSubmitRequired = static_cast<std::uint32_t>(
    offsetof(grserve_hal_submit_info_t, signal_event_count) +
    sizeof(((grserve_hal_submit_info_t*)nullptr)->signal_event_count));
thread_local std::string g_diagnostic;

void SetDiagnostic(std::string_view text) noexcept {
  try {
    g_diagnostic.assign(text.data(), std::min<std::size_t>(text.size(), 511U));
  } catch (...) {
  }
}

grserve_hal_status_t CopyString(std::string_view text, char* output,
                                std::uint32_t capacity,
                                std::uint32_t* required) noexcept {
  if (required == nullptr || (output == nullptr && capacity != 0U) ||
      (output != nullptr && capacity == 0U)) {
    return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
  }
  *required = static_cast<std::uint32_t>(text.size() + 1U);
  if (output == nullptr) {
    return GRSERVE_HAL_STATUS_OK;
  }
  const std::size_t copied =
      std::min<std::size_t>(text.size(), static_cast<std::size_t>(capacity - 1U));
  std::memcpy(output, text.data(), copied);
  output[copied] = '\0';
  return capacity >= text.size() + 1U ? GRSERVE_HAL_STATUS_OK
                                      : GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL;
}

grserve_hal_status_t BackendName(char* output, std::uint32_t capacity,
                                 std::uint32_t* required) noexcept {
  try {
    return CopyString("cpu-reference-hal", output, capacity, required);
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t BuildId(char* output, std::uint32_t capacity,
                             std::uint32_t* required) noexcept {
  try {
    return CopyString("cpu-reference-q16-v1", output, capacity, required);
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t LastError(char* output, std::uint32_t capacity,
                               std::uint32_t* required) noexcept {
  try {
    return CopyString(g_diagnostic, output, capacity, required);
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

template <class Handle>
grserve_hal_status_t Destroy(Handle* handle) noexcept {
  try {
    if (handle != nullptr && *handle != nullptr) {
      delete *handle;
      *handle = nullptr;
    }
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t RuntimeCreate(std::uint32_t,
                                   grserve_hal_runtime_t* output) noexcept {
  try {
    if (output == nullptr) return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    *output = new grserve_hal_runtime_s{};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t RuntimeDestroy(grserve_hal_runtime_t* value) noexcept {
  return Destroy(value);
}
grserve_hal_status_t DeviceCreate(grserve_hal_runtime_t parent, std::uint32_t,
                                  grserve_hal_device_t* output) noexcept {
  try {
    if (parent == nullptr || output == nullptr)
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    *output = new grserve_hal_device_s{parent};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t DeviceDestroy(grserve_hal_device_t* value) noexcept {
  return Destroy(value);
}
grserve_hal_status_t StreamCreate(grserve_hal_device_t parent, std::uint32_t,
                                  grserve_hal_stream_t* output) noexcept {
  try {
    if (parent == nullptr || output == nullptr)
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    *output = new grserve_hal_stream_s{parent};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t StreamDestroy(grserve_hal_stream_t* value) noexcept {
  return Destroy(value);
}
grserve_hal_status_t EventCreate(grserve_hal_device_t parent, std::uint32_t,
                                 grserve_hal_event_t* output) noexcept {
  try {
    if (parent == nullptr || output == nullptr)
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    *output = new grserve_hal_event_s{parent};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t EventDestroy(grserve_hal_event_t* value) noexcept {
  return Destroy(value);
}
grserve_hal_status_t BufferCreate(grserve_hal_device_t parent,
                                  std::uint64_t byte_count, std::uint64_t,
                                  std::uint32_t,
                                  grserve_hal_buffer_t* output) noexcept {
  try {
    if (parent == nullptr || output == nullptr || byte_count > SIZE_MAX)
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    *output = new grserve_hal_buffer_s{
        parent, std::vector<std::uint8_t>(static_cast<std::size_t>(byte_count))};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t BufferDestroy(grserve_hal_buffer_t* value) noexcept {
  return Destroy(value);
}
grserve_hal_status_t ExecutableCreate(grserve_hal_device_t parent,
                                      grserve_hal_executable_t* output) noexcept {
  try {
    if (parent == nullptr || output == nullptr)
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    *output = new grserve_hal_executable_s{parent, nullptr};
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t ExecutableDestroy(grserve_hal_executable_t* value) noexcept {
  return Destroy(value);
}

grserve_hal_status_t StreamCopy(grserve_hal_stream_t stream,
                                grserve_hal_buffer_t destination,
                                std::uint64_t destination_offset,
                                grserve_hal_buffer_t source,
                                std::uint64_t source_offset,
                                std::uint64_t count) noexcept {
  try {
    if (stream == nullptr || destination == nullptr || source == nullptr ||
        destination_offset > destination->bytes.size() ||
        source_offset > source->bytes.size() ||
        count > destination->bytes.size() - destination_offset ||
        count > source->bytes.size() - source_offset)
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    std::memmove(destination->bytes.data() + destination_offset,
                 source->bytes.data() + source_offset,
                 static_cast<std::size_t>(count));
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t StreamSubmit(grserve_hal_stream_t stream,
                                  const grserve_hal_submit_info_t* info) noexcept {
  try {
    if (stream == nullptr || info == nullptr || info->struct_size < kSubmitRequired ||
        (info->wait_event_count != 0U && info->wait_events == nullptr) ||
        (info->signal_event_count != 0U && info->signal_events == nullptr))
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    for (std::uint32_t index = 0; index < info->wait_event_count; ++index) {
      if (info->wait_events[index] == nullptr)
        return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
      if (!info->wait_events[index]->complete.load())
        return GRSERVE_HAL_STATUS_TIMEOUT;
    }
    for (std::uint32_t index = 0; index < info->signal_event_count; ++index) {
      if (info->signal_events[index] == nullptr)
        return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
      info->signal_events[index]->complete.store(true);
    }
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t EventRecord(grserve_hal_event_t event,
                                 grserve_hal_stream_t stream) noexcept {
  try {
    if (event == nullptr || stream == nullptr)
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    event->complete.store(true);
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t EventWait(grserve_hal_event_t event,
                               std::uint64_t) noexcept {
  try {
    if (event == nullptr) return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    return event->complete.load() ? GRSERVE_HAL_STATUS_OK
                                  : GRSERVE_HAL_STATUS_TIMEOUT;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t EventQuery(grserve_hal_event_t event,
                                grserve_hal_event_state_t* output) noexcept {
  try {
    if (event == nullptr || output == nullptr)
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    *output = event->complete.load() ? GRSERVE_HAL_EVENT_STATE_COMPLETE
                                     : GRSERVE_HAL_EVENT_STATE_PENDING;
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
grserve_hal_status_t StreamSynchronize(grserve_hal_stream_t stream,
                                       std::uint64_t) noexcept {
  try {
    return stream == nullptr ? GRSERVE_HAL_STATUS_INVALID_ARGUMENT
                             : GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t ExecutableLoad(grserve_hal_executable_t executable,
                                    const std::uint8_t* image,
                                    std::uint64_t image_size) noexcept {
  try {
    if (executable == nullptr || image == nullptr || image_size == 0U ||
        image_size > 4096U ||
        std::memchr(image, '\0', static_cast<std::size_t>(image_size)) != nullptr) {
      SetDiagnostic("invalid model image");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    const std::string path(reinterpret_cast<const char*>(image),
                           static_cast<std::size_t>(image_size));
    auto assets = grserve::LoadModelAssets(
        std::filesystem::path(path),
        {.abi_major = GRSERVE_HAL_ABI_MAJOR,
         .abi_minor = GRSERVE_HAL_ABI_MINOR,
         .capabilities = kCapabilities});
    if (!assets.ok()) {
      SetDiagnostic(assets.error()->message());
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    auto model = grserve::MiniQwen::Load(assets.value());
    if (!model.ok()) {
      SetDiagnostic(model.error()->message());
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    auto replacement = std::make_unique<grserve::MiniQwen>(std::move(model.value()));
    executable->model = std::move(replacement);
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    SetDiagnostic("model allocation failed");
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    SetDiagnostic("model load failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t ExecutableLaunch(
    grserve_hal_executable_t executable, grserve_hal_stream_t stream,
    const std::uint8_t* entry_name, std::uint64_t entry_size,
    const grserve_hal_launch_info_t* launch) noexcept {
  try {
    if (executable == nullptr || executable->model == nullptr || stream == nullptr) {
      SetDiagnostic("invalid executable launch");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    const grserve::MiniQwenHalLaunchArguments* args = nullptr;
    if (grserve::ValidateMiniQwenHalLaunch(entry_name, entry_size, launch,
                                          &args) !=
        GRSERVE_HAL_STATUS_OK) {
      SetDiagnostic("invalid model arguments");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    grserve::CanonicalState canonical = grserve::InitialCanonicalState();
    if (args->canonical_state_byte_count != 0U) {
      auto decoded = grserve::DecodeCanonicalState(
          std::span<const std::byte>(
              reinterpret_cast<const std::byte*>(args->canonical_state_bytes),
              static_cast<std::size_t>(args->canonical_state_byte_count)),
          1U, 64U);
      if (!decoded.ok()) {
        SetDiagnostic("invalid encoded state");
        return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
      }
      canonical = std::move(decoded.value());
    }
    if (!grserve::CanonicalStateMatchesStep(canonical, args->step_index) ||
        (args->step_index != 0U &&
         args->context_token_count != canonical.full_values.size() + 1U)) {
      SetDiagnostic("model step is out of sequence");
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    std::vector<grserve::TokenId> context;
    context.reserve(args->context_token_count);
    for (std::uint32_t index = 0; index < args->context_token_count; ++index)
      context.emplace_back(args->context_token_ids[index]);
    auto candidate = executable->model->ExecuteNext(context, canonical);
    if (!candidate.ok()) {
      SetDiagnostic(candidate.error()->message());
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    if (!candidate.value().next_state.has_value()) {
      return GRSERVE_HAL_STATUS_INTERNAL;
    }
    auto encoded =
        grserve::EncodeCanonicalState(1U, *candidate.value().next_state);
    if (!encoded.ok()) return GRSERVE_HAL_STATUS_INTERNAL;
    if (args->next_state_capacity < encoded.value().size()) {
      *args->out_next_state_byte_count = encoded.value().size();
      SetDiagnostic("next-state buffer too small");
      return GRSERVE_HAL_STATUS_BUFFER_TOO_SMALL;
    }
    *args->out_token_id = candidate.value().token.value();
    std::memcpy(args->out_canonical_digest,
                candidate.value().canonical_digest.data(), 64U);
    args->out_canonical_digest[64] = '\0';
    std::memcpy(args->out_next_state_bytes, encoded.value().data(),
                encoded.value().size());
    *args->out_next_state_byte_count = encoded.value().size();
    return GRSERVE_HAL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    SetDiagnostic("launch allocation failed");
    return GRSERVE_HAL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    SetDiagnostic("launch failed");
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t GetApi(const grserve_hal_api_request_t* request,
                            grserve_hal_api_t* output) noexcept {
  try {
    if (request == nullptr || output == nullptr ||
        request->struct_size < GRSERVE_HAL_API_REQUEST_REQUIRED_SIZE ||
        output->struct_size < GRSERVE_HAL_API_REQUIRED_SIZE)
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    if (request->abi_major != GRSERVE_HAL_ABI_MAJOR ||
        request->min_abi_minor > GRSERVE_HAL_ABI_MINOR)
      return GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI;
    if ((request->required_capabilities & ~kCapabilities) != 0U)
      return GRSERVE_HAL_STATUS_UNSUPPORTED;
    grserve_hal_api_t api{};
    api.struct_size = sizeof(api);
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
    const std::size_t caller_size = output->struct_size;
    std::memcpy(output, &api, std::min(caller_size, sizeof(api)));
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

}  // namespace

extern "C" GRSERVE_HAL_EXPORT grserve_hal_status_t grserve_hal_get_api(
    const grserve_hal_api_request_t* request, grserve_hal_api_t* output) {
  try {
    return GetApi(request, output);
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}
