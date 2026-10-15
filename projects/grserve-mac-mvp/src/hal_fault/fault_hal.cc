#include "hal_fault/fault_hook.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>

#include "grserve/hal.h"

namespace {

using GetApiFn = grserve_hal_status_t (*)(const grserve_hal_api_request_t*,
                                          grserve_hal_api_t*);

struct DelegatedHal {
  std::mutex mutex;
  void* library = nullptr;  // Intentionally retained for all delegated callback use.
  grserve_hal_api_t api{};
  bool initialized = false;
};

DelegatedHal& Delegate() {
  static DelegatedHal delegated;
  return delegated;
}

grserve::FaultPlan& EmptyPlan() {
  static grserve::FaultPlan plan(0U, {});
  return plan;
}

grserve::PlanFaultHook& Hook() {
  static grserve::PlanFaultHook hook(EmptyPlan());
  return hook;
}

std::mutex& LaunchMutex() {
  static std::mutex mutex;
  return mutex;
}

std::uint64_t& LaunchOrdinal() {
  static std::uint64_t ordinal = 0U;
  return ordinal;
}

std::uint64_t NextLaunchOrdinal() noexcept {
  std::lock_guard<std::mutex> lock(LaunchMutex());
  std::uint64_t& ordinal = LaunchOrdinal();
  if (ordinal == std::numeric_limits<std::uint64_t>::max()) {
    return 0U;
  }
  return ++ordinal;
}

grserve_hal_status_t StatusForPreEffect(const grserve::FaultEffect& effect) noexcept {
  switch (effect.action) {
    case grserve::FaultAction::kTimeout:
    case grserve::FaultAction::kBlock:
      // HAL has no virtual clock argument; block is represented without sleep.
      return GRSERVE_HAL_STATUS_TIMEOUT;
    case grserve::FaultAction::kFail:
    case grserve::FaultAction::kRankLoss:
      return GRSERVE_HAL_STATUS_INTERNAL;
    case grserve::FaultAction::kDropAck:
      return GRSERVE_HAL_STATUS_OK;
  }
  return GRSERVE_HAL_STATUS_INTERNAL;
}

grserve_hal_status_t ExecutableLaunch(grserve_hal_executable_t executable,
                                      grserve_hal_stream_t stream,
                                      const uint8_t* entry_name,
                                      uint64_t entry_name_byte_count,
                                      const grserve_hal_launch_info_t* launch_info) noexcept {
  try {
    const std::uint64_t ordinal = NextLaunchOrdinal();
    if (ordinal != 0U) {
      if (const auto effect = Hook().Before(grserve::RankId{0},
                                            grserve::FaultPhase::kHalLaunch,
                                            ordinal)) {
        return StatusForPreEffect(*effect);
      }
    }
    DelegatedHal& delegated = Delegate();
    grserve_hal_executable_launch_fn launch = nullptr;
    {
      std::lock_guard<std::mutex> lock(delegated.mutex);
      launch = delegated.api.executable_launch;
    }
    if (launch == nullptr) {
      return GRSERVE_HAL_STATUS_INTERNAL;
    }
    const grserve_hal_status_t status = launch(
        executable, stream, entry_name, entry_name_byte_count, launch_info);
    if (status != GRSERVE_HAL_STATUS_OK || ordinal == 0U) {
      return status;
    }
    // DropAck is a post-operation effect. The default public plan is empty;
    // higher-layer configuration will supply a non-empty hook in a later task.
    if (Hook().After(grserve::RankId{0}, grserve::FaultPhase::kHalLaunch,
                     ordinal)) {
      return GRSERVE_HAL_STATUS_TIMEOUT;
    }
    return status;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

grserve_hal_status_t DelegateApi(const grserve_hal_api_request_t* request,
                                 grserve_hal_api_t* out_api) noexcept {
  try {
    if (request == nullptr || out_api == nullptr ||
        request->struct_size < GRSERVE_HAL_API_REQUEST_REQUIRED_SIZE ||
        out_api->struct_size < GRSERVE_HAL_API_REQUIRED_SIZE) {
      return GRSERVE_HAL_STATUS_INVALID_ARGUMENT;
    }
    if (request->abi_major != GRSERVE_HAL_ABI_MAJOR ||
        request->min_abi_minor > GRSERVE_HAL_ABI_MINOR) {
      return GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI;
    }

    DelegatedHal& delegated = Delegate();
    std::lock_guard<std::mutex> lock(delegated.mutex);
    if (!delegated.initialized) {
      delegated.library = dlopen(GRSERVE_MOCK_HAL_PATH, RTLD_NOW | RTLD_LOCAL);
      if (delegated.library == nullptr) {
        return GRSERVE_HAL_STATUS_INTERNAL;
      }
      const auto get_api = reinterpret_cast<GetApiFn>(
          dlsym(delegated.library, "grserve_hal_get_api"));
      if (get_api == nullptr) {
        return GRSERVE_HAL_STATUS_INTERNAL;
      }
      grserve_hal_api_t underlying{};
      underlying.struct_size = static_cast<uint32_t>(sizeof(underlying));
      const grserve_hal_status_t status = get_api(request, &underlying);
      if (status != GRSERVE_HAL_STATUS_OK) {
        return status;
      }
      if (underlying.struct_size < GRSERVE_HAL_API_REQUIRED_SIZE ||
          underlying.executable_launch == nullptr) {
        return GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI;
      }
      delegated.api = underlying;
      delegated.initialized = true;
    }
    if ((request->required_capabilities & ~delegated.api.capabilities) != 0U) {
      return GRSERVE_HAL_STATUS_UNSUPPORTED;
    }
    grserve_hal_api_t wrapper = delegated.api;
    wrapper.executable_launch = ExecutableLaunch;
    std::memcpy(out_api, &wrapper,
                std::min<std::size_t>(out_api->struct_size, sizeof(wrapper)));
    return GRSERVE_HAL_STATUS_OK;
  } catch (...) {
    return GRSERVE_HAL_STATUS_INTERNAL;
  }
}

}  // namespace

extern "C" GRSERVE_HAL_EXPORT grserve_hal_status_t grserve_hal_get_api(
    const grserve_hal_api_request_t* request, grserve_hal_api_t* out_api) {
  return DelegateApi(request, out_api);
}
