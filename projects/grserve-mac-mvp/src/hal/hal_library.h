#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "grserve/error.h"
#include "grserve/hal.h"

namespace grserve {

// Internal HAL-boundary utility shared by loader and ownership wrappers.
std::string CopyHalDiagnostic(const grserve_hal_api_t& api);

class HalLibrary {
 public:
  static StatusOr<std::shared_ptr<HalLibrary>> Open(
      const std::filesystem::path& path, std::uint64_t required_capabilities);

  const grserve_hal_api_t& api() const noexcept { return api_; }

  HalLibrary(const HalLibrary&) = delete;
  HalLibrary& operator=(const HalLibrary&) = delete;
  ~HalLibrary();

 private:
  HalLibrary(void* native_library, grserve_hal_api_t api) noexcept
      : native_library_(native_library), api_(api) {}

  void* native_library_ = nullptr;
  grserve_hal_api_t api_ = {};
};

}  // namespace grserve
