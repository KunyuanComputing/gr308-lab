#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "grserve/error.h"

namespace grserve {

inline constexpr std::size_t kMiniQwenWeightCount = 456;
inline constexpr std::size_t kMiniQwenWeightBytes =
    kMiniQwenWeightCount * sizeof(std::int32_t);

struct ModelManifest {
  std::uint32_t schema_version;
  std::string model_id;
  std::uint32_t model_format_version;
  std::string topology;
  std::string dtype;
  std::uint32_t vocab_size;
  std::uint32_t hidden_size;
  std::uint32_t max_context_tokens;
  std::string weight_sha256;
  std::uint32_t golden_version;
  std::uint32_t hal_abi_major;
  std::uint32_t hal_abi_minor;
  std::uint64_t required_capabilities;
};

struct HalCompatibility {
  std::uint32_t abi_major;
  std::uint32_t abi_minor;
  std::uint64_t capabilities;
};

struct ModelLoadHooks {
  void (*after_manifest_open)(void* context) = nullptr;
  void (*before_manifest_read)(std::size_t allocation_size,
                               void* context) = nullptr;
  void* context = nullptr;
};

class ModelAssets {
 public:
  ModelAssets(ModelAssets&&) noexcept = default;
  ModelAssets& operator=(ModelAssets&&) = delete;
  ModelAssets(const ModelAssets&) = delete;
  ModelAssets& operator=(const ModelAssets&) = delete;

  [[nodiscard]] const ModelManifest& manifest() const noexcept {
    return manifest_;
  }
  [[nodiscard]] std::span<const std::byte> weights() const noexcept {
    return weights_;
  }

 private:
  friend StatusOr<ModelAssets> LoadModelAssets(const std::filesystem::path&,
                                                HalCompatibility);
  friend StatusOr<ModelAssets> LoadModelAssetsImpl(
      const std::filesystem::path&, HalCompatibility, const ModelLoadHooks*);
  ModelAssets(ModelManifest manifest, std::vector<std::byte> weights)
      : manifest_(std::move(manifest)), weights_(std::move(weights)) {}

  ModelManifest manifest_;
  std::vector<std::byte> weights_;
};

[[nodiscard]] StatusOr<ModelAssets> LoadModelAssets(
    const std::filesystem::path& model_directory,
    HalCompatibility hal_compatibility);

[[nodiscard]] StatusOr<ModelAssets> LoadModelAssetsForTesting(
    const std::filesystem::path& model_directory,
    HalCompatibility hal_compatibility, const ModelLoadHooks& hooks);

}  // namespace grserve
