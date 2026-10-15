#include "model/manifest.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <limits>
#include <memory>
#include <string_view>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/tokenizer.h>

#include "common/sha256.h"
#include "grserve/hal.h"
#include "model_manifest.pb.h"

namespace grserve {
namespace {

constexpr std::size_t kMaxManifestBytes = 64 * 1024;

struct FdCloser {
  void operator()(int* fd) const noexcept {
    if (fd != nullptr) {
      if (*fd >= 0) {
        close(*fd);
      }
      delete fd;
    }
  }
};
using UniqueFd = std::unique_ptr<int, FdCloser>;

UniqueFd OwnFd(int fd) { return UniqueFd(new int(fd)); }

struct OpenedRegular {
  UniqueFd fd;
  std::size_t validated_size;
};

Error InvalidAsset() {
  return Error::From(ErrorCode::kInvalidArgument, "invalid model assets");
}

StatusOr<OpenedRegular> OpenRegularAt(
    int directory_fd, const char* name, std::size_t maximum_size,
    std::optional<std::size_t> exact_size) {
  const int raw_fd = openat(directory_fd, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (raw_fd < 0) {
    return StatusOr<OpenedRegular>(InvalidAsset());
  }
  auto fd = OwnFd(raw_fd);
  struct stat metadata {};
  if (fstat(*fd, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
      metadata.st_size < 0 ||
      static_cast<std::uintmax_t>(metadata.st_size) > maximum_size ||
      (exact_size && static_cast<std::uintmax_t>(metadata.st_size) != *exact_size)) {
    return StatusOr<OpenedRegular>(InvalidAsset());
  }
  return StatusOr<OpenedRegular>(OpenedRegular{
      .fd = std::move(fd),
      .validated_size = static_cast<std::size_t>(metadata.st_size)});
}

StatusOr<std::vector<std::byte>> ReadExactly(int fd, std::size_t size) {
  std::vector<std::byte> bytes(size);
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = read(fd, bytes.data() + offset, bytes.size() - offset);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      return StatusOr<std::vector<std::byte>>(InvalidAsset());
    }
    offset += static_cast<std::size_t>(count);
  }
  std::byte extra{};
  ssize_t count;
  do {
    count = read(fd, &extra, 1);
  } while (count < 0 && errno == EINTR);
  if (count != 0) {
    return StatusOr<std::vector<std::byte>>(InvalidAsset());
  }
  return StatusOr<std::vector<std::byte>>(std::move(bytes));
}

struct HashedBytes {
  std::vector<std::byte> bytes;
  std::string sha256;
};

StatusOr<HashedBytes> ReadAndHashExactly(int fd, std::size_t size) {
  HashedBytes result{.bytes = std::vector<std::byte>(size), .sha256 = {}};
  Sha256 sha;
  std::size_t offset = 0;
  while (offset < result.bytes.size()) {
    const ssize_t count =
        read(fd, result.bytes.data() + offset, result.bytes.size() - offset);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      return StatusOr<HashedBytes>(InvalidAsset());
    }
    const auto count_as_size = static_cast<std::size_t>(count);
    const Status update = sha.Update(std::span<const std::byte>(
        result.bytes.data() + offset, count_as_size));
    if (!update.ok()) {
      return StatusOr<HashedBytes>(InvalidAsset());
    }
    offset += count_as_size;
  }
  std::byte extra{};
  ssize_t count;
  do {
    count = read(fd, &extra, 1);
  } while (count < 0 && errno == EINTR);
  if (count != 0) {
    return StatusOr<HashedBytes>(InvalidAsset());
  }
  auto digest = sha.FinalHex();
  if (!digest.ok()) {
    return StatusOr<HashedBytes>(InvalidAsset());
  }
  result.sha256 = std::move(digest.value());
  return StatusOr<HashedBytes>(std::move(result));
}

bool HasAllFields(const model::v1::ModelManifest& proto) {
  return proto.has_schema_version() && proto.has_model_id() &&
         proto.has_model_format_version() && proto.has_topology() &&
         proto.has_dtype() && proto.has_vocab_size() &&
         proto.has_hidden_size() && proto.has_max_context_tokens() &&
         proto.has_weight_sha256() && proto.has_golden_version() &&
         proto.has_hal_abi_major() && proto.has_hal_abi_minor() &&
         proto.has_required_capability();
}

bool IsLowerHexDigest(std::string_view digest) {
  if (digest.size() != 64) {
    return false;
  }
  for (char value : digest) {
    if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) {
      return false;
    }
  }
  return true;
}

StatusOr<ModelManifest> ParseAndValidate(std::string_view text) {
  class SilentErrorCollector final
      : public google::protobuf::io::ErrorCollector {
   public:
    void RecordError(int, google::protobuf::io::ColumnNumber,
                     absl::string_view) override {}
    void RecordWarning(int, google::protobuf::io::ColumnNumber,
                       absl::string_view) override {}
  } errors;
  model::v1::ModelManifest proto;
  google::protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&errors);
  parser.AllowUnknownField(false);
  if (!parser.ParseFromString(std::string(text), &proto) || !HasAllFields(proto) ||
      proto.schema_version() != 1 ||
      proto.model_id() != "mini-qwen38-hybrid" ||
      proto.model_format_version() != 1 ||
      proto.topology() != "GDN,GDN,GDN,FULL" || proto.dtype() != "Q16_16" ||
      proto.vocab_size() != 16 || proto.hidden_size() != 8 ||
      proto.max_context_tokens() != 64 || proto.golden_version() != 1 ||
      proto.hal_abi_major() != GRSERVE_HAL_ABI_MAJOR ||
      proto.hal_abi_minor() != GRSERVE_HAL_ABI_MINOR ||
      proto.required_capability() != "EXECUTABLE_LAUNCH" ||
      !IsLowerHexDigest(proto.weight_sha256())) {
    return StatusOr<ModelManifest>(InvalidAsset());
  }
  return StatusOr<ModelManifest>(ModelManifest{
      .schema_version = proto.schema_version(),
      .model_id = proto.model_id(),
      .model_format_version = proto.model_format_version(),
      .topology = proto.topology(),
      .dtype = proto.dtype(),
      .vocab_size = proto.vocab_size(),
      .hidden_size = proto.hidden_size(),
      .max_context_tokens = proto.max_context_tokens(),
      .weight_sha256 = proto.weight_sha256(),
      .golden_version = proto.golden_version(),
      .hal_abi_major = proto.hal_abi_major(),
      .hal_abi_minor = proto.hal_abi_minor(),
      .required_capabilities = GRSERVE_HAL_CAPABILITY_EXECUTABLES,
  });
}

}  // namespace

StatusOr<ModelAssets> LoadModelAssetsImpl(
    const std::filesystem::path& model_directory,
    HalCompatibility hal_compatibility, const ModelLoadHooks* hooks) {
  const int raw_directory_fd =
      open(model_directory.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  if (raw_directory_fd < 0) {
    return StatusOr<ModelAssets>(InvalidAsset());
  }
  auto directory_fd = OwnFd(raw_directory_fd);

  auto manifest_fd = OpenRegularAt(*directory_fd, "manifest.textproto",
                                   kMaxManifestBytes, std::nullopt);
  if (!manifest_fd.ok()) {
    return StatusOr<ModelAssets>(*manifest_fd.error());
  }
  if (hooks != nullptr && hooks->after_manifest_open != nullptr) {
    hooks->after_manifest_open(hooks->context);
  }
  if (hooks != nullptr && hooks->before_manifest_read != nullptr) {
    hooks->before_manifest_read(manifest_fd.value().validated_size,
                                hooks->context);
  }
  auto manifest_bytes = ReadExactly(*manifest_fd.value().fd,
                                    manifest_fd.value().validated_size);
  if (!manifest_bytes.ok()) {
    return StatusOr<ModelAssets>(*manifest_bytes.error());
  }
  const auto* manifest_chars =
      reinterpret_cast<const char*>(manifest_bytes.value().data());
  auto manifest = ParseAndValidate(
      std::string_view(manifest_chars, manifest_bytes.value().size()));
  if (!manifest.ok()) {
    return StatusOr<ModelAssets>(*manifest.error());
  }

  auto weights_fd = OpenRegularAt(*directory_fd, "weights.i32le",
                                  kMiniQwenWeightBytes, kMiniQwenWeightBytes);
  if (!weights_fd.ok()) {
    return StatusOr<ModelAssets>(*weights_fd.error());
  }
  auto weights =
      ReadAndHashExactly(*weights_fd.value().fd,
                         weights_fd.value().validated_size);
  if (!weights.ok()) {
    return StatusOr<ModelAssets>(*weights.error());
  }
  if (weights.value().sha256 != manifest.value().weight_sha256) {
    return StatusOr<ModelAssets>(InvalidAsset());
  }

  if (hal_compatibility.abi_major != manifest.value().hal_abi_major ||
      hal_compatibility.abi_minor < manifest.value().hal_abi_minor) {
    return StatusOr<ModelAssets>(Error::From(
        ErrorCode::kIncompatibleVersion, "incompatible HAL ABI"));
  }
  if ((hal_compatibility.capabilities &
       manifest.value().required_capabilities) !=
      manifest.value().required_capabilities) {
    return StatusOr<ModelAssets>(Error::From(
        ErrorCode::kUnsupportedCapability, "required HAL capability missing"));
  }
  return StatusOr<ModelAssets>(
      ModelAssets(std::move(manifest.value()),
                  std::move(weights.value().bytes)));
}

StatusOr<ModelAssets> LoadModelAssets(
    const std::filesystem::path& model_directory,
    HalCompatibility hal_compatibility) {
  return LoadModelAssetsImpl(model_directory, hal_compatibility, nullptr);
}

StatusOr<ModelAssets> LoadModelAssetsForTesting(
    const std::filesystem::path& model_directory,
    HalCompatibility hal_compatibility, const ModelLoadHooks& hooks) {
  return LoadModelAssetsImpl(model_directory, hal_compatibility, &hooks);
}

}  // namespace grserve
