#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>

#include "grserve/hal.h"
#include "model/manifest.h"

namespace grserve {
namespace {

static_assert(!std::is_copy_constructible_v<ModelAssets>);
static_assert(!std::is_copy_assignable_v<ModelAssets>);
static_assert(std::is_nothrow_move_constructible_v<ModelAssets>);
static_assert(!std::is_move_assignable_v<ModelAssets>);

struct HalBootstrapSpy {
  int runtime_create = 0;
  int executable_create = 0;
  int executable_launch = 0;
};

constexpr HalCompatibility CompatibleHal() {
  return HalCompatibility{.abi_major = GRSERVE_HAL_ABI_MAJOR,
                          .abi_minor = GRSERVE_HAL_ABI_MINOR,
                          .capabilities =
                              GRSERVE_HAL_CAPABILITY_EXECUTABLES};
}

StatusOr<ModelAssets> BootstrapModel(const std::filesystem::path& directory,
                                     HalCompatibility compatibility,
                                     HalBootstrapSpy& spy) {
  auto assets = LoadModelAssets(directory, compatibility);
  if (assets.ok()) {
    ++spy.runtime_create;
    ++spy.executable_create;
    ++spy.executable_launch;
  }
  return assets;
}

StatusOr<ModelAssets> BootstrapModelWithHooks(
    const std::filesystem::path& directory, HalCompatibility compatibility,
    const ModelLoadHooks& hooks, HalBootstrapSpy& spy) {
  auto assets = LoadModelAssetsForTesting(directory, compatibility, hooks);
  if (assets.ok()) {
    ++spy.runtime_create;
    ++spy.executable_create;
    ++spy.executable_launch;
  }
  return assets;
}

void ExpectHalUntouched(const HalBootstrapSpy& spy) {
  EXPECT_EQ(spy.runtime_create, 0);
  EXPECT_EQ(spy.executable_create, 0);
  EXPECT_EQ(spy.executable_launch, 0);
}

class UniqueTempDirectory {
 public:
  UniqueTempDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "grserve-model-test-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~UniqueTempDirectory() { std::filesystem::remove_all(path_); }
  UniqueTempDirectory(const UniqueTempDirectory&) = delete;
  UniqueTempDirectory& operator=(const UniqueTempDirectory&) = delete;
  UniqueTempDirectory(UniqueTempDirectory&&) = delete;
  UniqueTempDirectory& operator=(UniqueTempDirectory&&) = delete;

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

class TempModelDir : public testing::Test {
 protected:
  void SetUp() override {
    dir_ = temporary_directory_.path();
    std::filesystem::copy_file(
        std::filesystem::path(GRSERVE_MODEL_DIR) / "manifest.textproto",
        dir_ / "manifest.textproto");
    std::filesystem::copy_file(
        std::filesystem::path(GRSERVE_MODEL_DIR) / "weights.i32le",
        dir_ / "weights.i32le");
  }

  std::string ReadManifest() const {
    std::ifstream input(dir_ / "manifest.textproto");
    return {std::istreambuf_iterator<char>(input), {}};
  }

  void WriteManifest(const std::string& text) const {
    std::ofstream output(dir_ / "manifest.textproto", std::ios::trunc);
    output << text;
  }

  StatusOr<ModelAssets> Load(
      HalCompatibility compatibility = CompatibleHal()) {
    return BootstrapModel(dir_, compatibility, hal_spy_);
  }

  std::filesystem::path dir_;
  UniqueTempDirectory temporary_directory_;
  HalBootstrapSpy hal_spy_;
};

TEST_F(TempModelDir, LoadsFrozenAssets) {
  auto result = Load();
  ASSERT_TRUE(result.ok()) << result.error()->message();
  EXPECT_EQ(result.value().manifest().model_id, "mini-qwen38-hybrid");
  EXPECT_EQ(result.value().weights().size(), 1824u);
  EXPECT_EQ(hal_spy_.runtime_create, 1);
  EXPECT_EQ(hal_spy_.executable_create, 1);
  EXPECT_EQ(hal_spy_.executable_launch, 1);
}

TEST_F(TempModelDir, RejectsUnknownManifestField) {
  WriteManifest(ReadManifest() + "secret_unknown_field: 1\n");
  testing::internal::CaptureStderr();
  auto result = Load();
  const std::string stderr_output = testing::internal::GetCapturedStderr();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(stderr_output.find("secret_unknown_field"), std::string::npos);
  EXPECT_TRUE(stderr_output.empty());
  ExpectHalUntouched(hal_spy_);
}

struct ManifestGrowthContext {
  std::filesystem::path path;
  std::size_t allocation_size = std::numeric_limits<std::size_t>::max();
};

void GrowManifestAfterOpen(void* opaque) {
  const auto* context = static_cast<const ManifestGrowthContext*>(opaque);
  std::ofstream output(context->path, std::ios::app | std::ios::binary);
  output << std::string(70 * 1024, 'x');
}

void RecordManifestAllocation(std::size_t allocation_size, void* opaque) {
  auto* context = static_cast<ManifestGrowthContext*>(opaque);
  context->allocation_size = allocation_size;
}

TEST_F(TempModelDir, RejectsManifestGrowthAfterValidatedOpen) {
  ManifestGrowthContext context{.path = dir_ / "manifest.textproto"};
  const auto validated_size = std::filesystem::file_size(context.path);
  ModelLoadHooks hooks{.after_manifest_open = GrowManifestAfterOpen,
                       .before_manifest_read = RecordManifestAllocation,
                       .context = std::addressof(context)};
  auto result =
      BootstrapModelWithHooks(dir_, CompatibleHal(), hooks, hal_spy_);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(context.allocation_size, validated_size);
  EXPECT_LE(context.allocation_size, 64u * 1024u);
  ExpectHalUntouched(hal_spy_);
}

TEST_F(TempModelDir, RejectsDuplicateManifestField) {
  WriteManifest(ReadManifest() + "schema_version: 1\n");
  EXPECT_FALSE(Load().ok());
  ExpectHalUntouched(hal_spy_);
}

TEST_F(TempModelDir, RejectsMissingManifestField) {
  std::string text = ReadManifest();
  const auto start = text.find("golden_version:");
  ASSERT_NE(start, std::string::npos);
  text.erase(start, text.find('\n', start) - start + 1);
  WriteManifest(text);
  EXPECT_FALSE(Load().ok());
  ExpectHalUntouched(hal_spy_);
}

TEST_F(TempModelDir, RejectsInvalidFixedValue) {
  std::string text = ReadManifest();
  const auto pos = text.find("vocab_size: 16");
  ASSERT_NE(pos, std::string::npos);
  text.replace(pos, std::string("vocab_size: 16").size(), "vocab_size: 17");
  WriteManifest(text);
  EXPECT_FALSE(Load().ok());
  ExpectHalUntouched(hal_spy_);
}

TEST_F(TempModelDir, RejectsOverflow) {
  std::string text = ReadManifest();
  const auto pos = text.find("vocab_size: 16");
  ASSERT_NE(pos, std::string::npos);
  text.replace(pos, std::string("vocab_size: 16").size(),
               "vocab_size: 4294967296");
  WriteManifest(text);
  EXPECT_FALSE(Load().ok());
  ExpectHalUntouched(hal_spy_);
}

TEST_F(TempModelDir, RejectsWrongWeightSizeBeforeHashing) {
  std::filesystem::resize_file(dir_ / "weights.i32le", 4);
  auto result = Load();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), ErrorCode::kInvalidArgument);
  ExpectHalUntouched(hal_spy_);
}

TEST_F(TempModelDir, RejectsWrongWeightDigest) {
  std::fstream file(dir_ / "weights.i32le",
                    std::ios::binary | std::ios::in | std::ios::out);
  char byte = 0;
  file.read(&byte, 1);
  byte ^= 1;
  file.seekp(0);
  file.write(&byte, 1);
  file.close();
  EXPECT_FALSE(Load().ok());
  ExpectHalUntouched(hal_spy_);
}

TEST_F(TempModelDir, RejectsIncompatibleHalWithoutRuntimeInteraction) {
  auto result = Load(HalCompatibility{
      .abi_major = 2,
      .abi_minor = 0,
      .capabilities = GRSERVE_HAL_CAPABILITY_EXECUTABLES});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), ErrorCode::kIncompatibleVersion);
  ExpectHalUntouched(hal_spy_);
}

TEST_F(TempModelDir, RejectsMissingCapabilityWithoutRuntimeInteraction) {
  auto result = Load(HalCompatibility{.abi_major = GRSERVE_HAL_ABI_MAJOR,
                                      .abi_minor = GRSERVE_HAL_ABI_MINOR,
                                      .capabilities = 0});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->code(), ErrorCode::kUnsupportedCapability);
  ExpectHalUntouched(hal_spy_);
}

TEST_F(TempModelDir, RejectsSymlinkedWeightFile) {
  const auto real = dir_ / "real-weights";
  std::filesystem::rename(dir_ / "weights.i32le", real);
  std::filesystem::create_symlink(real.filename(), dir_ / "weights.i32le");
  EXPECT_FALSE(Load().ok());
  ExpectHalUntouched(hal_spy_);
}

TEST(ModelAssetsPathTest, RejectsNonDirectoryAndDoesNotLeakPath) {
  const auto secret = std::filesystem::temp_directory_path() /
                      "grserve-secret-model-name";
  std::ofstream(secret) << "not a directory";
  HalBootstrapSpy spy;
  auto result = BootstrapModel(secret, CompatibleHal(), spy);
  std::filesystem::remove(secret);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error()->message().find("grserve-secret-model-name"),
            std::string::npos);
  ExpectHalUntouched(spy);
}

TEST_F(TempModelDir, LoadsEquivalentPathContainingLexicalParent) {
  const auto traversing = dir_ / ".." / dir_.filename();
  auto result = BootstrapModel(traversing, CompatibleHal(), hal_spy_);
  ASSERT_TRUE(result.ok()) << result.error()->message();
  EXPECT_EQ(hal_spy_.runtime_create, 1);
  EXPECT_EQ(hal_spy_.executable_create, 1);
  EXPECT_EQ(hal_spy_.executable_launch, 1);
}

}  // namespace
}  // namespace grserve
