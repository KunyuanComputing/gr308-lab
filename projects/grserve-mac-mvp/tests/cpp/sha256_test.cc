#include <string_view>

#include <gtest/gtest.h>

#include "common/sha256.h"

namespace grserve {
class Sha256TestPeer {
 public:
  static Sha256 WithByteCount(std::uint64_t byte_count) {
    Sha256 sha;
    sha.total_bytes_ = byte_count;
    return sha;
  }
};

namespace {

TEST(Sha256Test, MatchesNistEmptyVector) {
  auto digest = Sha256Hex(std::string_view{});
  ASSERT_TRUE(digest.ok());
  EXPECT_EQ(digest.value(),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
}

TEST(Sha256Test, MatchesNistAbcVector) {
  auto digest = Sha256Hex("abc");
  ASSERT_TRUE(digest.ok());
  EXPECT_EQ(digest.value(),
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad");
}

TEST(Sha256Test, RejectsLengthAtTwoToTheSixtyFirstBytesAndStaysFailed) {
  Sha256 boundary =
      Sha256TestPeer::WithByteCount((std::uint64_t{1} << 61) - 2);
  Sha256 sha = Sha256TestPeer::WithByteCount((std::uint64_t{1} << 61) - 1);
  const std::array one_byte{std::byte{0}};
  EXPECT_TRUE(boundary.Update(one_byte).ok());
  const Status first = sha.Update(one_byte);
  ASSERT_FALSE(first.ok());
  EXPECT_EQ(first.error()->code(), ErrorCode::kInvalidArgument);
  const Status second = sha.Update(std::string_view{});
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(second.error()->code(), first.error()->code());
  auto digest = sha.FinalHex();
  ASSERT_FALSE(digest.ok());
  EXPECT_EQ(digest.error()->code(), first.error()->code());
}

TEST(Sha256Test, RejectsUpdateAfterFinalization) {
  Sha256 sha;
  ASSERT_TRUE(sha.Final().ok());
  const Status update = sha.Update("later");
  ASSERT_FALSE(update.ok());
  EXPECT_EQ(update.error()->code(), ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace grserve
