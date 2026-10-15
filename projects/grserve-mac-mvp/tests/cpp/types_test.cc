#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

#include "grserve/error.h"
#include "grserve/types.h"
#include "grserve/version.h"

static_assert(!std::is_convertible_v<grserve::RequestId, grserve::RankId>);
static_assert(!std::is_convertible_v<std::uint64_t, grserve::RequestId>);
static_assert(!std::is_default_constructible_v<grserve::StatusOr<int>>);
static_assert(!std::is_constructible_v<grserve::StatusOr<int>, grserve::Status>);
static_assert(!std::is_copy_constructible_v<
              grserve::StatusOr<std::unique_ptr<int>>>);
static_assert(!std::is_copy_assignable_v<grserve::StatusOr<std::unique_ptr<int>>>);
static_assert(std::is_move_constructible_v<
              grserve::StatusOr<std::unique_ptr<int>>>);
static_assert(std::is_move_assignable_v<grserve::StatusOr<std::unique_ptr<int>>>);

namespace {

class ThrowingMove {
 public:
  explicit ThrowingMove(int value) : value_(value) {}

  ThrowingMove(const ThrowingMove&) = delete;
  ThrowingMove& operator=(const ThrowingMove&) = delete;

  ThrowingMove(ThrowingMove&& other) {
    if (throw_on_move_) {
      throw std::runtime_error("move failed");
    }
    value_ = other.value_;
    other.value_ = -1;
  }
  ThrowingMove& operator=(ThrowingMove&& other) noexcept {
    value_ = other.value_;
    other.value_ = -1;
    return *this;
  }

  static void set_throw_on_move(bool value) { throw_on_move_ = value; }

 private:
  int value_;
  static inline bool throw_on_move_ = false;
};

class ConstSensitiveCopy {
 public:
  explicit ConstSensitiveCopy(int value) : value_(value) {}

  ConstSensitiveCopy(ConstSensitiveCopy& other) : value_(other.value_) {
    ++mutable_copies_;
    other.value_ = -1;
  }
  ConstSensitiveCopy(const ConstSensitiveCopy& other) : value_(other.value_) {
    ++const_copies_;
  }
  ConstSensitiveCopy(ConstSensitiveCopy&&) noexcept = default;
  ConstSensitiveCopy& operator=(const ConstSensitiveCopy&) = default;
  ConstSensitiveCopy& operator=(ConstSensitiveCopy&&) noexcept = default;

  [[nodiscard]] int value() const noexcept { return value_; }
  static void reset_copy_counts() {
    mutable_copies_ = 0;
    const_copies_ = 0;
  }
  [[nodiscard]] static int mutable_copies() { return mutable_copies_; }
  [[nodiscard]] static int const_copies() { return const_copies_; }

 private:
  int value_;
  static inline int mutable_copies_ = 0;
  static inline int const_copies_ = 0;
};

class ConstOnlyCopy {
 public:
  explicit ConstOnlyCopy(int value) : value_(value) {}

  ConstOnlyCopy(ConstOnlyCopy&) = delete;
  ConstOnlyCopy(const ConstOnlyCopy&) = default;
  ConstOnlyCopy(ConstOnlyCopy&&) noexcept = default;
  ConstOnlyCopy& operator=(const ConstOnlyCopy&) = default;
  ConstOnlyCopy& operator=(ConstOnlyCopy&&) noexcept = default;

  [[nodiscard]] int value() const noexcept { return value_; }

 private:
  int value_;
};

TEST(StrongId, PreservesValueEqualityAndOrder) {
  constexpr grserve::RequestId request_one{1};
  constexpr grserve::RequestId request_two{2};
  constexpr grserve::RankId rank{1};

  static_assert(request_one.value() == 1);
  static_assert(request_one == grserve::RequestId{1});
  static_assert(request_one < request_two);

  EXPECT_EQ(request_one.value(), 1U);
  EXPECT_NE(request_one, request_two);
  EXPECT_LT(request_one, request_two);
  EXPECT_EQ(rank.value(), 1U);
}

TEST(Status, RepresentsOnlyOkOrError) {
  const grserve::Status ok;
  EXPECT_TRUE(ok.ok());
  EXPECT_EQ(ok.error(), nullptr);
  EXPECT_TRUE(grserve::Status::Ok().ok());

  const auto error = grserve::Error::From(
      grserve::ErrorCode::kInvalidArgument, "bad request");
  const grserve::Status failed{error};
  EXPECT_FALSE(failed.ok());
  ASSERT_NE(failed.error(), nullptr);
  EXPECT_EQ(failed.error()->code(), grserve::ErrorCode::kInvalidArgument);
}

TEST(StatusOr, HoldsEitherValueOrError) {
  grserve::StatusOr<int> value{7};
  EXPECT_TRUE(value.ok());
  EXPECT_EQ(value.value(), 7);
  EXPECT_EQ(value.error(), nullptr);

  grserve::StatusOr<int> failed{
      grserve::Error::From(grserve::ErrorCode::kInternal, "broken")};
  EXPECT_FALSE(failed.ok());
  EXPECT_EQ(failed.value_if(), nullptr);
  ASSERT_NE(failed.error(), nullptr);
  EXPECT_EQ(failed.error()->message(), "broken");
}

TEST(StatusOr, SupportsMoveOnlyValues) {
  grserve::StatusOr<std::unique_ptr<int>> result{std::make_unique<int>(9)};
  ASSERT_TRUE(result.ok());
  ASSERT_NE(result.value(), nullptr);
  EXPECT_EQ(*result.value(), 9);

  std::unique_ptr<int> moved = std::move(result).value();
  ASSERT_NE(moved, nullptr);
  EXPECT_EQ(*moved, 9);
}

TEST(StatusOr, RetainsErrorWhenMoveAssignmentCannotConstructValue) {
  ThrowingMove::set_throw_on_move(false);
  grserve::StatusOr<ThrowingMove> source{ThrowingMove{1}};
  grserve::StatusOr<ThrowingMove> target{
      grserve::Error::From(grserve::ErrorCode::kInternal, "preserve me")};

  ThrowingMove::set_throw_on_move(true);
  EXPECT_THROW(target = std::move(source), std::runtime_error);
  ThrowingMove::set_throw_on_move(false);

  EXPECT_FALSE(target.ok());
  EXPECT_EQ(target.value_if(), nullptr);
  ASSERT_NE(target.error(), nullptr);
  EXPECT_EQ(target.error()->code(), grserve::ErrorCode::kInternal);
  EXPECT_EQ(target.error()->message(), "preserve me");
  EXPECT_FALSE(target.status().ok());
}

TEST(StatusOr, CopiesConstValueWithoutSelectingMutableCopyConstructor) {
  ConstSensitiveCopy::reset_copy_counts();
  const grserve::StatusOr<ConstSensitiveCopy> source{ConstSensitiveCopy{7}};

  const grserve::StatusOr<ConstSensitiveCopy> copy{source};

  ASSERT_TRUE(source.ok());
  ASSERT_TRUE(copy.ok());
  EXPECT_EQ(source.value().value(), 7);
  EXPECT_EQ(copy.value().value(), 7);
  EXPECT_EQ(ConstSensitiveCopy::mutable_copies(), 0);
  EXPECT_EQ(ConstSensitiveCopy::const_copies(), 1);
}

TEST(StatusOr, CopiesValueWithOnlyConstCopyConstructor) {
  const grserve::StatusOr<ConstOnlyCopy> source{ConstOnlyCopy{11}};

  const grserve::StatusOr<ConstOnlyCopy> copy{source};

  ASSERT_TRUE(copy.ok());
  EXPECT_EQ(copy.value().value(), 11);
}

TEST(Version, PublishesExactCompatibilityVersions) {
  EXPECT_EQ(grserve::kEngineVersion,
            (grserve::SemanticVersion{0, 1, 0}));
  EXPECT_EQ(grserve::kRpcVersion,
            (grserve::SemanticVersion{1, 0, 0}));
  EXPECT_EQ(grserve::kHalAbiVersion,
            (grserve::SemanticVersion{1, 0, 0}));
}

}  // namespace
