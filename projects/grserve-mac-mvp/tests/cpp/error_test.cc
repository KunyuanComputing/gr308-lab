#include <array>
#include <optional>
#include <type_traits>

#include <gtest/gtest.h>

#include "grserve/error.h"

namespace {

static_assert(std::is_nothrow_copy_constructible_v<grserve::Error>);
static_assert(std::is_nothrow_copy_assignable_v<grserve::Error>);

TEST(Error, RetryabilityIsStable) {
  using grserve::ErrorCode;
  EXPECT_FALSE(grserve::Error::From(ErrorCode::kInvalidArgument, "bad").retryable());
  EXPECT_TRUE(grserve::Error::From(ErrorCode::kResourceExhausted, "full").retryable());
  EXPECT_FALSE(grserve::Error::From(ErrorCode::kReplicaFatal, "fatal").retryable());
}

TEST(Error, UsesTheExactRetryabilityTable) {
  using grserve::ErrorCode;
  const std::array cases{
      std::pair{ErrorCode::kInvalidArgument, false},
      std::pair{ErrorCode::kIncompatibleVersion, false},
      std::pair{ErrorCode::kUnsupportedCapability, false},
      std::pair{ErrorCode::kResourceExhausted, true},
      std::pair{ErrorCode::kDeadlineExceeded, true},
      std::pair{ErrorCode::kRankFailure, true},
      std::pair{ErrorCode::kReplicaFatal, false},
      std::pair{ErrorCode::kInternal, false},
  };

  for (const auto& [code, retryable] : cases) {
    EXPECT_EQ(grserve::Error::From(code, "message").retryable(), retryable);
  }
}

TEST(Error, CarriesCodeMessageAndCorrelationIds) {
  const auto error = grserve::Error::From(
      grserve::ErrorCode::kRankFailure, "rank stopped", grserve::RequestId{12},
      grserve::TransactionId{34});

  EXPECT_EQ(error.code(), grserve::ErrorCode::kRankFailure);
  EXPECT_EQ(error.message(), "rank stopped");
  EXPECT_EQ(error.request_id(), std::optional{grserve::RequestId{12}});
  EXPECT_EQ(error.transaction_id(), std::optional{grserve::TransactionId{34}});
}

TEST(Error, RecodesAndScopesWithoutChangingTheSharedRootCause) {
  const auto source = grserve::Error::From(
      grserve::ErrorCode::kRankFailure,
      "rank 1 lost after commit decision", grserve::RequestId{1},
      grserve::TransactionId{9});
  const auto scoped = source.RecodeForRequest(
      grserve::ErrorCode::kReplicaFatal, grserve::RequestId{2});

  EXPECT_EQ(scoped.code(), grserve::ErrorCode::kReplicaFatal);
  EXPECT_EQ(scoped.message(), "rank 1 lost after commit decision");
  EXPECT_EQ(scoped.request_id(), std::optional{grserve::RequestId{2}});
  EXPECT_EQ(scoped.transaction_id(), std::nullopt);
  EXPECT_EQ(source.code(), grserve::ErrorCode::kRankFailure);
  EXPECT_EQ(source.request_id(), std::optional{grserve::RequestId{1}});
}

}  // namespace
