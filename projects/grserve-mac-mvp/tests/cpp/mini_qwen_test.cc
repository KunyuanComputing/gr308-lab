#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include "grserve/hal.h"
#include "model/manifest.h"
#include "model/mini_qwen.h"
#include "model/state_codec.h"

namespace {

using grserve::CanonicalState;
using grserve::HiddenVector;
using grserve::TokenId;

std::filesystem::path ModelDirectory() { return GRSERVE_MODEL_DIR; }

grserve::ModelAssets LoadAssets() {
  auto assets = grserve::LoadModelAssets(
      ModelDirectory(),
      {.abi_major = GRSERVE_HAL_ABI_MAJOR,
       .abi_minor = GRSERVE_HAL_ABI_MINOR,
       .capabilities = GRSERVE_HAL_CAPABILITY_EXECUTABLES});
  EXPECT_TRUE(assets.ok()) << assets.error()->message();
  return std::move(assets.value());
}

google::protobuf::Struct LoadGolden() {
  std::ifstream input(ModelDirectory() / "golden.json");
  const std::string json((std::istreambuf_iterator<char>(input)), {});
  google::protobuf::Struct document;
  EXPECT_TRUE(google::protobuf::util::JsonStringToMessage(json, &document).ok());
  return document;
}

const google::protobuf::Value& Field(const google::protobuf::Struct& value,
                                     const std::string& name) {
  const auto found = value.fields().find(name);
  EXPECT_NE(found, value.fields().end());
  return found->second;
}

std::int64_t Integer(const google::protobuf::Value& value) {
  return static_cast<std::int64_t>(value.number_value());
}

HiddenVector Vector(const google::protobuf::Value& value) {
  HiddenVector result{};
  EXPECT_EQ(value.list_value().values_size(), 8);
  for (int index = 0; index < 8; ++index) {
    result[static_cast<std::size_t>(index)] =
        static_cast<std::int32_t>(Integer(value.list_value().values(index)));
  }
  return result;
}

std::vector<HiddenVector> Vectors(const google::protobuf::Value& value) {
  std::vector<HiddenVector> result;
  for (const auto& entry : value.list_value().values()) {
    result.push_back(Vector(entry));
  }
  return result;
}

TEST(StateCodec, EmptyEncodingIsExplicitLittleEndianAndRoundTrips) {
  const CanonicalState state = grserve::InitialCanonicalState();
  auto encoded = grserve::EncodeCanonicalState(1U, state);
  ASSERT_TRUE(encoded.ok()) << encoded.error()->message();
  ASSERT_EQ(encoded.value().size(), 132U);
  const std::vector<std::byte> prefix = {
      std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0},
      std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
      std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
      std::byte{3}, std::byte{0}, std::byte{0}, std::byte{0},
      std::byte{8}, std::byte{0}, std::byte{0}, std::byte{0}};
  ASSERT_GE(encoded.value().size(), prefix.size());
  EXPECT_TRUE(std::equal(prefix.begin(), prefix.end(), encoded.value().begin()));
  auto decoded = grserve::DecodeCanonicalState(encoded.value(), 1U, 64U);
  ASSERT_TRUE(decoded.ok()) << decoded.error()->message();
  EXPECT_EQ(decoded.value(), state);
}

TEST(StateCodec, RejectsMalformedLengthsTrailingDataAndWrongVersion) {
  auto encoded = grserve::EncodeCanonicalState(1U, grserve::InitialCanonicalState());
  ASSERT_TRUE(encoded.ok());
  auto truncated = encoded.value();
  truncated.pop_back();
  EXPECT_FALSE(grserve::DecodeCanonicalState(truncated, 1U, 64U).ok());
  auto trailing = encoded.value();
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(grserve::DecodeCanonicalState(trailing, 1U, 64U).ok());
  auto huge = encoded.value();
  huge[4] = huge[5] = huge[6] = huge[7] = std::byte{0xff};
  EXPECT_FALSE(grserve::DecodeCanonicalState(huge, 1U, 64U).ok());
  EXPECT_FALSE(grserve::DecodeCanonicalState(encoded.value(), 2U, 64U).ok());
}

TEST(MiniQwen, MatchesEveryIndependentGoldenTokenFullStateAndDigest) {
  auto model = grserve::MiniQwen::Load(LoadAssets());
  ASSERT_TRUE(model.ok()) << model.error()->message();
  CanonicalState canonical = grserve::InitialCanonicalState();
  std::vector<TokenId> context = {TokenId{1}, TokenId{4}, TokenId{7}};
  const auto golden = LoadGolden();
  const auto& steps = Field(golden, "steps").list_value();
  ASSERT_EQ(steps.values_size(), 8);
  for (int index = 0; index < steps.values_size(); ++index) {
    auto candidate = model.value().ExecuteNext(context, canonical);
    ASSERT_TRUE(candidate.ok()) << candidate.error()->message();
    const auto& expected = steps.values(index).struct_value();
    EXPECT_EQ(candidate.value().token.value(),
              static_cast<std::uint32_t>(Integer(Field(expected, "token_id"))));
    ASSERT_TRUE(candidate.value().next_state.has_value());
    EXPECT_EQ(candidate.value().next_state->full_keys,
              Vectors(Field(expected, "full_keys")));
    EXPECT_EQ(candidate.value().next_state->full_values,
              Vectors(Field(expected, "full_values")));
    const auto expected_gdn = Vectors(Field(expected, "gdn_states"));
    ASSERT_EQ(expected_gdn.size(), 3U);
    for (std::size_t layer = 0; layer < 3U; ++layer) {
      EXPECT_EQ(candidate.value().next_state->gdn_states[layer], expected_gdn[layer]);
    }
    EXPECT_EQ(candidate.value().next_state->rng_state,
              std::stoull(Field(expected, "rng_state").string_value()));
    EXPECT_EQ(candidate.value().canonical_digest,
              Field(expected, "canonical_digest").string_value());
    context.push_back(candidate.value().token);
    ASSERT_TRUE(grserve::CommitCandidate(canonical, candidate.value()).ok());
    EXPECT_TRUE(candidate.value().consumed());
  }
}

TEST(MiniQwen, AbortIsPureAndCommitAdvancesExactlyOnce) {
  auto model = grserve::MiniQwen::Load(LoadAssets());
  ASSERT_TRUE(model.ok());
  CanonicalState canonical = grserve::InitialCanonicalState();
  const std::vector<TokenId> prompt = {TokenId{1}, TokenId{4}, TokenId{7}};
  auto aborted = model.value().ExecuteNext(prompt, canonical);
  ASSERT_TRUE(aborted.ok());
  EXPECT_EQ(canonical, grserve::InitialCanonicalState());
  auto retried = model.value().ExecuteNext(prompt, canonical);
  ASSERT_TRUE(retried.ok());
  EXPECT_EQ(retried.value().token, aborted.value().token);
  EXPECT_EQ(retried.value().canonical_digest,
            aborted.value().canonical_digest);
  EXPECT_EQ(retried.value().next_state, aborted.value().next_state);
  ASSERT_TRUE(retried.value().next_state.has_value());
  const CanonicalState expected = *retried.value().next_state;
  ASSERT_TRUE(grserve::CommitCandidate(canonical, retried.value()).ok());
  EXPECT_EQ(canonical, expected);
  EXPECT_NE(canonical, grserve::InitialCanonicalState());
}

TEST(MiniQwen, CandidateCanOnlyBeCommittedOnce) {
  auto model = grserve::MiniQwen::Load(LoadAssets());
  ASSERT_TRUE(model.ok());
  CanonicalState canonical = grserve::InitialCanonicalState();
  const std::vector<TokenId> prompt = {TokenId{1}, TokenId{4}, TokenId{7}};
  auto candidate = model.value().ExecuteNext(prompt, canonical);
  ASSERT_TRUE(candidate.ok());
  ASSERT_TRUE(candidate.value().next_state.has_value());
  const auto token = candidate.value().token;
  const auto digest = candidate.value().canonical_digest;

  ASSERT_TRUE(grserve::CommitCandidate(canonical, candidate.value()).ok());
  const CanonicalState committed = canonical;
  EXPECT_TRUE(candidate.value().consumed());
  EXPECT_EQ(candidate.value().token, token);
  EXPECT_EQ(candidate.value().canonical_digest, digest);
  const auto repeated = grserve::CommitCandidate(canonical, candidate.value());
  EXPECT_FALSE(repeated.ok());
  EXPECT_EQ(repeated.error()->code(), grserve::ErrorCode::kInvalidArgument);
  EXPECT_EQ(canonical, committed);
}

TEST(MiniQwen, SelfCommitIsRejectedWithoutConsumingOrMutatingState) {
  auto model = grserve::MiniQwen::Load(LoadAssets());
  ASSERT_TRUE(model.ok());
  const std::vector<TokenId> prompt = {TokenId{1}, TokenId{4}, TokenId{7}};
  auto candidate = model.value().ExecuteNext(
      prompt, grserve::InitialCanonicalState());
  ASSERT_TRUE(candidate.ok());
  ASSERT_TRUE(candidate.value().next_state.has_value());
  CanonicalState& aliased = *candidate.value().next_state;
  const CanonicalState before = aliased;
  const auto status = grserve::CommitCandidate(aliased, candidate.value());
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error()->code(), grserve::ErrorCode::kInvalidArgument);
  EXPECT_FALSE(candidate.value().consumed());
  EXPECT_EQ(aliased, before);
}

TEST(MiniQwen, MovingCandidateTransfersItsSingleCommitRight) {
  auto model = grserve::MiniQwen::Load(LoadAssets());
  ASSERT_TRUE(model.ok());
  const std::vector<TokenId> prompt = {TokenId{1}, TokenId{4}, TokenId{7}};
  auto source = model.value().ExecuteNext(prompt, grserve::InitialCanonicalState());
  ASSERT_TRUE(source.ok());
  grserve::CandidateStep moved = std::move(source.value());
  EXPECT_TRUE(source.value().consumed());

  CanonicalState canonical = grserve::InitialCanonicalState();
  EXPECT_FALSE(grserve::CommitCandidate(canonical, source.value()).ok());
  EXPECT_EQ(canonical, grserve::InitialCanonicalState());
  ASSERT_TRUE(moved.next_state.has_value());
  const CanonicalState expected = *moved.next_state;
  ASSERT_TRUE(grserve::CommitCandidate(canonical, moved).ok());
  EXPECT_EQ(canonical, expected);
  EXPECT_TRUE(moved.consumed());
}

TEST(MiniQwen, RejectsContextTokenAndStateShapeErrorsWithoutRngMutation) {
  auto model = grserve::MiniQwen::Load(LoadAssets());
  ASSERT_TRUE(model.ok());
  CanonicalState canonical = grserve::InitialCanonicalState();
  const auto initial_rng = canonical.rng_state;
  EXPECT_FALSE(model.value().ExecuteNext({}, canonical).ok());
  const std::vector<TokenId> invalid = {TokenId{16}};
  EXPECT_FALSE(model.value().ExecuteNext(invalid, canonical).ok());
  std::vector<TokenId> too_long(65U, TokenId{0});
  EXPECT_FALSE(model.value().ExecuteNext(too_long, canonical).ok());
  canonical.full_keys.push_back(HiddenVector{});
  const std::vector<TokenId> single = {TokenId{0}};
  EXPECT_FALSE(model.value().ExecuteNext(single, canonical).ok());
  EXPECT_EQ(canonical.rng_state, initial_rng);
}

}  // namespace
