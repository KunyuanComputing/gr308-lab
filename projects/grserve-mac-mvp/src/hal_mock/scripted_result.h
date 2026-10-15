#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "common/clock.h"
#include "grserve/error.h"
#include "grserve/types.h"

namespace grserve::hal_mock {

using CanonicalStateDigest = StrongId<struct CanonicalStateDigestTag, std::uint64_t>;

class ScriptedResult {
 public:
  ScriptedResult(TokenId candidate_token, CanonicalStateDigest state_digest,
                 Clock::duration virtual_duration) noexcept
      : candidate_token_(candidate_token),
        state_digest_(state_digest),
        virtual_duration_(virtual_duration) {}
  ScriptedResult(TokenId candidate_token, std::string digest_hex,
                 std::vector<std::byte> encoded_state,
                 std::vector<std::uint32_t> expected_context,
                 std::vector<std::byte> expected_predecessor) noexcept
      : candidate_token_(candidate_token),
        state_digest_(CanonicalStateDigest{0U}),
        digest_hex_(std::move(digest_hex)),
        encoded_state_(std::move(encoded_state)),
        expected_context_(std::move(expected_context)),
        expected_predecessor_(std::move(expected_predecessor)),
        virtual_duration_(Clock::duration::zero()) {}

  [[nodiscard]] TokenId candidate_token() const noexcept { return candidate_token_; }
  [[nodiscard]] CanonicalStateDigest state_digest() const noexcept {
    return state_digest_;
  }
  [[nodiscard]] Clock::duration virtual_duration() const noexcept {
    return virtual_duration_;
  }
  [[nodiscard]] const std::string& digest_hex() const noexcept {
    return digest_hex_;
  }
  [[nodiscard]] const std::vector<std::byte>& encoded_state() const noexcept {
    return encoded_state_;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& expected_context() const noexcept {
    return expected_context_;
  }
  [[nodiscard]] const std::vector<std::byte>& expected_predecessor() const noexcept {
    return expected_predecessor_;
  }

 private:
  TokenId candidate_token_;
  CanonicalStateDigest state_digest_;
  std::string digest_hex_;
  std::vector<std::byte> encoded_state_;
  std::vector<std::uint32_t> expected_context_;
  std::vector<std::byte> expected_predecessor_;
  Clock::duration virtual_duration_;
};

[[nodiscard]] StatusOr<std::vector<ScriptedResult>> LoadFrozenGolden(
    const std::filesystem::path& golden_path);

}  // namespace grserve::hal_mock
