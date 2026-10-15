#include "common/sha256.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace grserve {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

std::uint32_t LoadBigEndian(const std::byte* bytes) noexcept {
  return (std::to_integer<std::uint32_t>(bytes[0]) << 24) |
         (std::to_integer<std::uint32_t>(bytes[1]) << 16) |
         (std::to_integer<std::uint32_t>(bytes[2]) << 8) |
         std::to_integer<std::uint32_t>(bytes[3]);
}

}  // namespace

Sha256::Sha256() noexcept
    : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
             0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

void Sha256::Transform(const std::byte* block) noexcept {
  std::array<std::uint32_t, 64> words{};
  for (std::size_t i = 0; i < 16; ++i) {
    words[i] = LoadBigEndian(block + i * 4);
  }
  for (std::size_t i = 16; i < words.size(); ++i) {
    const auto s0 = std::rotr(words[i - 15], 7) ^
                    std::rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
    const auto s1 = std::rotr(words[i - 2], 17) ^
                    std::rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }

  auto a = state_[0];
  auto b = state_[1];
  auto c = state_[2];
  auto d = state_[3];
  auto e = state_[4];
  auto f = state_[5];
  auto g = state_[6];
  auto h = state_[7];
  for (std::size_t i = 0; i < words.size(); ++i) {
    const auto upper_e = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
    const auto choose = (e & f) ^ (~e & g);
    const auto temp1 = h + upper_e + choose + kRoundConstants[i] + words[i];
    const auto upper_a = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
    const auto majority = (a & b) ^ (a & c) ^ (b & c);
    const auto temp2 = upper_a + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

Status Sha256::Fail(std::string message) {
  if (!failure_) {
    failure_ = Error::From(ErrorCode::kInvalidArgument, std::move(message));
  }
  return Status{*failure_};
}

Status Sha256::Update(std::span<const std::byte> bytes) {
  if (failure_) {
    return Status{*failure_};
  }
  if (finalized_) {
    return Fail("SHA-256 already finalized");
  }
  constexpr std::uint64_t kMaximumInputBytes =
      (std::uint64_t{1} << 61) - 1;
  if (bytes.size() > kMaximumInputBytes - total_bytes_) {
    return Fail("SHA-256 input too long");
  }
  if (bytes.empty()) {
    return Status::Ok();
  }
  total_bytes_ += bytes.size();
  while (!bytes.empty()) {
    const auto count = std::min(buffer_.size() - buffered_, bytes.size());
    std::memcpy(buffer_.data() + buffered_, bytes.data(), count);
    buffered_ += count;
    bytes = bytes.subspan(count);
    if (buffered_ == buffer_.size()) {
      Transform(buffer_.data());
      buffered_ = 0;
    }
  }
  return Status::Ok();
}

Status Sha256::Update(std::string_view bytes) {
  return Update(std::as_bytes(std::span(bytes.data(), bytes.size())));
}

StatusOr<std::array<std::byte, 32>> Sha256::Final() {
  if (failure_) {
    return StatusOr<std::array<std::byte, 32>>(*failure_);
  }
  if (finalized_) {
    return StatusOr<std::array<std::byte, 32>>(digest_);
  }
  const std::uint64_t bit_count = total_bytes_ * 8;
  buffer_[buffered_++] = std::byte{0x80};
  if (buffered_ > 56) {
    std::fill(buffer_.begin() + buffered_, buffer_.end(), std::byte{0});
    Transform(buffer_.data());
    buffered_ = 0;
  }
  std::fill(buffer_.begin() + buffered_, buffer_.begin() + 56, std::byte{0});
  for (unsigned i = 0; i < 8; ++i) {
    buffer_[63 - i] = std::byte((bit_count >> (i * 8)) & 0xffu);
  }
  Transform(buffer_.data());
  for (std::size_t i = 0; i < state_.size(); ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      digest_[i * 4 + j] = std::byte((state_[i] >> (24 - j * 8)) & 0xffu);
    }
  }
  finalized_ = true;
  return StatusOr<std::array<std::byte, 32>>(digest_);
}

StatusOr<std::string> Sha256::FinalHex() {
  constexpr char kHex[] = "0123456789abcdef";
  auto digest = Final();
  if (!digest.ok()) {
    return StatusOr<std::string>(*digest.error());
  }
  std::string result(64, '0');
  for (std::size_t i = 0; i < digest.value().size(); ++i) {
    const auto value = std::to_integer<unsigned>(digest.value()[i]);
    result[i * 2] = kHex[value >> 4];
    result[i * 2 + 1] = kHex[value & 0xf];
  }
  return StatusOr<std::string>(std::move(result));
}

StatusOr<std::string> Sha256Hex(std::string_view bytes) {
  Sha256 sha;
  const Status update = sha.Update(bytes);
  if (!update.ok()) {
    return StatusOr<std::string>(*update.error());
  }
  return sha.FinalHex();
}

}  // namespace grserve
