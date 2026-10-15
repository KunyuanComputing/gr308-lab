#pragma once

#include <compare>
#include <cstdint>

namespace grserve {

template <class Tag, class Rep = std::uint64_t>
class StrongId {
 public:
  explicit constexpr StrongId(Rep value) : value_(value) {}

  [[nodiscard]] constexpr Rep value() const noexcept { return value_; }
  auto operator<=>(const StrongId&) const = default;

 private:
  Rep value_;
};

using RequestId = StrongId<struct RequestIdTag>;
using ReplicaId = StrongId<struct ReplicaIdTag>;
using RankId = StrongId<struct RankIdTag, std::uint32_t>;
using TransactionId = StrongId<struct TransactionIdTag>;
using Epoch = StrongId<struct EpochTag>;
using SequenceNumber = StrongId<struct SequenceNumberTag>;
using TokenId = StrongId<struct TokenIdTag, std::uint32_t>;

}  // namespace grserve
