#pragma once

#include <vector>

#include "grserve/types.h"

namespace grserve {

// Engine-owned digest.  It deliberately does not share a type with any HAL.
using StateDigest = StrongId<struct StateDigestTag>;

// Commands are pure values emitted to one configured rank.
struct Prepare {
  TransactionId transaction_id;
  RankId rank_id;
  Epoch epoch;
};

struct Execute {
  TransactionId transaction_id;
  RankId rank_id;
  Epoch epoch;
  TokenId candidate_token;
  // Immutable inference inputs. They cross into the rank worker by value so
  // the Controller owner never waits on backend execution.
  std::vector<TokenId> context_token_ids{};
  SequenceNumber sequence_number{0};
};

struct Commit {
  TransactionId transaction_id;
  RankId rank_id;
  Epoch epoch;
};

struct Abort {
  TransactionId transaction_id;
  RankId rank_id;
  Epoch epoch;
};

// Acks are routed by their transaction context, so rank and epoch are the
// required wire values.  PreparedAck's two-field aggregate form is stable.
struct PreparedAck {
  RankId rank_id;
  Epoch epoch;
};

struct ExecutedAck {
  RankId rank_id;
  Epoch epoch;
  TokenId candidate_token;
  StateDigest state_digest;
};

struct CommittedAck {
  RankId rank_id;
  Epoch epoch;
};

struct AbortedAck {
  RankId rank_id;
  Epoch epoch;
};

}  // namespace grserve
