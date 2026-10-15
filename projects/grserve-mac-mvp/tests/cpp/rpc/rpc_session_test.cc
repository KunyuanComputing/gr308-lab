#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engine/controller.h"
#include "engine/rank_worker.h"
#include "rpc/engine_rpc_server.h"
#include "rpc/frame_codec.h"
#include "rpc/rpc_session.h"

namespace {
using namespace grserve;
using namespace grserve::rpc;
namespace v1 = grserve::rpc::v1;

void WriteClient(int fd, const v1::ClientEnvelope& envelope) {
  std::string bytes;
  ASSERT_TRUE(envelope.SerializeToString(&bytes));
  const std::uint32_t length = htonl(static_cast<std::uint32_t>(bytes.size()));
  ASSERT_EQ(send(fd, &length, sizeof(length), 0), sizeof(length));
  ASSERT_EQ(send(fd, bytes.data(), bytes.size(), 0),
            static_cast<ssize_t>(bytes.size()));
}

v1::ServerEnvelope ReadServer(int fd) {
  std::uint32_t network_length = 0;
  EXPECT_EQ(recv(fd, &network_length, sizeof(network_length), MSG_WAITALL),
            sizeof(network_length));
  const std::uint32_t length = ntohl(network_length);
  EXPECT_GT(length, 0U);
  EXPECT_LE(length, kMaxFrameBytes);
  std::string bytes(length, '\0');
  EXPECT_EQ(recv(fd, bytes.data(), bytes.size(), MSG_WAITALL),
            static_cast<ssize_t>(bytes.size()));
  v1::ServerEnvelope envelope;
  EXPECT_TRUE(envelope.ParseFromString(bytes));
  return envelope;
}

v1::ClientEnvelope Completion(std::string request_id) {
  v1::ClientEnvelope envelope;
  envelope.mutable_protocol_version()->set_major(1);
  envelope.mutable_protocol_version()->set_minor(0);
  auto* request = envelope.mutable_completion();
  request->set_request_id(std::move(request_id));
  request->set_model_id("mini-qwen38-hybrid");
  request->add_prompt_token_ids(1);
  request->set_max_tokens(2);
  request->set_timeout_ms(1000);
  request->set_decode_mode(v1::DECODE_MODE_GREEDY);
  return envelope;
}

struct Fixture {
  Fixture() {
    auto admission_result = AdmissionController::Create(
        {{RankId{0}, 4}, {RankId{1}, 4}});
    EXPECT_TRUE(admission_result.ok());
    admission = std::move(admission_result).value();
    auto created = Controller::Create(
        admission, gate, completions,
        {{RankId{0}, [this](RankCommand command) {
            rank0.push_back(std::move(command));
            return Status::Ok();
          }},
         {RankId{1}, [this](RankCommand command) {
            rank1.push_back(std::move(command));
            return Status::Ok();
          }}});
    EXPECT_TRUE(created.ok());
    controller = std::move(created).value();
  }
  void PollBridge() {
    while (mailbox->PollOne(*controller)) {}
  }
  AdmissionController admission =
      std::move(AdmissionController::Create({{RankId{0}, 1}})).value();
  std::shared_ptr<ReplicaGate> gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  std::vector<RankCommand> rank0;
  std::vector<RankCommand> rank1;
  std::unique_ptr<Controller> controller;
  std::shared_ptr<ControllerMailbox> mailbox =
      std::make_shared<ControllerMailbox>();
};

class BlockingRankExecute {
 public:
  explicit BlockingRankExecute(std::thread::id owner) : owner_(owner) {}
  StatusOr<TokenId> Run(const Execute& execute) {
    std::unique_lock lock(mutex_);
    ++entered_;
    saw_nonempty_context_ = saw_nonempty_context_ ||
                            !execute.context_token_ids.empty();
    saw_sequence_ = saw_sequence_ || execute.sequence_number == SequenceNumber{1};
    all_calls_off_owner_ = all_calls_off_owner_ &&
                           std::this_thread::get_id() != owner_;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
    return StatusOr<TokenId>(TokenId{9});
  }
  void WaitForBothRanks() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return entered_ == 2; });
  }
  void Release() {
    std::lock_guard lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }
  bool saw_values() const {
    std::lock_guard lock(mutex_);
    return saw_nonempty_context_ && saw_sequence_ && all_calls_off_owner_;
  }
 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  int entered_{0};
  bool released_{false};
  bool saw_nonempty_context_{false};
  bool saw_sequence_{false};
  bool all_calls_off_owner_{true};
  std::thread::id owner_;
};

class BoundSink final : public OwnerBoundRequestSink {
 public:
  bool BindInternal(RequestId request) noexcept override {
    request_ = request;
    return true;
  }
  std::optional<RequestId> internal_request() const noexcept override {
    return request_;
  }
  bool TryAccepted(RequestId) noexcept override { return true; }
  bool TryToken(const TokenEvent&) noexcept override { return true; }
  bool TryFinished(const FinishedEvent&) noexcept override { return true; }
  bool TryError(const Error&) noexcept override { return true; }
 private:
  std::optional<RequestId> request_;
};

TEST(ControllerMailbox, SubmitBacklogIsHardBoundedAndCloseDrainsIt) {
  ControllerMailbox mailbox;
  auto sink = std::make_shared<BoundSink>();
  for (std::size_t index = 0; index < 1024; ++index) {
    ASSERT_TRUE(mailbox.Post(ControllerMailbox::Submit{
        {RequestId{0}, "model", {TokenId{1}}, 1, 1000, 1}, sink}));
    ASSERT_LE(mailbox.size_for_testing(),
              ControllerMailbox::capacity_for_testing());
  }
  EXPECT_FALSE(mailbox.Post(ControllerMailbox::Submit{
      {RequestId{0}, "model", {TokenId{1}}, 1, 1000, 1}, sink}));
  mailbox.Close();
  EXPECT_EQ(mailbox.size_for_testing(), 0U);
}

TEST(RpcSession, RankOwnedExecuteCannotBlockOwnerOrNonReadingClient) {
  CompletionQueue<RankCompletion> completions;
  auto execute =
      std::make_shared<BlockingRankExecute>(std::this_thread::get_id());
  RankWorker rank0(RankId{0}, completions, StateDigest{0}, {}, {},
                   [execute](const Execute& command) {
                     return execute->Run(command);
                   });
  RankWorker rank1(RankId{1}, completions, StateDigest{0}, {}, {},
                   [execute](const Execute& command) {
                     return execute->Run(command);
                   });
  auto admission = AdmissionController::Create({{RankId{0}, 1}, {RankId{1}, 1}});
  ASSERT_TRUE(admission.ok());
  auto gate = std::make_shared<ReplicaGate>();
  auto controller = Controller::Create(
      std::move(admission).value(), gate, completions,
      {{RankId{0}, [&rank0](RankCommand command) {
          return rank0.Enqueue(std::move(command));
        }},
       {RankId{1}, [&rank1](RankCommand command) {
          return rank1.Enqueue(std::move(command));
        }}});
  ASSERT_TRUE(controller.ok());
  auto mailbox = std::make_shared<ControllerMailbox>();
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  RpcSessionTestBarrier writer_barrier;
  auto session = RpcSession::StartForTesting(
      UniqueFd{sockets[0]}, mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 1,
       }, &writer_barrier);
  ASSERT_TRUE(session.ok());
  WriteClient(sockets[1], Completion("51"));
  mailbox->WaitForEventForTesting();
  ASSERT_TRUE(mailbox->PollOne(*controller.value()));
  writer_barrier.WaitUntilWriterBlocked();

  completions.WaitForValue();
  controller.value()->PollOnce();
  completions.WaitForValue();
  controller.value()->PollOnce();
  execute->WaitForBothRanks();
  controller.value()->PollOnce();
  EXPECT_TRUE(execute->saw_values());

  execute->Release();
  completions.WaitForValue();
  controller.value()->PollOnce();
  completions.WaitForValue();
  controller.value()->PollOnce();
  completions.WaitForValue();
  controller.value()->PollOnce();
  completions.WaitForValue();
  controller.value()->PollOnce();
  EXPECT_EQ(controller.value()->logical_publish_count(), 1U);

  writer_barrier.ReleaseWriter();
  close(sockets[1]);
  session.value()->Shutdown();
  static_cast<void>(controller.value()->Shutdown());
  rank0.Shutdown();
  rank1.Shutdown();
  completions.Close();
}

TEST(RpcSession, RejectsVersionAndFieldsBeforeControllerOrRanks) {
  Fixture fixture;
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  auto session = RpcSession::Start(
      UniqueFd{sockets[0]}, fixture.mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  ASSERT_TRUE(session.ok());
  auto bad_version = Completion("1");
  bad_version.mutable_protocol_version()->set_major(2);
  WriteClient(sockets[1], bad_version);
  auto response = ReadServer(sockets[1]);
  EXPECT_EQ(response.error().code(), v1::ERROR_CODE_INCOMPATIBLE_VERSION);
  EXPECT_TRUE(fixture.rank0.empty());

  WriteClient(sockets[1], Completion(std::string(129, 'x')));
  response = ReadServer(sockets[1]);
  EXPECT_EQ(response.error().code(), v1::ERROR_CODE_INVALID_ARGUMENT);
  EXPECT_TRUE(fixture.rank0.empty());
  close(sockets[1]);
  session.value()->Shutdown();
}

TEST(RpcSession, OpaqueWireIdsAreSessionLocalAndEchoedVerbatim) {
  Fixture fixture;
  int first[2];
  int second[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, first), 0);
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, second), 0);
  auto first_session = RpcSession::Start(
      UniqueFd{first[0]}, fixture.mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  auto second_session = RpcSession::Start(
      UniqueFd{second[0]}, fixture.mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  ASSERT_TRUE(first_session.ok());
  ASSERT_TRUE(second_session.ok());
  WriteClient(first[1], Completion("req-1"));
  WriteClient(second[1], Completion("req-1"));
  fixture.mailbox->WaitForEventCountForTesting(2);
  ASSERT_TRUE(fixture.mailbox->PollOne(*fixture.controller));
  ASSERT_TRUE(fixture.mailbox->PollOne(*fixture.controller));
  const auto first_reply = ReadServer(first[1]);
  const auto second_reply = ReadServer(second[1]);
  ASSERT_TRUE(first_reply.has_accepted());
  ASSERT_TRUE(second_reply.has_accepted());
  EXPECT_EQ(first_reply.accepted().request_id(), "req-1");
  EXPECT_EQ(second_reply.accepted().request_id(), "req-1");
  close(first[1]);
  close(second[1]);
  first_session.value()->Shutdown();
  second_session.value()->Shutdown();
}

TEST(RpcSession, AcceptedIsFirstAndTokenSequenceIsStrict) {
  Fixture fixture;
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  auto session = RpcSession::Start(
      UniqueFd{sockets[0]}, fixture.mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  ASSERT_TRUE(session.ok());
  WriteClient(sockets[1], Completion("7"));
  fixture.mailbox->WaitForEventForTesting();
  fixture.PollBridge();
  auto accepted = ReadServer(sockets[1]);
  ASSERT_TRUE(accepted.has_accepted());
  EXPECT_EQ(accepted.accepted().request_id(), "7");

  auto sink = session.value()->SinkForTesting(RequestId{7});
  ASSERT_TRUE(sink->TryToken({RequestId{7}, TransactionId{4},
                              SequenceNumber{1}, TokenId{9}, StateDigest{0}}));
  auto token = ReadServer(sockets[1]);
  EXPECT_EQ(token.token().sequence_number(), 1U);
  EXPECT_FALSE(sink->TryToken({RequestId{7}, TransactionId{5},
                               SequenceNumber{3}, TokenId{10}, StateDigest{0}}));
  fixture.mailbox->WaitForEventForTesting();
  fixture.PollBridge();
  pollfd peer{sockets[1], POLLIN | POLLHUP, 0};
  ASSERT_EQ(poll(&peer, 1, 0), 1);
  char byte = 0;
  EXPECT_EQ(recv(sockets[1], &byte, 1, 0), 0);
  close(sockets[1]);
  session.value()->Shutdown();
}

TEST(RpcSession, DuplicateAndWrongCancelFailStably) {
  Fixture fixture;
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  auto session = RpcSession::Start(
      UniqueFd{sockets[0]}, fixture.mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  ASSERT_TRUE(session.ok());
  v1::ClientEnvelope cancel;
  cancel.mutable_protocol_version()->set_major(1);
  cancel.mutable_cancel()->set_request_id("99");
  cancel.mutable_cancel()->set_reason(v1::CANCEL_REASON_CLIENT_DISCONNECTED);
  WriteClient(sockets[1], cancel);
  auto first = ReadServer(sockets[1]);
  EXPECT_EQ(first.error().code(), v1::ERROR_CODE_INVALID_ARGUMENT);
  WriteClient(sockets[1], cancel);
  auto second = ReadServer(sockets[1]);
  EXPECT_EQ(second.error().code(), v1::ERROR_CODE_INVALID_ARGUMENT);
  EXPECT_TRUE(fixture.rank0.empty());
  close(sockets[1]);
  session.value()->Shutdown();
}

TEST(RpcSession, FullQueueNeverBlocksControllerAndDetachesOnce) {
  Fixture fixture;
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  RpcSessionTestBarrier barrier;
  auto session = RpcSession::StartForTesting(
      UniqueFd{sockets[0]}, fixture.mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 1,
       }, &barrier);
  ASSERT_TRUE(session.ok());
  auto sink = session.value()->SinkForTesting(RequestId{11});
  ASSERT_TRUE(fixture.controller->Submit(
      {RequestId{11}, "mini-qwen38-hybrid", {TokenId{1}}, 2, 1000, 1}, sink).ok());
  barrier.WaitUntilWriterBlocked();
  ASSERT_TRUE(sink->TryToken({RequestId{11}, TransactionId{1}, SequenceNumber{1},
                              TokenId{1}, StateDigest{0}}));
  EXPECT_FALSE(sink->TryToken({RequestId{11}, TransactionId{2}, SequenceNumber{2},
                               TokenId{2}, StateDigest{0}}));
  fixture.mailbox->WaitForEventForTesting();
  fixture.PollBridge();
  EXPECT_EQ(fixture.controller->control_replay_size_for_testing(), 1U);
  fixture.controller->PollOnce();
  barrier.ReleaseWriter();
  close(sockets[1]);
  session.value()->Shutdown();
}

TEST(RpcSession, AcceptedQueueExhaustionStillReportsResourceExhausted) {
  Fixture fixture;
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  RpcSessionTestBarrier barrier;
  auto session = RpcSession::StartForTesting(
      UniqueFd{sockets[0]}, fixture.mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 1,
       }, &barrier);
  ASSERT_TRUE(session.ok());
  auto first = session.value()->SinkForTesting(RequestId{41});
  ASSERT_TRUE(first->TryAccepted(RequestId{41}));
  barrier.WaitUntilWriterBlocked();
  ASSERT_TRUE(first->TryToken({RequestId{41}, TransactionId{1}, SequenceNumber{1},
                               TokenId{1}, StateDigest{0}}));
  auto second = session.value()->SinkForTesting(RequestId{42});
  EXPECT_FALSE(second->TryAccepted(RequestId{42}));
  EXPECT_TRUE(second->TryError(Error::From(
      ErrorCode::kResourceExhausted, "accepted event queue is full", RequestId{42})));
  barrier.ReleaseWriter();
  auto accepted = ReadServer(sockets[1]);
  auto token = ReadServer(sockets[1]);
  auto error = ReadServer(sockets[1]);
  EXPECT_TRUE(accepted.has_accepted());
  EXPECT_TRUE(token.has_token());
  ASSERT_TRUE(error.has_error());
  EXPECT_EQ(error.error().code(), v1::ERROR_CODE_RESOURCE_EXHAUSTED);
  close(sockets[1]);
  session.value()->Shutdown();
}

TEST(RpcSession, HealthReflectsAdmissionAndReplicaGate) {
  Fixture fixture;
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  auto session = RpcSession::Start(
      UniqueFd{sockets[0]}, fixture.mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8,
       .replica_gate = fixture.gate});
  ASSERT_TRUE(session.ok());
  v1::ClientEnvelope health;
  health.mutable_protocol_version()->set_major(1);
  health.mutable_health();
  WriteClient(sockets[1], health);
  auto response = ReadServer(sockets[1]);
  EXPECT_TRUE(response.health().accepting_requests());
  fixture.gate->MarkFatal(Error::From(ErrorCode::kReplicaFatal, "fatal",
                                      RequestId{88}));
  WriteClient(sockets[1], health);
  response = ReadServer(sockets[1]);
  EXPECT_FALSE(response.health().accepting_requests());
  EXPECT_EQ(response.health().replica_state(), v1::REPLICA_STATE_FATAL);
  EXPECT_EQ(response.health().fatal_error().code(),
            v1::ERROR_CODE_REPLICA_FATAL);
  EXPECT_EQ(response.health().fatal_error().message(), "fatal");
  EXPECT_TRUE(response.health().fatal_error().request_id().empty());
  close(sockets[1]);
  session.value()->Shutdown();
}
}  // namespace
