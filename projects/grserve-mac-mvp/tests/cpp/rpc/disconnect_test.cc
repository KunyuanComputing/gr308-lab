#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engine/controller.h"
#include "rpc/rpc_session.h"
#include "rpc/frame_codec.h"

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
  std::uint32_t size = 0;
  EXPECT_EQ(recv(fd, &size, sizeof(size), MSG_WAITALL), sizeof(size));
  std::string bytes(ntohl(size), '\0');
  EXPECT_LE(bytes.size(), kMaxFrameBytes);
  EXPECT_EQ(recv(fd, bytes.data(), bytes.size(), MSG_WAITALL),
            static_cast<ssize_t>(bytes.size()));
  v1::ServerEnvelope envelope;
  EXPECT_TRUE(envelope.ParseFromString(bytes));
  return envelope;
}

v1::ClientEnvelope Completion(std::uint64_t id) {
  v1::ClientEnvelope envelope;
  envelope.mutable_protocol_version()->set_major(1);
  auto* request = envelope.mutable_completion();
  request->set_request_id(std::to_string(id));
  request->set_model_id("mini-qwen38-hybrid");
  request->add_prompt_token_ids(1);
  request->set_max_tokens(2);
  request->set_timeout_ms(1000);
  request->set_decode_mode(v1::DECODE_MODE_GREEDY);
  return envelope;
}

class Sink final : public RequestEventSink {
 public:
  bool TryAccepted(RequestId) noexcept override { return true; }
  bool TryToken(const TokenEvent&) noexcept override { ++tokens; return true; }
  bool TryFinished(const FinishedEvent&) noexcept override { return true; }
  bool TryError(const Error&) noexcept override { return true; }
  int tokens = 0;
};

struct Fixture {
  explicit Fixture(bool submit_initial = true) {
    auto admission = AdmissionController::Create({{RankId{0}, 2}, {RankId{1}, 2}});
    EXPECT_TRUE(admission.ok());
    capacity = admission.value();
    auto made = Controller::Create(
        admission.value(), gate, completions,
        {{RankId{0}, [this](RankCommand c) { rank0.push_back(c); return Status::Ok(); }},
         {RankId{1}, [this](RankCommand c) { rank1.push_back(c); return Status::Ok(); }}});
    EXPECT_TRUE(made.ok());
    controller = std::move(made).value();
    if (submit_initial) {
      EXPECT_TRUE(controller->Submit(
          {RequestId{21}, "mini-qwen38-hybrid", {TokenId{1}}, 2, 1000, 1}, sink).ok());
    }
  }
  void Push(RankCompletion c) { ASSERT_TRUE(completions.Push(std::move(c))); controller->PollOnce(); }
  Prepare prepare() const { return std::get<Prepare>(rank0.front()); }
  void ReachDecision() {
    const auto p = prepare();
    Push(PreparedAck{RankId{0}, p.epoch}); Push(PreparedAck{RankId{1}, p.epoch});
    Push(ExecutedAck{RankId{0}, p.epoch, TokenId{2}, StateDigest{3}});
    Push(ExecutedAck{RankId{1}, p.epoch, TokenId{2}, StateDigest{4}});
  }
  AdmissionController capacity = std::move(AdmissionController::Create({{RankId{0},1}})).value();
  std::shared_ptr<ReplicaGate> gate = std::make_shared<ReplicaGate>();
  CompletionQueue<RankCompletion> completions;
  std::vector<RankCommand> rank0, rank1;
  std::shared_ptr<Sink> sink = std::make_shared<Sink>();
  std::unique_ptr<Controller> controller;
};

TEST(RpcDisconnect, DetachBeforeDecisionAbortsWithoutPublish) {
  Fixture f;
  ASSERT_TRUE(f.controller->DetachTransport(RequestId{21}).ok());
  ASSERT_TRUE(f.controller->DetachTransport(RequestId{21}).ok());
  EXPECT_EQ(f.controller->last_terminal_state_for_testing(), TransactionState::kAborted);
  EXPECT_EQ(f.controller->logical_publish_count(), 0U);
  EXPECT_EQ(f.sink->tokens, 0);
  EXPECT_EQ(f.controller->control_replay_size_for_testing(), 1U);
}

TEST(RpcDisconnect, FinalExecuteBeforeDetachCommitsAndDropsExactlyOnce) {
  Fixture f;
  f.ReachDecision();
  const auto p = f.prepare();
  ASSERT_TRUE(f.controller->DetachTransport(RequestId{21}).ok());
  ASSERT_TRUE(f.controller->DetachTransport(RequestId{21}).ok());
  f.Push(CommittedAck{RankId{0}, p.epoch});
  f.Push(CommittedAck{RankId{1}, p.epoch});
  EXPECT_EQ(f.controller->last_terminal_state_for_testing(), TransactionState::kPublished);
  EXPECT_EQ(f.controller->logical_publish_count(), 1U);
  EXPECT_EQ(f.controller->network_delivery_count(), 0U);
  EXPECT_EQ(f.controller->last_delivery_outcome(), DeliveryOutcome::kDroppedClientDisconnect);
  EXPECT_EQ(f.sink->tokens, 0);
  EXPECT_EQ(f.controller->control_replay_size_for_testing(), 1U);
}

TEST(RpcDisconnect, SocketEofPostsOneOwnerThreadDetach) {
  Fixture f(false);
  auto mailbox = std::make_shared<ControllerMailbox>();
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  auto session = RpcSession::Start(
      UniqueFd{sockets[0]}, mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  ASSERT_TRUE(session.ok());
  WriteClient(sockets[1], Completion(31));
  mailbox->WaitForEventForTesting();
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  EXPECT_TRUE(ReadServer(sockets[1]).has_accepted());
  close(sockets[1]);
  mailbox->WaitForEventForTesting();
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  EXPECT_FALSE(mailbox->PollOne(*f.controller));
  EXPECT_EQ(f.controller->control_replay_size_for_testing(), 1U);
  session.value()->Shutdown();
}

TEST(RpcDisconnect, WriterEpipePostsOneOwnerThreadDetach) {
  Fixture f;
  auto mailbox = std::make_shared<ControllerMailbox>();
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  RpcSessionTestBarrier barrier;
  auto session = RpcSession::StartForTesting(
      UniqueFd{sockets[0]}, mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8,
       }, &barrier);
  ASSERT_TRUE(session.ok());
  auto sink = session.value()->SinkForTesting(RequestId{32});
  ASSERT_TRUE(f.controller->Submit(
      {RequestId{32}, "mini-qwen38-hybrid", {TokenId{1}}, 2, 1000, 1}, sink).ok());
  barrier.WaitUntilWriterBlocked();
  close(sockets[1]);
  barrier.ReleaseWriter();
  mailbox->WaitForEventForTesting();
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  EXPECT_FALSE(mailbox->PollOne(*f.controller));
  EXPECT_EQ(f.controller->control_replay_size_for_testing(), 1U);
  session.value()->Shutdown();
}

TEST(RpcDisconnect, ExplicitCancelIsNormalizedAndDuplicateFails) {
  Fixture f(false);
  auto mailbox = std::make_shared<ControllerMailbox>();
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  auto session = RpcSession::Start(
      UniqueFd{sockets[0]}, mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  ASSERT_TRUE(session.ok());
  WriteClient(sockets[1], Completion(33));
  mailbox->WaitForEventForTesting();
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  EXPECT_TRUE(ReadServer(sockets[1]).has_accepted());
  v1::ClientEnvelope cancel;
  cancel.mutable_protocol_version()->set_major(1);
  cancel.mutable_cancel()->set_request_id("33");
  cancel.mutable_cancel()->set_reason(v1::CANCEL_REASON_CLIENT_DISCONNECTED);
  WriteClient(sockets[1], cancel);
  mailbox->WaitForEventForTesting();
  pollfd before_owner{sockets[1], POLLIN, 0};
  EXPECT_EQ(poll(&before_owner, 1, 0), 0);
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  const auto ack = ReadServer(sockets[1]);
  ASSERT_TRUE(ack.has_cancel_ack());
  EXPECT_EQ(ack.cancel_ack().disposition(),
            v1::CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION);
  WriteClient(sockets[1], cancel);
  EXPECT_EQ(ReadServer(sockets[1]).error().code(),
            v1::ERROR_CODE_INVALID_ARGUMENT);
  close(sockets[1]);
  session.value()->Shutdown();
}

TEST(RpcDisconnect, CancelAfterDecisionWaitsForOwnerAndReportsCommitting) {
  Fixture f(false);
  auto mailbox = std::make_shared<ControllerMailbox>();
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  auto session = RpcSession::Start(
      UniqueFd{sockets[0]}, mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  ASSERT_TRUE(session.ok());
  WriteClient(sockets[1], Completion(61));
  mailbox->WaitForEventForTesting();
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  EXPECT_TRUE(ReadServer(sockets[1]).has_accepted());
  const auto prepare = std::get<Prepare>(f.rank0.front());
  f.Push(PreparedAck{RankId{0}, prepare.epoch});
  f.Push(PreparedAck{RankId{1}, prepare.epoch});
  f.Push(ExecutedAck{RankId{0}, prepare.epoch, TokenId{2}, StateDigest{3}});
  f.Push(ExecutedAck{RankId{1}, prepare.epoch, TokenId{2}, StateDigest{4}});
  ASSERT_TRUE(std::holds_alternative<Commit>(f.rank0.back()));

  v1::ClientEnvelope cancel;
  cancel.mutable_protocol_version()->set_major(1);
  cancel.mutable_cancel()->set_request_id("61");
  cancel.mutable_cancel()->set_reason(v1::CANCEL_REASON_SERVER_SHUTDOWN);
  WriteClient(sockets[1], cancel);
  mailbox->WaitForEventForTesting();
  pollfd before_owner{sockets[1], POLLIN, 0};
  EXPECT_EQ(poll(&before_owner, 1, 0), 0);
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  const auto ack = ReadServer(sockets[1]);
  ASSERT_TRUE(ack.has_cancel_ack());
  EXPECT_EQ(ack.cancel_ack().disposition(),
            v1::CANCEL_DISPOSITION_COMMITTING_DETACHED);

  f.Push(CommittedAck{RankId{0}, prepare.epoch});
  f.Push(CommittedAck{RankId{1}, prepare.epoch});
  EXPECT_EQ(f.controller->logical_publish_count(), 1U);
  EXPECT_EQ(f.controller->network_delivery_count(), 0U);
  close(sockets[1]);
  session.value()->Shutdown();
}

TEST(RpcDisconnect, DeadlineCancelAfterDecisionPreservesDeadlineSemantics) {
  Fixture f(false);
  auto mailbox = std::make_shared<ControllerMailbox>();
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  auto session = RpcSession::Start(
      UniqueFd{sockets[0]}, mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  ASSERT_TRUE(session.ok());
  WriteClient(sockets[1], Completion(62));
  mailbox->WaitForEventForTesting();
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  EXPECT_TRUE(ReadServer(sockets[1]).has_accepted());
  const auto prepare = std::get<Prepare>(f.rank0.front());
  f.Push(PreparedAck{RankId{0}, prepare.epoch});
  f.Push(PreparedAck{RankId{1}, prepare.epoch});
  f.Push(ExecutedAck{RankId{0}, prepare.epoch, TokenId{2}, StateDigest{3}});
  f.Push(ExecutedAck{RankId{1}, prepare.epoch, TokenId{2}, StateDigest{4}});

  v1::ClientEnvelope cancel;
  cancel.mutable_protocol_version()->set_major(1);
  cancel.mutable_cancel()->set_request_id("62");
  cancel.mutable_cancel()->set_reason(v1::CANCEL_REASON_DEADLINE);
  WriteClient(sockets[1], cancel);
  mailbox->WaitForEventForTesting();
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  const auto ack = ReadServer(sockets[1]);
  ASSERT_TRUE(ack.has_cancel_ack());
  EXPECT_EQ(ack.cancel_ack().disposition(),
            v1::CANCEL_DISPOSITION_COMMITTING_DETACHED);
  EXPECT_EQ(f.gate->health(), ReplicaHealth::kFatal);
  EXPECT_EQ(f.controller->logical_publish_count(), 0U);
  close(sockets[1]);
  session.value()->Shutdown();
}

TEST(RpcDisconnect, CompletionThenCancelPreservesReaderEventOrder) {
  Fixture f(false);
  auto mailbox = std::make_shared<ControllerMailbox>();
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  auto session = RpcSession::Start(
      UniqueFd{sockets[0]}, mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  ASSERT_TRUE(session.ok());
  WriteClient(sockets[1], Completion(70));
  v1::ClientEnvelope cancel;
  cancel.mutable_protocol_version()->set_major(1);
  cancel.mutable_cancel()->set_request_id("70");
  cancel.mutable_cancel()->set_reason(v1::CANCEL_REASON_CLIENT_DISCONNECTED);
  WriteClient(sockets[1], cancel);
  mailbox->WaitForEventCountForTesting(2);
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  EXPECT_TRUE(ReadServer(sockets[1]).has_accepted());
  const auto ack = ReadServer(sockets[1]);
  ASSERT_TRUE(ack.has_cancel_ack());
  EXPECT_EQ(ack.cancel_ack().disposition(),
            v1::CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION);
  close(sockets[1]);
  session.value()->Shutdown();
}

TEST(RpcDisconnect, PipelinedDuplicateCancelWaitsBehindOwnerDecision) {
  Fixture f(false);
  auto mailbox = std::make_shared<ControllerMailbox>();
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  auto session = RpcSession::Start(
      UniqueFd{sockets[0]}, mailbox,
      {.model_id = "mini-qwen38-hybrid", .outbound_capacity = 8});
  ASSERT_TRUE(session.ok());
  WriteClient(sockets[1], Completion(71));
  v1::ClientEnvelope cancel;
  cancel.mutable_protocol_version()->set_major(1);
  cancel.mutable_cancel()->set_request_id("71");
  cancel.mutable_cancel()->set_reason(v1::CANCEL_REASON_CLIENT_DISCONNECTED);
  WriteClient(sockets[1], cancel);
  WriteClient(sockets[1], cancel);
  mailbox->WaitForEventCountForTesting(2);
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  ASSERT_TRUE(mailbox->PollOne(*f.controller));
  EXPECT_TRUE(ReadServer(sockets[1]).has_accepted());
  EXPECT_TRUE(ReadServer(sockets[1]).has_cancel_ack());
  const auto duplicate = ReadServer(sockets[1]);
  ASSERT_TRUE(duplicate.has_error());
  EXPECT_EQ(duplicate.error().code(), v1::ERROR_CODE_INVALID_ARGUMENT);
  close(sockets[1]);
  session.value()->Shutdown();
}
}  // namespace
