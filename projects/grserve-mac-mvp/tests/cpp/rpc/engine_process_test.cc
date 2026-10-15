#include <arpa/inet.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "engine.pb.h"

namespace {
namespace v1 = grserve::rpc::v1;

class EngineProcess {
 public:
  explicit EngineProcess(const char* plugin = GRSERVE_MOCK_HAL_PATH,
                         int entered_fd = -1, int release_fd = -1) {
    std::array<char, 64> directory_template{};
    std::strcpy(directory_template.data(), "/tmp/grserve-engine-test.XXXXXX");
    const char* made = mkdtemp(directory_template.data());
    if (made == nullptr) return;
    directory_ = made;
    socket_ = directory_ + "/engine.sock";
    child_ = fork();
    if (child_ == 0) {
      if (entered_fd >= 0 && release_fd >= 0) {
        const std::string entered = std::to_string(entered_fd);
        const std::string release = std::to_string(release_fd);
        setenv("GRSERVE_BLOCKING_HAL_ENTERED_FD", entered.c_str(), 1);
        setenv("GRSERVE_BLOCKING_HAL_RELEASE_FD", release.c_str(), 1);
      }
      execl(GRSERVE_ENGINE_PATH, GRSERVE_ENGINE_PATH, "--socket",
            socket_.c_str(), "--model-dir", GRSERVE_MODEL_DIR,
            "--hal-plugin", plugin, "--ranks=2",
            "--max-concurrent-requests=2", "--max-state-slots-per-rank=2",
            static_cast<char*>(nullptr));
      _exit(127);
    }
  }
  bool TerminateWithin(std::chrono::milliseconds bound, int* status) {
    if (child_ <= 0) return false;
    if (kill(child_, SIGTERM) != 0) return false;
    const auto deadline = std::chrono::steady_clock::now() + bound;
    while (std::chrono::steady_clock::now() < deadline) {
      const pid_t result = waitpid(child_, status, WNOHANG);
      if (result == child_) {
        child_ = -1;
        return true;
      }
      if (result < 0 && errno != EINTR) return false;
      std::this_thread::yield();
    }
    kill(child_, SIGKILL);
    while (waitpid(child_, status, 0) < 0 && errno == EINTR) {}
    child_ = -1;
    return false;
  }
  const std::string& socket_path() const noexcept { return socket_; }
  ~EngineProcess() {
    if (child_ > 0) {
      kill(child_, SIGTERM);
      int status = 0;
      while (waitpid(child_, &status, 0) < 0 && errno == EINTR) {}
    }
    if (!socket_.empty()) unlink(socket_.c_str());
    if (!directory_.empty()) rmdir(directory_.c_str());
  }
  int Connect() {
    for (std::size_t attempt = 0; attempt < 200000; ++attempt) {
      const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
      if (fd < 0) return -1;
      sockaddr_un address{};
      address.sun_family = AF_UNIX;
      std::strncpy(address.sun_path, socket_.c_str(),
                   sizeof(address.sun_path) - 1);
      if (connect(fd, reinterpret_cast<const sockaddr*>(&address),
                  sizeof(address)) == 0) {
        return fd;
      }
      close(fd);
      int status = 0;
      if (child_ <= 0) return -1;
      if (waitpid(child_, &status, WNOHANG) == child_) {
        child_ = -1;
        return -1;
      }
      std::this_thread::yield();
    }
    return -1;
  }

 private:
  pid_t child_{-1};
  std::string directory_;
  std::string socket_;
};

bool WriteClient(int fd, const v1::ClientEnvelope& envelope) {
  std::string body;
  if (!envelope.SerializeToString(&body)) return false;
  const std::uint32_t size = htonl(static_cast<std::uint32_t>(body.size()));
  return send(fd, &size, sizeof(size), 0) == sizeof(size) &&
         send(fd, body.data(), body.size(), 0) ==
             static_cast<ssize_t>(body.size());
}

bool ReadServer(int fd, v1::ServerEnvelope* envelope) {
  std::uint32_t size = 0;
  if (recv(fd, &size, sizeof(size), MSG_WAITALL) != sizeof(size)) return false;
  std::string body(ntohl(size), '\0');
  return recv(fd, body.data(), body.size(), MSG_WAITALL) ==
             static_cast<ssize_t>(body.size()) &&
         envelope->ParseFromString(body);
}

TEST(EngineProcess, MockScriptExhaustionReturnsStableErrorWithoutCrash) {
  EngineProcess engine;
  const int client = engine.Connect();
  ASSERT_GE(client, 0);
  v1::ClientEnvelope envelope;
  envelope.mutable_protocol_version()->set_major(1);
  auto* request = envelope.mutable_completion();
  request->set_request_id("req-1");
  request->set_model_id("mini-qwen38-hybrid");
  for (const auto token : {1U, 4U, 7U}) request->add_prompt_token_ids(token);
  request->set_max_tokens(9);
  request->set_timeout_ms(10000);
  request->set_decode_mode(v1::DECODE_MODE_GREEDY);
  ASSERT_TRUE(WriteClient(client, envelope));

  std::vector<std::uint64_t> sequences;
  bool accepted = false;
  bool errored = false;
  while (!errored) {
    v1::ServerEnvelope response;
    ASSERT_TRUE(ReadServer(client, &response));
    if (response.has_accepted()) {
      accepted = true;
      EXPECT_EQ(response.accepted().request_id(), "req-1");
    }
    if (response.has_token()) {
      EXPECT_EQ(response.token().request_id(), "req-1");
      sequences.push_back(response.token().sequence_number());
    }
    if (response.has_error()) {
      EXPECT_EQ(response.error().request_id(), "req-1");
      EXPECT_EQ(response.error().code(), v1::ERROR_CODE_RANK_FAILURE);
      errored = true;
    }
    ASSERT_FALSE(response.has_finished());
  }
  EXPECT_TRUE(accepted);
  EXPECT_EQ(sequences,
            (std::vector<std::uint64_t>{1, 2, 3, 4, 5, 6, 7, 8}));
  close(client);
}

TEST(EngineProcess, SigtermFailStopsWhenHalLaunchNeverReturns) {
  int entered[2];
  int release[2];
  ASSERT_EQ(pipe(entered), 0);
  ASSERT_EQ(pipe(release), 0);
  EngineProcess engine(GRSERVE_BLOCKING_HAL_PATH, entered[1], release[0]);
  close(entered[1]);
  close(release[0]);
  const int client = engine.Connect();
  ASSERT_GE(client, 0);
  auto envelope = v1::ClientEnvelope{};
  envelope.mutable_protocol_version()->set_major(1);
  auto* request = envelope.mutable_completion();
  request->set_request_id("req-block");
  request->set_model_id("mini-qwen38-hybrid");
  request->add_prompt_token_ids(1);
  request->set_max_tokens(1);
  request->set_timeout_ms(10000);
  request->set_decode_mode(v1::DECODE_MODE_GREEDY);
  ASSERT_TRUE(WriteClient(client, envelope));
  v1::ServerEnvelope accepted;
  ASSERT_TRUE(ReadServer(client, &accepted));
  ASSERT_TRUE(accepted.has_accepted());
  EXPECT_EQ(accepted.accepted().request_id(), "req-block");

  pollfd entered_poll{entered[0], POLLIN, 0};
  ASSERT_EQ(poll(&entered_poll, 1, 2000), 1);
  char marker = 0;
  ASSERT_EQ(read(entered[0], &marker, 1), 1);
  int status = 0;
  EXPECT_TRUE(engine.TerminateWithin(std::chrono::milliseconds(1500), &status));
  EXPECT_NE(access(engine.socket_path().c_str(), F_OK), 0);
  close(client);
  close(entered[0]);
  close(release[1]);
}

TEST(EngineProcess, SigtermFailStopsAfterCancelHidesBlockedHalFromController) {
  int entered[2];
  int release[2];
  ASSERT_EQ(pipe(entered), 0);
  ASSERT_EQ(pipe(release), 0);
  EngineProcess engine(GRSERVE_BLOCKING_HAL_PATH, entered[1], release[0]);
  close(entered[1]);
  close(release[0]);
  const int client = engine.Connect();
  ASSERT_GE(client, 0);
  v1::ClientEnvelope completion;
  completion.mutable_protocol_version()->set_major(1);
  auto* request = completion.mutable_completion();
  request->set_request_id("req-cancel-block");
  request->set_model_id("mini-qwen38-hybrid");
  request->add_prompt_token_ids(1);
  request->set_max_tokens(1);
  request->set_timeout_ms(10000);
  request->set_decode_mode(v1::DECODE_MODE_GREEDY);
  ASSERT_TRUE(WriteClient(client, completion));
  v1::ServerEnvelope accepted;
  ASSERT_TRUE(ReadServer(client, &accepted));
  ASSERT_TRUE(accepted.has_accepted());
  pollfd entered_poll{entered[0], POLLIN, 0};
  ASSERT_EQ(poll(&entered_poll, 1, 2000), 1);
  char marker = 0;
  ASSERT_EQ(read(entered[0], &marker, 1), 1);

  v1::ClientEnvelope cancel;
  cancel.mutable_protocol_version()->set_major(1);
  cancel.mutable_cancel()->set_request_id("req-cancel-block");
  cancel.mutable_cancel()->set_reason(v1::CANCEL_REASON_CLIENT_DISCONNECTED);
  ASSERT_TRUE(WriteClient(client, cancel));
  v1::ServerEnvelope ack;
  ASSERT_TRUE(ReadServer(client, &ack));
  ASSERT_TRUE(ack.has_cancel_ack());
  ASSERT_EQ(ack.cancel_ack().disposition(),
            v1::CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION);

  int status = 0;
  EXPECT_TRUE(engine.TerminateWithin(std::chrono::milliseconds(1500), &status));
  EXPECT_NE(access(engine.socket_path().c_str(), F_OK), 0);
  close(client);
  close(entered[0]);
  close(release[1]);
}

}  // namespace
