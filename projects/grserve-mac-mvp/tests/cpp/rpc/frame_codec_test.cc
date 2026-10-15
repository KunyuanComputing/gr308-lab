#include "rpc/frame_codec.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <google/protobuf/unknown_field_set.h>
#include <gtest/gtest.h>

namespace grserve::rpc {
namespace {

using v1::ClientEnvelope;
using v1::ServerEnvelope;

class ScopedFd {
 public:
  explicit ScopedFd(int fd = -1) : fd_(fd) {}
  ~ScopedFd() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  int get() const { return fd_; }

 private:
  int fd_;
};

std::vector<std::uint8_t> EncodeFrame(const ClientEnvelope& envelope) {
  std::string body;
  EXPECT_TRUE(envelope.SerializeToString(&body));
  const std::uint32_t length = htonl(static_cast<std::uint32_t>(body.size()));
  std::vector<std::uint8_t> frame(sizeof(length) + body.size());
  std::memcpy(frame.data(), &length, sizeof(length));
  std::memcpy(frame.data() + sizeof(length), body.data(), body.size());
  return frame;
}

ClientEnvelope MakeClient() {
  ClientEnvelope envelope;
  envelope.mutable_protocol_version()->set_major(1);
  envelope.mutable_protocol_version()->set_minor(7);
  envelope.mutable_completion()->set_request_id("request-42");
  envelope.mutable_completion()->set_model_id("mini-qwen");
  envelope.mutable_completion()->add_prompt_token_ids(11);
  envelope.mutable_completion()->add_prompt_token_ids(12);
  return envelope;
}

ServerEnvelope MakeServer() {
  ServerEnvelope envelope;
  envelope.mutable_protocol_version()->set_major(1);
  envelope.mutable_accepted()->set_request_id("request-42");
  return envelope;
}

class ScriptedReader {
 public:
  static constexpr std::size_t kEintr = std::numeric_limits<std::size_t>::max();
  static constexpr std::size_t kEagain = kEintr - 1;

  ScriptedReader(std::vector<std::uint8_t> bytes,
                 std::vector<std::size_t> actions)
      : bytes_(std::move(bytes)), actions_(std::move(actions)) {}

  ssize_t Read(int, void* output, std::size_t count) {
    if (action_ < actions_.size()) {
      const std::size_t action = actions_[action_++];
      if (action == kEintr) {
        errno = EINTR;
        return -1;
      }
      if (action == kEagain) {
        errno = EAGAIN;
        return -1;
      }
      count = std::min(count, action);
    }
    count = std::min(count, bytes_.size() - offset_);
    if (count == 0) {
      return 0;
    }
    std::memcpy(output, bytes_.data() + offset_, count);
    offset_ += count;
    return static_cast<ssize_t>(count);
  }

  std::size_t consumed() const { return offset_; }

 private:
  std::vector<std::uint8_t> bytes_;
  std::vector<std::size_t> actions_;
  std::size_t offset_ = 0;
  std::size_t action_ = 0;
};

class ScriptedWriter {
 public:
  static constexpr std::size_t kEintr = std::numeric_limits<std::size_t>::max();
  static constexpr std::size_t kEagain = kEintr - 1;
  static constexpr std::size_t kEpipe = kEintr - 2;
  static constexpr std::size_t kConnectionReset = kEintr - 3;

  explicit ScriptedWriter(std::vector<std::size_t> actions)
      : actions_(std::move(actions)) {}

  ssize_t Write(int, const void* input, std::size_t count) {
    if (action_ < actions_.size()) {
      const std::size_t action = actions_[action_++];
      if (action >= kConnectionReset) {
        if (action == kEintr) errno = EINTR;
        if (action == kEagain) errno = EAGAIN;
        if (action == kEpipe) errno = EPIPE;
        if (action == kConnectionReset) errno = ECONNRESET;
        return -1;
      }
      count = std::min(count, action);
    }
    const auto* bytes = static_cast<const std::uint8_t*>(input);
    output_.insert(output_.end(), bytes, bytes + count);
    return static_cast<ssize_t>(count);
  }

  const std::vector<std::uint8_t>& output() const { return output_; }

 private:
  std::vector<std::size_t> actions_;
  std::vector<std::uint8_t> output_;
  std::size_t action_ = 0;
};

ScopedFd BlockingProbeFd() {
  int descriptors[2];
  EXPECT_EQ(pipe(descriptors), 0);
  close(descriptors[1]);
  return ScopedFd(descriptors[0]);
}

ssize_t WriteByteWithoutSigpipe(int fd, char byte) {
  sigset_t sigpipe_set;
  sigset_t previous_mask;
  sigset_t pending_before;
  if (sigemptyset(&sigpipe_set) != 0 ||
      sigaddset(&sigpipe_set, SIGPIPE) != 0) {
    return -1;
  }
  const int mask_error =
      pthread_sigmask(SIG_BLOCK, &sigpipe_set, &previous_mask);
  if (mask_error != 0) {
    errno = mask_error;
    return -1;
  }
  if (sigpending(&pending_before) != 0) {
    const int pending_error = errno;
    const int restore_error =
        pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr);
    errno = restore_error == 0 ? pending_error : restore_error;
    return -1;
  }
  const bool had_pending_sigpipe =
      sigismember(&pending_before, SIGPIPE) == 1;

  ssize_t result;
  do {
    result = write(fd, &byte, 1);
  } while (result < 0 && errno == EINTR);
  const int write_error = errno;
  int consume_error = 0;
  if (result < 0 && write_error == EPIPE && !had_pending_sigpipe) {
    sigset_t pending_after;
    if (sigpending(&pending_after) != 0) {
      consume_error = errno;
    } else if (sigismember(&pending_after, SIGPIPE) == 1) {
      int consumed_signal = 0;
      const int wait_error = sigwait(&sigpipe_set, &consumed_signal);
      if (wait_error != 0 || consumed_signal != SIGPIPE) {
        consume_error = wait_error != 0 ? wait_error : EIO;
      }
    }
  }
  const int restore_error =
      pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr);
  if (restore_error != 0) {
    errno = restore_error;
    return -1;
  }
  if (consume_error != 0) {
    errno = consume_error;
    return -1;
  }
  errno = write_error;
  return result;
}

void ExpectError(const Error* error, ErrorCode code, std::string_view message) {
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code(), code);
  EXPECT_EQ(error->message(), message);
}

TEST(FrameReaderTest, ReadsEveryHeaderSplitPoint) {
  const ClientEnvelope expected = MakeClient();
  const auto frame = EncodeFrame(expected);
  for (std::size_t split = 1; split < kFrameHeaderBytes; ++split) {
    SCOPED_TRACE(split);
    ScriptedReader input(frame, {split, kFrameHeaderBytes - split, frame.size()});
    FrameReader reader([&](int fd, void* data, std::size_t size) {
      return input.Read(fd, data, size);
    });
    auto result = reader.ReadOne(BlockingProbeFd().get());
    ASSERT_TRUE(result.ok()) << result.error()->message();
    ASSERT_TRUE(result.value().has_value());
    EXPECT_EQ(result.value()->SerializeAsString(), expected.SerializeAsString());
  }
}

TEST(FrameReaderTest, ReadsEveryBodySplitPoint) {
  const ClientEnvelope expected = MakeClient();
  const auto frame = EncodeFrame(expected);
  const std::size_t body_size = frame.size() - kFrameHeaderBytes;
  for (std::size_t split = 1; split < body_size; ++split) {
    SCOPED_TRACE(split);
    ScriptedReader input(frame, {kFrameHeaderBytes, split, body_size - split});
    FrameReader reader([&](int fd, void* data, std::size_t size) {
      return input.Read(fd, data, size);
    });
    auto result = reader.ReadOne(BlockingProbeFd().get());
    ASSERT_TRUE(result.ok()) << result.error()->message();
    ASSERT_TRUE(result.value().has_value());
    EXPECT_EQ(result.value()->SerializeAsString(), expected.SerializeAsString());
  }
}

TEST(FrameReaderTest, KeepsTwoCoalescedFramesSeparate) {
  const ClientEnvelope first = MakeClient();
  ClientEnvelope second = MakeClient();
  second.mutable_completion()->set_request_id("second");
  auto bytes = EncodeFrame(first);
  const auto second_bytes = EncodeFrame(second);
  bytes.insert(bytes.end(), second_bytes.begin(), second_bytes.end());
  ScriptedReader input(bytes, {});
  FrameReader reader([&](int fd, void* data, std::size_t size) {
    return input.Read(fd, data, size);
  });
  const ScopedFd probe = BlockingProbeFd();
  auto first_result = reader.ReadOne(probe.get());
  auto second_result = reader.ReadOne(probe.get());
  ASSERT_TRUE(first_result.ok());
  ASSERT_TRUE(second_result.ok());
  EXPECT_EQ(first_result.value()->SerializeAsString(), first.SerializeAsString());
  EXPECT_EQ(second_result.value()->SerializeAsString(), second.SerializeAsString());
}

TEST(FrameReaderTest, RetriesEintrForHeaderAndBody) {
  const ClientEnvelope expected = MakeClient();
  const auto frame = EncodeFrame(expected);
  ScriptedReader input(frame, {ScriptedReader::kEintr, kFrameHeaderBytes,
                               ScriptedReader::kEintr, frame.size()});
  FrameReader reader([&](int fd, void* data, std::size_t size) {
    return input.Read(fd, data, size);
  });
  auto result = reader.ReadOne(BlockingProbeFd().get());
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value()->SerializeAsString(), expected.SerializeAsString());
}

TEST(FrameReaderTest, ReportsCleanEofBeforeAnyHeaderByte) {
  ScriptedReader input({}, {});
  FrameReader reader([&](int fd, void* data, std::size_t size) {
    return input.Read(fd, data, size);
  });
  auto result = reader.ReadOne(BlockingProbeFd().get());
  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result.value().has_value());
}

TEST(FrameReaderTest, RejectsEofPartwayThroughHeader) {
  ScriptedReader input({0, 0, 0}, {});
  FrameReader reader([&](int fd, void* data, std::size_t size) {
    return input.Read(fd, data, size);
  });
  auto result = reader.ReadOne(BlockingProbeFd().get());
  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal,
              "frame read: EOF in 4-byte header");
}

TEST(FrameReaderTest, RejectsEofPartwayThroughBody) {
  const std::uint32_t length = htonl(2);
  std::vector<std::uint8_t> bytes(sizeof(length) + 1);
  std::memcpy(bytes.data(), &length, sizeof(length));
  bytes.back() = 1;
  ScriptedReader input(bytes, {});
  FrameReader reader([&](int fd, void* data, std::size_t size) {
    return input.Read(fd, data, size);
  });
  auto result = reader.ReadOne(BlockingProbeFd().get());
  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal, "frame read: EOF in body");
}

TEST(FrameReaderTest, RejectsZeroLengthBeforeParsing) {
  std::vector<std::uint8_t> bytes(kFrameHeaderBytes, 0);
  ScriptedReader input(bytes, {});
  FrameReader reader([&](int fd, void* data, std::size_t size) {
    return input.Read(fd, data, size);
  });
  auto result = reader.ReadOne(BlockingProbeFd().get());
  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInvalidArgument,
              "frame read: zero-length frame");
  EXPECT_EQ(input.consumed(), kFrameHeaderBytes);
}

TEST(FrameReaderTest, AcceptsExactlyFourMiB) {
  std::vector<std::uint8_t> body(kMaxFrameBytes, 0);
  body[0] = 0xA2;
  body[1] = 0x06;
  std::uint32_t payload = static_cast<std::uint32_t>(body.size() - 6);
  for (int index = 0; index < 4; ++index) {
    body[2 + index] = static_cast<std::uint8_t>((payload & 0x7fU) |
                                                (index == 3 ? 0 : 0x80U));
    payload >>= 7;
  }
  const std::uint32_t network_size = htonl(kMaxFrameBytes);
  std::vector<std::uint8_t> bytes(kFrameHeaderBytes + body.size());
  std::memcpy(bytes.data(), &network_size, sizeof(network_size));
  std::memcpy(bytes.data() + kFrameHeaderBytes, body.data(), body.size());
  ScriptedReader input(std::move(bytes), {});
  FrameReader reader([&](int fd, void* data, std::size_t size) {
    return input.Read(fd, data, size);
  });
  auto result = reader.ReadOne(BlockingProbeFd().get());
  ASSERT_TRUE(result.ok()) << result.error()->message();
  EXPECT_TRUE(result.value().has_value());
}

TEST(FrameReaderTest, RejectsOverFourMiBBeforeReadingBody) {
  const std::uint32_t network_size = htonl(kMaxFrameBytes + 1);
  std::vector<std::uint8_t> bytes(kFrameHeaderBytes + 8, 0x42);
  std::memcpy(bytes.data(), &network_size, sizeof(network_size));
  ScriptedReader input(std::move(bytes), {});
  FrameReader reader([&](int fd, void* data, std::size_t size) {
    return input.Read(fd, data, size);
  });
  auto result = reader.ReadOne(BlockingProbeFd().get());
  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInvalidArgument,
              "frame read: frame exceeds 4 MiB limit");
  EXPECT_EQ(input.consumed(), kFrameHeaderBytes);
}

TEST(FrameReaderTest, RejectsInvalidProtobufAndTrailingGarbage) {
  const std::array<std::vector<std::uint8_t>, 2> invalid_bodies = {
      std::vector<std::uint8_t>{0x80},
      std::vector<std::uint8_t>{0x0a, 0x00, 0x00},
  };
  for (const auto& body : invalid_bodies) {
    SCOPED_TRACE(body.size());
    const std::uint32_t network_size = htonl(static_cast<std::uint32_t>(body.size()));
    std::vector<std::uint8_t> bytes(kFrameHeaderBytes + body.size());
    std::memcpy(bytes.data(), &network_size, sizeof(network_size));
    std::memcpy(bytes.data() + kFrameHeaderBytes, body.data(), body.size());
    ScriptedReader input(std::move(bytes), {});
    FrameReader reader([&](int fd, void* data, std::size_t size) {
      return input.Read(fd, data, size);
    });
    auto result = reader.ReadOne(BlockingProbeFd().get());
    ASSERT_FALSE(result.ok());
    ExpectError(result.error(), ErrorCode::kInvalidArgument,
                "frame read: invalid ClientEnvelope protobuf");
  }
}

TEST(FrameReaderTest, EnforcesProtobufRecursionLimit) {
  constexpr std::size_t kGroups = kFrameProtoRecursionLimit + 1;
  std::vector<std::uint8_t> body;
  for (std::size_t index = 0; index < kGroups; ++index) {
    body.push_back(0xa3);
    body.push_back(0x06);
  }
  for (std::size_t index = 0; index < kGroups; ++index) {
    body.push_back(0xa4);
    body.push_back(0x06);
  }
  const std::uint32_t network_size = htonl(static_cast<std::uint32_t>(body.size()));
  std::vector<std::uint8_t> bytes(kFrameHeaderBytes + body.size());
  std::memcpy(bytes.data(), &network_size, sizeof(network_size));
  std::memcpy(bytes.data() + kFrameHeaderBytes, body.data(), body.size());
  ScriptedReader input(std::move(bytes), {});
  FrameReader reader([&](int fd, void* data, std::size_t size) {
    return input.Read(fd, data, size);
  });
  auto result = reader.ReadOne(BlockingProbeFd().get());
  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInvalidArgument,
              "frame read: invalid ClientEnvelope protobuf");
}

TEST(FrameReaderTest, TreatsEagainAsBackpressureOnBlockingDescriptor) {
  ScriptedReader input({}, {ScriptedReader::kEagain});
  FrameReader reader([&](int fd, void* data, std::size_t size) {
    return input.Read(fd, data, size);
  });
  auto result = reader.ReadOne(BlockingProbeFd().get());
  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kResourceExhausted,
              "frame read: blocking descriptor returned EAGAIN");
}

TEST(FrameReaderTest, RejectsNonblockingDescriptors) {
  int descriptors[2];
  ASSERT_EQ(pipe(descriptors), 0);
  ScopedFd reader_fd(descriptors[0]);
  ScopedFd writer_fd(descriptors[1]);
  ASSERT_NE(fcntl(reader_fd.get(), F_SETFL, O_NONBLOCK), -1);
  FrameReader reader;
  auto result = reader.ReadOne(reader_fd.get());
  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInvalidArgument,
              "frame read: nonblocking descriptors are unsupported");
}

TEST(FrameWriterTest, UsesFourByteNetworkOrderHeader) {
  ScriptedWriter output({});
  FrameWriter writer([&](int fd, const void* data, std::size_t size) {
    return output.Write(fd, data, size);
  });
  const ServerEnvelope envelope = MakeServer();
  const Status status = writer.WriteOne(BlockingProbeFd().get(), envelope);
  ASSERT_TRUE(status.ok());
  std::string body;
  ASSERT_TRUE(envelope.SerializeToString(&body));
  ASSERT_GE(output.output().size(), kFrameHeaderBytes);
  const std::array<std::uint8_t, 4> expected = {
      static_cast<std::uint8_t>((body.size() >> 24) & 0xff),
      static_cast<std::uint8_t>((body.size() >> 16) & 0xff),
      static_cast<std::uint8_t>((body.size() >> 8) & 0xff),
      static_cast<std::uint8_t>(body.size() & 0xff),
  };
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), output.output().begin()));
}

TEST(FrameWriterTest, UsesWriteForAnOrdinaryBlockingDescriptor) {
  int descriptors[2];
  ASSERT_EQ(pipe(descriptors), 0);
  ScopedFd read_fd(descriptors[0]);
  ScopedFd write_fd(descriptors[1]);
  const ServerEnvelope envelope = MakeServer();
  FrameWriter writer;
  const Status status = writer.WriteOne(write_fd.get(), envelope);
  ASSERT_TRUE(status.ok()) << status.error()->message();

  std::array<std::uint8_t, kFrameHeaderBytes> header{};
  ASSERT_EQ(read(read_fd.get(), header.data(), header.size()),
            static_cast<ssize_t>(header.size()));
  std::string body;
  ASSERT_TRUE(envelope.SerializeToString(&body));
  const std::array<std::uint8_t, 4> expected = {
      static_cast<std::uint8_t>((body.size() >> 24) & 0xff),
      static_cast<std::uint8_t>((body.size() >> 16) & 0xff),
      static_cast<std::uint8_t>((body.size() >> 8) & 0xff),
      static_cast<std::uint8_t>(body.size() & 0xff),
  };
  EXPECT_EQ(header, expected);
}

TEST(FrameWriterTest, SuppressesSigpipeForDisconnectedSocket) {
  int descriptors[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors), 0);
  ScopedFd writer_fd(descriptors[0]);
  ASSERT_EQ(close(descriptors[1]), 0);
  FrameWriter writer;
  const Status status = writer.WriteOne(writer_fd.get(), MakeServer());
  ASSERT_FALSE(status.ok());
  ExpectError(status.error(), ErrorCode::kInternal,
              "frame write: peer disconnected");
}

TEST(FrameWriterTest, SuppressesSigpipeForDisconnectedOrdinaryPipe) {
  int descriptors[2];
  ASSERT_EQ(pipe(descriptors), 0);
  int start_signal[2];
  if (pipe(start_signal) != 0) {
    const int pipe_error = errno;
    close(descriptors[0]);
    close(descriptors[1]);
    FAIL() << "synchronization pipe failed: " << strerror(pipe_error);
  }
  const std::string tested_fd_argument = std::to_string(descriptors[1]);
  const std::string start_fd_argument = std::to_string(start_signal[0]);
  const pid_t child = fork();
  if (child < 0) {
    const int fork_error = errno;
    close(descriptors[0]);
    close(descriptors[1]);
    close(start_signal[0]);
    close(start_signal[1]);
    FAIL() << "fork failed: " << strerror(fork_error);
  }
  if (child == 0) {
    const bool close_read_failed = close(descriptors[0]) != 0;
    const bool close_signal_write_failed = close(start_signal[1]) != 0;
    if (close_read_failed || close_signal_write_failed) {
      close(descriptors[1]);
      close(start_signal[0]);
      _exit(8);
    }
    execl(GRSERVE_FRAME_SIGPIPE_HELPER_PATH,
          GRSERVE_FRAME_SIGPIPE_HELPER_PATH, tested_fd_argument.c_str(),
          start_fd_argument.c_str(), static_cast<char*>(nullptr));
    close(descriptors[1]);
    close(start_signal[0]);
    _exit(9);
  }

  // The synchronization pipe makes the causal ordering explicit: all copies
  // of the tested pipe's read end are closed before the child calls WriteOne.
  const int close_read_result = close(descriptors[0]);
  const int close_write_result = close(descriptors[1]);
  const int close_signal_read_result = close(start_signal[0]);
  const char token = 1;
  const ssize_t notified = WriteByteWithoutSigpipe(start_signal[1], token);
  const int notify_error = errno;
  const int close_signal_write_result = close(start_signal[1]);
  int child_status = 0;
  pid_t waited;
  do {
    waited = waitpid(child, &child_status, 0);
  } while (waited < 0 && errno == EINTR);

  EXPECT_EQ(close_read_result, 0);
  EXPECT_EQ(close_write_result, 0);
  EXPECT_EQ(close_signal_read_result, 0);
  ASSERT_EQ(notified, 1) << strerror(notify_error);
  EXPECT_EQ(close_signal_write_result, 0);
  ASSERT_EQ(waited, child);
  ASSERT_TRUE(WIFEXITED(child_status))
      << (WIFSIGNALED(child_status) ? strsignal(WTERMSIG(child_status))
                                    : "unexpected child status");
  EXPECT_EQ(WEXITSTATUS(child_status), 0);
}

TEST(FrameWriterTest, RetriesEintrAndCompletesPartialWrites) {
  ScriptedWriter output({ScriptedWriter::kEintr, 1, 2, 3});
  FrameWriter writer([&](int fd, const void* data, std::size_t size) {
    return output.Write(fd, data, size);
  });
  const ServerEnvelope envelope = MakeServer();
  const Status status = writer.WriteOne(BlockingProbeFd().get(), envelope);
  ASSERT_TRUE(status.ok());
  std::string body;
  ASSERT_TRUE(envelope.SerializeToString(&body));
  EXPECT_EQ(output.output().size(), kFrameHeaderBytes + body.size());
}

TEST(FrameWriterTest, RejectsZeroByteWrite) {
  ScriptedWriter output({0});
  FrameWriter writer([&](int fd, const void* data, std::size_t size) {
    return output.Write(fd, data, size);
  });
  const Status status = writer.WriteOne(BlockingProbeFd().get(), MakeServer());
  ASSERT_FALSE(status.ok());
  ExpectError(status.error(), ErrorCode::kInternal,
              "frame write: write returned zero");
}

TEST(FrameWriterTest, ReturnsStablePeerDisconnectErrors) {
  for (const std::size_t action : {ScriptedWriter::kEpipe,
                                   ScriptedWriter::kConnectionReset}) {
    ScriptedWriter output({action});
    FrameWriter writer([&](int fd, const void* data, std::size_t size) {
      return output.Write(fd, data, size);
    });
    const Status status = writer.WriteOne(BlockingProbeFd().get(), MakeServer());
    ASSERT_FALSE(status.ok());
    ExpectError(status.error(), ErrorCode::kInternal,
                "frame write: peer disconnected");
  }
}

TEST(FrameWriterTest, TreatsEagainAsBackpressureOnBlockingDescriptor) {
  ScriptedWriter output({ScriptedWriter::kEagain});
  FrameWriter writer([&](int fd, const void* data, std::size_t size) {
    return output.Write(fd, data, size);
  });
  const Status status = writer.WriteOne(BlockingProbeFd().get(), MakeServer());
  ASSERT_FALSE(status.ok());
  ExpectError(status.error(), ErrorCode::kResourceExhausted,
              "frame write: blocking descriptor returned EAGAIN");
}

TEST(FrameWriterTest, RejectsNonblockingDescriptors) {
  int descriptors[2];
  ASSERT_EQ(pipe(descriptors), 0);
  ScopedFd reader_fd(descriptors[0]);
  ScopedFd writer_fd(descriptors[1]);
  ASSERT_NE(fcntl(writer_fd.get(), F_SETFL, O_NONBLOCK), -1);
  FrameWriter writer;
  const Status status = writer.WriteOne(writer_fd.get(), MakeServer());
  ASSERT_FALSE(status.ok());
  ExpectError(status.error(), ErrorCode::kInvalidArgument,
              "frame write: nonblocking descriptors are unsupported");
}

TEST(FrameWriterTest, RejectsEmptyAndOversizedMessagesBeforeWriting) {
  ScriptedWriter output({});
  FrameWriter writer([&](int fd, const void* data, std::size_t size) {
    return output.Write(fd, data, size);
  });
  const Status empty = writer.WriteOne(BlockingProbeFd().get(), ServerEnvelope{});
  ASSERT_FALSE(empty.ok());
  ExpectError(empty.error(), ErrorCode::kInvalidArgument,
              "frame write: zero-length frame");

  ServerEnvelope oversized;
  oversized.mutable_error()->set_message(std::string(kMaxFrameBytes, 'x'));
  const Status too_large = writer.WriteOne(BlockingProbeFd().get(), oversized);
  ASSERT_FALSE(too_large.ok());
  ExpectError(too_large.error(), ErrorCode::kInvalidArgument,
              "frame write: frame exceeds 4 MiB limit");
  EXPECT_TRUE(output.output().empty());
}

TEST(FrameWriterTest, AcceptsMessageSerializedToExactlyFourMiB) {
  ServerEnvelope envelope;
  auto* unknown = envelope.GetReflection()->MutableUnknownFields(&envelope);
  constexpr std::size_t kWireOverhead = 6;
  unknown->AddLengthDelimited(100, std::string(kMaxFrameBytes - kWireOverhead, 'x'));
  ASSERT_EQ(envelope.ByteSizeLong(), kMaxFrameBytes);
  ScriptedWriter output({});
  FrameWriter writer([&](int fd, const void* data, std::size_t size) {
    return output.Write(fd, data, size);
  });
  const Status status = writer.WriteOne(BlockingProbeFd().get(), envelope);
  ASSERT_TRUE(status.ok()) << status.error()->message();
  EXPECT_EQ(output.output().size(), kFrameHeaderBytes + kMaxFrameBytes);
}

}  // namespace
}  // namespace grserve::rpc
