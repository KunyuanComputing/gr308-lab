#include "rpc/frame_codec.h"

#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <unistd.h>

#include <cstdlib>

namespace {

using grserve::ErrorCode;
using grserve::Status;
using grserve::rpc::FrameWriter;
using grserve::rpc::v1::ServerEnvelope;

bool ParseDescriptor(const char* text, int* descriptor) {
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed < 0 ||
      parsed > INT_MAX) {
    return false;
  }
  *descriptor = static_cast<int>(parsed);
  return true;
}

ServerEnvelope MakeServer() {
  ServerEnvelope envelope;
  envelope.mutable_protocol_version()->set_major(1);
  envelope.mutable_accepted()->set_request_id("request-42");
  return envelope;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 10;
  int write_fd = -1;
  int start_fd = -1;
  if (!ParseDescriptor(argv[1], &write_fd) ||
      !ParseDescriptor(argv[2], &start_fd) || write_fd == start_fd) {
    return 11;
  }

  const bool signal_failed = signal(SIGPIPE, SIG_DFL) == SIG_ERR;
  sigset_t sigpipe_set;
  const bool mask_setup_failed = sigemptyset(&sigpipe_set) != 0 ||
                                 sigaddset(&sigpipe_set, SIGPIPE) != 0 ||
                                 sigprocmask(SIG_UNBLOCK, &sigpipe_set,
                                             nullptr) != 0;
  char token = 0;
  ssize_t received;
  do {
    received = read(start_fd, &token, 1);
  } while (received < 0 && errno == EINTR);
  const bool close_start_failed = close(start_fd) != 0;
  if (received != 1 || token != 1) {
    close(write_fd);
    return 12;
  }
  if (signal_failed || mask_setup_failed || close_start_failed) {
    close(write_fd);
    return 13;
  }

  FrameWriter writer;
  const Status status = writer.WriteOne(write_fd, MakeServer());
  if (close(write_fd) != 0) return 14;
  if (status.ok()) return 1;
  if (status.error()->code() != ErrorCode::kInternal) return 2;
  if (status.error()->message() != "frame write: peer disconnected") return 3;
  return 0;
}
