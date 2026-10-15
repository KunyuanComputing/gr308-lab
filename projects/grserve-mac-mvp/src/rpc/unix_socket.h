#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include <sys/types.h>

#include "grserve/error.h"

namespace grserve::rpc {

// GRServe deliberately caps pathname sockets at 100 bytes on both macOS and
// Linux. The terminating NUL must additionally fit the platform sun_path.
inline constexpr std::size_t kMaxUnixSocketPathBytes = 100;

// Linux creates and accepts descriptors atomically with CLOEXEC. Darwin has
// no SOCK_CLOEXEC/accept4, so this module sets FD_CLOEXEC before exposing each
// descriptor. The Engine must not concurrently fork/exec during that Darwin
// creation window (its process model does not spawn children after startup).

class UniqueFd {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(int fd) noexcept : fd_(fd) {}
  ~UniqueFd();

  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept;
  UniqueFd& operator=(UniqueFd&& other) noexcept;

  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] explicit operator bool() const noexcept { return fd_ >= 0; }
  [[nodiscard]] int release() noexcept;
  void reset(int replacement = -1) noexcept;

 private:
  int fd_ = -1;
};

class UnixListener {
 public:
  UnixListener() noexcept = default;
  ~UnixListener();

  UnixListener(const UnixListener&) = delete;
  UnixListener& operator=(const UnixListener&) = delete;
  UnixListener(UnixListener&& other) noexcept;
  UnixListener& operator=(UnixListener&& other) noexcept;

  [[nodiscard]] static StatusOr<UnixListener> Bind(std::string path,
                                                    int backlog = 128);
  [[nodiscard]] StatusOr<UniqueFd> Accept() const;
  [[nodiscard]] int get() const noexcept { return socket_.get(); }

 private:
  UnixListener(UniqueFd socket, UniqueFd parent, std::string basename,
               dev_t socket_device, ino_t socket_inode) noexcept;
  void Cleanup() noexcept;

  UniqueFd socket_;
  UniqueFd parent_;
  std::string basename_;
  dev_t socket_device_ = 0;
  ino_t socket_inode_ = 0;
  bool owns_path_ = false;
};

[[nodiscard]] Status ValidateUnixSocketPath(std::string_view path);
[[nodiscard]] StatusOr<UniqueFd> ConnectUnixSocket(std::string_view path);

#if defined(GRSERVE_RPC_TESTING)
namespace testing {

using SocketProbeHook = int (*)(std::string_view path);
void SetSocketProbeHook(SocketProbeHook hook) noexcept;

enum class SocketBindHookStage {
  kBeforeBind,
  kAfterBindBeforeIdentity,
  kAfterBoundIdentity,
  kBeforePermissionUpdate,
  kBeforeInstall,
  kAfterInstall,
};
using SocketBindHook = void (*)(SocketBindHookStage stage,
                                std::string_view socket_path,
                                std::string_view final_path);
void SetSocketBindHook(SocketBindHook hook) noexcept;

}  // namespace testing
#endif

}  // namespace grserve::rpc
