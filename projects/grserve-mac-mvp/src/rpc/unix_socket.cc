#include "rpc/unix_socket.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#if defined(__APPLE__)
#include <stdio.h>
#elif defined(__linux__)
#include <linux/fs.h>
#include <sys/random.h>
#include <sys/syscall.h>
#else
#error "GRServe Unix sockets require Darwin or Linux"
#endif

namespace grserve::rpc {
namespace {

#if defined(GRSERVE_RPC_TESTING)
std::atomic<testing::SocketProbeHook>& SocketProbeHookForTesting() {
  static std::atomic<testing::SocketProbeHook> hook{nullptr};
  return hook;
}

std::atomic<testing::SocketBindHook>& SocketBindHookForTesting() {
  static std::atomic<testing::SocketBindHook> hook{nullptr};
  return hook;
}

void RunSocketBindHookForTesting(testing::SocketBindHookStage stage,
                                 std::string_view socket_path,
                                 std::string_view final_path) {
  if (const testing::SocketBindHook hook =
          SocketBindHookForTesting().load(std::memory_order_acquire)) {
    hook(stage, socket_path, final_path);
  }
}
#endif

Error Failure(ErrorCode code, std::string message) {
  return Error::From(code, std::move(message));
}

Status InternalFailure(std::string message) {
  return Status{Failure(ErrorCode::kInternal, std::move(message))};
}

bool SameIdentity(const struct stat& left, const struct stat& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
         (left.st_mode & S_IFMT) == (right.st_mode & S_IFMT) &&
         left.st_uid == right.st_uid;
}

Status SetDescriptorPolicy(int fd) {
  const int descriptor_flags = fcntl(fd, F_GETFD);
  if (descriptor_flags < 0 ||
      fcntl(fd, F_SETFD, descriptor_flags | FD_CLOEXEC) < 0) {
    return InternalFailure("unix socket: failed to set FD_CLOEXEC");
  }
  const int status_flags = fcntl(fd, F_GETFL);
  if (status_flags < 0 ||
      fcntl(fd, F_SETFL, status_flags & ~O_NONBLOCK) < 0) {
    return InternalFailure("unix socket: failed to set blocking mode");
  }
  return Status::Ok();
}

Status SetSocketPolicy(int fd) {
  const Status descriptor = SetDescriptorPolicy(fd);
  if (!descriptor.ok()) return descriptor;
#if defined(__APPLE__) && defined(SO_NOSIGPIPE)
  const int enabled = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) !=
      0) {
    return InternalFailure("unix socket: failed to set SO_NOSIGPIPE");
  }
#endif
  return Status::Ok();
}

StatusOr<UniqueFd> NewStreamSocket() {
  int socket_type = SOCK_STREAM;
#if defined(SOCK_CLOEXEC)
  socket_type |= SOCK_CLOEXEC;
#endif
  const int fd = socket(AF_UNIX, socket_type, 0);
  if (fd < 0) {
    return StatusOr<UniqueFd>(
        Failure(ErrorCode::kInternal, "unix socket: socket creation failed"));
  }
  UniqueFd socket_fd(fd);
  const Status policy = SetSocketPolicy(fd);
  if (!policy.ok()) {
    return StatusOr<UniqueFd>(*policy.error());
  }
  return StatusOr<UniqueFd>(std::move(socket_fd));
}

struct PathParts {
  std::string parent;
  std::string basename;
};

StatusOr<PathParts> SplitPath(std::string_view path) {
  const std::size_t slash = path.rfind('/');
  if (slash == std::string_view::npos || slash + 1 == path.size()) {
    return StatusOr<PathParts>(Failure(
        ErrorCode::kInvalidArgument, "unix socket: path has no filename"));
  }
  PathParts parts;
  parts.parent = slash == 0 ? "/" : std::string(path.substr(0, slash));
  parts.basename = std::string(path.substr(slash + 1));
  return StatusOr<PathParts>(std::move(parts));
}

StatusOr<UniqueFd> OpenParentWithoutSymlinks(std::string_view parent) {
  int open_flags = O_RDONLY | O_DIRECTORY;
#if defined(O_CLOEXEC)
  open_flags |= O_CLOEXEC;
#endif
  const int root = open("/", open_flags);
  if (root < 0) {
    return StatusOr<UniqueFd>(Failure(
        ErrorCode::kInternal, "unix socket: could not open filesystem root"));
  }
  UniqueFd current(root);
  const Status root_policy = SetDescriptorPolicy(current.get());
  if (!root_policy.ok()) return StatusOr<UniqueFd>(*root_policy.error());
  if (parent == "/") return StatusOr<UniqueFd>(std::move(current));

  std::size_t begin = 1;
  bool first_component = true;
  while (begin < parent.size()) {
    const std::size_t end = parent.find('/', begin);
    const std::size_t length =
        (end == std::string_view::npos ? parent.size() : end) - begin;
    const std::string component(parent.substr(begin, length));
    if (component.empty() || component == "." || component == "..") {
      return StatusOr<UniqueFd>(Failure(
          ErrorCode::kInvalidArgument,
          "unix socket: parent path must be canonical"));
    }
    int component_flags = open_flags;
#if defined(O_NOFOLLOW)
    // /tmp is a trusted system symlink to /private/tmp on macOS. GRServe's
    // documented short socket location uses it; all user-controlled
    // components below /tmp are still opened without following symlinks.
#if defined(__APPLE__)
    if (!(first_component && component == "tmp")) {
      component_flags |= O_NOFOLLOW;
    }
#else
    component_flags |= O_NOFOLLOW;
#endif
#endif
    const int next = openat(current.get(), component.c_str(), component_flags);
    if (next < 0) {
      return StatusOr<UniqueFd>(Failure(
          ErrorCode::kInvalidArgument,
          "unix socket: parent must not contain symlinks"));
    }
    UniqueFd next_fd(next);
    const Status next_policy = SetDescriptorPolicy(next_fd.get());
    if (!next_policy.ok()) return StatusOr<UniqueFd>(*next_policy.error());
    current = std::move(next_fd);
    first_component = false;
    if (end == std::string_view::npos) break;
    begin = end + 1;
  }
  return StatusOr<UniqueFd>(std::move(current));
}

socklen_t MakeAddress(std::string_view path, sockaddr_un* address) {
  std::memset(address, 0, sizeof(*address));
  address->sun_family = AF_UNIX;
  std::memcpy(address->sun_path, path.data(), path.size());
  address->sun_path[path.size()] = '\0';
  const auto length = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + path.size() + 1);
#if defined(__APPLE__)
  address->sun_len = static_cast<unsigned char>(length);
#endif
  return length;
}

Status VerifyNamedParent(std::string_view parent_path,
                         const struct stat& opened_parent) {
  struct stat named_parent {};
  if (lstat(std::string(parent_path).c_str(), &named_parent) != 0 ||
      !S_ISDIR(named_parent.st_mode) ||
      !SameIdentity(named_parent, opened_parent)) {
    return InternalFailure("unix socket: parent changed during bind");
  }
  return Status::Ok();
}

enum class ProbeResult { kActive, kStale };

StatusOr<ProbeResult> ProbeExistingSocket(std::string_view path) {
  auto candidate = NewStreamSocket();
  if (!candidate.ok()) {
    return StatusOr<ProbeResult>(*candidate.error());
  }
  sockaddr_un address{};
  const socklen_t address_length = MakeAddress(path, &address);
  const int original_flags = fcntl(candidate.value().get(), F_GETFL);
  if (original_flags < 0 ||
      fcntl(candidate.value().get(), F_SETFL, original_flags | O_NONBLOCK) <
          0) {
    return StatusOr<ProbeResult>(Failure(
        ErrorCode::kInternal, "unix socket: stale probe setup failed"));
  }
  int result;
  do {
#if defined(GRSERVE_RPC_TESTING)
    if (const testing::SocketProbeHook hook =
            SocketProbeHookForTesting().load(std::memory_order_acquire)) {
      result = hook(path);
    } else {
#endif
      result = connect(candidate.value().get(),
                       reinterpret_cast<const sockaddr*>(&address),
                       address_length);
#if defined(GRSERVE_RPC_TESTING)
    }
#endif
  } while (result < 0 && errno == EINTR);
  if (result == 0 || errno == EISCONN || errno == EINPROGRESS ||
      errno == EALREADY || errno == EAGAIN || errno == EWOULDBLOCK) {
    return StatusOr<ProbeResult>(ProbeResult::kActive);
  }
  if (errno == ECONNREFUSED) {
    return StatusOr<ProbeResult>(ProbeResult::kStale);
  }
  return StatusOr<ProbeResult>(Failure(
      ErrorCode::kInternal, "unix socket: stale probe failed safely"));
}

Status PrepareSocketPath(int parent_fd, std::string_view basename,
                         std::string_view full_path) {
  struct stat initial {};
  if (fstatat(parent_fd, std::string(basename).c_str(), &initial,
              AT_SYMLINK_NOFOLLOW) != 0) {
    if (errno == ENOENT) return Status::Ok();
    return InternalFailure("unix socket: lstat failed");
  }
  if (!S_ISSOCK(initial.st_mode) || initial.st_uid != getuid()) {
    return Status{Failure(
        ErrorCode::kInvalidArgument,
        "unix socket: existing path is not an owned socket")};
  }

  auto probe = ProbeExistingSocket(full_path);
  if (!probe.ok()) return probe.status();
  if (probe.value() == ProbeResult::kActive) {
    return Status{Failure(ErrorCode::kResourceExhausted,
                          "unix socket: active listener already exists")};
  }

  struct stat confirmed {};
  if (fstatat(parent_fd, std::string(basename).c_str(), &confirmed,
              AT_SYMLINK_NOFOLLOW) != 0) {
    if (errno == ENOENT) return Status::Ok();
    return InternalFailure("unix socket: stale socket recheck failed");
  }
  if (!SameIdentity(initial, confirmed) || !S_ISSOCK(confirmed.st_mode) ||
      confirmed.st_uid != getuid()) {
    return InternalFailure("unix socket: path changed during stale check");
  }
  // The parent flock coordinates all GRServe binders. The repeated inode,
  // type, and uid check plus dirfd-relative unlink is the strongest portable
  // pathname-socket guard against non-cooperating same-uid processes; Unix
  // provides no atomic compare-and-unlink primitive.
  if (unlinkat(parent_fd, std::string(basename).c_str(), 0) != 0) {
    return InternalFailure("unix socket: stale socket unlink failed");
  }
  return Status::Ok();
}

std::mutex& BindMutex() {
  static std::mutex mutex;
  return mutex;
}

Status SetBoundSocketOwnerOnly(int parent_fd, std::string_view basename,
                               const struct stat& expected,
                               struct stat* confirmed,
                               std::string_view socket_path,
                               std::string_view final_path) {
  const std::string name(basename);
  struct stat before {};
  if (fstatat(parent_fd, name.c_str(), &before, AT_SYMLINK_NOFOLLOW) != 0 ||
      !SameIdentity(expected, before) || !S_ISSOCK(before.st_mode) ||
      before.st_uid != getuid()) {
    return InternalFailure(
        "unix socket: temporary path changed before permission update");
  }
#if defined(GRSERVE_RPC_TESTING)
  RunSocketBindHookForTesting(
      testing::SocketBindHookStage::kBeforePermissionUpdate, socket_path,
      final_path);
#endif
  // There is no POSIX compare-and-chmod or compare-and-unlink operation for
  // pathname sockets.  The 0700 parent excludes other uids and a random name
  // reduces accidental collision, while repeated identity checks narrow the
  // operation windows.  They cannot exclude a malicious same-uid process,
  // which can enumerate the directory and replace or chmod the temporary
  // entry between any pathname check and operation.  The public final
  // pathname is never passed to chmod.
  struct stat immediately_before {};
  if (fstatat(parent_fd, name.c_str(), &immediately_before,
              AT_SYMLINK_NOFOLLOW) != 0 ||
      !SameIdentity(expected, immediately_before) ||
      !S_ISSOCK(immediately_before.st_mode) ||
      immediately_before.st_uid != getuid()) {
    return InternalFailure(
        "unix socket: temporary path changed before permission update");
  }
  if (fchmodat(parent_fd, name.c_str(), 0600, AT_SYMLINK_NOFOLLOW) != 0) {
    return InternalFailure("unix socket: temporary permission update failed");
  }
  if (fstatat(parent_fd, name.c_str(), confirmed, AT_SYMLINK_NOFOLLOW) != 0 ||
      !SameIdentity(expected, *confirmed) || !S_ISSOCK(confirmed->st_mode) ||
      confirmed->st_uid != getuid()) {
    return InternalFailure(
        "unix socket: temporary path changed during permission update");
  }
  if ((confirmed->st_mode & 0777) != 0600) {
    return InternalFailure("unix socket: mode attestation failed");
  }
  return Status::Ok();
}

Status FillRandomBytes(unsigned char* data, std::size_t size) {
#if defined(__APPLE__)
  arc4random_buf(data, size);
  return Status::Ok();
#elif defined(__linux__)
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t count = getrandom(data + offset, size - offset, 0);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    return InternalFailure("unix socket: random name generation failed");
  }
  return Status::Ok();
#endif
}

constexpr std::string_view kTemporaryNameAlphabet =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

StatusOr<std::string> RandomTemporaryBasename(std::size_t length,
                                              std::size_t attempt,
                                              unsigned char one_byte_start) {
  if (length == 0) {
    return StatusOr<std::string>(Failure(
        ErrorCode::kInternal, "unix socket: no temporary name capacity"));
  }
  std::string basename(length, 'x');
  if (length == 1) {
    // A randomized start plus exhaustive traversal supports a 100-byte final
    // path whose parent leaves only one byte for the temporary basename.
    const std::size_t index =
        (static_cast<std::size_t>(one_byte_start) + attempt) %
        kTemporaryNameAlphabet.size();
    basename[0] = kTemporaryNameAlphabet[index];
    return StatusOr<std::string>(std::move(basename));
  }
  std::array<unsigned char, 32> random{};
  const std::size_t random_size = std::min(length, random.size());
  const Status generated = FillRandomBytes(random.data(), random_size);
  if (!generated.ok()) return StatusOr<std::string>(*generated.error());
  for (std::size_t index = 0; index < length; ++index) {
    const std::size_t alphabet_index =
        random[index % random_size] % kTemporaryNameAlphabet.size();
    basename[index] = kTemporaryNameAlphabet[alphabet_index];
  }
  return StatusOr<std::string>(std::move(basename));
}

struct TemporarySocket {
  UniqueFd socket;
  std::string basename;
  std::string path;
  struct stat identity {};
};

StatusOr<TemporarySocket> BindTemporarySocket(
    int parent_fd, std::string_view parent_path,
    std::string_view final_basename, std::string_view final_path) {
  const std::size_t prefix_length =
      parent_path == "/" ? 1 : parent_path.size() + 1;
  if (prefix_length >= kMaxUnixSocketPathBytes) {
    return StatusOr<TemporarySocket>(Failure(
        ErrorCode::kInternal, "unix socket: no temporary name capacity"));
  }
  const std::size_t available = kMaxUnixSocketPathBytes - prefix_length;
  const std::size_t basename_length = std::min<std::size_t>(available, 32);
  const std::size_t attempts =
      basename_length == 1 ? kTemporaryNameAlphabet.size() : 128;
  unsigned char one_byte_start = 0;
  if (basename_length == 1) {
    const Status generated = FillRandomBytes(&one_byte_start, 1);
    if (!generated.ok()) {
      return StatusOr<TemporarySocket>(*generated.error());
    }
  }

  for (std::size_t attempt = 0; attempt < attempts; ++attempt) {
    auto basename = RandomTemporaryBasename(basename_length, attempt,
                                            one_byte_start);
    if (!basename.ok()) return StatusOr<TemporarySocket>(*basename.error());
    if (basename.value() == final_basename) continue;
    std::string temporary_path =
        parent_path == "/" ? "/" + basename.value()
                           : std::string(parent_path) + "/" + basename.value();

    auto socket_result = NewStreamSocket();
    if (!socket_result.ok()) {
      return StatusOr<TemporarySocket>(*socket_result.error());
    }
    sockaddr_un address{};
    const socklen_t address_length = MakeAddress(temporary_path, &address);
#if defined(GRSERVE_RPC_TESTING)
    RunSocketBindHookForTesting(testing::SocketBindHookStage::kBeforeBind,
                                temporary_path, final_path);
#endif
    if (bind(socket_result.value().get(),
             reinterpret_cast<const sockaddr*>(&address),
             address_length) != 0) {
      if (errno == EADDRINUSE) continue;
      return StatusOr<TemporarySocket>(Failure(
          ErrorCode::kInternal, "unix socket: temporary bind failed"));
    }
#if defined(GRSERVE_RPC_TESTING)
    RunSocketBindHookForTesting(
        testing::SocketBindHookStage::kAfterBindBeforeIdentity, temporary_path,
        final_path);
#endif
    struct stat identity {};
    if (fstatat(parent_fd, basename.value().c_str(), &identity,
                AT_SYMLINK_NOFOLLOW) != 0 ||
        !S_ISSOCK(identity.st_mode) || identity.st_uid != getuid()) {
      return StatusOr<TemporarySocket>(Failure(
          ErrorCode::kInternal,
          "unix socket: could not attest temporary socket inode"));
    }
#if defined(GRSERVE_RPC_TESTING)
    RunSocketBindHookForTesting(
        testing::SocketBindHookStage::kAfterBoundIdentity, temporary_path,
        final_path);
#endif
    TemporarySocket temporary;
    temporary.socket = std::move(socket_result.value());
    temporary.basename = std::move(basename.value());
    temporary.path = std::move(temporary_path);
    temporary.identity = identity;
    return StatusOr<TemporarySocket>(std::move(temporary));
  }
  return StatusOr<TemporarySocket>(Failure(
      ErrorCode::kResourceExhausted,
      "unix socket: could not allocate temporary socket path"));
}

int RenameNoReplace(int parent_fd, const char* temporary_basename,
                    const char* final_basename) {
#if defined(__APPLE__)
  return renameatx_np(parent_fd, temporary_basename, parent_fd, final_basename,
                      RENAME_EXCL);
#elif defined(__linux__)
#if defined(SYS_renameat2)
  return static_cast<int>(
      syscall(SYS_renameat2, parent_fd, temporary_basename, parent_fd,
              final_basename, RENAME_NOREPLACE));
#else
  errno = ENOSYS;
  return -1;
#endif
#endif
}

void UnlinkIfSame(int parent_fd, std::string_view basename,
                  const struct stat& expected) {
  struct stat current {};
  const std::string name(basename);
  if (fstatat(parent_fd, name.c_str(), &current, AT_SYMLINK_NOFOLLOW) == 0 &&
      SameIdentity(current, expected) && S_ISSOCK(current.st_mode) &&
      current.st_uid == getuid()) {
    (void)unlinkat(parent_fd, name.c_str(), 0);
  }
}

}  // namespace

#if defined(GRSERVE_RPC_TESTING)
namespace testing {

void SetSocketProbeHook(SocketProbeHook hook) noexcept {
  SocketProbeHookForTesting().store(hook, std::memory_order_release);
}

void SetSocketBindHook(SocketBindHook hook) noexcept {
  SocketBindHookForTesting().store(hook, std::memory_order_release);
}

}  // namespace testing
#endif

UniqueFd::~UniqueFd() { reset(); }

UniqueFd::UniqueFd(UniqueFd&& other) noexcept : fd_(other.release()) {}

UniqueFd& UniqueFd::operator=(UniqueFd&& other) noexcept {
  if (this != &other) {
    reset(other.release());
  }
  return *this;
}

int UniqueFd::release() noexcept {
  const int released = fd_;
  fd_ = -1;
  return released;
}

void UniqueFd::reset(int replacement) noexcept {
  if (replacement == fd_) return;
  if (fd_ >= 0) {
    close(fd_);
  }
  fd_ = replacement;
}

Status ValidateUnixSocketPath(std::string_view path) {
  if (path.empty() || path.front() != '/') {
    return Status{Failure(ErrorCode::kInvalidArgument,
                          "unix socket: path must be absolute")};
  }
  if (path.find('\0') != std::string_view::npos) {
    return Status{Failure(ErrorCode::kInvalidArgument,
                          "unix socket: path contains NUL")};
  }
  constexpr std::size_t kPlatformCapacity = sizeof(sockaddr_un{}.sun_path) - 1;
  static_assert(kPlatformCapacity >= kMaxUnixSocketPathBytes,
                "GRServe requires a 100-byte Unix socket path capacity");
  if (path.size() > kMaxUnixSocketPathBytes) {
    return Status{Failure(ErrorCode::kInvalidArgument,
                          "unix socket: path exceeds 100-byte limit")};
  }
  return Status::Ok();
}

UnixListener::UnixListener(UniqueFd socket, UniqueFd parent,
                           std::string basename, dev_t socket_device,
                           ino_t socket_inode) noexcept
    : socket_(std::move(socket)),
      parent_(std::move(parent)),
      basename_(std::move(basename)),
      socket_device_(socket_device),
      socket_inode_(socket_inode),
      owns_path_(true) {}

UnixListener::~UnixListener() { Cleanup(); }

UnixListener::UnixListener(UnixListener&& other) noexcept
    : socket_(std::move(other.socket_)),
      parent_(std::move(other.parent_)),
      basename_(std::move(other.basename_)),
      socket_device_(other.socket_device_),
      socket_inode_(other.socket_inode_),
      owns_path_(other.owns_path_) {
  other.owns_path_ = false;
  other.socket_device_ = 0;
  other.socket_inode_ = 0;
}

UnixListener& UnixListener::operator=(UnixListener&& other) noexcept {
  if (this != &other) {
    Cleanup();
    socket_ = std::move(other.socket_);
    parent_ = std::move(other.parent_);
    basename_ = std::move(other.basename_);
    socket_device_ = other.socket_device_;
    socket_inode_ = other.socket_inode_;
    owns_path_ = other.owns_path_;
    other.owns_path_ = false;
    other.socket_device_ = 0;
    other.socket_inode_ = 0;
  }
  return *this;
}

void UnixListener::Cleanup() noexcept {
  if (owns_path_ && parent_ && !basename_.empty()) {
    struct stat current {};
    if (fstatat(parent_.get(), basename_.c_str(), &current,
                AT_SYMLINK_NOFOLLOW) == 0 &&
        S_ISSOCK(current.st_mode) && current.st_uid == getuid() &&
        current.st_dev == socket_device_ && current.st_ino == socket_inode_) {
      (void)unlinkat(parent_.get(), basename_.c_str(), 0);
    }
  }
  owns_path_ = false;
  socket_.reset();
  parent_.reset();
  basename_.clear();
  socket_device_ = 0;
  socket_inode_ = 0;
}

StatusOr<UnixListener> UnixListener::Bind(std::string path, int backlog) {
  // Serializing the discovery/probe/bind/listen sequence closes the in-process
  // startup window where a second binder could mistake a not-yet-listening
  // socket for stale. The parent directory flock provides the same startup
  // serialization across cooperating GRServe processes.
  std::lock_guard<std::mutex> lifecycle_lock(BindMutex());
  const Status path_status = ValidateUnixSocketPath(path);
  if (!path_status.ok()) {
    return StatusOr<UnixListener>(*path_status.error());
  }
  if (backlog <= 0) {
    return StatusOr<UnixListener>(Failure(
        ErrorCode::kInvalidArgument, "unix socket: backlog must be positive"));
  }
  auto parts = SplitPath(path);
  if (!parts.ok()) return StatusOr<UnixListener>(*parts.error());

  auto parent_result = OpenParentWithoutSymlinks(parts.value().parent);
  if (!parent_result.ok()) {
    return StatusOr<UnixListener>(*parent_result.error());
  }
  UniqueFd parent_fd(std::move(parent_result.value()));
  struct stat parent_stat {};
  if (fstat(parent_fd.get(), &parent_stat) != 0 ||
      !S_ISDIR(parent_stat.st_mode) || parent_stat.st_uid != getuid() ||
      (parent_stat.st_mode & 0777) != 0700) {
    return StatusOr<UnixListener>(Failure(
        ErrorCode::kInvalidArgument,
        "unix socket: parent must be owned by current user with mode 0700"));
  }
  int lock_result;
  do {
    lock_result = flock(parent_fd.get(), LOCK_EX);
  } while (lock_result != 0 && errno == EINTR);
  if (lock_result != 0) {
    return StatusOr<UnixListener>(Failure(
        ErrorCode::kInternal, "unix socket: parent lock failed"));
  }
  const Status named_parent =
      VerifyNamedParent(parts.value().parent, parent_stat);
  if (!named_parent.ok()) {
    return StatusOr<UnixListener>(*named_parent.error());
  }

  const Status prepared = PrepareSocketPath(
      parent_fd.get(), parts.value().basename, path);
  if (!prepared.ok()) return StatusOr<UnixListener>(*prepared.error());
  const Status parent_before_bind =
      VerifyNamedParent(parts.value().parent, parent_stat);
  if (!parent_before_bind.ok()) {
    return StatusOr<UnixListener>(*parent_before_bind.error());
  }

  auto temporary_result = BindTemporarySocket(
      parent_fd.get(), parts.value().parent, parts.value().basename, path);
  if (!temporary_result.ok()) {
    return StatusOr<UnixListener>(*temporary_result.error());
  }
  TemporarySocket temporary(std::move(temporary_result.value()));
  struct stat confirmed_socket {};
  const Status permissions = SetBoundSocketOwnerOnly(
      parent_fd.get(), temporary.basename, temporary.identity,
      &confirmed_socket, temporary.path, path);
  if (!permissions.ok()) {
    UnlinkIfSame(parent_fd.get(), temporary.basename, temporary.identity);
    return StatusOr<UnixListener>(*permissions.error());
  }
  const Status parent_after_bind =
      VerifyNamedParent(parts.value().parent, parent_stat);
  if (!parent_after_bind.ok()) {
    UnlinkIfSame(parent_fd.get(), temporary.basename, temporary.identity);
    return StatusOr<UnixListener>(*parent_after_bind.error());
  }
  if (listen(temporary.socket.get(), backlog) != 0) {
    UnlinkIfSame(parent_fd.get(), temporary.basename, temporary.identity);
    return StatusOr<UnixListener>(Failure(
        ErrorCode::kInternal, "unix socket: listen failed"));
  }
#if defined(GRSERVE_RPC_TESTING)
  RunSocketBindHookForTesting(testing::SocketBindHookStage::kBeforeInstall,
                              temporary.path, path);
#endif
  struct stat before_install {};
  if (fstatat(parent_fd.get(), temporary.basename.c_str(), &before_install,
              AT_SYMLINK_NOFOLLOW) != 0 ||
      !SameIdentity(confirmed_socket, before_install) ||
      !S_ISSOCK(before_install.st_mode) || before_install.st_uid != getuid() ||
      (before_install.st_mode & 0777) != 0600) {
    UnlinkIfSame(parent_fd.get(), temporary.basename, confirmed_socket);
    return StatusOr<UnixListener>(Failure(
        ErrorCode::kInternal,
        "unix socket: temporary path changed before install"));
  }
  if (RenameNoReplace(parent_fd.get(), temporary.basename.c_str(),
                      parts.value().basename.c_str()) != 0) {
    const int install_error = errno;
    UnlinkIfSame(parent_fd.get(), temporary.basename, temporary.identity);
    if (install_error == EEXIST) {
      return StatusOr<UnixListener>(Failure(
          ErrorCode::kResourceExhausted,
          "unix socket: final path occupied during install"));
    }
    return StatusOr<UnixListener>(Failure(
        ErrorCode::kInternal, "unix socket: atomic install failed"));
  }
#if defined(GRSERVE_RPC_TESTING)
  RunSocketBindHookForTesting(testing::SocketBindHookStage::kAfterInstall,
                              temporary.path, path);
#endif
  struct stat installed {};
  if (fstatat(parent_fd.get(), parts.value().basename.c_str(), &installed,
              AT_SYMLINK_NOFOLLOW) != 0 ||
      !SameIdentity(confirmed_socket, installed) ||
      !S_ISSOCK(installed.st_mode) || installed.st_uid != getuid() ||
      (installed.st_mode & 0777) != 0600) {
    UnlinkIfSame(parent_fd.get(), parts.value().basename, confirmed_socket);
    return StatusOr<UnixListener>(Failure(
        ErrorCode::kInternal,
        "unix socket: final path changed after install"));
  }
  const Status parent_after_install =
      VerifyNamedParent(parts.value().parent, parent_stat);
  if (!parent_after_install.ok()) {
    UnlinkIfSame(parent_fd.get(), parts.value().basename, confirmed_socket);
    return StatusOr<UnixListener>(*parent_after_install.error());
  }
  if (flock(parent_fd.get(), LOCK_UN) != 0) {
    UnlinkIfSame(parent_fd.get(), parts.value().basename, confirmed_socket);
    return StatusOr<UnixListener>(Failure(
        ErrorCode::kInternal, "unix socket: parent unlock failed"));
  }
  return StatusOr<UnixListener>(UnixListener(
      std::move(temporary.socket), std::move(parent_fd),
      std::move(parts.value().basename), confirmed_socket.st_dev,
      confirmed_socket.st_ino));
}

StatusOr<UniqueFd> UnixListener::Accept() const {
  if (!socket_) {
    return StatusOr<UniqueFd>(Failure(
        ErrorCode::kInvalidArgument, "unix socket: listener is closed"));
  }
  int accepted;
  do {
#if defined(__linux__) && defined(SOCK_CLOEXEC)
    accepted = accept4(socket_.get(), nullptr, nullptr, SOCK_CLOEXEC);
#else
    // Darwin has no accept4; set FD_CLOEXEC immediately after accept.
    accepted = accept(socket_.get(), nullptr, nullptr);
#endif
  } while (accepted < 0 && errno == EINTR);
  if (accepted < 0) {
    return StatusOr<UniqueFd>(Failure(
        ErrorCode::kInternal, "unix socket: accept failed"));
  }
  UniqueFd result(accepted);
  const Status policy = SetSocketPolicy(result.get());
  if (!policy.ok()) return StatusOr<UniqueFd>(*policy.error());
  return StatusOr<UniqueFd>(std::move(result));
}

StatusOr<UniqueFd> ConnectUnixSocket(std::string_view path) {
  const Status path_status = ValidateUnixSocketPath(path);
  if (!path_status.ok()) return StatusOr<UniqueFd>(*path_status.error());
  auto socket_result = NewStreamSocket();
  if (!socket_result.ok()) return socket_result;
  sockaddr_un address{};
  const socklen_t address_length = MakeAddress(path, &address);
  int result;
  do {
    result = connect(socket_result.value().get(),
                     reinterpret_cast<const sockaddr*>(&address),
                     address_length);
  } while (result < 0 && errno == EINTR);
  if (result != 0 && errno != EISCONN) {
    return StatusOr<UniqueFd>(Failure(
        ErrorCode::kInternal, "unix socket: connect failed"));
  }
  return socket_result;
}

}  // namespace grserve::rpc
