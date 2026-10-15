#include "rpc/unix_socket.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <barrier>
#include <cstddef>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

namespace grserve::rpc {
namespace {

class TempDir {
 public:
  TempDir() {
    std::array<char, 64> pattern{};
    std::strcpy(pattern.data(), "/tmp/grserve-test.XXXXXX");
    char* created = mkdtemp(pattern.data());
    EXPECT_NE(created, nullptr);
    if (created != nullptr) {
      path_ = created;
    }
  }
  ~TempDir() {
    if (!path_.empty()) {
      EXPECT_EQ(rmdir(path_.c_str()), 0) << strerror(errno);
    }
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  const std::string& path() const { return path_; }
  std::string SocketPath(std::string name = "engine.sock") const {
    return path_ + "/" + name;
  }

 private:
  std::string path_;
};

socklen_t UnixAddress(const std::string& path, sockaddr_un* address) {
  std::memset(address, 0, sizeof(*address));
  address->sun_family = AF_UNIX;
  std::memcpy(address->sun_path, path.data(), path.size());
  address->sun_path[path.size()] = '\0';
  const auto size = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + path.size() + 1);
#if defined(__APPLE__)
  address->sun_len = static_cast<unsigned char>(size);
#endif
  return size;
}

int BindRawSocket(const std::string& path, bool listen_socket) {
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);
  if (fd < 0) return fd;
  sockaddr_un address{};
  const socklen_t size = UnixAddress(path, &address);
  EXPECT_EQ(bind(fd, reinterpret_cast<const sockaddr*>(&address), size), 0)
      << strerror(errno);
  if (listen_socket) {
    EXPECT_EQ(listen(fd, 8), 0) << strerror(errno);
  }
  return fd;
}

class ScopedSocketProbeHook {
 public:
  explicit ScopedSocketProbeHook(testing::SocketProbeHook hook) {
    testing::SetSocketProbeHook(hook);
  }
  ~ScopedSocketProbeHook() { testing::SetSocketProbeHook(nullptr); }

  ScopedSocketProbeHook(const ScopedSocketProbeHook&) = delete;
  ScopedSocketProbeHook& operator=(const ScopedSocketProbeHook&) = delete;
};

class ScopedSocketBindHook {
 public:
  explicit ScopedSocketBindHook(testing::SocketBindHook hook) {
    testing::SetSocketBindHook(hook);
  }
  ~ScopedSocketBindHook() { testing::SetSocketBindHook(nullptr); }

  ScopedSocketBindHook(const ScopedSocketBindHook&) = delete;
  ScopedSocketBindHook& operator=(const ScopedSocketBindHook&) = delete;
};

class ScopedUmask {
 public:
  explicit ScopedUmask(mode_t mask) : original_(umask(mask)) {}
  ~ScopedUmask() { umask(original_); }

  ScopedUmask(const ScopedUmask&) = delete;
  ScopedUmask& operator=(const ScopedUmask&) = delete;

 private:
  mode_t original_;
};

int g_probe_replacement_fd = -1;
struct stat g_probe_replacement_stat {};
std::barrier<>* g_bind_barrier = nullptr;
struct stat g_bind_replacement_stat {};
std::string g_bind_replacement_path;
std::string g_observed_temporary_path;
std::string g_observed_final_path;
int g_install_competitor_fd = -1;
struct stat g_install_competitor_stat {};

void RecordTemporaryPaths(testing::SocketBindHookStage stage,
                          std::string_view socket_path,
                          std::string_view final_path) {
  if (stage != testing::SocketBindHookStage::kAfterBindBeforeIdentity) return;
  g_observed_temporary_path = std::string(socket_path);
  g_observed_final_path = std::string(final_path);
}

void CreateFinalFileAfterTemporaryBind(
    testing::SocketBindHookStage stage, std::string_view socket_path,
    std::string_view final_path) {
  if (stage != testing::SocketBindHookStage::kAfterBindBeforeIdentity) return;
  EXPECT_NE(socket_path, final_path);
  g_observed_temporary_path = std::string(socket_path);
  g_observed_final_path = std::string(final_path);
  const int fd = open(std::string(final_path).c_str(),
                      O_CREAT | O_EXCL | O_WRONLY, 0600);
  ASSERT_GE(fd, 0) << strerror(errno);
  constexpr std::string_view kContents = "final-competitor";
  EXPECT_EQ(write(fd, kContents.data(), kContents.size()),
            static_cast<ssize_t>(kContents.size()));
  EXPECT_EQ(fchmod(fd, 0640), 0) << strerror(errno);
  EXPECT_EQ(fstat(fd, &g_install_competitor_stat), 0) << strerror(errno);
  close(fd);
}

void CreateActiveFinalBeforeInstall(testing::SocketBindHookStage stage,
                                    std::string_view socket_path,
                                    std::string_view final_path) {
  if (stage != testing::SocketBindHookStage::kBeforeInstall) return;
  EXPECT_NE(socket_path, final_path);
  g_install_competitor_fd = BindRawSocket(std::string(final_path), true);
  if (g_install_competitor_fd >= 0) {
    EXPECT_EQ(lstat(std::string(final_path).c_str(),
                    &g_install_competitor_stat),
              0)
        << strerror(errno);
  }
}

void ReplaceFinalAfterInstall(testing::SocketBindHookStage stage,
                              std::string_view socket_path,
                              std::string_view final_path) {
  if (stage != testing::SocketBindHookStage::kAfterInstall) return;
  EXPECT_NE(socket_path, final_path);
  EXPECT_EQ(unlink(std::string(final_path).c_str()), 0) << strerror(errno);
  const int fd = open(std::string(final_path).c_str(),
                      O_CREAT | O_EXCL | O_WRONLY, 0600);
  ASSERT_GE(fd, 0) << strerror(errno);
  EXPECT_EQ(fchmod(fd, 0640), 0) << strerror(errno);
  EXPECT_EQ(fstat(fd, &g_install_competitor_stat), 0) << strerror(errno);
  close(fd);
}

void PauseBeforeBind(testing::SocketBindHookStage stage,
                     std::string_view /*socket_path*/,
                     std::string_view /*final_path*/) {
  if (stage != testing::SocketBindHookStage::kBeforeBind) return;
  g_bind_barrier->arrive_and_wait();
  g_bind_barrier->arrive_and_wait();
}

void ReplaceSocketAfterBoundIdentity(testing::SocketBindHookStage stage,
                                     std::string_view socket_path,
                                     std::string_view final_path) {
  if (stage != testing::SocketBindHookStage::kBeforePermissionUpdate) return;
  EXPECT_NE(socket_path, final_path);
  g_bind_replacement_path = std::string(socket_path);
  EXPECT_EQ(unlink(g_bind_replacement_path.c_str()), 0) << strerror(errno);
  const int fd = open(g_bind_replacement_path.c_str(),
                      O_CREAT | O_EXCL | O_WRONLY, 0600);
  ASSERT_GE(fd, 0) << strerror(errno);
  constexpr std::string_view kContents = "bind-replacement";
  EXPECT_EQ(write(fd, kContents.data(), kContents.size()),
            static_cast<ssize_t>(kContents.size()));
  EXPECT_EQ(fchmod(fd, 0640), 0) << strerror(errno);
  EXPECT_EQ(fstat(fd, &g_bind_replacement_stat), 0) << strerror(errno);
  close(fd);
}

void ReplaceTemporaryBeforeFirstIdentity(
    testing::SocketBindHookStage stage, std::string_view socket_path,
    std::string_view final_path) {
  if (stage != testing::SocketBindHookStage::kAfterBindBeforeIdentity) return;
  EXPECT_NE(socket_path, final_path);
  g_bind_replacement_path = std::string(socket_path);
  EXPECT_EQ(unlink(g_bind_replacement_path.c_str()), 0) << strerror(errno);
  const int fd = open(g_bind_replacement_path.c_str(),
                      O_CREAT | O_EXCL | O_WRONLY, 0600);
  ASSERT_GE(fd, 0) << strerror(errno);
  constexpr std::string_view kContents = "preidentity-replacement";
  EXPECT_EQ(write(fd, kContents.data(), kContents.size()),
            static_cast<ssize_t>(kContents.size()));
  EXPECT_EQ(fchmod(fd, 0640), 0) << strerror(errno);
  EXPECT_EQ(fstat(fd, &g_bind_replacement_stat), 0) << strerror(errno);
  close(fd);
}

void ReplaceTemporaryBeforeInstall(testing::SocketBindHookStage stage,
                                   std::string_view socket_path,
                                   std::string_view final_path) {
  if (stage != testing::SocketBindHookStage::kBeforeInstall) return;
  EXPECT_NE(socket_path, final_path);
  g_bind_replacement_path = std::string(socket_path);
  EXPECT_EQ(unlink(g_bind_replacement_path.c_str()), 0) << strerror(errno);
  const int fd = open(g_bind_replacement_path.c_str(),
                      O_CREAT | O_EXCL | O_WRONLY, 0600);
  ASSERT_GE(fd, 0) << strerror(errno);
  EXPECT_EQ(fchmod(fd, 0640), 0) << strerror(errno);
  EXPECT_EQ(fstat(fd, &g_bind_replacement_stat), 0) << strerror(errno);
  close(fd);
}

void WidenTemporaryBeforeInstall(testing::SocketBindHookStage stage,
                                 std::string_view socket_path,
                                 std::string_view final_path) {
  if (stage != testing::SocketBindHookStage::kBeforeInstall) return;
  EXPECT_NE(socket_path, final_path);
  g_observed_temporary_path = std::string(socket_path);
  EXPECT_EQ(chmod(g_observed_temporary_path.c_str(), 0666), 0)
      << strerror(errno);
}

void WidenFinalAfterInstall(testing::SocketBindHookStage stage,
                            std::string_view socket_path,
                            std::string_view final_path) {
  if (stage != testing::SocketBindHookStage::kAfterInstall) return;
  EXPECT_NE(socket_path, final_path);
  EXPECT_EQ(chmod(std::string(final_path).c_str(), 0666), 0) << strerror(errno);
}

int ReplaceProbedSocketWithFileAndReturnEnoent(std::string_view path) {
  EXPECT_EQ(unlink(std::string(path).c_str()), 0) << strerror(errno);
  const int fd = open(std::string(path).c_str(), O_CREAT | O_EXCL | O_WRONLY,
                      0600);
  EXPECT_GE(fd, 0) << strerror(errno);
  if (fd >= 0) {
    constexpr std::string_view kContents = "replacement-file";
    EXPECT_EQ(write(fd, kContents.data(), kContents.size()),
              static_cast<ssize_t>(kContents.size()));
    close(fd);
    EXPECT_EQ(lstat(std::string(path).c_str(), &g_probe_replacement_stat), 0)
        << strerror(errno);
  }
  errno = ENOENT;
  return -1;
}

int ReplaceProbedSocketWithListenerAndReturnEnoent(std::string_view path) {
  EXPECT_EQ(unlink(std::string(path).c_str()), 0) << strerror(errno);
  g_probe_replacement_fd = BindRawSocket(std::string(path), true);
  if (g_probe_replacement_fd >= 0) {
    EXPECT_EQ(lstat(std::string(path).c_str(), &g_probe_replacement_stat), 0)
        << strerror(errno);
  }
  errno = ENOENT;
  return -1;
}

int ReplaceProbedSocketWithFileAndReturnEacces(std::string_view path) {
  const int result = ReplaceProbedSocketWithFileAndReturnEnoent(path);
  errno = EACCES;
  return result;
}

void ExpectError(const Error* error, ErrorCode code, std::string_view message) {
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code(), code);
  EXPECT_EQ(error->message(), message);
}

std::size_t OpenFdCount() {
  std::size_t count = 0;
  const int limit = getdtablesize();
  for (int fd = 0; fd < limit; ++fd) {
    if (fcntl(fd, F_GETFD) != -1 || errno != EBADF) {
      ++count;
    }
  }
  return count;
}

TEST(UniqueFdTest, MoveAssignmentClosesOldFdAndNeverDoubleCloses) {
  int first_pipe[2];
  int second_pipe[2];
  ASSERT_EQ(pipe(first_pipe), 0);
  ASSERT_EQ(pipe(second_pipe), 0);
  close(first_pipe[1]);
  close(second_pipe[1]);
  const int old_destination = second_pipe[0];
  {
    UniqueFd source(first_pipe[0]);
    UniqueFd destination(old_destination);
    destination = std::move(source);
    EXPECT_EQ(source.get(), -1);
    EXPECT_EQ(destination.get(), first_pipe[0]);
    errno = 0;
    EXPECT_EQ(fcntl(old_destination, F_GETFD), -1);
    EXPECT_EQ(errno, EBADF);
  }
  errno = 0;
  EXPECT_EQ(fcntl(first_pipe[0], F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
}

TEST(UniqueFdTest, ResetToSameDescriptorIsANoop) {
  int descriptors[2];
  ASSERT_EQ(pipe(descriptors), 0);
  close(descriptors[1]);
  UniqueFd descriptor(descriptors[0]);
  descriptor.reset(descriptor.get());
  EXPECT_NE(fcntl(descriptor.get(), F_GETFD), -1);
}

TEST(UnixSocketPathTest, RejectsRelativeAndEmbeddedNulPaths) {
  auto relative = UnixListener::Bind("relative.sock");
  ASSERT_FALSE(relative.ok());
  ExpectError(relative.error(), ErrorCode::kInvalidArgument,
              "unix socket: path must be absolute");

  std::string embedded("/tmp/grserve\0hidden", 20);
  auto nul = UnixListener::Bind(embedded);
  ASSERT_FALSE(nul.ok());
  ExpectError(nul.error(), ErrorCode::kInvalidArgument,
              "unix socket: path contains NUL");
}

TEST(UnixSocketPathTest, Accepts100BytesAndRejects101BytesOnMacAndLinux) {
  static_assert(sizeof(sockaddr_un{}.sun_path) >= 101);
  TempDir directory;
  const std::size_t prefix = directory.path().size() + 1;
  const std::string path100 =
      directory.path() + "/" + std::string(100 - prefix, 'a');
  ASSERT_EQ(path100.size(), 100U);
  auto listener = UnixListener::Bind(path100);
  ASSERT_TRUE(listener.ok()) << listener.error()->message();
  listener.value() = UnixListener{};

  const std::string path101 =
      directory.path() + "/" + std::string(101 - prefix, 'b');
  ASSERT_EQ(path101.size(), 101U);
  auto too_long = UnixListener::Bind(path101);
  ASSERT_FALSE(too_long.ok());
  ExpectError(too_long.error(), ErrorCode::kInvalidArgument,
              "unix socket: path exceeds 100-byte limit");
}

TEST(UnixListenerTest, CreatesOwnerOnlyBlockingCloexecListenerAndConnection) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  auto listener = UnixListener::Bind(path, 16);
  ASSERT_TRUE(listener.ok()) << listener.error()->message();
  EXPECT_NE(fcntl(listener.value().get(), F_GETFD) & FD_CLOEXEC, 0);
  EXPECT_EQ(fcntl(listener.value().get(), F_GETFL) & O_NONBLOCK, 0);
  struct stat socket_stat {};
  ASSERT_EQ(lstat(path.c_str(), &socket_stat), 0);
  EXPECT_TRUE(S_ISSOCK(socket_stat.st_mode));
  EXPECT_EQ(socket_stat.st_uid, getuid());
  EXPECT_EQ(socket_stat.st_mode & 0777, 0600);

  auto client = ConnectUnixSocket(path);
  ASSERT_TRUE(client.ok()) << client.error()->message();
  auto accepted = listener.value().Accept();
  ASSERT_TRUE(accepted.ok()) << accepted.error()->message();
  EXPECT_NE(fcntl(client.value().get(), F_GETFD) & FD_CLOEXEC, 0);
  EXPECT_NE(fcntl(accepted.value().get(), F_GETFD) & FD_CLOEXEC, 0);
  EXPECT_EQ(fcntl(client.value().get(), F_GETFL) & O_NONBLOCK, 0);
  EXPECT_EQ(fcntl(accepted.value().get(), F_GETFL) & O_NONBLOCK, 0);
}

TEST(UnixListenerTest, BindDoesNotChangeProcessUmaskForConcurrentCreators) {
  TempDir directory;
  const std::string socket_path = directory.SocketPath();
  const std::string file_path = directory.SocketPath("concurrent-file");
  ScopedUmask process_umask(0022);
  std::barrier rendezvous(2);
  g_bind_barrier = &rendezvous;
  ScopedSocketBindHook hook(PauseBeforeBind);
  std::optional<UnixListener> listener;
  bool bind_succeeded = false;
  std::thread binder([&] {
    auto result = UnixListener::Bind(socket_path);
    bind_succeeded = result.ok();
    if (result.ok()) listener.emplace(std::move(result.value()));
  });

  rendezvous.arrive_and_wait();
  const int file = open(file_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0666);
  const int create_error = errno;
  struct stat created {};
  const int stat_result = file >= 0 ? fstat(file, &created) : -1;
  if (file >= 0) close(file);
  rendezvous.arrive_and_wait();
  binder.join();
  g_bind_barrier = nullptr;

  ASSERT_TRUE(bind_succeeded);
  ASSERT_GE(file, 0) << strerror(create_error);
  ASSERT_EQ(stat_result, 0);
  EXPECT_EQ(created.st_mode & 0777, 0644);
  listener.reset();
  ASSERT_EQ(unlink(file_path.c_str()), 0);
}

TEST(UnixListenerTest, PermissionStepDoesNotTouchReplacementPath) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  ScopedSocketBindHook hook(ReplaceSocketAfterBoundIdentity);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal,
              "unix socket: temporary path changed before permission update");
  EXPECT_NE(access(path.c_str(), F_OK), 0);
  ASSERT_FALSE(g_bind_replacement_path.empty());
  struct stat replacement {};
  ASSERT_EQ(lstat(g_bind_replacement_path.c_str(), &replacement), 0);
  EXPECT_TRUE(S_ISREG(replacement.st_mode));
  EXPECT_EQ(replacement.st_dev, g_bind_replacement_stat.st_dev);
  EXPECT_EQ(replacement.st_ino, g_bind_replacement_stat.st_ino);
  EXPECT_EQ(replacement.st_mode & 0777, 0640);
  const int fd = open(g_bind_replacement_path.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  std::array<char, 32> contents{};
  const ssize_t length = read(fd, contents.data(), contents.size());
  close(fd);
  ASSERT_EQ(length, 16);
  EXPECT_EQ(std::string_view(contents.data(), static_cast<std::size_t>(length)),
            "bind-replacement");
  ASSERT_EQ(unlink(g_bind_replacement_path.c_str()), 0);
  g_bind_replacement_path.clear();
}

TEST(UnixListenerTest, FirstIdentityCheckPreservesTemporaryReplacement) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  g_bind_replacement_path.clear();
  ScopedSocketBindHook hook(ReplaceTemporaryBeforeFirstIdentity);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal,
              "unix socket: could not attest temporary socket inode");
  EXPECT_NE(access(path.c_str(), F_OK), 0);
  ASSERT_FALSE(g_bind_replacement_path.empty());
  struct stat replacement {};
  ASSERT_EQ(lstat(g_bind_replacement_path.c_str(), &replacement), 0);
  EXPECT_TRUE(S_ISREG(replacement.st_mode));
  EXPECT_EQ(replacement.st_dev, g_bind_replacement_stat.st_dev);
  EXPECT_EQ(replacement.st_ino, g_bind_replacement_stat.st_ino);
  EXPECT_EQ(replacement.st_mode & 0777, 0640);
  const int fd = open(g_bind_replacement_path.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  std::array<char, 32> contents{};
  const ssize_t length = read(fd, contents.data(), contents.size());
  close(fd);
  ASSERT_EQ(length, 23);
  EXPECT_EQ(std::string_view(contents.data(), static_cast<std::size_t>(length)),
            "preidentity-replacement");
  ASSERT_EQ(unlink(g_bind_replacement_path.c_str()), 0);
  g_bind_replacement_path.clear();
}

TEST(UnixListenerTest, FinalFileCreatedAfterTemporaryBindWinsInstall) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  g_observed_temporary_path.clear();
  ScopedSocketBindHook hook(CreateFinalFileAfterTemporaryBind);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kResourceExhausted,
              "unix socket: final path occupied during install");
  ASSERT_FALSE(g_observed_temporary_path.empty());
  EXPECT_NE(access(g_observed_temporary_path.c_str(), F_OK), 0);
  struct stat competitor {};
  ASSERT_EQ(lstat(path.c_str(), &competitor), 0);
  EXPECT_TRUE(S_ISREG(competitor.st_mode));
  EXPECT_EQ(competitor.st_dev, g_install_competitor_stat.st_dev);
  EXPECT_EQ(competitor.st_ino, g_install_competitor_stat.st_ino);
  EXPECT_EQ(competitor.st_mode & 0777, 0640);
  const int fd = open(path.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  std::array<char, 32> contents{};
  const ssize_t length = read(fd, contents.data(), contents.size());
  close(fd);
  ASSERT_EQ(length, 16);
  EXPECT_EQ(std::string_view(contents.data(), static_cast<std::size_t>(length)),
            "final-competitor");
  ASSERT_EQ(unlink(path.c_str()), 0);
}

TEST(UnixListenerTest, ActiveListenerCreatedAfterStaleUnlinkWinsInstall) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  const int stale = BindRawSocket(path, false);
  ASSERT_GE(stale, 0);
  close(stale);
  g_install_competitor_fd = -1;
  ScopedSocketBindHook hook(CreateActiveFinalBeforeInstall);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kResourceExhausted,
              "unix socket: final path occupied during install");
  ASSERT_GE(g_install_competitor_fd, 0);
  struct stat competitor {};
  ASSERT_EQ(lstat(path.c_str(), &competitor), 0);
  EXPECT_TRUE(S_ISSOCK(competitor.st_mode));
  EXPECT_EQ(competitor.st_dev, g_install_competitor_stat.st_dev);
  EXPECT_EQ(competitor.st_ino, g_install_competitor_stat.st_ino);
  auto client = ConnectUnixSocket(path);
  EXPECT_TRUE(client.ok());
  close(g_install_competitor_fd);
  g_install_competitor_fd = -1;
  ASSERT_EQ(unlink(path.c_str()), 0);
}

TEST(UnixListenerTest, FileCreatedAfterStaleUnlinkWinsInstall) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  const int stale = BindRawSocket(path, false);
  ASSERT_GE(stale, 0);
  close(stale);
  g_observed_temporary_path.clear();
  ScopedSocketBindHook hook(CreateFinalFileAfterTemporaryBind);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kResourceExhausted,
              "unix socket: final path occupied during install");
  ASSERT_FALSE(g_observed_temporary_path.empty());
  EXPECT_NE(access(g_observed_temporary_path.c_str(), F_OK), 0);
  struct stat competitor {};
  ASSERT_EQ(lstat(path.c_str(), &competitor), 0);
  EXPECT_TRUE(S_ISREG(competitor.st_mode));
  EXPECT_EQ(competitor.st_dev, g_install_competitor_stat.st_dev);
  EXPECT_EQ(competitor.st_ino, g_install_competitor_stat.st_ino);
  EXPECT_EQ(competitor.st_mode & 0777, 0640);
  ASSERT_EQ(unlink(path.c_str()), 0);
}

TEST(UnixListenerTest, TemporaryReplacementBeforeRenameIsNotInstalled) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  g_bind_replacement_path.clear();
  ScopedSocketBindHook hook(ReplaceTemporaryBeforeInstall);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal,
              "unix socket: temporary path changed before install");
  EXPECT_NE(access(path.c_str(), F_OK), 0);
  ASSERT_FALSE(g_bind_replacement_path.empty());
  struct stat replacement {};
  ASSERT_EQ(lstat(g_bind_replacement_path.c_str(), &replacement), 0);
  EXPECT_TRUE(S_ISREG(replacement.st_mode));
  EXPECT_EQ(replacement.st_dev, g_bind_replacement_stat.st_dev);
  EXPECT_EQ(replacement.st_ino, g_bind_replacement_stat.st_ino);
  EXPECT_EQ(replacement.st_mode & 0777, 0640);
  ASSERT_EQ(unlink(g_bind_replacement_path.c_str()), 0);
  g_bind_replacement_path.clear();
}

TEST(UnixListenerTest, PermissionWideningBeforeRenameIsRejected) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  g_observed_temporary_path.clear();
  ScopedSocketBindHook hook(WidenTemporaryBeforeInstall);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal,
              "unix socket: temporary path changed before install");
  EXPECT_NE(access(path.c_str(), F_OK), 0);
  ASSERT_FALSE(g_observed_temporary_path.empty());
  EXPECT_NE(access(g_observed_temporary_path.c_str(), F_OK), 0);
}

TEST(UnixListenerTest, PermissionWideningAfterRenameIsRejectedAndCleaned) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  ScopedSocketBindHook hook(WidenFinalAfterInstall);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal,
              "unix socket: final path changed after install");
  EXPECT_NE(access(path.c_str(), F_OK), 0);
}

TEST(UnixListenerTest, ReplacementAfterInstallSurvivesListenerCleanup) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  ScopedSocketBindHook hook(ReplaceFinalAfterInstall);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal,
              "unix socket: final path changed after install");
  struct stat replacement {};
  ASSERT_EQ(lstat(path.c_str(), &replacement), 0);
  EXPECT_TRUE(S_ISREG(replacement.st_mode));
  EXPECT_EQ(replacement.st_dev, g_install_competitor_stat.st_dev);
  EXPECT_EQ(replacement.st_ino, g_install_competitor_stat.st_ino);
  EXPECT_EQ(replacement.st_mode & 0777, 0640);
  struct stat after_cleanup {};
  ASSERT_EQ(lstat(path.c_str(), &after_cleanup), 0);
  EXPECT_EQ(after_cleanup.st_dev, replacement.st_dev);
  EXPECT_EQ(after_cleanup.st_ino, replacement.st_ino);
  ASSERT_EQ(unlink(path.c_str()), 0);
}

TEST(UnixListenerTest, Supports100BytePathWithOneByteFinalBasename) {
  TempDir directory;
  const std::size_t parent_component_length =
      98 - directory.path().size() - 1;
  const std::string parent =
      directory.path() + "/" + std::string(parent_component_length, 'p');
  ASSERT_EQ(parent.size(), 98U);
  ASSERT_EQ(mkdir(parent.c_str(), 0700), 0);
  const std::string path = parent + "/s";
  ASSERT_EQ(path.size(), 100U);
  g_observed_temporary_path.clear();
  g_observed_final_path.clear();
  ScopedSocketBindHook hook(RecordTemporaryPaths);

  auto result = UnixListener::Bind(path);

  ASSERT_TRUE(result.ok()) << result.error()->message();
  EXPECT_EQ(g_observed_final_path, path);
  EXPECT_NE(g_observed_temporary_path, path);
  EXPECT_LE(g_observed_temporary_path.size(), 100U);
  EXPECT_EQ(g_observed_temporary_path.rfind('/'), 98U);
  result.value() = UnixListener{};
  ASSERT_EQ(rmdir(parent.c_str()), 0);
}

TEST(UnixListenerTest, RefusesActiveListenerWithoutDeletingIt) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  auto first = UnixListener::Bind(path);
  ASSERT_TRUE(first.ok());
  struct stat before {};
  ASSERT_EQ(lstat(path.c_str(), &before), 0);
  auto second = UnixListener::Bind(path);
  ASSERT_FALSE(second.ok());
  ExpectError(second.error(), ErrorCode::kResourceExhausted,
              "unix socket: active listener already exists");
  struct stat after {};
  ASSERT_EQ(lstat(path.c_str(), &after), 0);
  EXPECT_EQ(after.st_dev, before.st_dev);
  EXPECT_EQ(after.st_ino, before.st_ino);
  auto client = ConnectUnixSocket(path);
  ASSERT_TRUE(client.ok());
}

TEST(UnixListenerTest, ReplacesOnlyAStaleOwnedSocket) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  const int stale = BindRawSocket(path, false);
  ASSERT_GE(stale, 0);
  struct stat stale_stat {};
  ASSERT_EQ(lstat(path.c_str(), &stale_stat), 0);
  close(stale);

  auto listener = UnixListener::Bind(path);
  ASSERT_TRUE(listener.ok()) << listener.error()->message();
  struct stat replacement {};
  ASSERT_EQ(lstat(path.c_str(), &replacement), 0);
  EXPECT_NE(replacement.st_ino, stale_stat.st_ino);
  EXPECT_TRUE(S_ISSOCK(replacement.st_mode));
}

TEST(UnixListenerTest, ProbeEnoentPreservesReplacementFile) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  const int stale = BindRawSocket(path, false);
  ASSERT_GE(stale, 0);
  close(stale);
  ScopedSocketProbeHook hook(ReplaceProbedSocketWithFileAndReturnEnoent);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal,
              "unix socket: stale probe failed safely");
  struct stat replacement {};
  ASSERT_EQ(lstat(path.c_str(), &replacement), 0);
  EXPECT_TRUE(S_ISREG(replacement.st_mode));
  EXPECT_EQ(replacement.st_dev, g_probe_replacement_stat.st_dev);
  EXPECT_EQ(replacement.st_ino, g_probe_replacement_stat.st_ino);
  const int fd = open(path.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  std::array<char, 32> contents{};
  const ssize_t length = read(fd, contents.data(), contents.size());
  close(fd);
  ASSERT_EQ(length, 16);
  EXPECT_EQ(std::string_view(contents.data(), static_cast<std::size_t>(length)),
            "replacement-file");
  ASSERT_EQ(unlink(path.c_str()), 0);
}

TEST(UnixListenerTest, ProbeEnoentPreservesReplacementListener) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  const int stale = BindRawSocket(path, false);
  ASSERT_GE(stale, 0);
  struct stat stale_stat {};
  ASSERT_EQ(lstat(path.c_str(), &stale_stat), 0);
  close(stale);
  g_probe_replacement_fd = -1;
  ScopedSocketProbeHook hook(ReplaceProbedSocketWithListenerAndReturnEnoent);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal,
              "unix socket: stale probe failed safely");
  ASSERT_GE(g_probe_replacement_fd, 0);
  struct stat replacement {};
  ASSERT_EQ(lstat(path.c_str(), &replacement), 0);
  EXPECT_TRUE(S_ISSOCK(replacement.st_mode));
  EXPECT_NE(replacement.st_ino, stale_stat.st_ino);
  EXPECT_EQ(replacement.st_dev, g_probe_replacement_stat.st_dev);
  EXPECT_EQ(replacement.st_ino, g_probe_replacement_stat.st_ino);
  auto client = ConnectUnixSocket(path);
  EXPECT_TRUE(client.ok());
  close(g_probe_replacement_fd);
  g_probe_replacement_fd = -1;
  ASSERT_EQ(unlink(path.c_str()), 0);
}

TEST(UnixListenerTest, ProbeOtherErrorPreservesReplacement) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  const int stale = BindRawSocket(path, false);
  ASSERT_GE(stale, 0);
  close(stale);
  ScopedSocketProbeHook hook(ReplaceProbedSocketWithFileAndReturnEacces);

  auto result = UnixListener::Bind(path);

  ASSERT_FALSE(result.ok());
  ExpectError(result.error(), ErrorCode::kInternal,
              "unix socket: stale probe failed safely");
  struct stat replacement {};
  ASSERT_EQ(lstat(path.c_str(), &replacement), 0);
  EXPECT_TRUE(S_ISREG(replacement.st_mode));
  EXPECT_EQ(replacement.st_dev, g_probe_replacement_stat.st_dev);
  EXPECT_EQ(replacement.st_ino, g_probe_replacement_stat.st_ino);
  const int fd = open(path.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  std::array<char, 32> contents{};
  const ssize_t length = read(fd, contents.data(), contents.size());
  close(fd);
  ASSERT_EQ(length, 16);
  EXPECT_EQ(std::string_view(contents.data(), static_cast<std::size_t>(length)),
            "replacement-file");
  ASSERT_EQ(unlink(path.c_str()), 0);
}

TEST(UnixListenerTest, NeverChangesFileDirectoryOrSymlinkAtSocketPath) {
  TempDir directory;
  const std::string file_path = directory.SocketPath("file");
  const int file = open(file_path.c_str(), O_CREAT | O_WRONLY, 0644);
  ASSERT_GE(file, 0);
  close(file);
  struct stat file_before {};
  ASSERT_EQ(lstat(file_path.c_str(), &file_before), 0);
  auto file_result = UnixListener::Bind(file_path);
  ASSERT_FALSE(file_result.ok());
  struct stat file_after {};
  ASSERT_EQ(lstat(file_path.c_str(), &file_after), 0);
  EXPECT_EQ(file_after.st_ino, file_before.st_ino);
  EXPECT_TRUE(S_ISREG(file_after.st_mode));
  ASSERT_EQ(unlink(file_path.c_str()), 0);

  const std::string dir_path = directory.SocketPath("directory");
  ASSERT_EQ(mkdir(dir_path.c_str(), 0700), 0);
  auto dir_result = UnixListener::Bind(dir_path);
  ASSERT_FALSE(dir_result.ok());
  struct stat dir_after {};
  ASSERT_EQ(lstat(dir_path.c_str(), &dir_after), 0);
  EXPECT_TRUE(S_ISDIR(dir_after.st_mode));
  ASSERT_EQ(rmdir(dir_path.c_str()), 0);

  const std::string target_path = directory.SocketPath("target");
  const std::string link_path = directory.SocketPath("link");
  ASSERT_EQ(symlink(target_path.c_str(), link_path.c_str()), 0);
  struct stat link_before {};
  ASSERT_EQ(lstat(link_path.c_str(), &link_before), 0);
  auto link_result = UnixListener::Bind(link_path);
  ASSERT_FALSE(link_result.ok());
  struct stat link_after {};
  ASSERT_EQ(lstat(link_path.c_str(), &link_after), 0);
  EXPECT_EQ(link_after.st_ino, link_before.st_ino);
  EXPECT_TRUE(S_ISLNK(link_after.st_mode));
  ASSERT_EQ(unlink(link_path.c_str()), 0);
}

TEST(UnixListenerTest, RejectsUnsafeOrSymlinkParentDirectory) {
  TempDir directory;
  ASSERT_EQ(chmod(directory.path().c_str(), 0755), 0);
  auto unsafe = UnixListener::Bind(directory.SocketPath());
  ASSERT_FALSE(unsafe.ok());
  ExpectError(unsafe.error(), ErrorCode::kInvalidArgument,
              "unix socket: parent must be owned by current user with mode 0700");
  ASSERT_EQ(chmod(directory.path().c_str(), 0700), 0);

  const std::string real_parent = directory.SocketPath("real");
  const std::string link_parent = directory.SocketPath("parent-link");
  ASSERT_EQ(mkdir(real_parent.c_str(), 0700), 0);
  ASSERT_EQ(symlink(real_parent.c_str(), link_parent.c_str()), 0);
  auto through_link = UnixListener::Bind(link_parent + "/engine.sock");
  ASSERT_FALSE(through_link.ok());
  EXPECT_NE(access((real_parent + "/engine.sock").c_str(), F_OK), 0);
  ASSERT_EQ(unlink(link_parent.c_str()), 0);
  ASSERT_EQ(rmdir((directory.SocketPath("real")).c_str()), 0);
}

TEST(UnixListenerTest, RejectsSymlinkInIntermediateParentComponent) {
  TempDir directory;
  const std::string real = directory.SocketPath("real");
  const std::string child = real + "/child";
  const std::string link = directory.SocketPath("link");
  ASSERT_EQ(mkdir(real.c_str(), 0700), 0);
  ASSERT_EQ(mkdir(child.c_str(), 0700), 0);
  ASSERT_EQ(symlink(real.c_str(), link.c_str()), 0);
  auto result = UnixListener::Bind(link + "/child/engine.sock");
  EXPECT_FALSE(result.ok());
  if (result.ok()) result.value() = UnixListener{};
  EXPECT_NE(access((child + "/engine.sock").c_str(), F_OK), 0);
  ASSERT_EQ(unlink(link.c_str()), 0);
  ASSERT_EQ(rmdir(child.c_str()), 0);
  ASSERT_EQ(rmdir(real.c_str()), 0);
}

TEST(UnixListenerTest, DestructorDoesNotDeleteReplacementInode) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  std::optional<UnixListener> listener;
  {
    auto result = UnixListener::Bind(path);
    ASSERT_TRUE(result.ok());
    listener.emplace(std::move(result.value()));
  }
  ASSERT_EQ(unlink(path.c_str()), 0);
  const int replacement = BindRawSocket(path, true);
  ASSERT_GE(replacement, 0);
  struct stat before {};
  ASSERT_EQ(lstat(path.c_str(), &before), 0);
  listener.reset();
  struct stat after {};
  ASSERT_EQ(lstat(path.c_str(), &after), 0);
  EXPECT_EQ(after.st_ino, before.st_ino);
  auto client = ConnectUnixSocket(path);
  ASSERT_TRUE(client.ok());
  close(replacement);
  ASSERT_EQ(unlink(path.c_str()), 0);
}

TEST(UnixListenerTest, DestructorUsesDirfdAndDoesNotTouchReplacementParent) {
  TempDir directory;
  const std::string original_parent = directory.SocketPath("parent");
  const std::string moved_parent = directory.SocketPath("parent-old");
  ASSERT_EQ(mkdir(original_parent.c_str(), 0700), 0);
  const std::string path = original_parent + "/engine.sock";
  std::optional<UnixListener> listener;
  {
    auto result = UnixListener::Bind(path);
    ASSERT_TRUE(result.ok());
    listener.emplace(std::move(result.value()));
  }
  ASSERT_EQ(rename(original_parent.c_str(), moved_parent.c_str()), 0);
  ASSERT_EQ(mkdir(original_parent.c_str(), 0700), 0);
  const std::string replacement_path = original_parent + "/engine.sock";
  const int replacement = open(replacement_path.c_str(), O_CREAT | O_WRONLY, 0600);
  ASSERT_GE(replacement, 0);
  close(replacement);
  listener.reset();
  struct stat replacement_stat {};
  ASSERT_EQ(lstat(replacement_path.c_str(), &replacement_stat), 0);
  EXPECT_TRUE(S_ISREG(replacement_stat.st_mode));
  EXPECT_NE(access((moved_parent + "/engine.sock").c_str(), F_OK), 0);
  ASSERT_EQ(unlink(replacement_path.c_str()), 0);
  ASSERT_EQ(rmdir(original_parent.c_str()), 0);
  ASSERT_EQ(rmdir(moved_parent.c_str()), 0);
}

TEST(UnixListenerTest, ConcurrentBindersProduceExactlyOneOwner) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  std::barrier start(3);
  std::array<std::optional<UnixListener>, 2> listeners;
  std::array<bool, 2> succeeded{false, false};
  std::array<std::thread, 2> threads;
  for (std::size_t index = 0; index < threads.size(); ++index) {
    threads[index] = std::thread([&, index] {
      start.arrive_and_wait();
      auto result = UnixListener::Bind(path);
      succeeded[index] = result.ok();
      if (result.ok()) {
        listeners[index].emplace(std::move(result.value()));
      }
    });
  }
  start.arrive_and_wait();
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(static_cast<int>(succeeded[0]) + static_cast<int>(succeeded[1]), 1);
  EXPECT_EQ(access(path.c_str(), F_OK), 0);
}

TEST(UnixListenerTest, ConcurrentProcessBindersProduceExactlyOneOwner) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  std::array<std::array<int, 2>, 2> starts{};
  std::array<std::array<int, 2>, 2> holds{};
  std::array<std::array<int, 2>, 2> results{};
  for (std::size_t index = 0; index < 2; ++index) {
    ASSERT_EQ(pipe(starts[index].data()), 0);
    ASSERT_EQ(pipe(holds[index].data()), 0);
    ASSERT_EQ(pipe(results[index].data()), 0);
  }
  std::array<pid_t, 2> children{};
  for (std::size_t index = 0; index < children.size(); ++index) {
    children[index] = fork();
    ASSERT_GE(children[index], 0);
    if (children[index] == 0) {
      close(starts[index][1]);
      close(holds[index][1]);
      close(results[index][0]);
      char token = 0;
      if (read(starts[index][0], &token, 1) != 1) _exit(2);
      auto listener = UnixListener::Bind(path);
      const char succeeded = listener.ok() ? 1 : 0;
      if (write(results[index][1], &succeeded, 1) != 1) _exit(3);
      if (listener.ok() && read(holds[index][0], &token, 1) != 1) _exit(4);
      if (listener.ok()) listener.value() = UnixListener{};
      _exit(0);
    }
    close(starts[index][0]);
    close(holds[index][0]);
    close(results[index][1]);
  }
  const char token = 1;
  ASSERT_EQ(write(starts[0][1], &token, 1), 1);
  ASSERT_EQ(write(starts[1][1], &token, 1), 1);
  std::array<char, 2> succeeded{};
  ASSERT_EQ(read(results[0][0], &succeeded[0], 1), 1);
  ASSERT_EQ(read(results[1][0], &succeeded[1], 1), 1);
  EXPECT_EQ(static_cast<int>(succeeded[0]) + static_cast<int>(succeeded[1]), 1);
  struct stat winner_before {};
  ASSERT_EQ(lstat(path.c_str(), &winner_before), 0);
  auto client = ConnectUnixSocket(path);
  ASSERT_TRUE(client.ok()) << client.error()->message();
  struct stat winner_after {};
  ASSERT_EQ(lstat(path.c_str(), &winner_after), 0);
  EXPECT_EQ(winner_after.st_dev, winner_before.st_dev);
  EXPECT_EQ(winner_after.st_ino, winner_before.st_ino);
  for (std::size_t index = 0; index < children.size(); ++index) {
    if (succeeded[index] != 0) {
      ASSERT_EQ(write(holds[index][1], &token, 1), 1);
    }
    int status = 0;
    ASSERT_EQ(waitpid(children[index], &status, 0), children[index]);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    close(starts[index][1]);
    close(holds[index][1]);
    close(results[index][0]);
  }
}

TEST(UnixListenerTest, FailedBindsDoNotLeakDescriptors) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  const int file = open(path.c_str(), O_CREAT | O_WRONLY, 0600);
  ASSERT_GE(file, 0);
  close(file);
  const std::size_t before = OpenFdCount();
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto result = UnixListener::Bind(path);
    ASSERT_FALSE(result.ok());
  }
  EXPECT_EQ(OpenFdCount(), before);
  ASSERT_EQ(unlink(path.c_str()), 0);
}

TEST(UnixListenerTest, MoveTransfersPathCleanupOwnership) {
  TempDir directory;
  const std::string path = directory.SocketPath();
  auto result = UnixListener::Bind(path);
  ASSERT_TRUE(result.ok());
  UnixListener moved(std::move(result.value()));
  EXPECT_EQ(result.value().get(), -1);
  EXPECT_GE(moved.get(), 0);
  UnixListener destination;
  destination = std::move(moved);
  EXPECT_EQ(moved.get(), -1);
  EXPECT_GE(destination.get(), 0);
}

}  // namespace
}  // namespace grserve::rpc
