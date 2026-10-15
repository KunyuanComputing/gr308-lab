#include "rpc/engine_rpc_server.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <utility>

namespace grserve::rpc {

StatusOr<std::unique_ptr<EngineRpcServer>> EngineRpcServer::Create(
    UnixListener listener, Controller& controller, RpcSessionConfig config) {
  if (listener.get() < 0 || config.model_id.empty()) {
    return StatusOr<std::unique_ptr<EngineRpcServer>>(
        Error::From(ErrorCode::kInvalidArgument, "invalid Engine RPC server configuration"));
  }
  int wake[2];
  if (pipe(wake) != 0) {
    return StatusOr<std::unique_ptr<EngineRpcServer>>(
        Error::From(ErrorCode::kInternal, "could not create accept wake pipe"));
  }
  UniqueFd wake_read(wake[0]);
  UniqueFd wake_write(wake[1]);
  if (fcntl(wake_read.get(), F_SETFD, FD_CLOEXEC) != 0 ||
      fcntl(wake_write.get(), F_SETFD, FD_CLOEXEC) != 0) {
    return StatusOr<std::unique_ptr<EngineRpcServer>>(
        Error::From(ErrorCode::kInternal, "could not protect accept wake pipe"));
  }
  try {
    auto server = std::unique_ptr<EngineRpcServer>(
        new EngineRpcServer(std::move(listener), controller, std::move(config),
                            std::move(wake_read), std::move(wake_write)));
    server->accept_thread_ = std::thread(&EngineRpcServer::AcceptLoop, server.get());
    return StatusOr<std::unique_ptr<EngineRpcServer>>(std::move(server));
  } catch (...) {
    return StatusOr<std::unique_ptr<EngineRpcServer>>(
        Error::From(ErrorCode::kInternal, "could not start Engine RPC server"));
  }
}

EngineRpcServer::EngineRpcServer(UnixListener listener, Controller& controller,
                                 RpcSessionConfig config, UniqueFd wake_read,
                                 UniqueFd wake_write)
    : listener_(std::move(listener)), controller_(&controller),
      config_(std::move(config)), mailbox_(std::make_shared<ControllerMailbox>()),
      wake_read_(std::move(wake_read)), wake_write_(std::move(wake_write)) {}
EngineRpcServer::~EngineRpcServer() { Shutdown(); }

void EngineRpcServer::PollOnce() {
  constexpr std::size_t kOwnerBudget = 32;
  for (std::size_t item = 0; item < kOwnerBudget; ++item) {
    const bool mailbox_work = mailbox_->PollOne(*controller_);
    const bool completion_work = controller_->PollOnce();
    if (!mailbox_work && !completion_work) break;
  }
  std::lock_guard lock(sessions_mutex_);
  sessions_.erase(std::remove_if(sessions_.begin(), sessions_.end(),
      [](const auto& session) { return session->stopped(); }), sessions_.end());
}

void EngineRpcServer::StopAdmission() noexcept {
  mailbox_->StopAdmission();
  if (accepting_.exchange(false) && wake_write_) {
    const char wake = 1;
    ssize_t result;
    do { result = write(wake_write_.get(), &wake, sizeof(wake)); }
    while (result < 0 && errno == EINTR);
  }
}

bool EngineRpcServer::DrainComplete() const noexcept {
  return mailbox_->empty() && controller_->idle();
}

void EngineRpcServer::Shutdown() noexcept {
  try {
    StopAdmission();
    if (accept_thread_.joinable()) accept_thread_.join();
    std::vector<std::shared_ptr<RpcSession>> sessions;
    {
      std::lock_guard lock(sessions_mutex_);
      sessions.swap(sessions_);
    }
    for (auto& session : sessions) session->Shutdown();
    while (mailbox_->PollOne(*controller_)) {}
    static_cast<void>(controller_->PollOnce());
  } catch (...) {}
  mailbox_->Close();
  wake_write_.reset();
  wake_read_.reset();
  listener_ = UnixListener{};
}

void EngineRpcServer::AcceptLoop() noexcept {
  try {
    while (accepting_) {
      pollfd descriptors[2] = {{listener_.get(), POLLIN, 0},
                               {wake_read_.get(), POLLIN, 0}};
      int ready;
      do { ready = poll(descriptors, 2, -1); } while (ready < 0 && errno == EINTR);
      if (ready <= 0 || (descriptors[1].revents & POLLIN) != 0 || !accepting_) break;
      if ((descriptors[0].revents & POLLIN) == 0) continue;
      auto socket = listener_.Accept();
      if (!socket.ok()) break;
      if (!accepting_) break;
      {
        std::lock_guard lock(sessions_mutex_);
        sessions_.erase(std::remove_if(sessions_.begin(), sessions_.end(),
            [](const auto& session) { return session->stopped(); }), sessions_.end());
        if (sessions_.size() >= kMaxSessions) continue;
      }
      auto session = RpcSession::Start(std::move(socket).value(), mailbox_, config_);
      if (!session.ok()) continue;
      std::lock_guard lock(sessions_mutex_);
      sessions_.push_back(std::move(session).value());
    }
  } catch (...) {
    accepting_.store(false);
  }
}

}  // namespace grserve::rpc
