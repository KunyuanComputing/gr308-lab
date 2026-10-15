#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "engine/controller.h"
#include "rpc/rpc_session.h"
#include "rpc/unix_socket.h"

namespace grserve::rpc {

class EngineRpcServer {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<EngineRpcServer>> Create(
      UnixListener listener, Controller& controller, RpcSessionConfig config);
  EngineRpcServer(const EngineRpcServer&) = delete;
  EngineRpcServer& operator=(const EngineRpcServer&) = delete;
  ~EngineRpcServer();
  void PollOnce();
  void StopAdmission() noexcept;
  [[nodiscard]] bool DrainComplete() const noexcept;
  void Shutdown() noexcept;

 private:
  EngineRpcServer(UnixListener listener, Controller& controller,
                  RpcSessionConfig config, UniqueFd wake_read,
                  UniqueFd wake_write);
  void AcceptLoop() noexcept;
  UnixListener listener_;
  Controller* controller_;
  RpcSessionConfig config_;
  std::shared_ptr<ControllerMailbox> mailbox_;
  mutable std::mutex sessions_mutex_;
  std::vector<std::shared_ptr<RpcSession>> sessions_;
  std::atomic<bool> accepting_{true};
  UniqueFd wake_read_;
  UniqueFd wake_write_;
  std::thread accept_thread_;
  static constexpr std::size_t kMaxSessions = 128;
};

}  // namespace grserve::rpc
