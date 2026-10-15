#include <signal.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "engine/admission.h"
#include "common/sha256.h"
#include "engine/controller.h"
#include "engine/rank_worker.h"
#include "engine/replica_gate.h"
#include "grserve/hal.h"
#include "hal/hal_library.h"
#include "hal/hal_raii.h"
#include "model/manifest.h"
#include "model/mini_qwen.h"
#include "rpc/engine_rpc_server.h"
#include "rpc/unix_socket.h"

namespace {

volatile sig_atomic_t stop_requested = 0;
extern "C" void RequestStop(int) { stop_requested = 1; }

struct Options {
  std::string socket;
  std::filesystem::path model_dir;
  std::filesystem::path hal_plugin;
  std::uint32_t ranks{2};
  std::uint32_t max_concurrent_requests{8};
  std::uint32_t max_state_slots_per_rank{8};
  bool help{false};
};

void PrintHelp() {
  std::cout
      << "Usage: grserve-engine --socket PATH --model-dir DIR --hal-plugin FILE "
         "[--ranks=2] [--max-concurrent-requests N] "
         "[--max-state-slots-per-rank N]\n";
}

bool ParseUnsigned(std::string_view text, std::uint32_t* output) {
  std::uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value == 0) return false;
  *output = value;
  return true;
}

bool ParseOptions(int argc, char** argv, Options* options, std::string* error) {
  for (int index = 1; index < argc; ++index) {
    std::string_view argument(argv[index]);
    if (argument == "--help") { options->help = true; continue; }
    auto value = [&](std::string_view name) -> std::string_view {
      if (argument.starts_with(name) && argument.size() > name.size() &&
          argument[name.size()] == '=') return argument.substr(name.size() + 1);
      if (argument == name && index + 1 < argc) return argv[++index];
      return {};
    };
    if (auto text = value("--socket"); !text.empty()) options->socket = text;
    else if (auto text = value("--model-dir"); !text.empty()) options->model_dir = text;
    else if (auto text = value("--hal-plugin"); !text.empty()) options->hal_plugin = text;
    else if (auto text = value("--ranks"); !text.empty()) {
      if (!ParseUnsigned(text, &options->ranks)) { *error = "invalid --ranks"; return false; }
    } else if (auto text = value("--max-concurrent-requests"); !text.empty()) {
      if (!ParseUnsigned(text, &options->max_concurrent_requests)) {
        *error = "invalid --max-concurrent-requests"; return false;
      }
    } else if (auto text = value("--max-state-slots-per-rank"); !text.empty()) {
      if (!ParseUnsigned(text, &options->max_state_slots_per_rank)) {
        *error = "invalid --max-state-slots-per-rank"; return false;
      }
    } else {
      *error = "unknown or missing option value: " + std::string(argument);
      return false;
    }
  }
  if (options->help) return true;
  if (options->socket.empty() || options->model_dir.empty() ||
      options->hal_plugin.empty()) {
    *error = "--socket, --model-dir, and --hal-plugin are required";
    return false;
  }
  if (options->ranks != 2) {
    *error = "--ranks must be exactly 2 for the Mac MVP";
    return false;
  }
  constexpr std::uint32_t kMaximumRequests = 1024;
  if (options->max_concurrent_requests > kMaximumRequests ||
      options->max_state_slots_per_rank > kMaximumRequests) {
    *error = "request and state-slot limits must be at most 1024";
    return false;
  }
  return true;
}

class RankModelExecutor {
 public:
  RankModelExecutor(std::shared_ptr<grserve::HalLibrary> library,
                    grserve::HalDevice device, grserve::HalStream stream,
                    grserve::HalExecutable executable)
      : library_(std::move(library)), device_(std::move(device)),
        stream_(std::move(stream)), executable_(std::move(executable)) {}
  grserve::StatusOr<grserve::TokenId> Next(const grserve::Execute& execute) {
    try {
      if (execute.context_token_ids.empty() ||
          execute.sequence_number.value() == 0) {
        return grserve::StatusOr<grserve::TokenId>(grserve::Error::From(
            grserve::ErrorCode::kInvalidArgument,
            "rank execute input is empty", std::nullopt,
            execute.transaction_id));
      }
      std::vector<std::uint32_t> context;
      context.reserve(execute.context_token_ids.size() +
                      execute.sequence_number.value());
      for (auto token : execute.context_token_ids) {
        context.push_back(token.value());
      }
      std::vector<std::uint8_t> state;
      std::uint32_t token = 0;
      for (std::uint64_t step = 0;
           step < execute.sequence_number.value(); ++step) {
        std::array<char, 65> digest{};
        std::array<std::uint8_t, 8192> next_state{};
        std::uint64_t next_state_bytes = 0;
        grserve::MiniQwenHalLaunchArguments arguments{
            .struct_size = static_cast<std::uint32_t>(sizeof(arguments)),
            .step_index = static_cast<std::uint32_t>(step),
            .context_token_ids = context.data(),
            .context_token_count = static_cast<std::uint32_t>(context.size()),
            .canonical_state_bytes = state.empty() ? nullptr : state.data(),
            .canonical_state_byte_count = state.size(),
            .out_token_id = &token,
            .out_canonical_digest = digest.data(),
            .canonical_digest_capacity = digest.size(),
            .out_next_state_bytes = next_state.data(),
            .next_state_capacity = next_state.size(),
            .out_next_state_byte_count = &next_state_bytes};
        const grserve_hal_launch_info_t launch{
            .struct_size = static_cast<std::uint32_t>(sizeof(launch)),
            .grid_x = 1, .grid_y = 1, .grid_z = 1,
            .block_x = 1, .block_y = 1, .block_z = 1,
            .argument_data = &arguments,
            .argument_bytes = sizeof(arguments)};
        const auto status = library_->api().executable_launch(
            executable_.get(), stream_.get(),
            reinterpret_cast<const std::uint8_t*>(grserve::kMiniQwenEntryPoint),
            std::char_traits<char>::length(grserve::kMiniQwenEntryPoint), &launch);
        if (status != GRSERVE_HAL_STATUS_OK ||
            next_state_bytes > next_state.size()) {
          return grserve::StatusOr<grserve::TokenId>(grserve::Error::From(
              grserve::ErrorCode::kRankFailure, "HAL model launch failed",
              std::nullopt, execute.transaction_id));
        }
        state.assign(next_state.begin(), next_state.begin() + next_state_bytes);
        context.push_back(token);
      }
      return grserve::StatusOr<grserve::TokenId>(grserve::TokenId{token});
    } catch (...) {
      return grserve::StatusOr<grserve::TokenId>(grserve::Error::From(
          grserve::ErrorCode::kRankFailure, "HAL model execution failed",
          std::nullopt, execute.transaction_id));
    }
  }
 private:
  std::shared_ptr<grserve::HalLibrary> library_;
  grserve::HalDevice device_;
  grserve::HalStream stream_;
  grserve::HalExecutable executable_;
};

int Fail(const grserve::Error& error) {
  std::cerr << "grserve-engine: " << error.message() << '\n';
  return 1;
}

grserve::StatusOr<std::string> ManifestDigest(
    const std::filesystem::path& model_dir) {
  std::ifstream input(model_dir / "manifest.textproto", std::ios::binary);
  if (!input) {
    return grserve::StatusOr<std::string>(grserve::Error::From(
        grserve::ErrorCode::kInvalidArgument, "could not hash model manifest"));
  }
  grserve::Sha256 sha;
  std::array<char, 4096> bytes{};
  while (input) {
    input.read(bytes.data(), bytes.size());
    const auto count = input.gcount();
    if (count > 0) {
      const auto status = sha.Update(
          std::string_view(bytes.data(), static_cast<std::size_t>(count)));
      if (!status.ok()) return grserve::StatusOr<std::string>(*status.error());
    }
  }
  if (!input.eof()) {
    return grserve::StatusOr<std::string>(grserve::Error::From(
        grserve::ErrorCode::kInvalidArgument, "could not hash model manifest"));
  }
  return sha.FinalHex();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::string parse_error;
  if (!ParseOptions(argc, argv, &options, &parse_error)) {
    std::cerr << "grserve-engine: " << parse_error << '\n';
    PrintHelp();
    return 2;
  }
  if (options.help) { PrintHelp(); return 0; }

  struct sigaction action {};
  action.sa_handler = RequestStop;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGINT, &action, nullptr) != 0 ||
      sigaction(SIGTERM, &action, nullptr) != 0) {
    std::cerr << "grserve-engine: could not install signal handlers\n";
    return 1;
  }

  // The public socket is safely installed first, but accept does not begin
  // until every immutable manifest/HAL and replica gate below succeeds.
  auto listener = grserve::rpc::UnixListener::Bind(options.socket);
  if (!listener.ok()) return Fail(*listener.error());
  auto library = grserve::HalLibrary::Open(
      options.hal_plugin, GRSERVE_HAL_CAPABILITY_EXECUTABLES);
  if (!library.ok()) return Fail(*library.error());
  const auto& api = library.value()->api();
  auto assets = grserve::LoadModelAssets(
      options.model_dir,
      {api.abi_major, api.abi_minor, api.capabilities});
  if (!assets.ok()) return Fail(*assets.error());
  auto manifest_digest = ManifestDigest(options.model_dir);
  if (!manifest_digest.ok()) return Fail(*manifest_digest.error());
  auto model = grserve::MiniQwen::Load(assets.value());
  if (!model.ok()) return Fail(*model.error());

  auto runtime = grserve::HalRuntime::Create(library.value());
  if (!runtime.ok()) return Fail(*runtime.error());
  grserve::CompletionQueue<grserve::RankCompletion> completions;
  std::vector<std::unique_ptr<grserve::RankWorker>> workers;
  std::vector<grserve::ControllerRank> rank_endpoints;
  std::vector<grserve::RankCapacity> capacities;
  workers.reserve(options.ranks);
  rank_endpoints.reserve(options.ranks);
  capacities.reserve(options.ranks);
  const std::uint32_t slots = std::min(options.max_concurrent_requests,
                                       options.max_state_slots_per_rank);
  for (std::uint32_t rank = 0; rank < options.ranks; ++rank) {
    auto device = runtime.value().CreateDevice(0);
    if (!device.ok()) return Fail(*device.error());
    auto executable = device.value().CreateExecutable();
    if (!executable.ok()) return Fail(*executable.error());
    auto stream = device.value().CreateStream(0);
    if (!stream.ok()) return Fail(*stream.error());
    const std::string path = options.model_dir.string();
    if (api.executable_load(executable.value().get(),
                            reinterpret_cast<const std::uint8_t*>(path.data()),
                            path.size()) != GRSERVE_HAL_STATUS_OK) {
      return Fail(grserve::Error::From(grserve::ErrorCode::kInternal,
                                       "HAL executable load failed"));
    }
    auto executor = std::make_shared<RankModelExecutor>(
        library.value(), std::move(device).value(), std::move(stream).value(),
        std::move(executable).value());
    workers.push_back(std::make_unique<grserve::RankWorker>(
        grserve::RankId{rank}, completions, grserve::StateDigest{0},
        std::function<void()>{}, grserve::RankWorker::DigestTransition{},
        [executor](const grserve::Execute& command) {
          return executor->Next(command);
        }));
    auto* worker = workers.back().get();
    rank_endpoints.push_back({grserve::RankId{rank}, [worker](grserve::RankCommand command) {
      return worker->Enqueue(std::move(command));
    }});
    capacities.push_back({grserve::RankId{rank}, slots});
  }
  auto admission = grserve::AdmissionController::Create(std::move(capacities));
  if (!admission.ok()) return Fail(*admission.error());
  auto gate = std::make_shared<grserve::ReplicaGate>();
  auto controller = grserve::Controller::Create(
      std::move(admission).value(), gate, completions,
      std::move(rank_endpoints));
  if (!controller.ok()) return Fail(*controller.error());
  grserve::rpc::RpcSessionConfig session_config{
      .model_id = assets.value().manifest().model_id,
      .outbound_capacity = 64,
      .engine_build_id = "0.1.0",
      .manifest_sha256 = std::move(manifest_digest).value(),
      .max_context_tokens = assets.value().manifest().max_context_tokens,
      .vocab_size = assets.value().manifest().vocab_size,
      .replica_gate = gate};
  auto server = grserve::rpc::EngineRpcServer::Create(
      std::move(listener).value(), *controller.value(), std::move(session_config));
  if (!server.ok()) return Fail(*server.error());

  while (!stop_requested) {
    server.value()->PollOnce();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  server.value()->StopAdmission();
  const auto drain_deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(250);
  while (std::chrono::steady_clock::now() < drain_deadline) {
    server.value()->PollOnce();
    const bool ranks_quiescent = std::all_of(
        workers.begin(), workers.end(),
        [](const auto& worker) { return worker->quiescent(); });
    if (server.value()->DrainComplete() && ranks_quiescent) break;
    std::this_thread::yield();
  }
  const bool ranks_quiescent = std::all_of(
      workers.begin(), workers.end(),
      [](const auto& worker) { return worker->quiescent(); });
  const bool drained = server.value()->DrainComplete() && ranks_quiescent;
  server.value()->Shutdown();
  if (!drained) {
    // A HAL ABI call has no cancellation contract. After every externally
    // visible endpoint and the identity-checked socket inode are closed,
    // fail-stop lets the OS reclaim blocked HAL threads and handles.
    std::_Exit(2);
  }
  static_cast<void>(controller.value()->Shutdown());
  for (auto& worker : workers) worker->Shutdown();
  completions.Close();
  return 0;
}
