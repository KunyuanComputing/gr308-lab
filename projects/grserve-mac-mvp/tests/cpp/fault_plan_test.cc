#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <locale>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "hal/hal_library.h"
#include "hal/hal_raii.h"
#include "hal_fault/fault_hook.h"
#include "hal_fault/fault_plan.h"

namespace {

using grserve::FaultAction;
using grserve::FaultPhase;
using grserve::FaultPlan;
using grserve::FaultRule;
using grserve::RankId;

std::string DefinedExports(const char* library_path) {
#if defined(__APPLE__)
  const std::string command = std::string("nm -gU '") + library_path + "'";
#else
  const std::string command = std::string("nm -D --defined-only '") + library_path + "'";
#endif
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return {};
  }
  std::string output;
  std::array<char, 512> buffer{};
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  return pclose(pipe) == 0 ? output : std::string{};
}

class GroupingNumpunct final : public std::numpunct<char> {
 private:
  char do_thousands_sep() const override { return '_'; }
  std::string do_grouping() const override { return "\3"; }
};

class GlobalLocaleRestore final {
 public:
  GlobalLocaleRestore() : original_(std::locale()) {}
  ~GlobalLocaleRestore() { std::locale::global(original_); }

 private:
  std::locale original_;
};

int g_hook_commit_failures_remaining = 0;

void FailConfiguredHookCommit() {
  if (g_hook_commit_failures_remaining > 0) {
    --g_hook_commit_failures_remaining;
    throw std::bad_alloc();
  }
}

TEST(FaultPlan, MatchesFrozenExplicitOccurrenceExample) {
  FaultRule rule{
      .rank = RankId{1}, .phase = FaultPhase::kCommit, .occurrence = 2,
      .action = FaultAction::kDropAck, .delay_ticks = 0};
  const char* configured_seed = std::getenv("GRSERVE_FAULT_PLAN_SEED");
  const std::uint64_t seed =
      configured_seed == nullptr ? 17U : std::stoull(configured_seed);
  FaultPlan plan(seed, {rule});
  EXPECT_FALSE(plan.Match(RankId{1}, FaultPhase::kCommit).has_value());
  EXPECT_EQ(plan.Match(RankId{1}, FaultPhase::kCommit)->action,
            FaultAction::kDropAck);
  EXPECT_EQ(plan.CanonicalDigest(),
            "seed=" + std::to_string(seed) +
                ";rank=1,phase=commit,occurrence=2,action=drop_ack,delay=0");
}

TEST(FaultPlan, RejectsMalformedAndDuplicateRules) {
  const FaultRule valid{RankId{4}, FaultPhase::kReserve, 1,
                        FaultAction::kFail, 0};
  EXPECT_FALSE(FaultPlan::Create(1, {{RankId{4}, FaultPhase::kReserve, 0,
                                      FaultAction::kFail, 0}}).ok());
  EXPECT_FALSE(FaultPlan::Create(1, {valid, valid}).ok());
  EXPECT_FALSE(FaultPlan::Create(
                   1, {{RankId{4}, static_cast<FaultPhase>(99), 1,
                        FaultAction::kFail, 0}})
                   .ok());
  EXPECT_FALSE(FaultPlan::Create(
                   1, {{RankId{4}, FaultPhase::kReserve, 1,
                        static_cast<FaultAction>(99), 0}})
                   .ok());
}

TEST(FaultPlan, FormatsAndMatchesEveryActionAndPhase) {
  constexpr std::array<FaultPhase, 6> phases{
      FaultPhase::kReserve, FaultPhase::kPrepare, FaultPhase::kExecute,
      FaultPhase::kCommit, FaultPhase::kAbort, FaultPhase::kHalLaunch};
  constexpr std::array<FaultAction, 5> actions{
      FaultAction::kFail, FaultAction::kBlock, FaultAction::kTimeout,
      FaultAction::kDropAck, FaultAction::kRankLoss};
  constexpr std::array<const char*, 6> phase_names{
      "reserve", "prepare", "execute", "commit", "abort", "hal_launch"};
  constexpr std::array<const char*, 5> action_names{
      "fail", "block", "timeout", "drop_ack", "rank_loss"};
  std::vector<FaultRule> rules;
  for (std::size_t i = 0; i < phases.size(); ++i) {
    rules.push_back(FaultRule{RankId{static_cast<std::uint32_t>(20 + i)}, phases[i],
                              1, actions[i % actions.size()], i});
  }
  FaultPlan plan(9, rules);
  const std::string digest = plan.CanonicalDigest();
  for (std::size_t i = 0; i < phases.size(); ++i) {
    EXPECT_STREQ(FaultPhaseText(phases[i]), phase_names[i]);
    EXPECT_STREQ(FaultActionText(actions[i % actions.size()]),
                 action_names[i % actions.size()]);
    const auto effect = plan.Match(RankId{static_cast<std::uint32_t>(20 + i)}, phases[i]);
    ASSERT_TRUE(effect.has_value());
    EXPECT_EQ(effect->action, actions[i % actions.size()]);
    EXPECT_NE(digest.find(std::string("phase=") + phase_names[i]), std::string::npos);
  }
}

TEST(FaultPlan, CountsRankAndPhaseOccurrencesIndependently) {
  FaultPlan plan(3, {{RankId{1}, FaultPhase::kCommit, 2, FaultAction::kFail, 7},
                     {RankId{1}, FaultPhase::kPrepare, 1, FaultAction::kTimeout, 8},
                     {RankId{2}, FaultPhase::kCommit, 1, FaultAction::kRankLoss, 9}});
  EXPECT_FALSE(plan.Match(RankId{1}, FaultPhase::kCommit));
  EXPECT_EQ(plan.Match(RankId{1}, FaultPhase::kPrepare)->action,
            FaultAction::kTimeout);
  EXPECT_EQ(plan.Match(RankId{2}, FaultPhase::kCommit)->action,
            FaultAction::kRankLoss);
  EXPECT_EQ(plan.Match(RankId{1}, FaultPhase::kCommit)->action,
            FaultAction::kFail);
  EXPECT_FALSE(plan.Match(RankId{3}, FaultPhase::kCommit));
  EXPECT_FALSE(plan.Match(RankId{1}, FaultPhase::kAbort));
  EXPECT_EQ(plan.EffectAt(RankId{1}, FaultPhase::kCommit,
                           std::numeric_limits<std::uint64_t>::max())
                .has_value(), false);
}

TEST(FaultHook, MapsActionsToPrincipledInjectionPointsIdempotently) {
  FaultPlan plan(88, {{RankId{1}, FaultPhase::kReserve, 1, FaultAction::kFail, 1},
                      {RankId{1}, FaultPhase::kReserve, 2, FaultAction::kBlock, 2},
                      {RankId{1}, FaultPhase::kReserve, 3, FaultAction::kTimeout, 3},
                      {RankId{1}, FaultPhase::kReserve, 4, FaultAction::kRankLoss, 4},
                      {RankId{1}, FaultPhase::kCommit, 2, FaultAction::kDropAck, 5}});
  grserve::PlanFaultHook hook(plan);
  for (std::uint64_t ordinal = 1; ordinal <= 4; ++ordinal) {
    const auto before = hook.Before(RankId{1}, FaultPhase::kReserve, ordinal);
    ASSERT_TRUE(before.has_value());
    EXPECT_EQ(before->occurrence, ordinal);
    EXPECT_FALSE(hook.After(RankId{1}, FaultPhase::kReserve, ordinal));
  }
  EXPECT_FALSE(hook.Before(RankId{1}, FaultPhase::kCommit, 2));
  const auto dropped = hook.After(RankId{1}, FaultPhase::kCommit, 2);
  ASSERT_TRUE(dropped.has_value());
  EXPECT_EQ(dropped->action, FaultAction::kDropAck);
  EXPECT_EQ(hook.After(RankId{1}, FaultPhase::kCommit, 2)->action,
            FaultAction::kDropAck);
  EXPECT_EQ(hook.ObservedDigest(),
            "seed=88;rank=1,phase=reserve,ordinal=1,point=before,action=fail,delay=1"
            ";rank=1,phase=reserve,ordinal=2,point=before,action=block,delay=2"
            ";rank=1,phase=reserve,ordinal=3,point=before,action=timeout,delay=3"
            ";rank=1,phase=reserve,ordinal=4,point=before,action=rank_loss,delay=4"
            ";rank=1,phase=commit,ordinal=2,point=after,action=drop_ack,delay=5");
}

TEST(FaultHook, EmptyPlanDoesNotRetainUnmatchedResponses) {
  FaultPlan plan(4, {});
  grserve::PlanFaultHook hook(plan);
  for (std::uint64_t ordinal = 1; ordinal <= 10000; ++ordinal) {
    EXPECT_FALSE(hook.Before(RankId{1}, FaultPhase::kCommit, ordinal));
    EXPECT_FALSE(hook.After(RankId{1}, FaultPhase::kCommit, ordinal));
  }
  EXPECT_EQ(hook.CachedResponseCountForTesting(), 0U);
  EXPECT_EQ(hook.ObservedEventCountForTesting(), 0U);
  EXPECT_EQ(hook.ObservedDigest(), "seed=4");
}

TEST(FaultHook, DoesNotRetainSideFilteredResponses) {
  FaultPlan plan(4, {{RankId{1}, FaultPhase::kCommit, 1,
                     FaultAction::kDropAck, 0}});
  grserve::PlanFaultHook hook(plan);
  EXPECT_FALSE(hook.Before(RankId{1}, FaultPhase::kCommit, 1));
  EXPECT_EQ(hook.CachedResponseCountForTesting(), 0U);
  EXPECT_EQ(hook.ObservedEventCountForTesting(), 0U);
}

TEST(FaultHook, AllocationFailureDoesNotPartiallyCommitResponseOrEvent) {
  FaultPlan plan(5, {{RankId{2}, FaultPhase::kExecute, 1, FaultAction::kFail, 9}});
  g_hook_commit_failures_remaining = 1;
  grserve::PlanFaultHook hook(plan, FailConfiguredHookCommit);
  EXPECT_THROW(hook.Before(RankId{2}, FaultPhase::kExecute, 1), std::bad_alloc);
  EXPECT_EQ(hook.CachedResponseCountForTesting(), 0U);
  EXPECT_EQ(hook.ObservedEventCountForTesting(), 0U);
  EXPECT_EQ(hook.ObservedDigest(), "seed=5");
  const auto recovered = hook.Before(RankId{2}, FaultPhase::kExecute, 1);
  ASSERT_TRUE(recovered.has_value());
  EXPECT_EQ(recovered->action, FaultAction::kFail);
  EXPECT_EQ(hook.CachedResponseCountForTesting(), 1U);
  EXPECT_EQ(hook.ObservedEventCountForTesting(), 1U);
  EXPECT_EQ(hook.Before(RankId{2}, FaultPhase::kExecute, 1)->action,
            FaultAction::kFail);
  EXPECT_EQ(hook.CachedResponseCountForTesting(), 1U);
  EXPECT_EQ(hook.ObservedEventCountForTesting(), 1U);
  EXPECT_EQ(hook.ObservedDigest(),
            "seed=5;rank=2,phase=execute,ordinal=1,point=before,action=fail,delay=9");
}

TEST(FaultDigest, IgnoresGlobalNumericGroupingLocale) {
  const std::locale original = std::locale();
  {
    GlobalLocaleRestore restore;
    std::locale::global(std::locale(std::locale::classic(), new GroupingNumpunct));
    FaultPlan plan(1234567,
                   {{RankId{1234}, FaultPhase::kCommit, 1234567,
                     FaultAction::kDropAck, 9876543}});
    grserve::PlanFaultHook hook(plan);
    EXPECT_EQ(plan.CanonicalDigest(),
              "seed=1234567;rank=1234,phase=commit,occurrence=1234567,action=drop_ack,delay=9876543");
    ASSERT_TRUE(hook.After(RankId{1234}, FaultPhase::kCommit, 1234567));
    EXPECT_EQ(hook.ObservedDigest(),
              "seed=1234567;rank=1234,phase=commit,ordinal=1234567,point=after,action=drop_ack,delay=9876543");
  }
  EXPECT_TRUE(std::locale() == original);
}

TEST(FaultPlan, IsReproducibleAcrossRunsAndSeedDoesNotShiftRules) {
  const std::vector<FaultRule> rules{{RankId{6}, FaultPhase::kExecute, 2,
                                      FaultAction::kTimeout, 12}};
  std::string plan_digest;
  std::string event_digest;
  for (int i = 0; i < 100; ++i) {
    FaultPlan plan(10, rules);
    grserve::PlanFaultHook hook(plan);
    EXPECT_FALSE(plan.Match(RankId{6}, FaultPhase::kExecute));
    ASSERT_TRUE(plan.Match(RankId{6}, FaultPhase::kExecute));
    ASSERT_TRUE(hook.Before(RankId{6}, FaultPhase::kExecute, 2));
    if (i == 0) {
      plan_digest = plan.CanonicalDigest();
      event_digest = hook.ObservedDigest();
    } else {
      EXPECT_EQ(plan.CanonicalDigest(), plan_digest);
      EXPECT_EQ(hook.ObservedDigest(), event_digest);
    }
  }
  FaultPlan first(10, rules);
  FaultPlan second(11, rules);
  EXPECT_EQ(first.CanonicalDigest(),
            "seed=10;rank=6,phase=execute,occurrence=2,action=timeout,delay=12");
  EXPECT_EQ(second.CanonicalDigest(),
            "seed=11;rank=6,phase=execute,occurrence=2,action=timeout,delay=12");
  EXPECT_FALSE(first.Match(RankId{6}, FaultPhase::kExecute));
  EXPECT_FALSE(second.Match(RankId{6}, FaultPhase::kExecute));
  EXPECT_EQ(first.Match(RankId{6}, FaultPhase::kExecute)->action,
            second.Match(RankId{6}, FaultPhase::kExecute)->action);
}

TEST(FaultHal, LoadsAndDelegatesLifecycleAndLaunch) {
  const auto opened = grserve::HalLibrary::Open(FAULT_HAL_PATH,
      GRSERVE_HAL_CAPABILITY_EXECUTABLES | GRSERVE_HAL_CAPABILITY_STREAMS);
  ASSERT_TRUE(opened.ok()) << opened.error()->message();
  auto runtime_created = grserve::HalRuntime::Create(opened.value());
  ASSERT_TRUE(runtime_created.ok()) << runtime_created.error()->message();
  auto runtime = std::move(runtime_created).value();
  auto device_created = runtime.CreateDevice(0);
  ASSERT_TRUE(device_created.ok()) << device_created.error()->message();
  auto device = std::move(device_created).value();
  auto stream_created = device.CreateStream(0);
  ASSERT_TRUE(stream_created.ok()) << stream_created.error()->message();
  auto stream = std::move(stream_created).value();
  auto executable_created = device.CreateExecutable();
  ASSERT_TRUE(executable_created.ok()) << executable_created.error()->message();
  auto executable = std::move(executable_created).value();
  const std::array<std::uint8_t, 1> image{0};
  ASSERT_EQ(opened.value()->api().executable_load(executable.get(), image.data(), image.size()),
            GRSERVE_HAL_STATUS_OK);
  grserve_hal_launch_info_t launch{};
  launch.struct_size = static_cast<std::uint32_t>(sizeof(launch));
  const std::array<std::uint8_t, 1> entry{'x'};
  EXPECT_EQ(opened.value()->api().executable_launch(executable.get(), stream.get(),
                                                    entry.data(), entry.size(), &launch),
            GRSERVE_HAL_STATUS_OK);
}

TEST(FaultHal, RechecksEveryNegotiationAfterDelegateInitialization) {
  void* library = dlopen(FAULT_HAL_PATH, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(library, nullptr);
  const auto get_api = reinterpret_cast<grserve_hal_status_t (*) (
      const grserve_hal_api_request_t*, grserve_hal_api_t*)>(
      dlsym(library, "grserve_hal_get_api"));
  ASSERT_NE(get_api, nullptr);
  grserve_hal_api_request_t good_request{};
  good_request.struct_size = static_cast<std::uint32_t>(sizeof(good_request));
  good_request.abi_major = GRSERVE_HAL_ABI_MAJOR;
  grserve_hal_api_t good_api{};
  good_api.struct_size = static_cast<std::uint32_t>(sizeof(good_api));
  ASSERT_EQ(get_api(&good_request, &good_api), GRSERVE_HAL_STATUS_OK);
  grserve_hal_api_request_t incompatible = good_request;
  incompatible.abi_major = GRSERVE_HAL_ABI_MAJOR + 1U;
  grserve_hal_api_t rejected_api{};
  rejected_api.struct_size = static_cast<std::uint32_t>(sizeof(rejected_api));
  EXPECT_EQ(get_api(&incompatible, &rejected_api),
            GRSERVE_HAL_STATUS_INCOMPATIBLE_ABI);
  incompatible = good_request;
  incompatible.required_capabilities = UINT64_C(1) << 63;
  EXPECT_EQ(get_api(&incompatible, &rejected_api), GRSERVE_HAL_STATUS_UNSUPPORTED);
  dlclose(library);
}

TEST(FaultHal, ExportsOnlyTheAbiEntryPoint) {
  const std::string output = DefinedExports(FAULT_HAL_PATH);
  ASSERT_FALSE(output.empty());
  std::istringstream lines(output);
  std::string line;
  std::vector<std::string> symbols;
  while (std::getline(lines, line)) {
    const std::size_t symbol_start = line.find_last_of(" \t");
    ASSERT_NE(symbol_start, std::string::npos) << line;
    std::string symbol = line.substr(symbol_start + 1U);
    if (!symbol.empty() && symbol.front() == '_') {
      symbol.erase(0, 1);
    }
    symbols.push_back(std::move(symbol));
  }
  ASSERT_EQ(symbols.size(), 1U);
  EXPECT_EQ(symbols.front(), "grserve_hal_get_api");
}

}  // namespace
