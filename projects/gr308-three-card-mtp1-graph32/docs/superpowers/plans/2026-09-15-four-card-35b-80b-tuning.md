# Four-Card 35B and 80B Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the approved two-card 35B TPS campaign concurrently with a TP4 Qwen3-Next-80B-A3B-Instruct 1024/1024 short matrix, stopping the 80B workflow before formal testing.

**Architecture:** Make workload shape, reserved-device checks, unused-MTP handling, and container identity explicit per job. Reuse the existing safety-gated worker for both campaigns, with exact cross-campaign container allowlists. Launch two independent transient systemd services so a local candidate failure remains scoped unless global core health fails.

**Tech Stack:** Python 3, multiprocessing, Docker, systemd, vLLM 0.18.0 FANT plugin, JSON benchmark evidence, pytest-style tests.

---

### Task 1: Generalize Per-Job Safety and Workload Shape

**Files:**
- Modify: `scripts/gr308_parallel_goal.py`
- Modify: `tests/test_gr308_parallel_goal.py`

- [ ] **Step 1: Add failing tests**

```python
def test_job_can_disable_reserved_device_check_for_coordinated_full_card_use():
    module = load_module()
    job = module.Job("m", (4, 5, 6, 7), 22120, "16-31", Path("/tmp/o"),
                     "c", 1, "TP4", reserved_devices=())
    assert job.reserved_devices == ()
    assert module.card_cleanup_ok([540] * 8, job.devices, job.reserved_devices)


def test_job_carries_exact_workload_shape():
    module = load_module()
    job = module.Job("m", (4, 5, 6, 7), 22120, "16-31", Path("/tmp/o"),
                     "c", 1, "TP4", input_tokens=1024, output_tokens=1024,
                     allow_unused_mtp=True)
    assert (job.input_tokens, job.output_tokens) == (1024, 1024)
    assert job.allow_unused_mtp is True
```

- [ ] **Step 2: Verify focused tests fail**

Run `python3 -m pytest tests/test_gr308_parallel_goal.py -v`.

Expected: FAIL because the new `Job` fields and scoped cleanup signature do not exist.

- [ ] **Step 3: Add defaulted `Job` fields**

```python
class Job(NamedTuple):
    # existing fields unchanged
    short: bool = False
    reserved_devices: tuple[int, ...] = RESERVED_DEVICES
    input_tokens: int = 3072
    output_tokens: int = 1024
    allow_unused_mtp: bool = False
```

Accept `reserved_devices` in `card_cleanup_ok`, defaulting to
`RESERVED_DEVICES`. In `run_worker`, use `job.reserved_devices` for admission,
monitoring, cleanup flags, and never read the global tuple for those decisions.

- [ ] **Step 4: Configure the benchmark contract in the worker process**

Immediately after importing `goal_bench_contract`, set:

```python
goal_bench_contract.INPUT_TOKENS = job.input_tokens
goal_bench_contract.OUTPUT_TOKENS = job.output_tokens
goal_bench_contract.REQUESTS = short_request_count(1) if job.short else 64
```

Keep the existing per-concurrency short request override inside the case loop so
C16 uses 32 short requests. Change the unused native-MTP rejection to:

```python
if (not job.mtp_tokens and cap["native_mtp_weight_evidence"]
        and not job.allow_unused_mtp):
    raise RuntimeError("baseline-no-native-mtp job unexpectedly has native MTP weights")
```

- [ ] **Step 5: Verify and commit**

```bash
python3 -m pytest tests/test_gr308_parallel_goal.py -v
python3 -m py_compile scripts/gr308_parallel_goal.py
git add scripts/gr308_parallel_goal.py tests/test_gr308_parallel_goal.py
git commit -m "Make GR308 worker workload and reservations job scoped"
```

Expected: tests PASS and compilation exits 0.

### Task 2: Extend Adaptive Specs for Concurrent 35B and TP4 80B

**Files:**
- Modify: `scripts/gr308_adaptive_short_matrix.py`
- Modify: `tests/test_adaptive_short_matrix.py`

- [ ] **Step 1: Add a failing construction test**

```python
def test_80b_spec_builds_tp4_eager_1024_workload_without_disabling_custom_ar():
    module = load_module()
    spec = module.Spec(
        name="probe", model="Qwen3-Next-80B-A3B-Instruct",
        devices=(4, 5, 6, 7), port=22120, cpu_set="16-31", mtp=1,
        graph=(4, 8, 16, 32), batched_tokens=2048, block_size=1088,
        kv_bytes=module.FIXED_KV, tp_size=4, enforce_eager=True,
        container_name="gr308-80b-short", reserved_devices=(),
        input_tokens=1024, output_tokens=1024,
    )
    args = module.serve_args(spec)
    job = module._job(spec, Path("/tmp/results"))
    assert "--enforce-eager" in args
    assert "--compilation-config" not in args
    assert "--disable-custom-all-reduce" not in args
    assert job.devices == (4, 5, 6, 7)
    assert (job.input_tokens, job.output_tokens) == (1024, 1024)
```

- [ ] **Step 2: Run and observe failure**

Run `python3 -m pytest tests/test_adaptive_short_matrix.py -v`.

Expected: FAIL on missing spec fields/model mapping.

- [ ] **Step 3: Add fields and mapping**

Append these defaulted fields to `Spec`:

```python
container_name: str | None = None
reserved_devices: tuple[int, ...] = base.RESERVED_DEVICES
input_tokens: int = 3072
output_tokens: int = 1024
allow_unused_mtp: bool = False
enforce_eager: bool = False
```

Add `Qwen3-Next-80B-A3B-Instruct` to `CONTAINER_NAMES`. Pass every field through
`_job`. Resolve `spec.container_name` before the model mapping.

- [ ] **Step 4: Encode eager mode safely**

In `serve_args`, when `spec.enforce_eager` is true, delete the
`--compilation-config` flag and its JSON value, then append `--enforce-eager`.
When false, retain the existing graph configuration. Preserve the final guard
that raises if `--disable-custom-all-reduce` appears.

- [ ] **Step 5: Verify and commit**

```bash
python3 -m pytest tests/test_adaptive_short_matrix.py -v
python3 -m py_compile scripts/gr308_adaptive_short_matrix.py
git add scripts/gr308_adaptive_short_matrix.py tests/test_adaptive_short_matrix.py
git commit -m "Support TP4 80B adaptive specs"
```

### Task 3: Implement the 35B Two-Card Campaign

**Files:**
- Create: `scripts/gr308_35b_two_card_tps.py`
- Create: `tests/test_gr308_35b_two_card_tps.py`

- [ ] **Step 1: Add tests for the approved eight candidates**

Assert two four-spec lanes on `(0, 1)` and `(2, 3)`, unique lane container
names, MTP3, block1088, automatic KV, empty reserved devices, and the exact
graph/batch/async matrix recorded in the approved design.

- [ ] **Step 2: Verify tests fail**

Run `python3 -m pytest tests/test_gr308_35b_two_card_tps.py -v`.

Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement the short lanes and metric enrichment**

Create graph candidates for baseline, small graph sizes, Graph128, and the full
graph set. Create scheduler candidates for baseline, async, batch1536, and
batch2560. Read C1/C16 benchmark JSON to add Median TTFT and Decode Avg TPS.

- [ ] **Step 4: Implement top-two cross-card validation and formal winner**

Rank by C16 aggregate throughput, apply the 3% band, cross-run the top two on
opposite cards, rank by mean C16 throughput, then run the winner with
`short=False`. The 35B campaign may perform its approved formal confirmation.

- [ ] **Step 5: Add the exact shared allowlist**

Include `gr308-35b-tps-lane0`, `gr308-35b-tps-lane1`,
`gr308-35b-tps-cross`, `gr308-35b-tps-formal`, `gr308-80b-short`, and the three
known non-GPU background container names.

- [ ] **Step 6: Verify and commit**

```bash
python3 -m pytest tests/test_gr308_35b_two_card_tps.py -v
python3 -m py_compile scripts/gr308_35b_two_card_tps.py
git add scripts/gr308_35b_two_card_tps.py tests/test_gr308_35b_two_card_tps.py
git commit -m "Implement two-card 35B TPS campaign"
```

### Task 4: Implement the 80B Short-Only Adaptive Matrix

**Files:**
- Create: `scripts/gr308_80b_short_tuning.py`
- Create: `tests/test_gr308_80b_short_tuning.py`

- [ ] **Step 1: Add exact-matrix and terminal-state tests**

Assert A-G names and parameter transitions, TP4 devices `(4,5,6,7)`, input and
output 1024, MTP limited to 0/1, no disabled custom all-reduce, and a root
summary schema with no formal-run field or formal invocation.

- [ ] **Step 2: Verify tests fail**

Run `python3 -m pytest tests/test_gr308_80b_short_tuning.py -v`.

Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement A-D and adaptive graph selection**

Run eager no-MTP, eager MTP1, Graph32 MTP1, and Graph64 MTP1 sequentially.
Allow unused MTP weights only for A. Select the highest C16 Decode Avg TPS among
passing graph candidates.

- [ ] **Step 4: Implement E-G without formal execution**

Run automatic KV from the selected graph, then batch4096, choose the better
batch by C16 Decode Avg TPS, and run async. Rank all passing rows with the 3%
tie rule. Write `matrix-summary.json` and exit after cleanup. Do not construct or
call a `short=False` spec.

- [ ] **Step 5: Add the shared allowlist and verify**

Use the same exact container-name set from Task 3. Run:

```bash
python3 -m pytest tests/test_gr308_80b_short_tuning.py -v
python3 -m py_compile scripts/gr308_80b_short_tuning.py
git diff --check
```

Expected: tests PASS, compilation exits 0, and diff check is clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/gr308_80b_short_tuning.py tests/test_gr308_80b_short_tuning.py
git commit -m "Implement Qwen3 Next 80B short tuning matrix"
```

### Task 5: Deploy, Admit, and Launch Both Campaigns

**Files:**
- Deploy the four modified/created scripts to `/data/infra/benchmarks/gr308-specdecode-refresh-20260911/`
- Remote output: `/data/infra/benchmarks/gr308-35b-two-card-tps-20260915/`
- Remote output: `/data/infra/benchmarks/gr308-80b-short-tuning-20260915/`

- [ ] **Step 1: Run complete local verification**

```bash
python3 -m pytest tests -v
python3 -m py_compile scripts/*.py
git diff --check
```

Expected: all tests PASS, all scripts compile, and diff check is clean.

- [ ] **Step 2: Verify remote admission**

Confirm 8/8 cores OK, all eight logical devices idle, ports 22100/22110/22120
free, no GPU test containers running, model manifests present, and the 80B
directory contains its config, index, and 41 safetensor shards.

- [ ] **Step 3: Deploy scripts and start both units**

Start `gr308-35b-two-card-tps-20260915.service` and
`gr308-80b-short-tuning-20260915.service` with the benchmark directory as their
working directory.

- [ ] **Step 4: Verify first launches**

Check frozen launch JSON: 35B uses devices 0-1 and 2-3 in its two lanes; 80B
uses devices 4-7; all use BF16; no launch contains
`--disable-custom-all-reduce`; 80B records 1024/1024; and all active containers
belong to the exact allowlist.

### Task 6: Completion Boundaries

**Files:**
- Read both remote summary files and hardware telemetry.

- [ ] **Step 1: Confirm 80B stops after short selection**

The 80B unit must become inactive after A-G, devices 4-7 must return to idle,
and no formal 80B directory or 64-request result may exist.

- [ ] **Step 2: Confirm 35B follows its approved terminal path**

The 35B campaign continues through cross-card validation and its formal winner
unless a safety failure stops it.

- [ ] **Step 3: Report evidence**

Report 80B short candidates against the screenshot's C1/C16 TTFT and Decode Avg
TPS reference, identify the selected configuration, and request explicit user
confirmation before any 80B formal test. Report 35B progress or final evidence
separately.
