# Qwen3.6-35B-A3B Two-Card TPS Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an evidence-preserving two-card Qwen3.6-35B-A3B TPS sweep, cross-check the top two candidates on opposite cards, and formally confirm the stable winner.

**Architecture:** Extend the reusable adaptive `Spec` with an optional per-instance container name so two TP2 instances of the same model can coexist. A focused campaign runner starts two independent short-test lanes, collects identical C1/C16 metrics, cross-runs the top two candidates, applies the 3% tie rule, and runs one 64-request formal confirmation. The existing worker remains responsible for correctness, health, temperature, container, and reserved-card safety gates.

**Tech Stack:** Python 3, multiprocessing, systemd transient services, Docker, vLLM 0.18.0 with FANT plugin, pytest-style unit tests, JSON evidence files.

---

### Task 1: Support Two Same-Model Container Instances

**Files:**
- Modify: `scripts/gr308_adaptive_short_matrix.py:31-54`
- Modify: `scripts/gr308_adaptive_short_matrix.py:121-130`
- Test: `tests/test_adaptive_short_matrix.py`

- [ ] **Step 1: Write the failing test**

Add a test proving that a spec-level container name overrides the legacy model mapping:

```python
def test_spec_container_name_allows_parallel_same_model_instances():
    module = load_module()
    spec = module.Spec(
        "lane1", "Qwen3.6-35B-A3B", (2, 3), 22110, "8-15", 3,
        (4, 8, 16, 32, 64), 2048, 1088, None,
        container_name="gr308-35b-tps-lane1",
    )

    job = module._job(spec, Path("/tmp/results"))

    assert job.container_name == "gr308-35b-tps-lane1"
    assert job.devices == (2, 3)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python3 -m pytest tests/test_adaptive_short_matrix.py::test_spec_container_name_allows_parallel_same_model_instances -v
```

Expected: FAIL because `Spec` does not accept `container_name`.

- [ ] **Step 3: Add the optional field and use it in `_job`**

Append the field after the existing defaulted fields:

```python
class Spec(NamedTuple):
    # existing fields remain unchanged
    short: bool = True
    container_name: str | None = None
```

Change `_job` to resolve the name once:

```python
container_name = spec.container_name or CONTAINER_NAMES[spec.model]
return base.Job(
    spec.model, tuple(spec.devices), spec.port, spec.cpu_set,
    root / f"{spec.name.lower()}-{spec.model.lower().replace('.', '-')}",
    container_name, spec.mtp, execution, spec.graph, spec.short,
)
```

- [ ] **Step 4: Run the adaptive-runner tests**

Run:

```bash
python3 -m pytest tests/test_adaptive_short_matrix.py -v
python3 -m py_compile scripts/gr308_adaptive_short_matrix.py
```

Expected: all adaptive tests PASS and compilation exits 0.

- [ ] **Step 5: Commit the reusable runner change**

```bash
git add scripts/gr308_adaptive_short_matrix.py tests/test_adaptive_short_matrix.py
git commit -m "Support parallel same-model tuning lanes"
```

### Task 2: Define the Focused Matrix and Ranking Rules

**Files:**
- Create: `scripts/gr308_35b_two_card_tps.py`
- Create: `tests/test_gr308_35b_two_card_tps.py`

- [ ] **Step 1: Write matrix and ranking tests**

Create tests with direct module loading and these assertions:

```python
def test_lanes_isolate_graph_and_scheduler_dimensions():
    module = load_module()
    lanes = module.lane_specs()
    assert len(lanes["graph"]) == 4
    assert len(lanes["scheduler"]) == 4
    assert {spec.devices for spec in lanes["graph"]} == {(0, 1)}
    assert {spec.devices for spec in lanes["scheduler"]} == {(2, 3)}
    assert all(spec.mtp == 3 and spec.block_size == 1088 for specs in lanes.values() for spec in specs)
    assert all(spec.kv_bytes is None and spec.short for specs in lanes.values() for spec in specs)
    assert len({spec.container_name for specs in lanes.values() for spec in specs}) == 2


def test_rank_uses_c16_then_c1_and_ttft_inside_three_percent_band():
    module = load_module()
    rows = [
        {"name": "raw-top", "status": "passed", "c16_tps": 200.0, "c1_tps": 40.0,
         "c16_median_ttft_ms": 5000.0},
        {"name": "balanced", "status": "passed", "c16_tps": 196.0, "c1_tps": 45.0,
         "c16_median_ttft_ms": 4000.0},
    ]
    assert module.rank_candidates(rows)[0]["name"] == "balanced"
```

- [ ] **Step 2: Run tests and verify both fail**

```bash
python3 -m pytest tests/test_gr308_35b_two_card_tps.py -v
```

Expected: FAIL because the campaign module does not exist.

- [ ] **Step 3: Implement exact lane definitions**

Create `lane_specs()` with these eight specs:

```python
def lane_specs() -> dict[str, list[matrix.Spec]]:
    fixed = dict(model="Qwen3.6-35B-A3B", mtp=3, block_size=1088,
                 kv_bytes=None, max_num_seqs=16, short=True)
    graph = [
        make_spec("G-base", (0, 1), 22100, "0-7", (4, 8, 16, 32, 64), 2048,
                  "gr308-35b-tps-lane0", **fixed),
        make_spec("G-small", (0, 1), 22100, "0-7", (1, 2, 4, 8, 16, 32, 64), 2048,
                  "gr308-35b-tps-lane0", **fixed),
        make_spec("G-128", (0, 1), 22100, "0-7", (4, 8, 16, 32, 64, 128), 2048,
                  "gr308-35b-tps-lane0", **fixed),
        make_spec("G-full", (0, 1), 22100, "0-7", (1, 2, 4, 8, 16, 32, 64, 128), 2048,
                  "gr308-35b-tps-lane0", **fixed),
    ]
    scheduler = [
        make_spec("S-base", (2, 3), 22110, "8-15", (4, 8, 16, 32, 64), 2048,
                  "gr308-35b-tps-lane1", **fixed),
        make_spec("S-async", (2, 3), 22110, "8-15", (4, 8, 16, 32, 64), 2048,
                  "gr308-35b-tps-lane1", async_scheduling=True, **fixed),
        make_spec("S-batch1536", (2, 3), 22110, "8-15", (4, 8, 16, 32, 64), 1536,
                  "gr308-35b-tps-lane1", **fixed),
        make_spec("S-batch2560", (2, 3), 22110, "8-15", (4, 8, 16, 32, 64), 2560,
                  "gr308-35b-tps-lane1", **fixed),
    ]
    return {"graph": graph, "scheduler": scheduler}
```

`make_spec` must construct `matrix.Spec` using keyword arguments so defaulted
fields cannot be shifted by future schema additions.

- [ ] **Step 4: Implement metric enrichment and ranking**

Read `c1/benchmark.json` and `c16/benchmark.json` next to each result and add
median TTFT, decode TPS (`1000 / mean_tpot_ms`), completed, and failed. Exclude
failed rows. Rank candidates within 97% of the highest C16 aggregate TPS by C1
aggregate TPS, then by lower C16 median TTFT:

```python
def rank_candidates(rows: list[dict]) -> list[dict]:
    valid = [row for row in rows if row.get("status") == "passed"]
    if len(valid) < 2:
        raise RuntimeError("fewer than two valid short candidates")
    ceiling = max(row["c16_tps"] for row in valid)
    band = [row for row in valid if row["c16_tps"] >= ceiling * 0.97]
    preferred = sorted(band, key=lambda row: (-row["c1_tps"], row["c16_median_ttft_ms"]))
    remainder = sorted((row for row in valid if row not in band), key=lambda row: -row["c16_tps"])
    return preferred + remainder
```

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest tests/test_gr308_35b_two_card_tps.py -v
python3 -m py_compile scripts/gr308_35b_two_card_tps.py
git add scripts/gr308_35b_two_card_tps.py tests/test_gr308_35b_two_card_tps.py
git commit -m "Add focused two-card 35B TPS matrix"
```

Expected: tests PASS, compilation exits 0, and commit succeeds.

### Task 3: Add Parallel Lanes, Cross-Card Runs, and Formal Confirmation

**Files:**
- Modify: `scripts/gr308_35b_two_card_tps.py`
- Modify: `tests/test_gr308_35b_two_card_tps.py`

- [ ] **Step 1: Write cross-card transformation tests**

```python
def test_cross_spec_moves_candidate_to_opposite_card():
    module = load_module()
    source = module.lane_specs()["graph"][0]
    crossed = module.cross_spec(source, "cross-g-base")
    assert crossed.devices == (2, 3)
    assert crossed.port == 22110
    assert crossed.cpu_set == "8-15"
    assert crossed.container_name == "gr308-35b-tps-cross"
    assert crossed.graph == source.graph
    assert crossed.batched_tokens == source.batched_tokens


def test_formal_spec_uses_64_request_worker_mode():
    module = load_module()
    source = module.lane_specs()["scheduler"][0]
    formal = module.formal_spec(source)
    assert formal.devices == (0, 1)
    assert formal.port == 22100
    assert formal.short is False
```

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
python3 -m pytest tests/test_gr308_35b_two_card_tps.py -v
```

Expected: FAIL because `cross_spec` and `formal_spec` do not exist.

- [ ] **Step 3: Implement the transformations**

```python
def cross_spec(source: matrix.Spec, name: str) -> matrix.Spec:
    if tuple(source.devices) == (0, 1):
        placement = {"devices": (2, 3), "port": 22110, "cpu_set": "8-15"}
    else:
        placement = {"devices": (0, 1), "port": 22100, "cpu_set": "0-7"}
    return matrix._spec_like(source, name, container_name="gr308-35b-tps-cross", **placement)


def formal_spec(source: matrix.Spec) -> matrix.Spec:
    return matrix._spec_like(
        source, "formal-winner", devices=(0, 1), port=22100, cpu_set="0-7",
        container_name="gr308-35b-tps-formal", short=False,
    )
```

- [ ] **Step 4: Implement orchestration and evidence writing**

Create the output root once. Add every lane, cross, formal, and known non-GPU
background container name to the allowed-job list. Start two multiprocessing
workers for the short lanes and join both. Load their lane summaries, rank valid
rows, and cross-run the top two sequentially on their opposite cards.

For each candidate, compute:

```python
candidate["cross_mean_c16_tps"] = (
    candidate["c16_tps"] + cross_row["c16_tps"]
) / 2.0
```

Discard a candidate if its cross run fails. Apply the 3% band to
`cross_mean_c16_tps`, then prefer higher mean C1 TPS and lower mean C16 TTFT.
Run `formal_spec(winner)` through `matrix.run_spec`. Write:

```text
campaign.json
lanes/graph/summary.json
lanes/scheduler/summary.json
cross/summary.json
formal/summary.json
summary.json
```

The root summary is `complete` only if the formal row passes and both formal
cases report 64 completed requests and zero failed requests.

- [ ] **Step 5: Run all local verification**

```bash
python3 -m pytest tests/test_gr308_35b_two_card_tps.py tests/test_adaptive_short_matrix.py tests/test_gr308_parallel_goal.py -v
python3 -m py_compile scripts/gr308_35b_two_card_tps.py scripts/gr308_adaptive_short_matrix.py scripts/gr308_parallel_goal.py
git diff --check
```

Expected: all tests PASS, compilation exits 0, and `git diff --check` has no output.

- [ ] **Step 6: Commit orchestration**

```bash
git add scripts/gr308_35b_two_card_tps.py tests/test_gr308_35b_two_card_tps.py
git commit -m "Orchestrate 35B TPS selection and formal confirmation"
```

### Task 4: Deploy and Start the Campaign

**Files:**
- Deploy: `scripts/gr308_parallel_goal.py`
- Deploy: `scripts/gr308_adaptive_short_matrix.py`
- Deploy: `scripts/gr308_35b_two_card_tps.py`
- Create remotely: `/data/infra/benchmarks/gr308-35b-two-card-tps-20260915/`

- [ ] **Step 1: Run read-only remote admission checks**

Verify all of the following before deployment:

```bash
fantgpu-smi --query-core-health
fantgpu-smi
docker ps --format '{{.Names}} {{.Status}}'
ss -ltnp | grep -E ':(22100|22110) '
```

Expected: 8/8 core status OK; logical devices 0-3 and 6-7 are in the accepted
idle range; ports 22100 and 22110 are free; no GPU test container is running.

- [ ] **Step 2: Copy the verified scripts**

```bash
scp -P 22 scripts/gr308_parallel_goal.py \
  scripts/gr308_adaptive_short_matrix.py \
  scripts/gr308_35b_two_card_tps.py \
  root@192.0.2.11:/data/infra/benchmarks/gr308-specdecode-refresh-20260911/
```

Expected: all three transfers complete successfully.

- [ ] **Step 3: Start a transient systemd unit**

```bash
systemd-run --unit=gr308-35b-two-card-tps-20260915 \
  --property=WorkingDirectory=/data/infra/benchmarks/gr308-specdecode-refresh-20260911 \
  --property=StandardOutput=journal \
  --property=StandardError=journal \
  /usr/bin/python3 \
  /data/infra/benchmarks/gr308-specdecode-refresh-20260911/gr308_35b_two_card_tps.py
```

Expected: systemd reports `Running as unit`.

- [ ] **Step 4: Verify both first-lane launches**

Read both `launch.json` files and assert:

```python
assert launch0["logical_functions"] == [0, 1]
assert launch1["logical_functions"] == [2, 3]
assert "--disable-custom-all-reduce" not in launch0["vllm_args"]
assert "--disable-custom-all-reduce" not in launch1["vllm_args"]
assert "--kv-cache-memory-bytes" not in launch0["vllm_args"]
assert "--kv-cache-memory-bytes" not in launch1["vllm_args"]
```

Expected: all assertions pass, both containers remain running, and logical
devices 6 and 7 remain idle.

### Task 5: Completion Verification and Result Handoff

**Files:**
- Read remotely: `/data/infra/benchmarks/gr308-35b-two-card-tps-20260915/summary.json`
- Read remotely: formal winner `benchmark.json` and `result.json`

- [ ] **Step 1: Verify terminal service state and summaries**

```bash
systemctl is-active gr308-35b-two-card-tps-20260915 || true
docker ps --format '{{.Names}} {{.Status}}'
```

Expected after completion: service inactive, no campaign container running,
and root summary status `complete`.

- [ ] **Step 2: Verify formal evidence**

Check that the formal result has status
`native_cases_passed_pending_independent_audit`, both C1 and C16 contain 64
completed requests and zero failures, and frozen launch arguments keep custom
all-reduce enabled.

- [ ] **Step 3: Verify hardware cleanup**

Run `fantgpu-smi --query-core-health` and parse `fantgpu-smi` telemetry.

Expected: 8/8 cores OK, logical devices 0-7 in the accepted idle-memory range,
and no inference process listed.

- [ ] **Step 4: Report the result**

Report the short candidates, cross-card means, formal C1/C16 aggregate TPS,
Decode Avg TPS, Median TTFT, acceptance rate/length, comparison against the
previous formal G1 result, failed candidates with exact causes, and the remote
evidence directory. State that a single formal run supports descriptive rather
than statistical claims.
