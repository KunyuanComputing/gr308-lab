# GR308 Three-Card MTP1 / Graph32 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run isolated, disconnect-safe C1/C16 benchmarks for Qwen3.6-35B-A3B, Qwen3.5-9B, and Qwen3-30B-A3B on three physical GR308 cards with Graph32, using MTP1 only where native MTP weights exist.

**Architecture:** Add a dedicated parallel runner derived from the proven remote `run_goal_model.py`, plus a small orchestrator that performs one global admission check and starts three registered card-scoped workers. Workers use separate device pairs, ports, CPU sets, container names, and output directories while sharing the existing immutable model and runtime inputs.

**Tech Stack:** Python 3.10, Docker, vLLM/vllm_fant, fantgpu-smi, systemd, unittest/pytest-compatible tests.

---

### Task 1: Test the card-scoped safety and command contract

**Files:**
- Create: `tests/test_gr308_parallel_goal.py`
- Create: `scripts/gr308_parallel_goal.py`

- [ ] Write tests asserting that Graph capture sizes equal `[32]`, MTP1 is added for 35B/9B, MTP is absent for 30B, the device pairs are `(0,1)`, `(2,3)`, `(4,5)`, ports and CPU sets are distinct, and devices `(6,7)` remain reserved.
- [ ] Run `python3 -m pytest tests/test_gr308_parallel_goal.py -q`; expect failures because the module does not exist.
- [ ] Implement pure configuration helpers `build_serve_args()`, `build_jobs()`, `allowed_container_names()`, and `card_cleanup_ok()` in `scripts/gr308_parallel_goal.py`.
- [ ] Re-run the focused tests; expect all tests to pass.
- [ ] Commit only the new runner and its tests.

### Task 2: Implement isolated workers and orchestration

**Files:**
- Modify: `scripts/gr308_parallel_goal.py`
- Modify: `tests/test_gr308_parallel_goal.py`

- [ ] Add a failing test showing that a worker accepts only the three registered container names and rejects an unexpected fourth container.
- [ ] Add worker execution by reusing the remote benchmark modules for model inspection, correctness checks, C1/C16 requests, metrics, telemetry, and evidence writing.
- [ ] Scope memory assertions and cleanup to the worker's device pair; keep all-device core-health checks and enforce 540 MiB on reserved devices 6/7.
- [ ] Add an orchestrator that performs a single global idle/health preflight, starts all workers, records PIDs/status, and stops all workers if any device health query ceases to be OK.
- [ ] Run `python3 -m pytest tests/test_gr308_parallel_goal.py -q` and `python3 -m py_compile scripts/gr308_parallel_goal.py`; expect success.
- [ ] Commit the implementation and tests.

### Task 3: Deploy and dry-run on the GR308 host

**Files:**
- Deploy: `/data/infra/benchmarks/gr308-specdecode-refresh-20260911/run_parallel_graph32_20260914.py`
- Create at runtime: `/data/infra/benchmarks/gr308-multicard-mtp1-graph32-20260914/`

- [ ] Copy the verified runner to the host without modifying the existing benchmark scripts.
- [ ] Run its configuration-only mode and verify the exact mapping: 35B→0,1/port 22100; 9B→2,3/port 22110; 30B→4,5/port 22120; reserved→6,7.
- [ ] Recheck `fantgpu-smi`, core health, running containers, disk space, model headers, and port availability; abort without launching on any mismatch.
- [ ] Verify 35B/9B report native MTP evidence and 30B reports no native MTP evidence.

### Task 4: Launch and verify live progress

**Files:**
- Runtime evidence: `/data/infra/benchmarks/gr308-multicard-mtp1-graph32-20260914/**`

- [ ] Start the orchestrator as a named systemd service so execution survives SSH disconnect.
- [ ] Confirm all three containers become registered and only device pairs 0/1, 2/3, and 4/5 allocate memory.
- [ ] Confirm each launch record contains BF16, 4096 service context, max sequences 16, batch tokens 2048, fixed KV allocation, Graph `[32]`, and the correct MTP policy.
- [ ] Confirm model identity/correctness gates and the first native benchmark heartbeat; if startup is still in progress, report the live stage without claiming completion.

### Task 5: Completion audit

**Files:**
- Read: `/data/infra/benchmarks/gr308-multicard-mtp1-graph32-20260914/**/result.json`

- [ ] Wait for all workers to finish or fail, preserving raw evidence.
- [ ] Verify every successful case completed 64/64 requests at both C1 and C16 and generated the requested output length.
- [ ] Calculate MTP acceptance and Graph evidence from before/after metrics and logs.
- [ ] Verify all containers stopped, all eight logical devices returned to 540 MiB, and core health is OK.
- [ ] Report throughput, latency, correctness, acceptance, Graph activity, failures, and the explicit 30B no-MTP limitation.
