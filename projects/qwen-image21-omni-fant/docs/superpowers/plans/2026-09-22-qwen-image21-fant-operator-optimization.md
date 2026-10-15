# Qwen-Image-2.1 FANT Operator Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify and retain numerically correct GR308 operator implementations that improve Qwen-Image-2.1 TP2 attention or elementwise latency by at least 5% without running end-to-end image generation.

**Architecture:** Add one isolated benchmark harness that loads a real checkpoint block, creates deterministic TP2 inputs, and runs interchangeable attention and elementwise candidates against the existing eager reference. Keep candidate discovery, timing, correctness checks, and JSON evidence in the harness; record accepted and rejected candidates in the existing test report.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, torch_fant, FHCCL, Diffusers QwenImage21, optional `torch.compile`/`flex_attention`, Docker.

---

## File structure

- Create `scripts/attention_operator_bench.py`: real-checkpoint SDPA/backend/flex/chunked candidate benchmark.
- Create `scripts/elementwise_operator_bench.py`: Norm, modulation, RoPE and residual candidate benchmark.
- Modify `scripts/tp2_block_probe.py`: add real-block vendor operator comparison and stable timing.
- Modify `docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md`: record raw measurements, failures and decisions.
- Modify `README.md`: identify the selected operator path and invocation.

### Task 1: Freeze the attention benchmark

**Files:**
- Create: `scripts/attention_operator_bench.py`

- [x] **Step 1: Implement deterministic real-checkpoint inputs**

Load transformer block 0, shard 32 heads to 16 heads per rank, and create BF16 Q/K/V with CPU seed `20260922`. Support `SEQ_LEN=1024|4096`, `WARMUP>=3`, and `ITERATIONS>=10`.

- [x] **Step 2: Implement correctness reporting**

For each candidate print finite status, MAE, maximum absolute error, exact fraction and peak memory against eager `F.scaled_dot_product_attention`.

- [x] **Step 3: Implement robust timing**

Synchronize the FANT device around each sample, use barriers outside candidate timing, and report median, p10, p90 and all samples. Randomize candidate order after warmup to reduce order bias.

- [x] **Step 4: Verify the harness**

Run:

```bash
python3 -m compileall -q scripts/attention_operator_bench.py
```

Expected: exit 0. On GR308 run sequence 1024 first; the eager candidate must pass with zero self-difference and at least ten timing samples.

- [x] **Step 5: Commit**

```bash
git add scripts/attention_operator_bench.py
git commit -m "perf: add FANT attention operator benchmark"
```

### Task 2: Test available fused attention backends

**Files:**
- Modify: `scripts/attention_operator_bench.py`
- Modify: `docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md`

- [x] **Step 1: Inventory runtime backends**

Record availability of `flash_attn`, attention-related `torch_fant` and `torch.ops.fant` symbols, PyTorch SDPA backend controls, `flex_attention`, and compatible vLLM custom ops.

- [x] **Step 2: Add explicit SDPA backend candidates**

For every backend exposed on FANT, run it in an isolated try/except candidate and record unsupported backend errors instead of aborting the benchmark.

- [x] **Step 3: Add compiled flex candidate**

Compile a fixed-shape flex attention callable before timed iterations. Exclude compilation time and reject the candidate if compilation falls back to dense FP32, exceeds memory, or fails the numerical gate.

- [x] **Step 4: Run both sequence sizes**

Execute sequence 1024, then 4096 only for candidates that remain healthy. Preserve stdout under `logs/attention-operator-*.log`.

- [x] **Step 5: Record decision and commit**

Document every candidate's median, error and status, then commit:

```bash
git add scripts/attention_operator_bench.py docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md
git commit -m "perf: evaluate FANT fused attention paths"
```

### Task 3: Test exact chunked attention fallback

**Files:**
- Modify: `scripts/attention_operator_bench.py`
- Modify: `docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md`

- [x] **Step 1: Implement online-softmax chunking**

Process fixed query and key chunks without materializing the full score matrix. Accumulate running maximum, normalization sum and value result in FP32, then return BF16 in the reference layout. Expose chunk sizes 128, 256 and 512.

- [x] **Step 2: Run the numerical gate at sequence 1024**

Reject any chunk size with non-finite output, MAE above 0.01 or maximum absolute error above 0.25.

- [x] **Step 3: Benchmark surviving candidates**

Run at least ten iterations at sequence 1024. Run sequence 4096 only when the candidate is no slower than 1.25 times eager at 1024, since slower candidates cannot meet the acceptance gate.

- [x] **Step 4: Record decision and commit**

Retain code only when it is useful for speed or memory; otherwise record the negative result and remove the candidate implementation before commit.

### Task 4: Test elementwise fusion

**Files:**
- Create: `scripts/elementwise_operator_bench.py`
- Modify: `docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md`

- [x] **Step 1: Add eager reference groups**

Benchmark attention LayerNorm plus modulation, Q/K RMSNorm, attention residual, MLP LayerNorm plus modulation, SiLU multiply and MLP residual using the real Block dimensions.

- [x] **Step 2: Add compiled candidates**

Create fixed-shape compiled callables for each group. Compile before measurement and record graph breaks or unsupported operators.

- [x] **Step 3: Run correctness and timing gates**

Use at least ten iterations at sequence 4096. Require finite output, MAE at most 0.01, maximum absolute error at most 0.25, and at least 5% median speedup.

- [x] **Step 4: Record decision and commit**

```bash
git add scripts/elementwise_operator_bench.py docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md
git commit -m "perf: evaluate FANT elementwise fusion"
```

### Task 5: Verify and summarize accepted operators

**Files:**
- Modify: `README.md`
- Modify: `docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md`

- [x] **Step 1: Re-run accepted candidates**

Run the frozen baseline and each accepted candidate in a fresh container. Confirm the result crosses the 5% speed gate twice and remains within the numerical gate.

- [x] **Step 2: Verify repository state**

Run:

```bash
python3 -m pytest -q
python3 -m compileall -q src scripts tests
git diff --check
```

Expected: seven or more tests pass, compileall exits 0, and diff check emits no output.

- [x] **Step 3: Document the selected path**

Report medians, variability, numerical differences, peak memory, rejected candidates, runtime versions, and the fact that no end-to-end image test was run.

- [x] **Step 4: Commit**

```bash
git add README.md docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md scripts
git commit -m "docs: summarize Qwen Image operator optimization"
```
