# Qwen-Image-2.1 FTUCA SDPA Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and benchmark a forward-only BF16 fused SDPA prototype for the fixed Qwen-Image-2.1 TP2 shape on GR308 without running end-to-end generation.

**Architecture:** Compile a standalone PyTorch custom-op shared library with `/usr/local/fant/bin/hipcc`, load it with `torch.ops.load_library`, and expose `fant_sdpa::forward`. Begin with a device-copy smoke kernel, then replace it with tiled online Softmax using BF16x2 loads and FP32 accumulation. A deterministic Python harness compares it with PyTorch math SDPA and enforces numerical and performance gates before any sequence-4096 run.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, torch_fant PrivateUse1, FTUCA/HIP clang 17, C++17, BF16 device headers, Docker.

---

## File structure

- Create `native/fant_sdpa/CMakeLists.txt`: configure PyTorch include/library paths and build the shared operator with `hipcc`.
- Create `native/fant_sdpa/fant_sdpa.cpp`: validate FANT tensors, register `fant_sdpa::forward`, allocate output, and launch the device kernel.
- Create `native/fant_sdpa/fant_sdpa_kernel.hip`: implement the copy smoke kernel and then fixed-shape online-softmax SDPA.
- Create `scripts/fant_sdpa_sdk_probe.py`: report SDK, compiler, headers, BF16 support, and extension loading as structured JSON.
- Create `scripts/fant_sdpa_bench.py`: deterministic correctness, memory, and latency benchmark.
- Create `tests/test_fant_sdpa_bench.py`: test numerical gates, summary statistics, and shape validation without FANT hardware.
- Modify `docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md`: record compiler evidence, results, and the keep/reject decision.

### Task 1: Freeze benchmark gates and result formatting

**Files:**
- Create: `tests/test_fant_sdpa_bench.py`
- Create: `scripts/fant_sdpa_bench.py`

- [ ] **Step 1: Write failing unit tests for numerical and speed gates**

```python
from scripts.fant_sdpa_bench import compare, percentile_summary, passes_gates


def test_compare_and_gates_accept_small_bf16_error():
    import torch
    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    candidate = torch.tensor([1.005, 1.995], dtype=torch.float32)
    metrics = compare(reference, candidate)
    assert metrics["finite"]
    assert passes_gates(metrics, candidate_median=0.90, reference_median=1.0)


def test_speed_gate_requires_five_percent():
    metrics = {"finite": True, "mae": 0.0, "max_abs": 0.0}
    assert not passes_gates(metrics, candidate_median=0.951, reference_median=1.0)
    assert passes_gates(metrics, candidate_median=0.95, reference_median=1.0)


def test_percentile_summary_is_deterministic():
    result = percentile_summary([0.5, 0.1, 0.4, 0.2, 0.3])
    assert result["median_seconds"] == 0.3
    assert result["p10_seconds"] == 0.1
    assert result["p90_seconds"] == 0.5
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python3 -m pytest -q tests/test_fant_sdpa_bench.py`

Expected: import failure because `scripts/fant_sdpa_bench.py` does not exist.

- [ ] **Step 3: Implement hardware-independent helpers and CLI validation**

Implement:

```python
def compare(reference, candidate):
    delta = (reference.float() - candidate.float()).abs()
    return {
        "finite": bool(candidate.isfinite().all()),
        "mae": delta.mean().item(),
        "max_abs": delta.max().item(),
        "exact_fraction": (delta == 0).float().mean().item(),
    }


def passes_gates(metrics, candidate_median, reference_median):
    return (
        metrics["finite"]
        and metrics["mae"] <= 0.01
        and metrics["max_abs"] <= 0.25
        and candidate_median <= reference_median * 0.95
    )
```

Use nearest-rank p10/p90 and require `SEQ_LEN in {16, 128, 1024, 4096}`, `HEADS == 16`, and `HEAD_DIM == 128`.

- [ ] **Step 4: Run the tests and commit**

Run: `python3 -m pytest -q tests/test_fant_sdpa_bench.py`

Expected: all new tests pass.

```bash
git add scripts/fant_sdpa_bench.py tests/test_fant_sdpa_bench.py
git commit -m "test: define FTUCA SDPA benchmark gates"
```

### Task 2: Probe compiler and custom-op loading

**Files:**
- Create: `scripts/fant_sdpa_sdk_probe.py`
- Create: `native/fant_sdpa/CMakeLists.txt`
- Create: `native/fant_sdpa/fant_sdpa.cpp`
- Create: `native/fant_sdpa/fant_sdpa_kernel.hip`

- [ ] **Step 1: Add a structured SDK probe**

The script must verify these exact paths and print a single `FANT_SDPA_SDK` JSON object:

```python
REQUIRED = (
    "/usr/local/fant/bin/hipcc",
    "/usr/local/fant/include/hip/hip_bfloat16.h",
    "/usr/local/fant/include/hip/devicelib/bfloat16/bfloat162_arithemtic.h",
    "/usr/local/fant/include/fhblas/fhblas.h",
    "/usr/local/fant/lib/libfhblas.so.4.0",
)
```

Also record `torch.__version__`, `torch_fant.__file__`, compiler version, and whether the built library exposes `torch.ops.fant_sdpa.copy_smoke`.

- [ ] **Step 2: Add the initial operator registration**

Register schemas in `fant_sdpa.cpp`:

```cpp
TORCH_LIBRARY(fant_sdpa, m) {
  m.def("copy_smoke(Tensor input) -> Tensor");
  m.def("forward(Tensor query, Tensor key, Tensor value) -> Tensor");
}
TORCH_LIBRARY_IMPL(fant_sdpa, PrivateUse1, m) {
  m.impl("copy_smoke", &copy_smoke);
  m.impl("forward", &sdpa_forward);
}
```

`copy_smoke` must reject non-FANT, non-BF16, or non-contiguous input, allocate `at::empty_like`, and launch a BF16 device-copy kernel on the current stream.

- [ ] **Step 3: Build with hipcc**

Configure CMake with paths printed by `torch.utils.cpp_extension.include_paths()` and `library_paths()`. Compile both translation units with `hipcc -std=c++17 -fPIC`, link a shared library against `torch`, `torch_cpu`, `c10`, `torch_python`, and the FANT runtime, and preserve the PyTorch ABI flag returned by `torch._C._GLIBCXX_USE_CXX11_ABI`.

Run inside the existing image:

```bash
cmake -S /work/native/fant_sdpa -B /work/build/fant_sdpa \
  -DPYTHON_EXECUTABLE=$(command -v python)
cmake --build /work/build/fant_sdpa -j2
```

Expected: `/work/build/fant_sdpa/libfant_sdpa.so` exists.

- [ ] **Step 4: Run the device-copy smoke test on logical device 6**

```python
torch.ops.load_library("/work/build/fant_sdpa/libfant_sdpa.so")
torch.fant.set_device(6)
x = torch.randn((4096,), dtype=torch.bfloat16).to("fant:6")
y = torch.ops.fant_sdpa.copy_smoke(x)
torch.fant.synchronize()
assert torch.equal(x.cpu(), y.cpu())
```

Expected: exact equality, finite output, and no change to existing containers.

- [ ] **Step 5: Commit the reproducible build probe**

```bash
git add native/fant_sdpa scripts/fant_sdpa_sdk_probe.py
git commit -m "feat: add FTUCA PyTorch operator build probe"
```

### Task 3: Implement fixed-shape online-softmax SDPA

**Files:**
- Modify: `native/fant_sdpa/fant_sdpa.cpp`
- Modify: `native/fant_sdpa/fant_sdpa_kernel.hip`
- Modify: `scripts/fant_sdpa_bench.py`

- [ ] **Step 1: Enforce the fixed operator contract**

In `sdpa_forward`, require three contiguous tensors with equal shape, BF16 dtype, FANT device, rank 4, batch 1, heads 16, head dimension 128, and sequence in `{16, 128, 1024, 4096}`. Allocate an output with the query layout and launch on its device/current stream.

- [ ] **Step 2: Implement the diagnostic kernel**

For one `(head, query)` row per block, calculate all 128-element QK dot products with BF16x2 loads and FP32 accumulation. Apply the scale constant `0.08838834764831845f`. Maintain online state:

```cpp
new_max = max(running_max, tile_max);
old_scale = expf(running_max - new_max);
running_sum = running_sum * old_scale + tile_exp_sum;
output_acc[d] = output_acc[d] * old_scale + tile_weighted_v[d];
running_max = new_max;
```

After all K/V tiles, store `output_acc[d] / running_sum` as BF16. Synchronize within a block around shared K/V tiles and use warp reductions for score maxima and sums.

- [ ] **Step 3: Validate diagnostic sequences**

Run `SEQ_LEN=16`, then `SEQ_LEN=128`, each with a fixed CPU seed of `20260922`. Require finite output, MAE no greater than 0.01, and maximum absolute error no greater than 0.25 against math SDPA.

- [ ] **Step 4: Diagnose numerical failure before performance work**

If a gate fails, emit the failing head/query/channel, reference value, candidate value, running maximum, and running sum. Correct scaling, BF16 conversion, bounds, or reduction ordering and rerun both diagnostic sequences. Do not loosen the gates.

- [ ] **Step 5: Commit the correct fixed-shape kernel**

```bash
git add native/fant_sdpa scripts/fant_sdpa_bench.py
git commit -m "feat: implement fixed-shape FTUCA online SDPA"
```

### Task 4: Run the sequence-1024 performance gate

**Files:**
- Modify: `scripts/fant_sdpa_bench.py`
- Modify: `docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md`

- [ ] **Step 1: Complete synchronized benchmark sampling**

Create deterministic tensors shaped `[1, 16, 1024, 128]`, warm each candidate three times, and collect ten alternating samples with `torch.fant.synchronize()` immediately before and after each operation. Record allocated and peak bytes after resetting peak statistics.

- [ ] **Step 2: Benchmark in a fresh container on logical device 6**

Compare:

```python
reference = lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
candidate = lambda: torch.ops.fant_sdpa.forward(q, k, v)
```

Expected: structured `FANT_SDPA_BENCH` JSON containing both timing distributions and numerical metrics.

- [ ] **Step 3: Apply the hard decision gate**

Keep the candidate only if it meets all numerical limits and its median is at most 95% of the reference median. If slower, run one profiler pass to distinguish launch overhead, dot-product throughput, Softmax, or memory traffic. Permit one bounded change to tile size, vector width, or work assignment, then repeat the fresh-container benchmark.

- [ ] **Step 4: Record and commit the 1024 result**

Document compiler version, device, exact shape, medians, p10/p90, errors, peak memory, and the keep/reject decision.

```bash
git add scripts/fant_sdpa_bench.py docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md
git commit -m "perf: evaluate FTUCA SDPA at sequence 1024"
```

### Task 5: Validate 4096 and a real TP2 block only after acceptance

**Files:**
- Modify: `scripts/tp2_block_probe.py`
- Modify: `docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md`
- Modify: `README.md`

- [ ] **Step 1: Repeat the accepted candidate twice at sequence 4096**

Run in two fresh containers on logical device 6. Require both runs to meet the numerical gate and at least 5% median speedup. Record peak memory to verify that the full 512 MiB BF16 score matrix is not retained.

- [ ] **Step 2: Add an opt-in TP2 block mode**

Add `ATTENTION_MODE=math|fant_sdpa` to `tp2_block_probe.py`. Load the shared library only for `fant_sdpa`; leave `math` as default. Preserve the existing RMSNorm and RoPE modes.

- [ ] **Step 3: Run real Block 0 on logical devices 6 and 7**

Use ten iterations, `RMS_NORM_MODE=vllm_fant`, `ROTARY_MODE=iformer`, and `COMPARE_EAGER=1`. Require optimized-versus-reference MAE no greater than 0.01, maximum absolute error no greater than 0.25 for the attention output, finite final output, and at least 5% block-latency improvement.

- [ ] **Step 4: Verify and document**

Run:

```bash
python3 -m pytest -q
python3 -m compileall -q src scripts tests
git diff --check
```

Document that no end-to-end image test was run. If any acceptance condition fails, retain only the benchmark/probe evidence and do not wire the candidate into `tp2_full_dit.py`.

- [ ] **Step 5: Commit the final decision**

```bash
git add README.md docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md scripts/tp2_block_probe.py
git commit -m "docs: conclude FTUCA fused SDPA prototype"
```
