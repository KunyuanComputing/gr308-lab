# Qwen-Image-2.1 GR308 Custom FTUCA SDPA Design

## Objective

Build and benchmark a forward-only custom SDPA prototype for the Qwen-Image-2.1 TP2 DiT path on GR308. The prototype succeeds only if it is numerically acceptable and at least 5% faster than the current PyTorch math SDPA at the same shape. This phase runs isolated operator tests only and does not generate an image.

## Confirmed runtime facts

- The current `aten::scaled_dot_product_attention` dispatch for `PrivateUse1/fant` is PyTorch's composite math kernel, not a FANT fused kernel.
- Its `mm`, `bmm`, and `_softmax` primitives are implemented by `torch_fant`.
- The installed FTUCA SDK contains BF16 and BF16x2 device headers, warp shuffle/reduction support, dynamic shared memory, and `fhBLAS` BF16 GEMM support.
- No public FTUCA MMA/MFMA header or existing fused attention entry point has been found.
- The target TP2 shape is BF16 `[1, 16, S, 128]` per logical chip, initially with `S=1024`, then `S=4096` only after the first performance gate passes.

## Approaches considered

### Compose existing BMM and Softmax

Wrap `torch_fant` BMM, Softmax, and BMM in a custom namespace. This verifies registration but retains the full score matrix and three separate kernels, so it cannot provide the intended memory traffic reduction.

### Hybrid fhBLAS plus custom Softmax

Call BF16 strided batched GEMM for QK, run a custom Softmax kernel, then call GEMM for PV. This provides a useful explicit baseline and may allow limited launch tuning, but still writes the full score matrix.

### Native FTUCA online-softmax kernel

Tile K and V, calculate QK scores, maintain a running FP32 maximum and normalization sum, and accumulate the output before writing BF16 once. This is the selected experimental path because it removes the full score matrix. Without a public matrix-core intrinsic it may be slower than fhBLAS, so it is guarded by an early performance gate.

## Scope

The first native kernel supports:

- forward inference only;
- BF16 Q, K, V and BF16 output;
- FP32 dot-product and online-softmax accumulation;
- batch size 1;
- 16 local attention heads;
- head dimension 128;
- sequence lengths 1024 and 4096;
- no dropout, causal mask, arbitrary attention mask, backward pass, or segment packing.

Unsupported shapes must fail with a clear error rather than silently fall back. Segment support and general batch support are deferred until the fixed-shape kernel passes both gates.

## Components

### SDK capability probe

A reproducible script records compiler path and version, target architecture, BF16 type compilation, extension linkage, device launch, and PyTorch tensor pointer access. Failure here stops native kernel work and leaves a documented SDK gap.

### PyTorch extension

A small extension registers `torch.ops.fant_sdpa.forward(q, k, v)`. The C++ boundary validates device, dtype, contiguity, rank, head count, sequence length, and head dimension. It launches on the current FANT stream and returns a tensor in `[1, 16, S, 128]` layout.

### Reference and benchmark harness

The harness compares the custom result against `torch.nn.functional.scaled_dot_product_attention`, synchronizes around every sample, uses at least three warmups and ten measured iterations, and records median, p10, p90, peak memory, finite status, MAE, maximum absolute error, and exact fraction.

### Native kernel

The kernel assigns work by head and query tile. It reads K/V in tiles, computes dot products scaled by `1/sqrt(128)`, applies online Softmax with FP32 running state, rescales the running output when the maximum changes, and converts the final accumulator to BF16. Shared memory and BF16x2 loads are used where supported by the compiler.

## Gates

### Build and correctness gate

The extension must compile inside the existing FANT image and operate on logical device 6 without changing running services. Required numerical limits are:

- all outputs finite;
- MAE no greater than 0.01;
- maximum absolute error no greater than 0.25.

Testing starts at a small diagnostic sequence and then sequence 1024.

### Performance gate

At sequence 1024, the custom kernel must be at least 5% faster than math SDPA. A result within 5% is recorded as inconclusive and is not integrated. A slower result stops sequence-4096 native testing unless profiling identifies one bounded implementation defect.

If it passes, repeat in a fresh container and then test sequence 4096 twice. Integration into the TP2 block remains opt-in until a real-block comparison passes the same numerical limits and improves block latency by at least 5%.

## Failure handling

Compiler, linker, device launch, unsupported instruction, out-of-memory, and numerical failures are captured as structured results. The prototype never replaces global PyTorch SDPA registration. It uses a separate namespace and an explicit feature flag, so failure cannot affect the existing BF16 TP2 path.

## Expected outcome

The investigation will produce one of two useful results: a validated faster fixed-shape fused SDPA candidate, or precise evidence that the public FTUCA instruction surface cannot outperform the current `torch_fant`/fhBLAS path without a vendor matrix-core or fused-attention interface.
