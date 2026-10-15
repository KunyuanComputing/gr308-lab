# Analysis report

## Scope and metric normalization

This analysis covers Qwen3.5-9B, Qwen3-30B-A3B, Qwen3.6-35B-A3B, and Qwen3-Next-80B-A3B-Instruct on GR308. Qwen3.6-27B is excluded because it was not requested in the final summary.

Two throughput metrics are kept separate:

- `Decode Avg TPS = 1000 / mean_tpot_ms`, measured per request.
- `aggregate_output_tps`, measured across the complete service instance.

The external reference image supplies TTFT and Decode Avg TPS but not aggregate throughput. Roofline utilization therefore uses only locally measured aggregate throughput.

## Principal findings

- 9B achieves the highest one-card C16 aggregate throughput: 247.16 tok/s. Its dual-TP1/DP2 layout is the dominant gain, while MTP4 improves the decode path.
- 30B reaches 225.67 tok/s in the formal C16 run and 320.90 tok/s at C32. EP is a clear regression in the tested software stack; pure TP2 remains preferable.
- 35B reaches 180.64 tok/s at C16 and 208.08 tok/s at C32 with MTP4. MTP depth and batched-token capacity matter more than the tested MoE backend or EP changes.
- 80B reaches 255.93 tok/s at C16 with TP4/MTP3, but uses two physical cards. TP4+EP4 is slower, and TP2×DP2×EP cannot complete engine profiling reliably in the current stack.

## Evidence strength

- 9B and 30B C1/C16 values are 64-request formal runs with zero failures.
- 35B MTP4 and 80B MTP3 values are single short-test candidates (16 C1 requests and 32 C16 requests). They are best measured values, not stability-qualified deployment claims.
- No confidence intervals or significance claims are made because the candidate configurations do not have three or more matched independent repetitions.

## Decision

Retain the recommended configurations in the main report. Treat sub-1% differences such as 30B Triton grouped GEMM versus auto backend, or 35B batch 6144 versus 4096 at C32, as operationally equivalent until repeated. Prefer the configuration with lower complexity and better TTFT.
