# Analysis report

## Scope

This analysis covers Qwen3.5-9B, Qwen3-30B-A3B, Qwen3.6-35B-A3B, and Qwen3-Next-80B-A3B-Instruct on four physical GR308 cards. It consolidates the external C1/C16 comparison table, the parameter-ablation campaigns, the EP/MoE backend attempts, and the final C32 short tests.

It also includes the user-supplied RTX 5090 comparison for 35B and 9B only. The 35B RTX 5090 value is the eight-card aggregate divided by eight; the original run used 4 x TP2 and is not an independently reproduced single-card result. The 9B row is closer to a like-for-like comparison at BF16, one card, and C16, but the request workload differs.

The primary decision metric is aggregate output throughput. Median TTFT and per-request Decode Avg TPS are retained as guardrails so that throughput gains are not presented without their latency cost.

## Evidence validation

- Qwen3.5-9B and Qwen3-30B-A3B final C1/C16 values are 64-request formal runs with zero failures.
- The 35B MTP4 and 80B MTP3 selections are single short-test candidates. Their C1/C16 values are not stability-qualified production claims.
- C32 values for all four models are single short runs with 64 completed requests and zero failures.
- The first 9B C32 attempt is invalid and excluded because per-replica `max-num-seqs=32` caused a non-contiguous `ssm_state_indices` failure. The corrected run uses 16 sequences per DP replica.
- No confidence interval or significance claim is permitted because the selected short-test candidates do not have at least three matched repetitions.

## Principal findings

- 30B is the strongest one-card peak-throughput candidate: 320.90 tok/s at C32 and 40.1% of the estimated batch roof.
- 9B reaches 287.02 tok/s at C32, but its median TTFT rises to 8.81 seconds; C16 is the more balanced service point.
- 35B reaches 180.64/208.08 tok/s at C16/C32 with MTP4. Graph128, batch6144, Triton grouped GEMM, and EP2 do not clear the 3% adoption threshold.
- 80B reaches the highest total C32 throughput, 339.26 tok/s, but requires two physical cards and therefore provides 169.63 tok/s per card.
- EP is a regression for all completed MoE comparisons. The TP2 x DP2 x EP2 80B route also remains blocked by a firmware semaphore timeout during profiling.
- Custom all-reduce remains enabled in every recommended configuration.
- For 35B C16, GR308 reaches 180.64 tok/s versus an RTX 5090 per-card normalized value of 498.78 tok/s, or 36.22%. The RTX 5090 value is 3990.23/8 from a 4 x TP2 system and must not be described as a standalone one-card benchmark.
- For 9B C16, GR308 reaches 247.16 tok/s versus RTX 5090 at 444.40 tok/s, or 55.62%. Both are BF16, one-card, C16 results, but the 5090 workload uses ShareGPT with maximum output 256 rather than fixed 3072/1024.

## Claim candidates

- Claim:
  - 30B is the best current one-card high-concurrency deployment candidate among the tested configurations.
  - Source evidence: 320.90 tok/s at C32; one physical card; 40.1% estimated batch-roof efficiency.
  - Allowed wording: “best measured one-card C32 throughput in this campaign.”
  - Forbidden stronger wording: “universally fastest model on GR308.”
  - Uncertainty: single short C32 run; different model capabilities are not normalized.
  - Next check: three matched C32 repetitions with fixed NUMA binding.
  - Decision: keep.

- Claim:
  - MTP is the dominant tested optimization for 35B and 80B.
  - Source evidence: 35B MTP3 to MTP4 and 80B no-MTP to MTP1/2/3 ablations.
  - Allowed wording: “dominant gain within the tested parameter family.”
  - Forbidden stronger wording: “MTP is the only remaining optimization.”
  - Uncertainty: kernel-level alternatives were not exhaustively implemented.
  - Next check: record draft acceptance and verify-kernel time across repetitions.
  - Decision: keep.

- Claim:
  - EP should not be enabled in the current production candidates.
  - Source evidence: 30B EP2, 35B EP2, and 80B EP4 all regress in aggregate throughput; 80B TP2 x EP2 does not complete profiling.
  - Allowed wording: “not recommended on the tested software stack and topology.”
  - Forbidden stronger wording: “EP can never improve these models.”
  - Uncertainty: future firmware, dispatch/combine kernels, or interconnect changes may reverse the result.
  - Next check: re-evaluate after native token-sorting and grouped-GEMM fusion becomes available.
  - Decision: keep.

- Claim:
  - At C16, the reported GR308/RTX 5090 ratios are 36.22% for 35B and 55.62% for 9B.
  - Source evidence: 35B GR308 180.64 vs RTX 5090 normalized 498.78; 9B GR308 247.16 vs RTX 5090 444.40.
  - Allowed wording: “35B per-card normalized ratio” and “9B same-concurrency ratio.”
  - Forbidden stronger wording: “35B standalone single-card comparison.”
  - Uncertainty: 35B is derived from an eight-card 4 x TP2 run; 9B request shapes differ.
  - Next check: reproduce fixed 3072/1024 C16 directly on one RTX 5090.
  - Decision: keep with caveat.

## Decision

Retain the final configuration table in the main PDF. Use C16 as the balanced operating point and C32 as the peak-throughput point. Require repeated acceptance testing before treating the 35B and 80B short-test candidates as production-qualified.
