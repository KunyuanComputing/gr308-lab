# Statistics appendix

## Formulas

- Decode Avg TPS: `1000 / mean_tpot_ms`.
- System roof efficiency: `best C16 aggregate output TPS / ideal batch roof TPS`.
- Decode-stage equivalent efficiency: `best C16 aggregate output TPS / decode-stage roof TPS`.
- Reference delta: `(best / reference - 1) × 100%`; for TTFT, a negative delta is better.

## Selected C1/C16 comparisons

| Model | Concurrency | Reference TTFT | Best TTFT | TTFT delta | Reference Decode TPS | Best Decode TPS | Decode delta | Best aggregate TPS | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 35B | 1 | 1731.87 | 1537.19 | -11.2% | 52.91 | 50.75 | -4.1% | 47.21 | short |
| 35B | 16 | 3487.11 | 3321.47 | -4.7% | 12.74 | 13.93 | +9.4% | 180.64 | short |
| 9B | 1 | 1545.07 | 2335.93 | +51.2% | 45.09 | 54.64 | +21.2% | 48.62 | formal |
| 9B | 16 | 2487.47 | 3501.62 | +40.8% | 11.97 | 18.90 | +57.9% | 247.16 | formal |
| 30B | 1 | 1245.71 | 1054.43 | -15.4% | 26.85 | 26.43 | -1.5% | 25.76 | formal |
| 30B | 16 | 3384.84 | 4225.69 | +24.8% | 11.70 | 15.22 | +30.1% | 225.67 | formal |
| 80B | 1 | 1072.45 | 607.98 | -43.3% | 33.23 | 48.52 | +46.0% | 47.13 | short |
| 80B | 16 | 6048.71 | 3535.56 | -41.5% | 15.02 | 19.71 | +31.2% | 255.93 | short |

## C32 results

| Model | Recommended configuration | Median TTFT | Decode Avg TPS | Aggregate TPS | Completed/failed |
| --- | --- | ---: | ---: | ---: | ---: |
| 35B | TP2, MTP4, Graph 4–64, batch 4096, auto KV | 4909.15 | 8.20 | 208.08 | 64/0 |
| 30B | TP2, auto MoE, Graph 4–32, batch 4096, block 800, 8 GiB KV, async | 5616.56 | 11.33 | 320.90 | 64/0 |

The 35B batch-6144 C32 run reached 209.08 tok/s, only 0.48% above batch 4096, while C16 fell by 0.50% and TTFT increased. It is recorded as a single-cell maximum, not the recommended deployment setting. The 30B Triton grouped-GEMM C32 run reached 321.35 tok/s, only 0.14% above auto backend; this is below the campaign's 3% adoption threshold.

## Roofline comparison

| Model | Ideal batch roof | Best C16 aggregate | System efficiency | Decode-stage roof | Decode equivalent efficiency | Per physical card |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 35B | 1066 | 180.64 | 16.9% | 864.31 | 20.9% | 180.64 |
| 9B | 896 | 247.16 | 27.6% | 731.24 | 33.8% | 247.16 |
| 30B | 800 | 225.67 | 28.2% | 739.90 | 30.5% | 225.67 |
| 80B | 1420 | 255.93 | 18.0% | 1267.65 | 20.2% | 127.97 |

The roof values are planning estimates from the prior calculation, not direct hardware counters. Efficiency should be used for relative deployment screening, not as a utilization counter.

## Qwen3.5-9B C1 single-replica efficiency

At concurrency 1, the dual-TP1/DP2 service activates only one TP1 replica for the request. Against the estimated 56 tok/s single-replica roof:

- Per-request decode efficiency: `54.6398 / 56 = 97.6%`.
- End-to-end aggregate output efficiency: `48.6222 / 56 = 86.8%`.

These C1 ratios are not comparable to the C16 full-card system-roof efficiency, `247.1625 / 896 = 27.6%`, because concurrency, active replicas, metric numerator, and theoretical denominator all differ.
