# Statistics appendix

## Metric formulas

- Decode Avg TPS = `1000 / mean_tpot_ms`.
- Aggregate output throughput = completed output tokens divided by benchmark duration.
- Batch-roof efficiency = aggregate output throughput divided by the estimated ideal batch roof.
- Reference delta = `(selected / reference - 1) x 100%`.
- A negative TTFT delta is an improvement; a positive throughput delta is an improvement.

## Final selected results

| Model | Concurrency | Median TTFT (ms) | Decode Avg TPS | Aggregate output TPS | Completed/failed | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.6-35B-A3B | 1 | 1537.19 | 50.75 | 47.21 | 16/0 | short |
| Qwen3.6-35B-A3B | 16 | 3321.47 | 13.93 | 180.64 | 32/0 | short |
| Qwen3.6-35B-A3B | 32 | 4909.15 | 8.20 | 208.08 | 64/0 | short |
| Qwen3.5-9B | 1 | 2335.93 | 54.64 | 48.62 | 64/0 | formal |
| Qwen3.5-9B | 16 | 3501.62 | 18.90 | 247.16 | 64/0 | formal |
| Qwen3.5-9B | 32 | 8806.52 | 12.62 | 287.02 | 64/0 | short |
| Qwen3-30B-A3B | 1 | 1054.43 | 26.43 | 25.76 | 64/0 | formal |
| Qwen3-30B-A3B | 16 | 4225.69 | 15.22 | 225.67 | 64/0 | formal |
| Qwen3-30B-A3B | 32 | 5616.56 | 11.33 | 320.90 | 64/0 | short |
| Qwen3-Next-80B-A3B-Instruct | 1 | 607.98 | 48.52 | 47.13 | 16/0 | short |
| Qwen3-Next-80B-A3B-Instruct | 16 | 3535.56 | 19.71 | 255.93 | 32/0 | short |
| Qwen3-Next-80B-A3B-Instruct | 32 | 4141.24 | 13.74 | 339.26 | 64/0 | short |

## C16 to C32 scaling

| Model | C16 aggregate | C32 aggregate | Throughput change | C16 TTFT | C32 TTFT | TTFT change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 35B | 180.64 | 208.08 | +15.2% | 3321.47 | 4909.15 | +47.8% |
| 9B | 247.16 | 287.02 | +16.1% | 3501.62 | 8806.52 | +151.5% |
| 30B | 225.67 | 320.90 | +42.2% | 4225.69 | 5616.56 | +32.9% |
| 80B | 255.93 | 339.26 | +32.6% | 3535.56 | 4141.24 | +17.1% |

## Roof comparison

### C1 single-request roof

| Model | C1 theoretical roof | Aggregate output TPS | Decode Avg TPS | System efficiency | Pure Decode efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| 35B | 336 | 47.21 | 50.75 | 14.1% | 15.1% |
| 9B | 56 | 48.62 | 54.64 | 86.8% | 97.6% |
| 30B | 305 | 25.76 | 26.43 | 8.4% | 8.7% |
| 80B | 672 | 47.13 | 48.52 | 7.0% | 7.2% |

The 9B C1 run activates one TP1 replica inside the dual-TP1/DP2 service. Its 97.6% ratio therefore describes single-replica Decode efficiency, not full-card C16/C32 batch utilization.

### C16/C32 batch roof

| Model | Physical cards | Estimated ideal batch roof | C16 efficiency | C32 efficiency | C32 per-card throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| 35B | 1 | 1066 | 16.9% | 19.5% | 208.08 |
| 9B | 1 | 896 | 27.6% | 32.0% | 287.02 |
| 30B | 1 | 800 | 28.2% | 40.1% | 320.90 |
| 80B | 2 | 1420 | 18.0% | 23.9% | 169.63 |

The roof values are planning estimates rather than direct GPU-counter utilization. They support relative deployment screening but cannot establish hardware efficiency with metrology-grade precision.

## RTX 5090 comparison supplied by the user

| Model | Concurrency | GR308 per-card throughput | RTX 5090 per-card throughput | GR/5090 | Data validity |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3.6-35B-A3B | C16/card | 180.64 | 498.78 | 36.22% | RTX 5090 value is 3990.23/8 from an eight-card 4 x TP2 run; not standalone single-card execution |
| Qwen3.5-9B | C16 | 247.16 | 444.40 | 55.62% | Same model, BF16, one card, C16; ShareGPT/output≤256 vs fixed 3072/1024 |

Only these two models are retained in the RTX 5090 section. The 35B ratio is a per-card system normalization, while the 9B ratio is closer to a same-device-count comparison.

## Statistical limitations

The ablation matrices are mostly single short runs. There are no matched repeated runs, standard deviations, confidence intervals, or justified inferential tests. Differences below 3% are therefore treated as operational ties. Claims of statistical significance are blocked until at least three controlled repetitions are available.
