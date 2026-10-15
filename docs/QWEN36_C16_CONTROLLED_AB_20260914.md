# Qwen3.6-27B GR308 C16 controlled A/B (2026-09-14)

## Scope

- Model: `/data/models/Qwen3.6-27B`
- Precision: BF16
- Hardware: logical GPUs 2,3; TP2
- Workload: random input 3072, output 1024, 64 requests, concurrency 16
- Execution: FANT Graph `FULL_DECODE_ONLY`
- Vendor vLLM image: `sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc`
- Raw results: `/data/infra/benchmarks/gr308-qwen36-c16-ab64-20260914T1450/`

## Results

| Case | Single change | Status | Output throughput | Median TPOT | Finding |
|---|---|---:|---:|---:|---|
| A | Teammate baseline reproduced | Pass, 64/64 | 53.574 tok/s | 309.707 ms | Within +4.0% of prior 51.518 tok/s |
| B | Remove `fantHipSettings.cfg` | Startup failure | N/A | N/A | HIP OOM while allocating 512 MiB during KV initialization |
| C | Override KV blocks to 152 | Pass, 64/64 | 53.587 tok/s | 309.649 ms | +0.023% vs A; KV block count is not the throughput cause |
| D1 | MTP1, cold direct C16 | Deadlock | N/A | N/A | First wave never returned; GPU utilization fell to 0% |
| D2 | MTP1, isolated cold retry | Same deadlock | N/A | N/A | Reproduced the same failure sequence |

## Evidence and interpretation

1. The old 18--19 tok/s result is not a model or hardware ceiling. Reusing the teammate's
   vendor launch environment produces 53.574 tok/s on the same TP2 pair.
2. Reducing effective KV blocks from roughly 327 to 152 changes throughput by only 0.023%
   and produces an almost identical request timeline. With no preemption, KV capacity is not
   the explanation for 18 versus 53 tok/s.
3. The current FANT settings file is operationally required for this 8 GiB KV configuration.
   Without it, the runtime emits verbose PAL defaults and fails during KV initialization even
   though `fantgpu-smi` reports 23.68 GiB free.
4. MTP1 did use Graph32: logs repeatedly recorded `32 -> 32`, `Runtime Mode FULL`.
   Before the deadlock, generation throughput was about 142--150 tok/s and draft acceptance
   was 90--96%. Near the first wave boundary, acceptance dropped to 11--18%, then 0%; GPU
   utilization fell to 0 while all 16 requests remained running, followed by repeated
   `No available shared memory broadcast block found in 60 seconds` messages.
5. The FANT settings hashes are identical between this run and the teammate result:
   `c8d77445c099d1ac40a4a09f00c3052eb3ca86ab11ec53942bb8f46036990150`.
   The confirmed workflow difference is that the teammate's successful MTP1 C16 result was
   preceded in the same service by a full C1 x 64 run lasting 3596.9 seconds. The cold direct
   C16 path therefore cannot yet be considered stable.

## Current conclusion

- The immediately reusable non-speculative baseline is about 53.6 output tok/s per TP2 replica.
- FANT runtime configuration is required; KV=152 is not a throughput limiter for this workload.
- The reported 67.75 tok/s MTP1 result is real in the earlier stateful sequence, but cold direct
  C16 reproduced a completion-boundary deadlock twice. Before adopting it, test whether a minimal
  C1 priming request is sufficient; if not, reproduce the full C1 x 64 -> C16 sequence and treat
  the behavior as a vendor runtime lifecycle bug.

## Cleanup

All test containers were stopped. All eight logical GPUs returned to 540 MiB used memory and
reported core health `OK` after the run.
