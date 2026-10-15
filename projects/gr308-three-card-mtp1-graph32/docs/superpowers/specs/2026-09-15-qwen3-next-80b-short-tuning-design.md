# Qwen3-Next-80B-A3B Short Tuning Design

## Objective

Tune Qwen3-Next-80B-A3B-Instruct for high decode throughput using the two
physical GR308 cards not assigned to the concurrent 35B campaign. This phase
performs short screening only. It must stop after selecting a candidate and
must not start a formal test without new user confirmation.

## Comparison Contract

- Model directory: `/data/models/Qwen3-Next-80B-A3B-Instruct`
- Served model name: `Qwen3-Next-80B-A3B-Instruct`
- TP4, BF16
- Input 1024 tokens, output 1024 tokens
- Concurrency 1 and 16
- Short workload: 16 C1 requests and 32 C16 requests
- Primary metric: C16 Decode Avg TPS, then C16 aggregate output throughput
- Secondary metrics: C1 Decode Avg TPS and Median TTFT
- Reference table:
  - C1: Median TTFT 1072.45 ms, Decode Avg TPS 33.23 tokens/s
  - C16: Median TTFT 6048.71 ms, Decode Avg TPS 15.02 tokens/s

## Fixed Safety Requirements

- Custom all-reduce remains enabled. Never pass `--disable-custom-all-reduce`.
- The previous 80B launch records that disabled custom all-reduce are evidence
  only and must not be copied as executable configuration.
- Stop the affected run on any core-health failure, temperature at or above 80
  C, backend fatal marker, unexpected GPU container, or incomplete cleanup.
- Preserve frozen launch arguments, logs, telemetry, correctness output, and
  per-case benchmark JSON for every candidate.
- All four physical cards are intentionally assigned: physical cards 0 and 1
  to 35B; physical cards 2 and 3 to 80B. The old global rule that reserves
  logical devices 6 and 7 must be disabled only for these coordinated campaigns.
- Each campaign must whitelist the other campaign's exact container names while
  continuing to reject any additional GPU container.

## Short Matrix

Use logical devices 4, 5, 6, and 7 with CPU set 16-31 and a dedicated port and
container name. Run candidates sequentially because each TP4 server consumes
both assigned physical cards.

1. `A-eager-no-mtp`: eager, no MTP, batch 2048, fixed 8 GiB KV.
2. `B-eager-mtp1`: eager, MTP1, batch 2048, fixed 8 GiB KV.
3. `C-graph32`: MTP1, Graph `[4,8,16,32]`, batch 2048, fixed 8 GiB KV.
4. `D-graph64`: MTP1, Graph `[4,8,16,32,64]`, batch 2048, fixed 8 GiB KV.
5. `E-auto-kv`: best passing graph candidate with automatic KV.
6. `F-batch4096`: automatic KV, batch 4096, otherwise the best passing graph
   configuration.
7. `G-async`: best passing automatic-KV and batch configuration with async
   scheduling enabled.

Do not test MTP2 in this phase. The installed model and prior successful record
only establish MTP1 as supported, and MTP2 adds avoidable backend risk before
the safer dimensions are screened.

## Adaptive Selection

After A-D, select the graph candidate with the highest C16 Decode Avg TPS.
Continue E-F from that graph. Compare E and F to choose the batch setting before
running G.

At the end, exclude any candidate that fails correctness, has incomplete
requests, triggers a safety gate, or does not release hardware. Rank the valid
candidates by C16 Decode Avg TPS. Treat results within 3% of the top value as a
tie; inside the band prefer higher C16 aggregate output throughput, then higher
C1 Decode Avg TPS, then lower C16 Median TTFT.

## Interaction with the 35B Campaign

Start the 35B two-card short lanes on logical devices 0-3 and the 80B short
matrix on logical devices 4-7. Both runners share an exact allowlist containing
the two 35B lane containers, the 80B container, and the known non-GPU background
containers. Neither runner treats the other campaign's assigned memory as an
idle-card violation.

If the 80B run causes global core-health failure or a communication failure that
affects the 35B cards, stop both campaigns and retain evidence. A failure local
to one campaign stops only that campaign unless global health is affected.

## Terminal State

This phase ends after writing the short-matrix summary and releasing logical
devices 4-7. Report every candidate, the selected configuration, comparison to
the reference table, failures, and hardware health. Do not create or launch a
formal 80B workload. Formal-test design and execution require an explicit user
confirmation after review of the short results.
