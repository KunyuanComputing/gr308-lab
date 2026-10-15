# Qwen3.6-35B-A3B Two-Card TPS Tuning Design

## Objective

Improve Qwen3.6-35B-A3B decode throughput on GR308 while preserving output
correctness and the established safety constraints. C16 decode TPS and aggregate
output throughput are primary. C1 metrics are diagnostic rather than selection
gates.

## Fixed Configuration

- BF16, input 3072 tokens, output 1024 tokens
- TP2 on one physical GR308 card per server
- MTP3, block size 1088, automatic KV allocation
- custom all-reduce enabled; never pass `--disable-custom-all-reduce`
- Graph full-decode-only mode
- short cases use 16 C1 requests and 32 C16 requests
- logical devices 6 and 7 remain reserved and idle
- stop the affected run on core-health failure, 80 C temperature, backend fatal
  marker, unexpected GPU container, or reserved-card memory change

## Two Parallel Lanes

Lane 0 uses logical devices 0 and 1 and isolates graph capture coverage:

1. Existing baseline: Graph `[4,8,16,32,64]`, batch 2048, synchronous scheduling.
2. Add small batches: Graph `[1,2,4,8,16,32,64]`.
3. Extend large batches: Graph `[4,8,16,32,64,128]`.
4. Cover both ends: Graph `[1,2,4,8,16,32,64,128]`.

Lane 1 uses logical devices 2 and 3 and isolates scheduling and batch size:

1. Card-local baseline: Graph `[4,8,16,32,64]`, batch 2048, synchronous.
2. Async scheduling with the same graph and batch.
3. Batch 1536, synchronous scheduling.
4. Batch 2560, synchronous scheduling.

Physical card 2 is unused by this campaign. Physical card 3 is the explicit
reserved card.

## Selection and Cross-Validation

Rank valid candidates by C16 aggregate output throughput. A candidate is valid
only when correctness passes, all requests complete, no safety failure occurs,
and hardware is released after shutdown.

Take the top two candidates. Run each once on the opposite physical card using
the same short workload. Rank by the mean C16 aggregate throughput across the
original and cross-card runs. Treat differences below 3% as effectively tied;
within that band prefer higher C1 throughput, then lower C16 median TTFT.

## Formal Confirmation

Run the winning configuration with 64 requests at C1 and 64 requests at C16.
The formal result is accepted only if correctness passes, all 128 requests
complete, the custom all-reduce remains enabled in the frozen launch record,
and all logical devices return to the idle-memory range with 8/8 core health OK.

## Exclusions

- MTP2 and MTP4: already below MTP3 for C16 throughput.
- Triton MoE backend: no improvement over the current path.
- communication fusion: current plugin fails because `_C.rms_norm` is absent.
- batch 4096: previously unstable and did not improve the known high-throughput
  baseline.
- EP: not part of this model's validated TP2 path.

## Outputs

Write immutable evidence under a new timestamped benchmark directory, including
the campaign specification, frozen launch arguments, server logs, telemetry,
per-case benchmark JSON, cross-card comparison, final selection, and formal
result summary.
