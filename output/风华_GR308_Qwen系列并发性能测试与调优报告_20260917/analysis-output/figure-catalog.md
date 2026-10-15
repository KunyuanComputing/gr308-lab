# Figure catalog

## figure-01-reference-comparison.png

- Purpose: compare the selected C1/C16 TTFT and Decode Avg TPS with the user-provided external table.
- Data source: `data/report_data.json`, `reference` and `selected` sections.
- Caption requirement: state that 9B and 30B change parallel topology relative to the external table.
- Key observation: 30B and 9B gain C16 Decode TPS at a TTFT cost; 80B improves both C1/C16 latency and Decode TPS.
- Interpretation caveat: models have different input lengths and card counts.

## figure-02-c16-c32-throughput.png

- Purpose: show peak-throughput scaling and normalize C32 throughput by physical card count.
- Data source: selected C16/C32 aggregate throughput.
- Key observation: 30B is the highest one-card C32 result; 80B has the highest total result but lower per-card throughput.
- Decision implication: use 30B for throughput/cost and 80B only when model capability justifies two-card use.

## figure-03-roof-efficiency.png

- Purpose: compare C1 system/pure-Decode efficiency and C16/C32 aggregate throughput with their corresponding estimated roofs.
- Data source: `roof` and `selected` sections in `data/report_data.json`.
- Key observation: 9B reaches 97.6% of its C1 single-replica Decode roof, while 30B reaches the highest C32 batch-roof ratio, 40.1%.
- Interpretation caveat: estimated roof ratios are not hardware-counter utilization.

## figure-04-tuning-paths.png

- Purpose: summarize the main C16 ablation path for each model.
- Data source: `tuning` section in `data/report_data.json`.
- Key observation: dual TP1/DP2 is the main 9B gain; MTP is the main 35B/80B gain; EP regresses.
- Decision implication: stop expanding the simple batch/Graph search and move future work to kernel, routing, communication, and scheduling layers.

## figure-05-rtx5090-comparison.png

- Purpose: compare the user-supplied C16 per-card throughput for 35B and 9B on GR308 and RTX 5090.
- Data source: `rtx5090.35B` and `rtx5090.9B` in `data/report_data.json`.
- Key observation: the reported GR308/RTX 5090 ratios are 36.22% for 35B and 55.62% for 9B.
- Interpretation caveat: the 35B RTX 5090 value is an eight-card aggregate divided by eight from a 4 x TP2 run; the 9B workloads use different request shapes.
