#!/usr/bin/env python3
"""Run parallel TTFT-oriented short matrices for Qwen 9B and 30B."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import time
import traceback

import gr308_adaptive_short_matrix as matrix
import gr308_parallel_goal as base


OUTPUT_ROOT = Path("/data/infra/benchmarks/gr308-ttft-short-matrix-20260915")
BACKGROUND_CONTAINERS = (
    "glm53-int4-download",
    "dsv4-gptq-download",
    "glm53-gptq-convert",
)


def background_job(name: str) -> base.Job:
    return base.Job("background-no-gpu", (6, 7), 0, "0", OUTPUT_ROOT / name,
                    name, 0, "background")


def specs(kind: str) -> list[matrix.Spec]:
    if kind == "9b":
        common = ("Qwen3.5-9B", (2, 3), 22110, "8-15", 4,
                  (4, 8, 16, 32), 800, None)
    elif kind == "30b":
        common = ("Qwen3-30B-A3B", (4, 5), 22120, "16-23", 0,
                  (4, 8, 16, 32), 800, None)
    else:
        raise ValueError(kind)
    model, devices, port, cpu_set, mtp, graph, block, kv = common
    return [
        matrix.Spec("A-batch3072-seq16", model, devices, port, cpu_set, mtp,
                    graph, 3072, block, kv, max_num_seqs=16),
        matrix.Spec("B-batch4096-seq16", model, devices, port, cpu_set, mtp,
                    graph, 4096, block, kv, max_num_seqs=16),
        matrix.Spec("C-batch4096-seq24", model, devices, port, cpu_set, mtp,
                    graph, 4096, block, kv, max_num_seqs=24),
        matrix.Spec("D-batch4096-seq32", model, devices, port, cpu_set, mtp,
                    graph, 4096, block, kv, max_num_seqs=32),
    ]


def add_latency_metrics(row: dict) -> dict:
    if row.get("status") != "passed":
        return row
    result_dir = Path(row["result_path"]).parent
    for concurrency in (1, 16):
        payload = json.loads((result_dir / f"c{concurrency}" / "benchmark.json").read_text())
        row[f"c{concurrency}_median_ttft_ms"] = payload["median_ttft_ms"]
        row[f"c{concurrency}_decode_tps"] = 1000.0 / payload["mean_tpot_ms"]
    return row


def pick_low_ttft(rows: list[dict], concurrency: int) -> dict:
    baseline = next(row for row in rows if row["name"] == "B-batch4096-seq16")
    floor = baseline[f"c{concurrency}_tps"] * 0.95
    eligible = [
        row for row in rows
        if row.get("status") == "passed" and row[f"c{concurrency}_tps"] >= floor
    ]
    return min(
        eligible,
        key=lambda row: (row[f"c{concurrency}_median_ttft_ms"],
                         -row[f"c{concurrency}_tps"]),
    )


def worker(kind: str, allowed: list[base.Job]) -> None:
    root = OUTPUT_ROOT / kind
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    try:
        for spec in specs(kind):
            row = add_latency_metrics(matrix.run_spec(spec, root, allowed))
            rows.append(row)
        payload = {
            "status": "complete",
            "matrix": kind,
            "rows": rows,
            "best_c1": pick_low_ttft(rows, 1),
            "best_c16": pick_low_ttft(rows, 16),
            "selection_rule": "lowest median TTFT with aggregate TPS >= 95% of B",
            "finished": time.time(),
        }
        exit_code = 0
    except Exception as exc:
        payload = {
            "status": "failed", "matrix": kind, "rows": rows,
            "error": repr(exc), "traceback": traceback.format_exc(),
            "finished": time.time(),
        }
        exit_code = 1
    (root / "matrix-summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    raise SystemExit(exit_code)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    allowed = matrix._allowed_jobs(OUTPUT_ROOT)
    allowed.extend(background_job(name) for name in BACKGROUND_CONTAINERS)
    processes = {
        kind: multiprocessing.Process(target=worker, args=(kind, allowed), name=f"ttft-{kind}")
        for kind in ("9b", "30b")
    }
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()
    payload = {
        "status": "complete" if all(p.exitcode == 0 for p in processes.values()) else "failed",
        "worker_exitcodes": {kind: p.exitcode for kind, p in processes.items()},
        "finished": time.time(),
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
