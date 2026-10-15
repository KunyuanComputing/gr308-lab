#!/usr/bin/env python3
"""Run the final high-throughput formal campaign on three GR308 cards."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import time
import traceback

import gr308_adaptive_short_matrix as matrix
import gr308_parallel_goal as base


OUTPUT_ROOT = Path("/data/infra/benchmarks/gr308-final-high-throughput-20260915")
BACKGROUND_CONTAINERS = (
    "glm53-int4-download",
    "dsv4-gptq-download",
    "glm53-gptq-convert",
)


def final_specs() -> dict[str, matrix.Spec]:
    return {
        "35b": matrix.Spec(
            "final-high-throughput", "Qwen3.6-35B-A3B", (0, 1), 22100, "0-7",
            3, (4, 8, 16, 32, 64), 2048, 1088, None,
            max_num_seqs=16, short=False,
        ),
        "9b": matrix.Spec(
            "final-high-throughput", "Qwen3.5-9B", (2, 3), 22110, "8-15",
            4, (4, 8, 16, 32), 4096, 800, None,
            tp_size=1, dp_size=2, max_num_seqs=16, short=False,
        ),
        "30b": matrix.Spec(
            "final-high-throughput", "Qwen3-30B-A3B", (4, 5), 22120, "16-23",
            0, (4, 8, 16, 32), 4096, 800, matrix.FIXED_KV,
            async_scheduling=True, max_num_seqs=16, short=False,
        ),
    }


def background_job(name: str) -> base.Job:
    return base.Job("background-no-gpu", (6, 7), 0, "0", OUTPUT_ROOT / name,
                    name, 0, "background")


def worker(kind: str, spec: matrix.Spec, allowed: list[base.Job]) -> None:
    root = OUTPUT_ROOT / kind
    root.mkdir(parents=True, exist_ok=False)
    try:
        row = matrix.run_spec(spec, root, allowed)
        payload = {
            "status": "complete" if row.get("status") == "passed" else "failed",
            "model": kind,
            "formal_requests_per_concurrency": 64,
            "row": row,
            "finished": time.time(),
        }
        exit_code = 0 if payload["status"] == "complete" else 1
    except Exception as exc:
        payload = {
            "status": "failed", "model": kind, "error": repr(exc),
            "traceback": traceback.format_exc(), "finished": time.time(),
        }
        exit_code = 1
    (root / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    raise SystemExit(exit_code)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    specs = final_specs()
    allowed = matrix._allowed_jobs(OUTPUT_ROOT)
    allowed.extend(background_job(name) for name in BACKGROUND_CONTAINERS)
    (OUTPUT_ROOT / "campaign.json").write_text(json.dumps({
        "goal": "final high-throughput formal benchmark",
        "input_tokens": 3072,
        "output_tokens": 1024,
        "concurrencies": [1, 16],
        "requests_per_concurrency": 64,
        "custom_all_reduce_enabled": True,
        "reserved_devices": [6, 7],
        "specs": {kind: spec._asdict() for kind, spec in specs.items()},
        "started": time.time(),
    }, indent=2, default=str) + "\n")

    processes = {
        kind: multiprocessing.Process(target=worker, args=(kind, spec, allowed), name=f"final-{kind}")
        for kind, spec in specs.items()
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
