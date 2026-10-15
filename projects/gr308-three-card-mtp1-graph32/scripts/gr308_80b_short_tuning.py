#!/usr/bin/env python3
"""Short-only adaptive tuning for Qwen3-Next-80B-A3B-Instruct."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).parent))
import gr308_adaptive_short_matrix as matrix
import gr308_parallel_goal as base


OUTPUT_ROOT = Path("/data/infra/benchmarks/gr308-80b-short-tuning-20260915")
ALL_CONTAINER_NAMES = (
    "gr308-35b-tps-lane0", "gr308-35b-tps-lane1", "gr308-35b-tps-cross",
    "gr308-35b-tps-formal", "gr308-80b-short", "glm53-int4-download",
    "dsv4-gptq-download", "glm53-gptq-convert",
)


def make_spec(name: str, *, mtp: int, graph: tuple[int, ...], kv_bytes: int | None,
              batched_tokens: int = 2048, enforce_eager: bool = False,
              async_scheduling: bool = False, allow_unused_mtp: bool = False) -> matrix.Spec:
    return matrix.Spec(
        name=name, model="Qwen3-Next-80B-A3B-Instruct", devices=(4, 5, 6, 7),
        port=22120, cpu_set="16-31", mtp=mtp, graph=graph,
        batched_tokens=batched_tokens, block_size=1088, kv_bytes=kv_bytes,
        async_scheduling=async_scheduling, tp_size=4, max_num_seqs=16,
        short=True, container_name="gr308-80b-short", reserved_devices=(),
        input_tokens=1024, output_tokens=1024,
        allow_unused_mtp=allow_unused_mtp, enforce_eager=enforce_eager,
    )


def initial_specs() -> list[matrix.Spec]:
    return [
        make_spec("A-eager-no-mtp", mtp=0, graph=(32,), kv_bytes=matrix.FIXED_KV,
                  enforce_eager=True, allow_unused_mtp=True),
        make_spec("B-eager-mtp1", mtp=1, graph=(32,), kv_bytes=matrix.FIXED_KV,
                  enforce_eager=True),
        make_spec("C-graph32", mtp=1, graph=(4, 8, 16, 32), kv_bytes=matrix.FIXED_KV),
        make_spec("D-graph64", mtp=1, graph=(4, 8, 16, 32, 64), kv_bytes=matrix.FIXED_KV),
    ]


def followup_specs(graph_source: matrix.Spec) -> list[matrix.Spec]:
    e = matrix._spec_like(graph_source, "E-auto-kv", kv_bytes=None)
    f = matrix._spec_like(e, "F-batch4096", batched_tokens=4096)
    return [e, f]


def enrich(row: dict) -> dict:
    if row.get("status") != "passed":
        return row
    root = Path(row["result_path"]).parent
    for concurrency in (1, 16):
        data = json.loads((root / f"c{concurrency}" / "benchmark.json").read_text())
        row[f"c{concurrency}_median_ttft_ms"] = data["median_ttft_ms"]
        row[f"c{concurrency}_decode_tps"] = 1000.0 / data["mean_tpot_ms"]
        row[f"c{concurrency}_completed"] = data["completed"]
        row[f"c{concurrency}_failed"] = data["failed"]
    return row


def rank_candidates(rows: list[dict]) -> list[dict]:
    valid = [row for row in rows if row.get("status") == "passed"]
    if not valid:
        raise RuntimeError("no passing 80B short candidate")
    ceiling = max(row["c16_decode_tps"] for row in valid)
    band = [row for row in valid if row["c16_decode_tps"] >= ceiling * 0.97]
    preferred = sorted(
        band,
        key=lambda row: (-row["c16_tps"], -row["c1_decode_tps"], row["c16_median_ttft_ms"]),
    )
    remainder = sorted(
        (row for row in valid if row not in band),
        key=lambda row: -row["c16_decode_tps"],
    )
    return preferred + remainder


def dummy_job(name: str) -> base.Job:
    return base.Job("allowed", (), 0, "0", OUTPUT_ROOT / "allowed" / name,
                    name, 0, "allowed", reserved_devices=())


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    allowed = [dummy_job(name) for name in ALL_CONTAINER_NAMES]
    rows: list[dict] = []
    try:
        seeds = initial_specs()
        (OUTPUT_ROOT / "campaign.json").write_text(json.dumps({
            "goal": "80B short-only throughput tuning",
            "input_tokens": 1024, "output_tokens": 1024,
            "concurrencies": [1, 16], "short_requests": {"1": 16, "16": 32},
            "formal_test_authorized": False,
            "initial_specs": [spec._asdict() for spec in seeds],
            "started": time.time(),
        }, indent=2, default=str) + "\n")

        for spec in seeds:
            rows.append(enrich(matrix.run_spec(spec, OUTPUT_ROOT, allowed)))

        graph_rows = [row for row in rows if row["name"] in {"C-graph32", "D-graph64"}]
        graph_best = rank_candidates(graph_rows)[0]
        graph_source = matrix.Spec(**graph_best["spec"])
        e_spec, f_spec = followup_specs(graph_source)
        e_row = enrich(matrix.run_spec(e_spec, OUTPUT_ROOT, allowed))
        f_row = enrich(matrix.run_spec(f_spec, OUTPUT_ROOT, allowed))
        rows.extend([e_row, f_row])

        batch_best = rank_candidates([e_row, f_row])[0]
        batch_source = matrix.Spec(**batch_best["spec"])
        g_spec = matrix._spec_like(batch_source, "G-async", async_scheduling=True)
        rows.append(enrich(matrix.run_spec(g_spec, OUTPUT_ROOT, allowed)))

        winner = rank_candidates(rows)[0]
        payload = {
            "status": "complete", "phase": "short-only",
            "formal_test_authorized": False, "rows": rows, "best": winner,
            "reference": {
                "c1": {"median_ttft_ms": 1072.45, "decode_tps": 33.23},
                "c16": {"median_ttft_ms": 6048.71, "decode_tps": 15.02},
            },
            "finished": time.time(),
        }
        code = 0
    except Exception as exc:
        payload = {
            "status": "failed", "phase": "short-only",
            "formal_test_authorized": False, "rows": rows,
            "error": repr(exc), "traceback": traceback.format_exc(),
            "finished": time.time(),
        }
        code = 1
    (OUTPUT_ROOT / "matrix-summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
