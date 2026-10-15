#!/usr/bin/env python3
"""Two-card Qwen3.6-35B-A3B TPS sweep, cross-check, and formal confirmation."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).parent))
import gr308_adaptive_short_matrix as matrix
import gr308_parallel_goal as base


OUTPUT_ROOT = Path("/data/infra/benchmarks/gr308-35b-two-card-tps-20260915")
ALL_CONTAINER_NAMES = (
    "gr308-35b-tps-lane0", "gr308-35b-tps-lane1", "gr308-35b-tps-cross",
    "gr308-35b-tps-formal", "gr308-80b-short", "glm53-int4-download",
    "dsv4-gptq-download", "glm53-gptq-convert",
)


def make_spec(name, devices, port, cpu_set, graph, batch, container_name,
              *, async_scheduling=False) -> matrix.Spec:
    return matrix.Spec(
        name=name, model="Qwen3.6-35B-A3B", devices=devices, port=port,
        cpu_set=cpu_set, mtp=3, graph=graph, batched_tokens=batch,
        block_size=1088, kv_bytes=None, async_scheduling=async_scheduling,
        tp_size=2, max_num_seqs=16, short=True,
        container_name=container_name, reserved_devices=(),
        input_tokens=3072, output_tokens=1024,
    )


def lane_specs() -> dict[str, list[matrix.Spec]]:
    return {
        "graph": [
            make_spec("G-base", (0, 1), 22100, "0-7", (4, 8, 16, 32, 64), 2048,
                      "gr308-35b-tps-lane0"),
            make_spec("G-small", (0, 1), 22100, "0-7", (1, 2, 4, 8, 16, 32, 64), 2048,
                      "gr308-35b-tps-lane0"),
            make_spec("G-128", (0, 1), 22100, "0-7", (4, 8, 16, 32, 64, 128), 2048,
                      "gr308-35b-tps-lane0"),
            make_spec("G-full", (0, 1), 22100, "0-7", (1, 2, 4, 8, 16, 32, 64, 128), 2048,
                      "gr308-35b-tps-lane0"),
        ],
        "scheduler": [
            make_spec("S-base", (2, 3), 22110, "8-15", (4, 8, 16, 32, 64), 2048,
                      "gr308-35b-tps-lane1"),
            make_spec("S-async", (2, 3), 22110, "8-15", (4, 8, 16, 32, 64), 2048,
                      "gr308-35b-tps-lane1", async_scheduling=True),
            make_spec("S-batch1536", (2, 3), 22110, "8-15", (4, 8, 16, 32, 64), 1536,
                      "gr308-35b-tps-lane1"),
            make_spec("S-batch2560", (2, 3), 22110, "8-15", (4, 8, 16, 32, 64), 2560,
                      "gr308-35b-tps-lane1"),
        ],
    }


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
    if len(valid) < 2:
        raise RuntimeError("fewer than two valid short candidates")
    ceiling = max(row["c16_tps"] for row in valid)
    band = [row for row in valid if row["c16_tps"] >= ceiling * 0.97]
    preferred = sorted(band, key=lambda row: (-row["c1_tps"], row["c16_median_ttft_ms"]))
    remainder = sorted((row for row in valid if row not in band), key=lambda row: -row["c16_tps"])
    return preferred + remainder


def cross_spec(source: matrix.Spec, name: str) -> matrix.Spec:
    placement = (
        {"devices": (2, 3), "port": 22110, "cpu_set": "8-15"}
        if tuple(source.devices) == (0, 1)
        else {"devices": (0, 1), "port": 22100, "cpu_set": "0-7"}
    )
    return matrix._spec_like(
        source, name, container_name="gr308-35b-tps-cross", **placement,
    )


def formal_spec(source: matrix.Spec) -> matrix.Spec:
    return matrix._spec_like(
        source, "formal-winner", devices=(0, 1), port=22100, cpu_set="0-7",
        container_name="gr308-35b-tps-formal", short=False,
    )


def dummy_job(name: str) -> base.Job:
    return base.Job("allowed", (), 0, "0", OUTPUT_ROOT / "allowed" / name,
                    name, 0, "allowed", reserved_devices=())


def run_lane(name: str, specs: list[matrix.Spec], allowed: list[base.Job]) -> None:
    root = OUTPUT_ROOT / "lanes" / name
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for spec in specs:
            rows.append(enrich(matrix.run_spec(spec, root, allowed)))
        payload = {"status": "complete", "lane": name, "rows": rows,
                   "finished": time.time()}
        code = 0
    except Exception as exc:
        payload = {"status": "failed", "lane": name, "rows": rows,
                   "error": repr(exc), "traceback": traceback.format_exc(),
                   "finished": time.time()}
        code = 1
    (root / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    raise SystemExit(code)


def select_cross(records: list[dict]) -> dict:
    valid = [record for record in records if record["cross"].get("status") == "passed"]
    if not valid:
        raise RuntimeError("no passing cross-card candidate")
    top = max(record["mean_c16_tps"] for record in valid)
    band = [record for record in valid if record["mean_c16_tps"] >= top * 0.97]
    return max(band, key=lambda record: (
        record["mean_c1_tps"], -record["mean_c16_ttft_ms"], record["mean_c16_tps"],
    ))


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    lanes = lane_specs()
    allowed = [dummy_job(name) for name in ALL_CONTAINER_NAMES]
    (OUTPUT_ROOT / "campaign.json").write_text(json.dumps({
        "goal": "two-card 35B TPS tuning", "lanes": {
            name: [spec._asdict() for spec in specs] for name, specs in lanes.items()
        }, "started": time.time(),
    }, indent=2, default=str) + "\n")

    processes = {
        name: multiprocessing.Process(target=run_lane, args=(name, specs, allowed), name=name)
        for name, specs in lanes.items()
    }
    for process in processes.values():
        process.start()
    for process in processes.values():
        process.join()

    rows = []
    for name in lanes:
        summary = json.loads((OUTPUT_ROOT / "lanes" / name / "summary.json").read_text())
        rows.extend(summary["rows"])
    ranked = rank_candidates(rows)

    cross_root = OUTPUT_ROOT / "cross"
    cross_root.mkdir()
    cross_records = []
    for original in ranked[:2]:
        source = matrix.Spec(**original["spec"])
        cross = enrich(matrix.run_spec(cross_spec(source, f"X-{original['name']}"), cross_root, allowed))
        record = {"original": original, "cross": cross}
        if cross.get("status") == "passed":
            record.update(
                mean_c16_tps=(original["c16_tps"] + cross["c16_tps"]) / 2.0,
                mean_c1_tps=(original["c1_tps"] + cross["c1_tps"]) / 2.0,
                mean_c16_ttft_ms=(original["c16_median_ttft_ms"]
                                   + cross["c16_median_ttft_ms"]) / 2.0,
            )
        cross_records.append(record)
    winner = select_cross(cross_records)
    (cross_root / "summary.json").write_text(json.dumps({
        "status": "complete", "records": cross_records,
        "winner": winner["original"]["name"],
    }, indent=2, default=str) + "\n")

    formal_root = OUTPUT_ROOT / "formal"
    formal_root.mkdir()
    source = matrix.Spec(**winner["original"]["spec"])
    formal = enrich(matrix.run_spec(formal_spec(source), formal_root, allowed))
    complete = (
        formal.get("status") == "passed"
        and formal.get("c1_completed") == 64 and formal.get("c16_completed") == 64
        and formal.get("c1_failed") == 0 and formal.get("c16_failed") == 0
    )
    (formal_root / "summary.json").write_text(json.dumps({
        "status": "complete" if complete else "failed", "row": formal,
    }, indent=2, default=str) + "\n")
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps({
        "status": "complete" if complete else "failed", "short_rows": rows,
        "cross_records": cross_records, "winner": winner["original"]["name"],
        "formal": formal, "finished": time.time(),
    }, indent=2, default=str) + "\n")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
