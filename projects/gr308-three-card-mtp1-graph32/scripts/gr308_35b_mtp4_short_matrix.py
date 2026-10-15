#!/usr/bin/env python3
"""Four-card, short-only Qwen3.6-35B MTP3/MTP4 tuning matrix."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import queue as queue_module
import sys
import time
import traceback
import urllib.request

sys.path.insert(0, str(Path(__file__).parent))
import gr308_adaptive_short_matrix as matrix
import gr308_parallel_goal as base


OUTPUT_ROOT = Path("/data/infra/benchmarks/gr308-35b-mtp4-short-matrix-ready-20260915")
STARTUP_READY_TIMEOUT_S = 900
READY_POLL_S = 3
CONTAINER_NAMES = tuple(f"gr308-35b-mtp4-card{i}" for i in range(4))
ALL_CONTAINER_NAMES = CONTAINER_NAMES + (
    "glm53-int4-download", "dsv4-gptq-download", "glm53-gptq-convert",
)
PLACEMENTS = (
    ((0, 1), 22100, "0-7", CONTAINER_NAMES[0]),
    ((2, 3), 22110, "8-15", CONTAINER_NAMES[1]),
    ((4, 5), 22120, "16-23", CONTAINER_NAMES[2]),
    ((6, 7), 22130, "24-31", CONTAINER_NAMES[3]),
)


def make_spec(name: str, placement: int, *, mtp: int, graph: tuple[int, ...],
              batched_tokens: int) -> matrix.Spec:
    devices, port, cpu_set, container_name = PLACEMENTS[placement]
    return matrix.Spec(
        name=name, model="Qwen3.6-35B-A3B", devices=devices, port=port,
        cpu_set=cpu_set, mtp=mtp, graph=graph, batched_tokens=batched_tokens,
        block_size=1088, kv_bytes=None, async_scheduling=False,
        tp_size=2, max_num_seqs=16, short=True,
        container_name=container_name, reserved_devices=(),
        input_tokens=3072, output_tokens=1024,
    )


def wave_specs() -> list[list[matrix.Spec]]:
    g64 = (4, 8, 16, 32, 64)
    return [
        [
            make_spec("A-mtp4-g64-b2048", 0, mtp=4, graph=g64, batched_tokens=2048),
            make_spec("B-mtp4-g128-b2048", 1, mtp=4,
                      graph=g64 + (128,), batched_tokens=2048),
            make_spec("C-mtp4-g64-b3072", 2, mtp=4, graph=g64, batched_tokens=3072),
            make_spec("D-mtp4-g64-b4096", 3, mtp=4, graph=g64, batched_tokens=4096),
        ],
        [
            make_spec("E-mtp3-g64-b2048", 0, mtp=3, graph=g64, batched_tokens=2048),
            make_spec("F-mtp3-g64-b4096", 1, mtp=3, graph=g64, batched_tokens=4096),
        ],
    ]


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


def select_winners(rows: list[dict]) -> dict[str, dict]:
    valid = [row for row in rows if row.get("status") == "passed"]
    if not valid:
        raise RuntimeError("no passing 35B short candidate")
    return {
        "c1_decode": max(valid, key=lambda row: row["c1_decode_tps"]),
        "c16_decode": max(valid, key=lambda row: row["c16_decode_tps"]),
        "c16_aggregate": max(valid, key=lambda row: row["c16_tps"]),
    }


def dummy_job(name: str) -> base.Job:
    return base.Job("allowed", (), 0, "0", OUTPUT_ROOT / "allowed" / name,
                    name, 0, "allowed", reserved_devices=())


def run_worker(spec: matrix.Spec, allowed: list[base.Job], result_queue) -> None:
    try:
        row = enrich(matrix.run_spec(spec, OUTPUT_ROOT, allowed))
    except Exception as exc:
        row = {
            "name": spec.name, "status": "failed", "spec": spec._asdict(),
            "error": repr(exc), "traceback": traceback.format_exc(),
        }
    result_queue.put((spec.name, row))


def wait_for_ready_or_exit(spec: matrix.Spec, process) -> str:
    deadline = time.monotonic() + STARTUP_READY_TIMEOUT_S
    url = f"http://127.0.0.1:{spec.port}/v1/models"
    while time.monotonic() < deadline:
        if not process.is_alive():
            process.join()
            return "exited"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                models = json.load(response)
            if models.get("data", [{}])[0].get("id") == spec.model:
                return "ready"
        except Exception:
            pass
        time.sleep(READY_POLL_S)
    raise TimeoutError(f"startup readiness deadline for {spec.name}")


def start_wave_processes(specs: list[matrix.Spec], allowed: list[base.Job], result_queue,
                         *, process_factory=None, readiness_waiter=None):
    process_factory = process_factory or multiprocessing.Process
    readiness_waiter = readiness_waiter or wait_for_ready_or_exit
    processes = []
    for index, spec in enumerate(specs):
        process = process_factory(
            target=run_worker, args=(spec, allowed, result_queue), name=spec.name,
        )
        process.start()
        processes.append(process)
        if index + 1 < len(specs):
            readiness_waiter(spec, process)
    return processes


def run_wave(specs: list[matrix.Spec], allowed: list[base.Job]) -> list[dict]:
    result_queue = multiprocessing.Queue()
    processes = start_wave_processes(specs, allowed, result_queue)
    for process in processes:
        process.join()

    received = {}
    for _ in processes:
        try:
            name, row = result_queue.get(timeout=10)
            received[name] = row
        except queue_module.Empty:
            break
    return [
        received.get(spec.name, {
            "name": spec.name, "status": "failed", "spec": spec._asdict(),
            "error": "worker exited without returning a result",
        })
        for spec in specs
    ]


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    waves = wave_specs()
    allowed = [dummy_job(name) for name in ALL_CONTAINER_NAMES]
    (OUTPUT_ROOT / "campaign.json").write_text(json.dumps({
        "goal": "35B MTP4 decode TPS short tuning with MTP3 controls",
        "input_tokens": 3072, "output_tokens": 1024,
        "concurrencies": [1, 16], "short_requests": {"1": 16, "16": 32},
        "formal_test_authorized": False,
        "startup_ready_timeout_s": STARTUP_READY_TIMEOUT_S,
        "waves": [[spec._asdict() for spec in wave] for wave in waves],
        "started": time.time(),
    }, indent=2, default=str) + "\n")

    rows = []
    try:
        for wave in waves:
            rows.extend(run_wave(wave, allowed))
        winners = select_winners(rows)
        payload = {
            "status": "complete", "phase": "short-only",
            "formal_test_authorized": False, "rows": rows,
            "winners": winners, "finished": time.time(),
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
