#!/usr/bin/env python3
"""Adaptive short matrices for Qwen 35B, 9B, and 30B on three GR308 cards."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import NamedTuple


sys.path.insert(0, str(Path(__file__).parent))
import gr308_parallel_goal as base


FIXED_KV = 8 * 1024**3
ORIGINAL_BUILD_SERVE_ARGS = base.build_serve_args
CONTAINER_NAMES = {
    "Qwen3.6-35B-A3B": "gr308-qwen3-6-35b-a3b-mtp1-graph32",
    "Qwen3.5-9B": "gr308-qwen3-5-9b-mtp1-graph32",
    "Qwen3-30B-A3B": "gr308-qwen3-30b-a3b-baseline-graph32",
    "Qwen3-Next-80B-A3B-Instruct": "gr308-qwen3-next-80b-a3b-instruct",
}


class Spec(NamedTuple):
    name: str
    model: str
    devices: tuple[int, ...]
    port: int
    cpu_set: str
    mtp: int
    graph: tuple[int, ...]
    batched_tokens: int
    block_size: int
    kv_bytes: int | None
    moe_backend: str | None = None
    async_scheduling: bool = False
    tp_size: int = 2
    dp_size: int = 1
    fuse_gemm_comms: bool = False
    fuse_allreduce_rms: bool = False
    max_num_seqs: int = 16
    short: bool = True
    container_name: str | None = None
    reserved_devices: tuple[int, ...] = base.RESERVED_DEVICES
    input_tokens: int = 3072
    output_tokens: int = 1024
    allow_unused_mtp: bool = False
    enforce_eager: bool = False


def execution_name(spec: Spec) -> str:
    if spec.tp_size > 1:
        return f"TP{spec.tp_size}"
    return "dual-TP1" if spec.dp_size == 2 else "TP1"


def initial_specs(matrix: str) -> list[Spec]:
    if matrix == "35b":
        common = ("Qwen3.6-35B-A3B", (0, 1), 22100, "0-7")
        return [
            Spec("B", *common, 1, (4, 8, 16, 32), 2048, 1088, FIXED_KV),
            Spec("C", *common, 3, (4, 8, 16, 32, 64), 2048, 1088, FIXED_KV),
        ]
    if matrix == "9b":
        common = ("Qwen3.5-9B", (2, 3), 22110, "8-15")
        return [Spec("B", *common, 3, (4, 8, 16, 32, 64), 2048, 800, FIXED_KV)]
    if matrix == "30b":
        common = ("Qwen3-30B-A3B", (4, 5), 22120, "16-23")
        return [
            Spec("B", *common, 0, (4, 8, 16, 32), 2048, 800, FIXED_KV),
            Spec("C", *common, 0, (4, 8, 16, 32), 2048, 800, FIXED_KV,
                 moe_backend="triton"),
        ]
    raise ValueError(matrix)


def _set_option(args: list[str], flag: str, value: str) -> None:
    args[args.index(flag) + 1] = value


def serve_args(spec: Spec) -> list[str]:
    job = base.Job(
        spec.model, tuple(spec.devices), spec.port, spec.cpu_set, Path("/tmp/unused"),
        CONTAINER_NAMES[spec.model], spec.mtp,
        execution_name(spec),
        spec.graph, True,
    )
    args = ORIGINAL_BUILD_SERVE_ARGS(job)
    _set_option(args, "--max-num-batched-tokens", str(spec.batched_tokens))
    _set_option(args, "--max-num-seqs", str(spec.max_num_seqs))
    _set_option(args, "--block-size", str(spec.block_size))
    _set_option(args, "--tensor-parallel-size", str(spec.tp_size))
    if spec.kv_bytes is None:
        index = args.index("--kv-cache-memory-bytes")
        del args[index:index + 2]
    else:
        _set_option(args, "--kv-cache-memory-bytes", str(spec.kv_bytes))
    if spec.moe_backend:
        args.extend(["--moe-backend", spec.moe_backend])
    if spec.async_scheduling:
        args[args.index("--no-async-scheduling")] = "--async-scheduling"
    if spec.dp_size > 1:
        args.extend(["--data-parallel-size", str(spec.dp_size)])
    if spec.fuse_gemm_comms or spec.fuse_allreduce_rms:
        index = args.index("--compilation-config") + 1
        compilation = json.loads(args[index])
        compilation["pass_config"] = {
            "fuse_gemm_comms": spec.fuse_gemm_comms,
            "fuse_allreduce_rms": spec.fuse_allreduce_rms,
        }
        args[index] = json.dumps(compilation)
    if spec.enforce_eager:
        index = args.index("--compilation-config")
        del args[index:index + 2]
        args.append("--enforce-eager")
    if "--disable-custom-all-reduce" in args:
        raise RuntimeError("custom all-reduce must remain enabled")
    return args


def choose_best(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("status") == "passed"]
    if not valid:
        raise RuntimeError("no passing candidate")
    top_c16 = max(row["c16_tps"] for row in valid)
    band = [row for row in valid if row["c16_tps"] >= top_c16 * 0.97]
    return max(band, key=lambda row: (row["c1_tps"], row["c16_tps"]))


def _job(spec: Spec, root: Path) -> base.Job:
    execution = execution_name(spec)
    container_name = spec.container_name or CONTAINER_NAMES[spec.model]
    return base.Job(
        spec.model, tuple(spec.devices), spec.port, spec.cpu_set,
        root / f"{spec.name.lower()}-{spec.model.lower().replace('.', '-')}",
        container_name, spec.mtp, execution, spec.graph, spec.short,
        reserved_devices=tuple(spec.reserved_devices),
        input_tokens=spec.input_tokens,
        output_tokens=spec.output_tokens,
        allow_unused_mtp=spec.allow_unused_mtp,
    )


def _allowed_jobs(root: Path) -> list[base.Job]:
    return [
        base.Job("Qwen3.6-35B-A3B", (0, 1), 22100, "0-7", root / "allowed35",
                 CONTAINER_NAMES["Qwen3.6-35B-A3B"], 1, "TP2"),
        base.Job("Qwen3.5-9B", (2, 3), 22110, "8-15", root / "allowed9",
                 CONTAINER_NAMES["Qwen3.5-9B"], 1, "TP2"),
        base.Job("Qwen3-30B-A3B", (4, 5), 22120, "16-23", root / "allowed30",
                 CONTAINER_NAMES["Qwen3-30B-A3B"], 0, "TP2"),
    ]


def _row_from_result(name: str, result_path: Path, spec: Spec | None = None) -> dict:
    if not result_path.exists():
        return {"name": name, "status": "failed", "reason": "missing result"}
    result = json.loads(result_path.read_text())
    cases = {case["concurrency"]: case for case in result.get("cases", [])}
    row = {
        "name": name,
        "status": "passed" if result.get("status") == "native_cases_passed_pending_independent_audit" else "failed",
        "result_path": str(result_path),
        "c1_tps": cases.get(1, {}).get("output_tokens_per_s"),
        "c16_tps": cases.get(16, {}).get("output_tokens_per_s"),
    }
    if spec:
        row["spec"] = spec._asdict()
    if row["status"] == "failed":
        row["reason"] = result.get("error", result.get("status"))
    return row


def run_spec(spec: Spec, root: Path, allowed: list[base.Job]) -> dict:
    job = _job(spec, root)
    subprocess.run(["docker", "rm", job.container_name], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base.build_serve_args = lambda unused: serve_args(spec)
    ok = base.run_worker(job, allowed, root)
    row = _row_from_result(spec.name, job.output_dir / "result.json", spec)
    if ok != (row["status"] == "passed"):
        raise RuntimeError("worker/result status mismatch")
    subprocess.run(["docker", "rm", job.container_name], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return row


def _spec_like(source: Spec, name: str, **changes) -> Spec:
    values = source._asdict()
    values.update(changes, name=name)
    return Spec(**values)


def run_35b(root: Path, baseline_root: Path, allowed: list[base.Job]) -> list[dict]:
    rows = [_row_from_result("A", baseline_root / "qwen3-6-35b-a3b-mtp1-graph32/result.json")]
    for spec in initial_specs("35b"):
        rows.append(run_spec(spec, root, allowed))
    best = choose_best(rows)
    source = next((Spec(**row["spec"]) for row in rows if row["name"] == best["name"] and "spec" in row),
                  Spec("A", "Qwen3.6-35B-A3B", (0, 1), 22100, "0-7", 1, (32,), 2048, 1088, FIXED_KV))
    d = _spec_like(source, "D", batched_tokens=4096)
    rows.append(run_spec(d, root, allowed))
    e800 = _spec_like(d, "E-block800", block_size=800)
    if d.block_size != 800:
        rows.append(run_spec(e800, root, allowed))
    best_e = choose_best([row for row in rows if row["name"] in {"D", "E-block800"}])
    e_source = d if best_e["name"] == "D" else e800
    rows.append(run_spec(_spec_like(e_source, "F-auto-kv", kv_bytes=None), root, allowed))
    return rows


def run_9b(root: Path, baseline_root: Path, allowed: list[base.Job]) -> list[dict]:
    baseline = baseline_root / "qwen3-5-9b-mtp3-graph4-32-short/result.json"
    rows = [_row_from_result("A", baseline)]
    b = initial_specs("9b")[0]
    rows.append(run_spec(b, root, allowed))
    graph_choice = choose_best(rows)
    best_graph = (4, 8, 16, 32) if graph_choice["name"] == "A" else b.graph
    c = _spec_like(b, "C", graph=best_graph, batched_tokens=4096)
    rows.append(run_spec(c, root, allowed))
    graph_baseline = rows[0] if graph_choice["name"] == "A" else rows[1]
    best_batch = 4096 if choose_best([graph_baseline, rows[2]])["name"] == "C" else 2048
    d = _spec_like(c, "D-auto-kv", batched_tokens=best_batch, kv_bytes=None)
    rows.append(run_spec(d, root, allowed))
    mtp_rows = [rows[-1]]
    for mtp in (2, 4):
        spec = _spec_like(d, f"E-mtp{mtp}", mtp=mtp)
        row = run_spec(spec, root, allowed)
        rows.append(row)
        mtp_rows.append(row)
    best_mtp = choose_best(mtp_rows)
    best_spec = Spec(**best_mtp["spec"])
    rows.append(run_spec(_spec_like(best_spec, "F-tp1", tp_size=1, dp_size=1), root, allowed))
    rows.append(run_spec(_spec_like(best_spec, "F-dual-tp1", tp_size=1, dp_size=2), root, allowed))
    return rows


def run_30b(root: Path, baseline_root: Path, allowed: list[base.Job]) -> list[dict]:
    rows = [_row_from_result("A", baseline_root / "qwen3-30b-a3b-tp2-noep-graph32/result.json")]
    seeds = initial_specs("30b")
    for spec in seeds:
        rows.append(run_spec(spec, root, allowed))
    backend_best = choose_best(rows[1:])
    source = Spec(**backend_best["spec"])
    d = _spec_like(source, "D-block1088", block_size=1088)
    rows.append(run_spec(d, root, allowed))
    block_best = choose_best([backend_best, rows[-1]])
    block_source = Spec(**block_best["spec"])
    e = _spec_like(block_source, "E-batch4096", batched_tokens=4096)
    rows.append(run_spec(e, root, allowed))
    rows.append(run_spec(_spec_like(e, "F-async", async_scheduling=True), root, allowed))
    return rows


def _worker(kind: str, root: Path, baseline_root: Path, allowed: list[base.Job]) -> None:
    path = root / kind
    path.mkdir(parents=True, exist_ok=False)
    try:
        if kind == "35b":
            rows = run_35b(path, baseline_root, allowed)
        elif kind == "9b":
            rows = run_9b(path, Path("/data/infra/benchmarks/gr308-qwen35-short-tuning-20260914"), allowed)
        else:
            rows = run_30b(path, Path("/data/infra/benchmarks/gr308-30b-tp2-noep-graph32-20260914"), allowed)
        payload = {"status": "complete", "matrix": kind, "rows": rows,
                   "best": choose_best(rows), "finished": time.time()}
    except Exception as exc:
        payload = {"status": "failed", "matrix": kind, "error": repr(exc),
                   "traceback": traceback.format_exc(), "finished": time.time()}
    (path / "matrix-summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    raise SystemExit(0 if payload["status"] == "complete" else 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    if root.exists():
        raise RuntimeError(f"output root exists: {root}")
    root.mkdir(parents=True)
    allowed = _allowed_jobs(root)
    processes = []
    baseline = Path("/data/infra/benchmarks/gr308-multicard-mtp1-graph32-20260914")
    for kind in ("35b", "9b", "30b"):
        process = multiprocessing.get_context("fork").Process(
            target=_worker, args=(kind, root, baseline, allowed), name=f"matrix-{kind}")
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    summary = {"status": "complete" if all(p.exitcode == 0 for p in processes) else "complete_with_failures",
               "worker_exitcodes": {p.name: p.exitcode for p in processes}, "finished": time.time()}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(0 if summary["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
