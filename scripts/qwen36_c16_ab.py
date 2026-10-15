#!/usr/bin/env python3
"""Run short, sequential Qwen3.6-27B C16 A/B tests on one GR308 TP2 pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request


IMAGE = "sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc"
MODEL = "/data/models/Qwen3.6-27B"
FANT_CFG = "/etc/fant/fantHipSettings.cfg"
CACHE = "/data/infra/benchmarks/.grserve-qwen36-work/runtime-cache"


def run(command: list[str], *, check: bool = True, timeout: int | None = None,
        stdout=None) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, timeout=timeout,
                          stdout=stdout, stderr=subprocess.STDOUT)


def api_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
            payload = json.load(response)
        return payload["data"][0]["id"] == "Qwen3.6-27B"
    except Exception:
        return False


def serve_args(port: int, *, mtp: bool, kv152: bool) -> list[str]:
    args = [
        "vllm", "serve", "/model", "--served-model-name", "Qwen3.6-27B",
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", "2", "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.9", "--max-model-len", "4096",
        "--max-num-seqs", "16", "--max-num-batched-tokens", "2048",
        "--kv-cache-memory-bytes", "8589934592",
        "--no-enable-prefix-caching", "--no-async-scheduling",
        "--disable-custom-all-reduce", "--cudagraph-metrics",
        "--no-enable-log-requests", "--compilation-config",
        json.dumps({
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8, 16, 32, 64, 128, 256],
        }),
        "--language-model-only", "--block-size", "800",
    ]
    if kv152:
        args.extend(["--num-gpu-blocks-override", "152"])
    if mtp:
        args.extend(["--speculative-config", json.dumps({
            "method": "mtp", "num_speculative_tokens": 1,
        })])
    return args


def docker_args(name: str, port: int, out: Path, *, fant: bool,
                mtp: bool, kv152: bool) -> list[str]:
    args = [
        "docker", "run", "-d", "--name", name, "--restart", "no",
        "--network", "host", "--ipc", "private", "--shm-size", "8g",
        "--privileged", "--security-opt", "label=disable", "--memory", "256g",
        "--cpuset-cpus", "56-63", "--log-opt", "max-size=40m",
        "--log-opt", "max-file=2", "-e", "CUDA_VISIBLE_DEVICES=2,3",
        "-e", "OMP_NUM_THREADS=2", "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1", "-e", "XDG_CACHE_HOME=/cache/xdg",
        "-e", "TORCH_EXTENSIONS_DIR=/cache/torch_extensions",
        "-v", "/usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro",
        "-v", "/usr/local/fant/include:/usr/local/fant/include:ro",
        "-v", "/usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro",
        "-v", f"{MODEL}:/model:ro", "-v", f"{CACHE}:/cache",
        "-v", f"{out}:/results",
    ]
    if fant:
        args.extend(["-v", f"{FANT_CFG}:/etc/fant/fantHipSettings.cfg:ro"])
    args.extend([IMAGE, "timeout", "--signal=TERM", "--kill-after=20", "7200"])
    args.extend(serve_args(port, mtp=mtp, kv152=kv152))
    return args


def bench_args(cid: str, port: int, filename: str) -> list[str]:
    return [
        "docker", "exec", cid, "timeout", "--signal=TERM", "--kill-after=20", "3600",
        "vllm", "bench", "serve", "--backend", "vllm", "--base-url",
        f"http://127.0.0.1:{port}", "--model", "Qwen3.6-27B",
        "--tokenizer", "/model", "--dataset-name", "random",
        "--random-input-len", "3072", "--random-output-len", "1024",
        "--random-range-ratio", "0", "--num-prompts", "64",
        "--max-concurrency", "16", "--ignore-eos", "--request-rate", "inf",
        "--temperature", "0", "--seed", "2026", "--percentile-metrics",
        "ttft,tpot,itl", "--metric-percentiles", "25,50,75,90,99",
        "--save-result", "--save-detailed", "--result-dir", "/results",
        "--result-filename", filename,
    ]


def wait_for_idle_memory(timeout: int = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        query = subprocess.run(
            ["fantgpu-smi", "--showmeminfo", "vram"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout
        if "Used" not in query or all("540" in row for row in query.splitlines() if "Used" in row):
            return
        time.sleep(2)


def run_case(root: Path, index: int, case: dict) -> dict:
    label = case["label"]
    out = root / label
    out.mkdir(parents=True, exist_ok=False)
    port = 21410 + index
    name = f"gr308-qwen36-ab-{label}"
    launch = docker_args(name, port, out, fant=case["fant"],
                         mtp=case["mtp"], kv152=case["kv152"])
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    cid = ""
    follower = None
    started = time.time()
    result = {**case, "status": "failed", "started": started}
    try:
        cid = subprocess.check_output(launch, text=True).strip()
        server_log = (out / "server.log").open("w")
        follower = subprocess.Popen(["docker", "logs", "-f", "--timestamps", cid],
                                    stdout=server_log, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline and not api_ready(port):
            if subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", cid],
                              text=True, stdout=subprocess.PIPE).stdout.strip() != "true":
                raise RuntimeError("server exited during startup")
            time.sleep(3)
        if not api_ready(port):
            raise TimeoutError("server startup exceeded 900 seconds")
        result["startup_s"] = time.time() - started

        warmup = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps({"model": "Qwen3.6-27B", "prompt": "warmup",
                             "max_tokens": 32, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(warmup, timeout=300) as response:
            if response.status != 200:
                raise RuntimeError(f"warmup failed: HTTP {response.status}")

        telemetry = (out / "telemetry.jsonl").open("w")
        monitor = subprocess.Popen(
            ["bash", "-lc", "while true; do date +%s; fantgpu-smi; sleep 5; done"],
            stdout=telemetry, stderr=subprocess.STDOUT, text=True,
        )
        try:
            filename = "benchmark.json"
            bench_log = (out / "benchmark.log").open("w")
            benchmark_started = time.time()
            completed = run(bench_args(cid, port, filename), timeout=3700, stdout=bench_log)
            result["benchmark_s"] = time.time() - benchmark_started
            result["benchmark_returncode"] = completed.returncode
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
            telemetry.close()
        payload = json.loads((out / "benchmark.json").read_text())
        result.update(
            status="passed",
            output_throughput=payload.get("output_throughput"),
            total_token_throughput=payload.get("total_token_throughput"),
            median_ttft_ms=payload.get("median_ttft_ms"),
            median_tpot_ms=payload.get("median_tpot_ms"),
            completed=payload.get("completed"),
        )
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if cid:
            run(["docker", "stop", "--timeout", "20", cid], check=False, timeout=60)
        if follower:
            try:
                follower.wait(timeout=10)
            except subprocess.TimeoutExpired:
                follower.terminate()
        run(["docker", "rm", "-f", name], check=False, timeout=60)
        wait_for_idle_memory()
        result["finished"] = time.time()
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--labels", nargs="+")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = [
        {"label": "a-full-baseline", "fant": True, "kv152": False, "mtp": False},
        {"label": "b-no-fant-config", "fant": False, "kv152": False, "mtp": False},
        {"label": "c-kv152", "fant": True, "kv152": True, "mtp": False},
        {"label": "d-mtp1", "fant": True, "kv152": False, "mtp": True},
        {"label": "d-mtp1-retry", "fant": True, "kv152": False, "mtp": True},
    ]
    if args.labels:
        requested = set(args.labels)
        cases = [case for case in cases if case["label"] in requested]
        if {case["label"] for case in cases} != requested:
            raise SystemExit(f"unknown labels: {sorted(requested - {case['label'] for case in cases})}")
    summary_path = args.output_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    for index, case in enumerate(cases):
        summary.append(run_case(args.output_root, index, case))
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if summary[-1]["status"] != "passed":
            break


if __name__ == "__main__":
    main()
