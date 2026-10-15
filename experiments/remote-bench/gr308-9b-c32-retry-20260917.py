#!/usr/bin/env python3
"""C32 retry for tuned Qwen3.5-9B with 16 sequences per DP replica."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import traceback
import urllib.request


IMAGE = "sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc"
ROOT = Path("/data/infra/benchmarks/gr308-9b-c32-retry-20260917")
CACHE = "/data/infra/benchmarks/.grserve-qwen36-work/runtime-cache"
RUNTIME_CFG = "/etc/fant/fantHipSettings.cfg"
START_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()
STOP = threading.Event()
OWNED: set[str] = set()
OWNED_LOCK = threading.Lock()


def emit(**row):
    with PRINT_LOCK:
        print(json.dumps({"time": time.time(), **row}, ensure_ascii=False), flush=True)


def run(cmd, timeout=60, check=True):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode:
        raise RuntimeError(f"command rc={proc.returncode}: {cmd[:5]}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return proc


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def gpu_snapshot():
    raw = run(["sudo", "-n", "fantgpu-smi"], timeout=20).stdout
    memories = [int(x) for x in re.findall(r"(\d+)MiB / 57344MiB", raw)]
    temps = [int(x) for x in re.findall(r"\b(\d+)C\s", raw)]
    return {"time": time.time(), "memories_mib": memories, "temperatures_c": temps, "raw": raw}


def health_snapshot():
    raw = run(["sudo", "-n", "fantgpu-smi", "--query-core-health"], timeout=20).stdout
    ok = len(re.findall(r"Device \d+:\d+ Status:\[OK\]", raw)) == 8
    return {"ok": ok, "raw": raw}


def check_port(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def compilation(sizes):
    return json.dumps({
        "mode": "NONE",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": sizes,
    })


def base_args(v):
    args = [
        "vllm", "serve", "/model",
        "--served-model-name", v["model"],
        "--host", "127.0.0.1", "--port", str(v["port"]),
        "--tensor-parallel-size", str(v["tp"]),
        "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9",
        "--max-model-len", str(v["context"]),
        "--max-num-seqs", str(v["max_seqs"]),
        "--max-num-batched-tokens", str(v["batch_tokens"]),
        "--no-enable-prefix-caching",
        "--cudagraph-metrics", "--no-enable-log-requests",
        "--compilation-config", compilation(v["graphs"]),
        "--block-size", str(v["block"]),
    ]
    # Custom all-reduce intentionally stays enabled: never add --disable-custom-all-reduce.
    args.append("--async-scheduling" if v.get("async_scheduling") else "--no-async-scheduling")
    if v.get("kv_gib") is not None:
        args += ["--kv-cache-memory-bytes", str(v["kv_gib"] * 1024**3)]
    if v.get("language_model_only"):
        args.append("--language-model-only")
    if v.get("mtp"):
        args += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": v["mtp"]})]
    if v.get("moe_backend"):
        args += ["--moe-backend", v["moe_backend"]]
    if v.get("performance_mode"):
        args += ["--performance-mode", v["performance_mode"]]
    if v.get("dp"):
        args += ["--data-parallel-size", str(v["dp"])]
    if v.get("expert_parallel"):
        args.append("--enable-expert-parallel")
    if v.get("expert_placement"):
        args += ["--expert-placement-strategy", v["expert_placement"]]
    return args


def launch(v, out):
    check_port(v["port"])
    name = "gr308-tune-" + v["lane"] + "-" + v["name"]
    run(["docker", "rm", "-f", name], check=False)
    serve = base_args(v)
    cmd = [
        "docker", "run", "-d", "--name", name, "--restart", "no",
        "--network", "host", "--ipc", "private", "--shm-size", "8g",
        "--privileged", "--security-opt", "label=disable",
        "--memory", "256g", "--cpuset-cpus", v["cpus"],
        "--log-opt", "max-size=80m", "--log-opt", "max-file=2",
        "-e", "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, v["devices"])),
        "-e", "OMP_NUM_THREADS=2", "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1", "-e", "XDG_CACHE_HOME=/cache/xdg",
        "-e", "TORCH_EXTENSIONS_DIR=/cache/torch_extensions",
    ]
    for path in ("/usr/lib/x86_64-linux-gnu/fantgpu-fh3", "/usr/local/fant/include", "/usr/local/fant/lib/llvm"):
        cmd += ["-v", path + ":" + path + ":ro"]
    cmd += [
        "-v", f"/data/models/{v['model']}:/model:ro",
        "-v", CACHE + ":/cache",
        "-v", RUNTIME_CFG + ":/etc/fant/fantHipSettings.cfg:ro",
        IMAGE, "timeout", "--signal=TERM", "--kill-after=20", "14400", *serve,
    ]
    write_json(out / "launch.json", {
        "variant": v, "docker_args": cmd, "vllm_args": serve,
        "custom_all_reduce_disabled": False,
        "moe_mapping": v.get("moe_mapping"),
    })
    cid = run(cmd, timeout=120).stdout.strip()
    with OWNED_LOCK:
        OWNED.add(cid)
    emit(stage="server_started", lane=v["lane"], variant=v["name"], container_id=cid, devices=v["devices"])
    return cid, serve


def wait_ready(v, cid, out):
    deadline = time.monotonic() + 1200
    log_path = out / "server.log"
    while time.monotonic() < deadline and not STOP.is_set():
        logs = run(["docker", "logs", cid], timeout=30, check=False)
        log_path.write_text(logs.stdout + logs.stderr)
        state = run(["docker", "inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", cid], check=False).stdout.strip()
        if not state.startswith("true"):
            raise RuntimeError(f"server exited during startup: {state}\n{(logs.stdout + logs.stderr)[-6000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{v['port']}/v1/models", timeout=3) as response:
                payload = json.load(response)
            ids = [x.get("id") for x in payload.get("data", [])]
            if v["model"] in ids:
                write_json(out / "models.json", payload)
                emit(stage="ready", lane=v["lane"], variant=v["name"], startup_s=1200-(deadline-time.monotonic()))
                return
        except Exception:
            pass
        time.sleep(4)
    raise TimeoutError(f"startup timeout for {v['name']}")


def benchmark(v, cid, out, concurrency):
    requests = {1: 16, 16: 32, 32: 64}[concurrency]
    result_name = f"c{concurrency}.json"
    inside = "/tmp/" + v["name"] + "-" + result_name
    cmd = [
        "docker", "exec", cid, "timeout", "--signal=TERM", "--kill-after=20", "5400",
        "vllm", "bench", "serve", "--backend", "vllm",
        "--base-url", f"http://127.0.0.1:{v['port']}",
        "--model", v["model"], "--tokenizer", "/model",
        "--dataset-name", "random", "--random-input-len", str(v["input_tokens"]),
        "--random-output-len", str(v["output_tokens"]), "--random-range-ratio", "0",
        "--num-prompts", str(requests), "--max-concurrency", str(concurrency),
        "--ignore-eos", "--request-rate", "inf", "--temperature", "0", "--seed", "2026",
        "--percentile-metrics", "ttft,tpot,itl", "--metric-percentiles", "25,50,75,90,99",
        "--save-result", "--save-detailed", "--result-dir", "/tmp", "--result-filename", Path(inside).name,
    ]
    emit(stage="benchmark_started", lane=v["lane"], variant=v["name"], concurrency=concurrency,
         requests=requests, input_tokens=v["input_tokens"], output_tokens=v["output_tokens"])
    began = time.monotonic()
    # The GR308 host has occasionally returned an OCI setns error before the
    # exec'd client process exists. Retry only that exact pre-exec condition;
    # never retry a benchmark that may have issued requests.
    proc = None
    for attempt in range(1, 6):
        proc = run(cmd, timeout=5500, check=False)
        combined = proc.stdout + proc.stderr
        if proc.returncode != 128 or "OCI runtime exec failed" not in combined or "setns" not in combined:
            break
        emit(stage="benchmark_exec_retry", lane=v["lane"], variant=v["name"],
             concurrency=concurrency, attempt=attempt, error=combined[-1000:])
        time.sleep(3)
    assert proc is not None
    (out / f"c{concurrency}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"benchmark c{concurrency} failed rc={proc.returncode}: {(proc.stdout + proc.stderr)[-5000:]}")
    run(["docker", "cp", f"{cid}:{inside}", str(out / result_name)], timeout=60)
    data = json.loads((out / result_name).read_text())
    if data.get("completed") != requests or data.get("failed") != 0:
        raise RuntimeError(f"incomplete benchmark c{concurrency}: completed={data.get('completed')} failed={data.get('failed')}")
    if data.get("output_lens") != [v["output_tokens"]] * requests:
        raise RuntimeError(f"short output in c{concurrency}")
    row = {
        "concurrency": concurrency,
        "requests": requests,
        "duration_s": data.get("duration"),
        "output_throughput": data.get("output_throughput"),
        "median_ttft_ms": data.get("median_ttft_ms", data.get("p50_ttft_ms")),
        "median_tpot_ms": data.get("median_tpot_ms", data.get("p50_tpot_ms")),
        "wall_s": time.monotonic() - began,
        "completed": data.get("completed"),
        "failed": data.get("failed"),
    }
    emit(stage="benchmark_complete", lane=v["lane"], variant=v["name"], **row)
    return row


def stop_server(cid, out):
    logs = run(["docker", "logs", cid], timeout=30, check=False)
    (out / "server.log").write_text(logs.stdout + logs.stderr)
    run(["docker", "stop", "--timeout", "20", cid], timeout=60, check=False)
    inspect = run(["docker", "inspect", cid], timeout=30, check=False)
    (out / "container-inspect.json").write_text(inspect.stdout or "[]\n")
    with OWNED_LOCK:
        OWNED.discard(cid)


def run_variant(v):
    out = ROOT / v["lane"] / v["name"]
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "variant": v, "cases": [], "started": time.time()}
    cid = None
    try:
        # Serialize communicator/model initialization; benchmark phases run concurrently.
        with START_LOCK:
            cid, _ = launch(v, out)
            wait_ready(v, cid, out)
        for concurrency in v["concurrencies"]:
            result["cases"].append(benchmark(v, cid, out, concurrency))
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        emit(stage="variant_failed", lane=v["lane"], variant=v["name"], error=repr(exc))
    finally:
        if cid:
            stop_server(cid, out)
        result["finished"] = time.time()
        write_json(out / "result.json", result)
    return result


def lane_worker(lane, variants):
    rows = []
    for v in variants:
        if STOP.is_set():
            break
        rows.append(run_variant(v))
    write_json(ROOT / lane / "lane-result.json", rows)
    emit(stage="lane_complete", lane=lane, passed=sum(r["status"] == "passed" for r in rows), total=len(rows))
    return rows


def monitor():
    while not STOP.wait(15):
        try:
            snap = gpu_snapshot()
            health = health_snapshot()
            running = run(["docker", "ps", "-q", "--no-trunc"], timeout=20).stdout.split()
            with OWNED_LOCK:
                unexpected = sorted(set(running) - OWNED)
            with (ROOT / "telemetry.jsonl").open("a") as handle:
                handle.write(json.dumps({**snap, "health_ok": health["ok"], "unexpected": unexpected}) + "\n")
            if not health["ok"] or unexpected or (snap["temperatures_c"] and max(snap["temperatures_c"]) >= 80):
                raise RuntimeError(f"safety gate: health={health['ok']} unexpected={unexpected} temps={snap['temperatures_c']}")
        except Exception as exc:
            emit(stage="safety_stop", error=repr(exc))
            STOP.set()
            with OWNED_LOCK:
                ids = list(OWNED)
            for cid in ids:
                run(["docker", "stop", "--timeout", "20", cid], timeout=60, check=False)
            return


def matrix():
    common9 = dict(
        lane="9b", model="Qwen3.5-9B", devices=[0, 1], cpus="0-15", port=22400,
        tp=1, dp=2, context=4096, max_seqs=16, batch_tokens=4096,
        graphs=[4, 8, 16, 32], block=800, kv_gib=None,
        async_scheduling=False, language_model_only=True, mtp=4,
        input_tokens=3072, output_tokens=1024,
    )
    return {
        "9b": [
            {**common9, "name": "dual-tp1-dp2-mtp4-c32", "concurrencies": [32],
             "moe_mapping": "two TP1 data-parallel replicas on one physical GR308; MTP4"},
        ],
    }


def preflight():
    ROOT.mkdir(parents=True, exist_ok=False)
    snap = gpu_snapshot()
    health = health_snapshot()
    running = run(["docker", "ps", "-q"], timeout=20).stdout.split()
    if snap["memories_mib"] != [540] * 8 or not health["ok"] or running:
        raise RuntimeError(f"preflight failed memory={snap['memories_mib']} health={health['ok']} running={running}")
    for name in ("Qwen3.5-9B",):
        if not (Path("/data/models") / name / "config.json").exists():
            raise FileNotFoundError(name)
    write_json(ROOT / "preflight.json", {"gpu": snap, "health": health, "running": running, "image": IMAGE})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-matrix", action="store_true")
    args = parser.parse_args()
    plan = matrix()
    if args.print_matrix:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    preflight()
    write_json(ROOT / "matrix.json", plan)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    result = {"status": "failed", "started": time.time(), "lanes": {}}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            futures = {pool.submit(lane_worker, lane, rows): lane for lane, rows in plan.items()}
            for future in concurrent.futures.as_completed(futures):
                lane = futures[future]
                result["lanes"][lane] = future.result()
        result["status"] = "complete"
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        STOP.set()
    finally:
        STOP.set()
        watcher.join(timeout=30)
        with OWNED_LOCK:
            ids = list(OWNED)
        for cid in ids:
            run(["docker", "stop", "--timeout", "20", cid], timeout=60, check=False)
        result["finished"] = time.time()
        result["hardware_after"] = gpu_snapshot()
        result["health_after"] = health_snapshot()
        write_json(ROOT / "campaign-result.json", result)
        emit(stage="campaign_complete", status=result["status"])


if __name__ == "__main__":
    main()
