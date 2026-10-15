#!/usr/bin/env python3
"""Run the 2026-09-14 three-card GR308 Graph32 benchmark campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import traceback
from typing import NamedTuple


RESERVED_DEVICES = (6, 7)
IDLE_MEMORY_MIN_MIB = 500
IDLE_MEMORY_MAX_MIB = 540


class Job(NamedTuple):
    model: str
    devices: tuple[int, ...]
    port: int
    cpu_set: str
    output_dir: Path
    container_name: str
    mtp_tokens: int
    execution: str
    graph_capture_sizes: tuple[int, ...] = (32,)
    short: bool = False
    reserved_devices: tuple[int, ...] = RESERVED_DEVICES
    input_tokens: int = 3072
    output_tokens: int = 1024
    allow_unused_mtp: bool = False


def build_jobs(output_root: Path) -> list[Job]:
    definitions = [
        ("Qwen3.6-35B-A3B", (0, 1), 22100, "0-7", 1, "TP2"),
        ("Qwen3.5-9B", (2, 3), 22110, "8-15", 1, "TP2"),
        ("Qwen3-30B-A3B", (4, 5), 22120, "16-23", 0, "EP+TP2"),
    ]
    jobs = []
    for model, devices, port, cpu_set, mtp_tokens, execution in definitions:
        slug = model.lower().replace(".", "-")
        label = f"{slug}-{'mtp1' if mtp_tokens else 'baseline'}-graph32"
        jobs.append(Job(model, devices, port, cpu_set, output_root / label,
                        f"gr308-{label}", mtp_tokens, execution))
    return jobs


def build_qwen35_short_job(output_root: Path, mtp_tokens: int) -> Job:
    label = f"qwen3-5-9b-mtp{mtp_tokens}-graph4-32-short"
    return Job(
        "Qwen3.5-9B", (2, 3), 22110, "8-15", output_root / label,
        "gr308-qwen3-5-9b-mtp1-graph32", mtp_tokens, "TP2",
        (4, 8, 16, 32), True,
    )


def short_request_count(concurrency: int) -> int:
    return {1: 16, 16: 32}[concurrency]


def profile_name(mtp_tokens: int) -> str:
    return f"mtp{mtp_tokens}" if mtp_tokens else "baseline-no-native-mtp"


def build_serve_args(job: Job) -> list[str]:
    args = [
        "vllm", "serve", "/model", "--served-model-name", job.model,
        "--host", "127.0.0.1", "--port", str(job.port),
        "--tensor-parallel-size", "2", "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.9", "--max-model-len", "4096",
        "--max-num-seqs", "16", "--max-num-batched-tokens", "2048",
        "--kv-cache-memory-bytes", str(8 * 1024**3),
        "--no-enable-prefix-caching", "--no-async-scheduling",
        "--cudagraph-metrics",
        "--no-enable-log-requests", "--compilation-config",
        json.dumps({
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": list(job.graph_capture_sizes),
        }),
    ]
    if job.model.startswith(("Qwen3.5-", "Qwen3.6-")):
        args.append("--language-model-only")
    if job.execution == "EP+TP2":
        args.append("--enable-expert-parallel")
    args.extend(["--block-size", "1088" if job.model == "Qwen3.6-35B-A3B" else "800"])
    if job.mtp_tokens:
        args.extend(["--speculative-config", json.dumps({
            "method": "mtp", "num_speculative_tokens": job.mtp_tokens,
        })])
    return args


def allowed_container_names(jobs: list[Job]) -> set[str]:
    return {job.container_name for job in jobs}


def containers_allowed(running: set[str], allowed: set[str]) -> bool:
    return running <= allowed


def memory_idle(memory_mib: int) -> bool:
    """Accept the observed clean-driver baseline range across cold and warm boots."""
    return IDLE_MEMORY_MIN_MIB <= memory_mib <= IDLE_MEMORY_MAX_MIB


def card_cleanup_ok(
    memories_mib: list[int],
    devices: tuple[int, ...],
    reserved_devices: tuple[int, ...] = RESERVED_DEVICES,
) -> bool:
    if len(memories_mib) != 8:
        return False
    return (all(memory_idle(memories_mib[index]) for index in devices)
            and all(memory_idle(memories_mib[index]) for index in reserved_devices))


def all_core_health_ok(raw: str) -> bool:
    rows = re.findall(r"Device (\d+):\d+ Status:\[([^]]+)\]", raw)
    return (len(rows) == 8 and {int(index) for index, _ in rows} == set(range(8))
            and all(status == "OK" for _, status in rows))


def run_worker(job: Job, jobs: list[Job], campaign_root: Path) -> bool:
    """Run one card-scoped model job and retain all evidence on failure."""
    import goal_bench_contract
    from goal_bench_contract import admit_capacity, run_case
    from goal_preflight import capabilities
    from inventory import inspect_model
    from run_legacy_multi import IMAGE, CLIENT, QUALITY, command, load, quality, save, smi

    goal_bench_contract.INPUT_TOKENS = job.input_tokens
    goal_bench_contract.OUTPUT_TOKENS = job.output_tokens
    goal_bench_contract.REQUESTS = short_request_count(1) if job.short else 64

    out = job.output_dir
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "failed",
        "model": job.model,
        "profile": profile_name(job.mtp_tokens),
        "execution": job.execution,
        "graph_capture_sizes": list(job.graph_capture_sizes),
        "started": time.time(),
        "logical_functions": list(job.devices),
        "physical_cards": sorted({device // 2 for device in job.devices}),
        "container": job.container_name,
        "cases": [],
    }
    cid = None
    follower = None
    monitor = None
    failures: list[str] = []
    stop = threading.Event()
    current_case = {"value": None}
    serve = build_serve_args(job)
    started = time.monotonic()
    try:
        model_root = Path("/data/models") / job.model
        manifest = inspect_model(model_root)
        if not manifest["weights_header_qualified"]:
            raise RuntimeError("weights failed header admission")
        config = json.loads((model_root / "config.json").read_text())
        weight_map = json.loads((model_root / "model.safetensors.index.json").read_text())["weight_map"]
        cap = capabilities(config, weight_map)
        if job.mtp_tokens and not cap["native_mtp_weight_evidence"]:
            raise RuntimeError("native MTP weights are absent")
        if (not job.mtp_tokens and cap["native_mtp_weight_evidence"]
                and not job.allow_unused_mtp):
            raise RuntimeError("baseline-no-native-mtp job unexpectedly has native MTP weights")
        save(out, "model-manifest.json", manifest)
        save(out, "model-capabilities.json", cap)

        initial = smi()
        health = command(["sudo", "-n", "fantgpu-smi", "--query-core-health"]).stdout
        if not all_core_health_ok(health):
            raise RuntimeError("device core health admission failed")
        if not card_cleanup_ok(initial["memories_mib"], job.devices, job.reserved_devices):
            raise RuntimeError("target or reserved card is not idle")
        save(out, "hardware-before.json", {**initial, "health": health})

        if command(["docker", "inspect", job.container_name], check=False).returncode == 0:
            raise RuntimeError("container name already exists")

        sources = [
            (CLIENT, "client-frozen.py"),
            (QUALITY, "quality-frozen.py"),
            (Path(__file__), "runner-frozen.py"),
            (Path(__file__).parent / "goal_bench_contract.py", "bench-frozen.py"),
            (Path(__file__).parent / "goal_preflight.py", "preflight-frozen.py"),
            (Path(__file__).parent / "inventory.py", "inventory-frozen.py"),
            (Path(__file__).parent / "run_legacy_multi.py", "helper-frozen.py"),
            (Path("/etc/fant/fantHipSettings.cfg"), "fantHipSettings.cfg"),
        ]
        for source, destination in sources:
            shutil.copyfile(source, out / destination)

        launch = [
            "docker", "run", "-d", "--name", job.container_name,
            "--restart", "no", "--network", "host", "--ipc", "private",
            "--shm-size", "8g", "--privileged", "--security-opt", "label=disable",
            "--memory", "256g", "--cpuset-cpus", job.cpu_set,
            "--log-opt", "max-size=40m", "--log-opt", "max-file=2",
            "-e", "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, job.devices)),
            "-e", "OMP_NUM_THREADS=2", "-e", "HF_HUB_OFFLINE=1",
            "-e", "TRANSFORMERS_OFFLINE=1", "-e", "XDG_CACHE_HOME=/cache/xdg",
            "-e", "TORCH_EXTENSIONS_DIR=/cache/torch_extensions",
        ]
        for path in ("/usr/lib/x86_64-linux-gnu/fantgpu-fh3",
                     "/usr/local/fant/include", "/usr/local/fant/lib/llvm"):
            launch.extend(["-v", f"{path}:{path}:ro"])
        launch.extend([
            "-v", f"{model_root}:/model:ro",
            "-v", f"{out / 'fantHipSettings.cfg'}:/etc/fant/fantHipSettings.cfg:ro",
            "-v", "/data/infra/benchmarks/.grserve-qwen36-work/runtime-cache:/cache",
            IMAGE, "timeout", "--signal=TERM", "--kill-after=20", "34000", *serve,
        ])
        save(out, "launch.json", {
            "docker_args": launch,
            "vllm_args": serve,
            "logical_functions": list(job.devices),
            "physical_cards": result["physical_cards"],
            "profile": result["profile"],
            "mtp_tokens": job.mtp_tokens,
            "graph_capture_sizes": list(job.graph_capture_sizes),
            "source_hashes": {
                destination: hashlib.sha256((out / destination).read_bytes()).hexdigest()
                for _, destination in sources
            },
        })

        cid = command(launch).stdout.strip()
        result["container_id"] = cid
        follower = subprocess.Popen(
            ["docker", "logs", "--follow", "--timestamps", cid],
            stdout=(out / "server.log").open("w"), stderr=subprocess.STDOUT,
        )

        allowed = allowed_container_names(jobs)

        def watch() -> None:
            while not stop.is_set():
                try:
                    row = smi()
                    with (out / "telemetry.jsonl").open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    if max(row["temperatures_c"][index] for index in job.devices) >= 80:
                        raise RuntimeError("target temperature limit")
                    if any(not memory_idle(row["memories_mib"][index]) for index in job.reserved_devices):
                        raise RuntimeError("reserved physical card changed")
                    if time.monotonic() - started >= 33300:
                        raise RuntimeError("overall run deadline")
                    running = set(command(["docker", "ps", "--format", "{{.Names}}"]).stdout.split())
                    if not containers_allowed(running, allowed):
                        raise RuntimeError(f"unexpected running container: {sorted(running - allowed)}")
                    raw = command(["sudo", "-n", "fantgpu-smi", "--query-core-health"]).stdout
                    if not all_core_health_ok(raw):
                        raise RuntimeError("device core health failed")
                    logs = (out / "server.log").read_text(errors="replace")
                    fatal = ("HIP error: out of memory", "Guilty of lockup",
                             "Compute Context has been reset", "Segfault encountered")
                    if any(marker in logs for marker in fatal):
                        raise RuntimeError("backend fatal error")
                except Exception as exc:
                    failures.append(repr(exc))
                    if cid:
                        command(["docker", "stop", "--timeout", "20", cid], check=False)
                    return
                stop.wait(15)

        monitor = threading.Thread(target=watch, daemon=True)
        monitor.start()

        client = load(out / "client-frozen.py", f"client_{job.port}")
        client.ROOT = out
        client.MODEL = job.model
        client.PORT = job.port
        quality_module = load(out / "quality-frozen.py", f"quality_{job.port}")

        def safety_check() -> None:
            if failures:
                raise RuntimeError(f"safety monitor failed: {failures}")
            running = command(["docker", "inspect", "--format", "{{.State.Running}}", cid]).stdout.strip()
            if running != "true":
                raise RuntimeError("server exited")

        boot = time.monotonic()
        while time.monotonic() - boot < 900:
            safety_check()
            try:
                models = client.api("/v1/models", timeout=2)
            except Exception:
                time.sleep(3)
                continue
            if models["data"][0]["id"] != job.model:
                raise RuntimeError("wrong server model identity")
            save(out, "models.json", models)
            break
        else:
            raise TimeoutError("model startup deadline")
        result["startup_s"] = time.monotonic() - boot
        save(out, "capacity-admission.json", admit_capacity(serve, (out / "server.log").read_text(), 16))

        correctness = quality(client, quality_module, out, {"vision": False, "reasoning": False})
        result["correctness"] = correctness
        if not all(item["passed"] for item in correctness):
            raise RuntimeError("correctness gate failed")

        help_text = command(["docker", "exec", cid, "vllm", "bench", "serve", "--help=all"], timeout=60).stdout
        (out / "bench-help.txt").write_text(help_text)
        required_flags = ("--random-range-ratio", "--tokenizer", "--save-detailed", "--seed")
        if not all(flag in help_text for flag in required_flags):
            raise RuntimeError("benchmark CLI contract failed")

        for concurrency in (1, 16):
            if job.short:
                import goal_bench_contract
                goal_bench_contract.REQUESTS = short_request_count(concurrency)
            current_case["value"] = concurrency
            (out / f"c{concurrency}-metrics-before.prom").write_text(client.api("/metrics"))
            case = run_case(
                container_id=cid, model=job.model, concurrency=concurrency,
                port=job.port, output_dir=out / f"c{concurrency}", serve_args=serve,
                server_log=(out / "server.log").read_text(), safety_check=safety_check,
            )
            (out / f"c{concurrency}-metrics-after.prom").write_text(client.api("/metrics"))
            result["cases"].append(case)
            save(out, "progress.json", result)
        result["status"] = "native_cases_passed_pending_independent_audit"
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=45)
        if cid:
            command(["docker", "stop", "--timeout", "20", cid], check=False)
            inspected = command(["docker", "inspect", cid], check=False)
            if inspected.returncode == 0:
                save(out, "container-after-stop.json", json.loads(inspected.stdout))
        if follower:
            try:
                follower.wait(timeout=10)
            except subprocess.TimeoutExpired:
                follower.terminate()
        deadline = time.monotonic() + 90
        while True:
            after = smi()
            if (card_cleanup_ok(after["memories_mib"], job.devices, job.reserved_devices)
                    or time.monotonic() >= deadline):
                break
            time.sleep(2)
        health = command(["sudo", "-n", "fantgpu-smi", "--query-core-health"]).stdout
        save(out, "hardware-after.json", {**after, "health": health})
        result.update(
            target_memory_released=all(memory_idle(after["memories_mib"][i]) for i in job.devices),
            reserved_card_idle=all(memory_idle(after["memories_mib"][i]) for i in job.reserved_devices),
            health_ok=all_core_health_ok(health),
            finished=time.time(),
            safety_failures=failures,
        )
        save(out, "result.json", result)
    return result["status"] == "native_cases_passed_pending_independent_audit"


def execute_campaign(output_root: Path) -> int:
    from run_goal_model import check_port_available
    from run_legacy_multi import command, save, smi

    jobs = build_jobs(output_root)
    if output_root.exists():
        raise RuntimeError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)

    initial = smi()
    health = command(["sudo", "-n", "fantgpu-smi", "--query-core-health"]).stdout
    running = set(command(["docker", "ps", "--format", "{{.Names}}"]).stdout.split())
    if running:
        raise RuntimeError(f"containers already running: {sorted(running)}")
    if (len(initial["memories_mib"]) != 8
            or not all(memory_idle(value) for value in initial["memories_mib"])
            or not all_core_health_ok(health)):
        raise RuntimeError("global idle/health admission failed")
    for job in jobs:
        check_port_available(job.port)
        if "--disable-custom-all-reduce" in build_serve_args(job):
            raise RuntimeError("custom all-reduce must remain enabled")
        if command(["docker", "inspect", job.container_name], check=False).returncode == 0:
            raise RuntimeError(f"stale container exists: {job.container_name}")
    save(output_root, "campaign.json", {
        "status": "starting", "started": time.time(), "hardware_before": initial,
        "health_before": health, "reserved_devices": list(RESERVED_DEVICES),
        "jobs": [job._asdict() | {"output_dir": str(job.output_dir),
                                  "serve_args": build_serve_args(job)} for job in jobs],
    })

    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_worker_entry, args=(job, jobs, output_root),
                                 name=f"worker-{job.model}") for job in jobs]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    results = []
    for job, process in zip(jobs, processes):
        result_path = job.output_dir / "result.json"
        payload = json.loads(result_path.read_text()) if result_path.exists() else {
            "status": "missing_result", "model": job.model,
        }
        payload["worker_exitcode"] = process.exitcode
        results.append(payload)
    final = smi()
    final_health = command(["sudo", "-n", "fantgpu-smi", "--query-core-health"]).stdout
    summary = {
        "status": "complete" if all(p["status"] == "native_cases_passed_pending_independent_audit"
                                      for p in results) else "complete_with_failures",
        "finished": time.time(), "results": results,
        "hardware_after": final, "health_after": final_health,
    }
    save(output_root, "summary.json", summary)
    return 0 if summary["status"] == "complete" else 1


def _worker_entry(job: Job, jobs: list[Job], campaign_root: Path) -> None:
    raise SystemExit(0 if run_worker(job, jobs, campaign_root) else 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    jobs = build_jobs(args.output_root)
    if args.print_config:
        print(json.dumps([job._asdict() | {"serve_args": build_serve_args(job)}
                          for job in jobs], ensure_ascii=False, indent=2, default=str))
        return
    raise SystemExit(execute_campaign(args.output_root))


if __name__ == "__main__":
    main()
