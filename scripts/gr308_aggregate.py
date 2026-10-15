#!/usr/bin/env python3
"""Aggregate and validate the frozen GR308 vLLM baseline evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


T975_DF2 = 4.3026527299


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def case_entry(phase: str, repeat: str, case_dir: Path) -> dict:
    summary = read_json(case_dir / "summary.json")
    rc = int((case_dir / "client.rc").read_text().strip())
    statuses = summary.get("statuses", {})
    requests = int(summary["requests"])
    ok = int(statuses.get("ok", 0))
    return {
        "phase": phase,
        "repeat": repeat,
        "concurrency": int(summary["concurrency"]),
        "input_tokens": int(summary["input_tokens"]),
        "output_tokens": int(summary["output_tokens"]),
        "requests": requests,
        "ok": ok,
        "success_rate": ok / requests,
        "duration_s": float(summary["measurement_duration_s"]),
        "validated_output_tok_s": float(summary["raw_completed_output_tokens_per_s"]),
        "client_rc": rc,
        "path": str(case_dir),
    }


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    analysis = root / "analysis"
    outputs = [
        analysis / "run-level.csv",
        analysis / "request-level.csv",
        analysis / "telemetry-physical.csv",
        analysis / "baseline-summary.json",
        analysis / "validation.json",
        root / "REPORT_ZH.md",
    ]
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    cases: list[tuple[str, str, Path]] = []
    for concurrency in (1, 8, 16, 32, 64):
        name = "c1-rerun-01" if concurrency == 1 else f"c{concurrency}"
        cases.append(("rapid", "rapid", root / "eager" / "rapid" / name))
    for repeat in ("r1-retry01", "r3", "r5"):
        for concurrency in (1, 8, 16, 32, 64):
            cases.append(("formal", repeat, root / "eager" / "formal" / repeat / f"c{concurrency}"))
    stability = root / "eager" / "stability-retry02" / "c64"
    cases.append(("stability", "stability-retry02", stability))

    run_rows = [case_entry(*item) for item in cases]
    extension_summary = read_json(stability / "extension-summary.json")
    extension_rc = int((stability / "extension-client.rc").read_text().strip())
    run_rows.append({
        "phase": "stability-extension",
        "repeat": "stability-retry02",
        "concurrency": int(extension_summary["concurrency"]),
        "input_tokens": int(extension_summary["input_tokens"]),
        "output_tokens": int(extension_summary["output_tokens"]),
        "requests": int(extension_summary["requests"]),
        "ok": int(extension_summary["statuses"].get("ok", 0)),
        "success_rate": int(extension_summary["statuses"].get("ok", 0)) / int(extension_summary["requests"]),
        "duration_s": float(extension_summary["measurement_duration_s"]),
        "validated_output_tok_s": float(extension_summary["raw_completed_output_tokens_per_s"]),
        "client_rc": extension_rc,
        "path": str(stability),
    })

    with outputs[0].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    request_fields = [
        "phase", "repeat", "concurrency", "request_id", "deployment_id", "status",
        "input_tokens", "cached_input_tokens", "output_tokens", "ttft_ms", "tpot_ms",
        "e2e_ms", "scheduled_arrival_epoch_ms", "actual_start_epoch_ms",
        "client_schedule_delay_ms", "http_status", "finish_reason", "stream_done",
        "error_reason", "synthetic_ignore_eos", "warmup",
    ]
    request_count = 0
    with outputs[1].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=request_fields, extrasaction="ignore")
        writer.writeheader()
        for phase, repeat, case_dir in cases:
            concurrency = int(read_json(case_dir / "summary.json")["concurrency"])
            with (case_dir / "requests.jsonl").open(encoding="utf-8") as records:
                for line in records:
                    record = json.loads(line)
                    writer.writerow({"phase": phase, "repeat": repeat, "concurrency": concurrency, **record})
                    request_count += 1
        with (stability / "extension-requests.jsonl").open(encoding="utf-8") as records:
            for line in records:
                record = json.loads(line)
                writer.writerow({"phase": "stability-extension", "repeat": "stability-retry02", "concurrency": 64, **record})
                request_count += 1

    telemetry_fields = [
        "phase", "repeat", "concurrency", "timestamp", "index", "pci_bus_id", "serial_number",
        "memory_used_mib", "memory_total_mib", "gpu_util_pct", "memory_util_pct",
        "power_w", "temperature_c",
    ]
    telemetry_count = 0
    with outputs[2].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=telemetry_fields)
        writer.writeheader()
        for phase, repeat, case_dir in cases:
            summary = read_json(case_dir / "summary.json")
            telemetry_path = case_dir / "telemetry.csv"
            if not telemetry_path.exists():
                continue
            with telemetry_path.open(encoding="utf-8", errors="replace") as telemetry:
                for raw in csv.reader(telemetry):
                    if len(raw) < 10:
                        continue
                    values = [value.strip() for value in raw]
                    try:
                        index = int(values[1])
                    except ValueError:
                        continue
                    if index not in (0, 2, 4, 6):
                        continue
                    writer.writerow(dict(zip(telemetry_fields, [phase, repeat, summary["concurrency"], *values[:10]])))
                    telemetry_count += 1

    formal_stats = {}
    for concurrency in (1, 8, 16, 32, 64):
        values = [row["validated_output_tok_s"] for row in run_rows if row["phase"] == "formal" and row["concurrency"] == concurrency]
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        margin = T975_DF2 * std / math.sqrt(len(values))
        formal_stats[str(concurrency)] = {
            "values": values,
            "n": len(values),
            "mean": mean,
            "median": statistics.median(values),
            "sample_std": std,
            "cv_pct": std / mean * 100,
            "descriptive_t95_ci": [mean - margin, mean + margin],
        }

    rapid = {str(row["concurrency"]): row["validated_output_tok_s"] for row in run_rows if row["phase"] == "rapid"}
    stability_main = next(row for row in run_rows if row["phase"] == "stability")
    stability_ext = next(row for row in run_rows if row["phase"] == "stability-extension")
    post = read_json(root / "eager" / "correctness-post-retry02" / "summary.json")

    request_metrics = {}
    with outputs[1].open(newline="", encoding="utf-8") as handle:
        grouped: dict[tuple[str, str, int], list[dict]] = {}
        for row in csv.DictReader(handle):
            grouped.setdefault((row["phase"], row["repeat"], int(row["concurrency"])), []).append(row)
    for key, records in grouped.items():
        ok = [record for record in records if record["status"] == "ok"]
        label = "/".join(map(str, key))
        request_metrics[label] = {
            "ok": len(ok),
            "ttft_ms_p50": percentile([float(x["ttft_ms"]) for x in ok], 0.50),
            "ttft_ms_p95": percentile([float(x["ttft_ms"]) for x in ok], 0.95),
            "tpot_ms_p50": percentile([float(x["tpot_ms"]) for x in ok], 0.50),
            "tpot_ms_p95": percentile([float(x["tpot_ms"]) for x in ok], 0.95),
            "e2e_ms_p50": percentile([float(x["e2e_ms"]) for x in ok], 0.50),
            "e2e_ms_p95": percentile([float(x["e2e_ms"]) for x in ok], 0.95),
        }

    checks = {
        "all_selected_client_rc_zero": all(row["client_rc"] == 0 for row in run_rows),
        "all_selected_requests_ok": all(row["ok"] == row["requests"] for row in run_rows),
        "formal_three_blocks_each": all(item["n"] == 3 for item in formal_stats.values()),
        "formal_cv_le_5_pct": all(item["cv_pct"] <= 5 for item in formal_stats.values()),
        "stability_main_1024_ok": stability_main["ok"] == 1024,
        "stability_extension_512_ok": stability_ext["ok"] == 512,
        "post_sentinel_12_of_12": post.get("all_cases_pass") is True and post.get("passed") == 12,
        "request_rows_match": request_count == sum(row["requests"] for row in run_rows),
        "physical_telemetry_rows_present": telemetry_count > 0,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scope": "vLLM Lane E only; SGLang excluded",
        "model": "Qwen3.8-27B",
        "model_served_name": "qwen3.8-27b",
        "rapid_validated_output_tok_s": rapid,
        "formal_validated_output_tok_s": formal_stats,
        "stability": {
            "main": stability_main,
            "extension": stability_ext,
            "combined_requests": stability_main["requests"] + stability_ext["requests"],
            "combined_ok": stability_main["ok"] + stability_ext["ok"],
            "combined_duration_s": stability_main["duration_s"] + stability_ext["duration_s"],
        },
        "request_latency": request_metrics,
        "correctness_post": {"passed": post.get("passed"), "total": post.get("cases_total")},
        "excluded_attempts": [
            {"path": "eager/rapid/c1", "reason": "SSH-interrupted partial; replaced by c1-rerun-01"},
            {"path": "eager/formal/r1", "reason": "FHCCL/NCCL topology initialization failure; replaced by r1-retry01"},
            {"path": "eager/stability", "reason": "extension incorrectly skipped because jq was unavailable"},
            {"path": "eager/stability-retry01", "reason": "FHCCL/NCCL topology initialization failure"},
        ],
        "caveats": [
            "Synthetic fixed-token ignore_eos throughput diagnostic; not a production SLA or quality score.",
            "Engine-cold means a fresh container and engine process; host page cache and driver state may remain warm.",
            "Formal confidence intervals are descriptive with n=3 and are not evidence for small performance differences.",
            "GPU telemetry uses physical representatives 0/2/4/6; sibling functions are not double-counted.",
        ],
    }
    with outputs[3].open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    validation = {"schema_version": 1, "passed": passed, "checks": checks, "selected_formal_blocks": ["r1-retry01", "r3", "r5"], "selected_stability": "stability-retry02"}
    with outputs[4].open("x", encoding="utf-8") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        "# GR308 vLLM Baseline 实验报告", "", f"结论：{'通过' if passed else '未通过'}。本报告仅覆盖 vLLM Lane E；SGLang 未纳入本轮 baseline。", "",
        "## 冻结配置", "", "- 模型：`/data/models/Qwen3.8-27B`（served name `qwen3.8-27b`）", "- BF16，TP=4，设备 0/2/4/6，eager execution", "- `max-model-len=32768`，`gpu-memory-utilization=0.40`，`block-size=800`", "- 接口仅绑定 `127.0.0.1:8000`", "",
        "## Rapid（128→64）", "", "| 并发 | validated output tok/s |", "|---:|---:|",
    ]
    for concurrency in (1, 8, 16, 32, 64):
        lines.append(f"| {concurrency} | {rapid[str(concurrency)]:.4f} |")
    lines += ["", "## Formal（1024→256，3 个 block）", "", "| 并发 | 原始三点 tok/s | mean | median | CV | 描述性 95% t-CI |", "|---:|---|---:|---:|---:|---:|"]
    for concurrency in (1, 8, 16, 32, 64):
        item = formal_stats[str(concurrency)]
        values = ", ".join(f"{value:.4f}" for value in item["values"])
        ci = item["descriptive_t95_ci"]
        lines.append(f"| {concurrency} | {values} | {item['mean']:.4f} | {item['median']:.4f} | {item['cv_pct']:.3f}% | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    combined_s = stability_main["duration_s"] + stability_ext["duration_s"]
    lines += [
        "", "## 稳定性与正确性", "",
        f"- 有效替代段：`stability-retry02`；主段 1024/1024，{stability_main['validated_output_tok_s']:.4f} output tok/s，{stability_main['duration_s']:.1f} 秒。",
        f"- 延长段 512/512，{stability_ext['validated_output_tok_s']:.4f} output tok/s，{stability_ext['duration_s']:.1f} 秒；合计 {combined_s:.1f} 秒。",
        f"- 性能后正确性哨兵：{post.get('passed')}/{post.get('cases_total')} 通过。", "",
        "## 排除与故障记录", "",
        "- Rapid 原始 C1 因 SSH 中断仅形成部分证据，使用 `c1-rerun-01`。",
        "- Formal 原始 R1 在 FHCCL/NCCL 拓扑初始化时失败，使用 `r1-retry01`。",
        "- 首个 stability 因系统缺少 `jq` 错误跳过延长段，保留但排除；`stability-retry01` 又在 FHCCL/NCCL 初始化失败，最终使用 `stability-retry02`。", "",
        "## 解释边界", "",
        "这些数字来自固定 token-ID、`ignore_eos=true` 的合成吞吐诊断，不代表业务质量或生产 SLA。所谓 engine-cold 仅表示新容器与新 engine 进程；宿主 page cache 和驱动状态可能仍为热态。Formal 只有 3 个 block，置信区间仅作描述。", "",
        "机器可读明细见 `analysis/baseline-summary.json`、`analysis/validation.json`、`analysis/run-level.csv`、`analysis/request-level.csv` 和 `analysis/telemetry-physical.csv`。", "",
    ]
    with outputs[5].open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(json.dumps({"passed": passed, "run_rows": len(run_rows), "request_rows": request_count, "telemetry_rows": telemetry_count, "formal_max_cv_pct": max(x["cv_pct"] for x in formal_stats.values())}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
