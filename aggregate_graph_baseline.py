#!/usr/bin/env python3
"""Aggregate and validate the GR308 FANT Graph baseline evidence."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path


T975 = {2: 4.3026527299, 3: 3.1824463053, 4: 2.7764451052}
CONCURRENCIES = (1, 8, 16, 32, 64)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def case_entry(phase: str, repeat: str, case_dir: Path) -> dict:
    summary = read_json(case_dir / "summary.json")
    statuses = summary.get("statuses", {})
    requests = int(summary["requests"])
    return {
        "phase": phase,
        "repeat": repeat,
        "concurrency": int(summary["concurrency"]),
        "input_tokens": int(summary["input_tokens"]),
        "output_tokens": int(summary["output_tokens"]),
        "requests": requests,
        "ok": int(statuses.get("ok", 0)),
        "duration_s": float(summary["measurement_duration_s"]),
        "validated_output_tok_s": float(summary["raw_completed_output_tokens_per_s"]),
        "client_rc": int((case_dir / "client.rc").read_text().strip()),
        "path": str(case_dir),
    }


def case_passed(row: dict) -> bool:
    return row["client_rc"] == 0 and row["ok"] == row["requests"]


def descriptive_stats(values: list[float]) -> dict:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    result = {
        "values": values,
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "sample_std": std,
        "cv_pct": (std / mean * 100) if mean else None,
    }
    if len(values) in (3, 4, 5):
        margin = T975[len(values) - 1] * std / math.sqrt(len(values))
        result["descriptive_t95_ci"] = [mean - margin, mean + margin]
    return result


def selected(root: Path, *parts: str) -> Path:
    path = root.joinpath("graph32", *parts, "selected")
    if not path.exists():
        raise FileNotFoundError(f"missing selected evidence: {path}")
    return path.resolve()


def write_json_exclusive(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    outputs = {
        "runs": analysis / "run-level.csv",
        "requests": analysis / "request-level.csv",
        "telemetry": analysis / "telemetry-physical.csv",
        "summary": analysis / "baseline-summary.json",
        "validation": analysis / "validation.json",
        "report": root / "REPORT_ZH.md",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    cases: list[tuple[str, str, Path]] = []
    rapid_root = selected(root, "rapid")
    for c in CONCURRENCIES:
        cases.append(("rapid", "rapid", rapid_root / f"c{c}"))
    formal_repeats = [repeat for repeat in ("r1", "r3", "r5", "r2", "r4") if (root / "graph32" / "formal" / repeat / "selected").exists()]
    for repeat in formal_repeats:
        repeat_root = selected(root, "formal", repeat)
        for c in CONCURRENCIES:
            cases.append(("formal", repeat, repeat_root / f"c{c}"))
    stability_root = selected(root, "stability")
    cases.append(("stability", "stability", stability_root / "c64"))

    rows = [case_entry(*case) for case in cases]
    stability_dir = stability_root / "c64"
    extension_present = (stability_dir / "extension-summary.json").exists()
    extension_row = None
    if extension_present:
        ext = read_json(stability_dir / "extension-summary.json")
        extension_row = {
            "phase": "stability-extension",
            "repeat": "stability",
            "concurrency": int(ext["concurrency"]),
            "input_tokens": int(ext["input_tokens"]),
            "output_tokens": int(ext["output_tokens"]),
            "requests": int(ext["requests"]),
            "ok": int(ext.get("statuses", {}).get("ok", 0)),
            "duration_s": float(ext["measurement_duration_s"]),
            "validated_output_tok_s": float(ext["raw_completed_output_tokens_per_s"]),
            "client_rc": int((stability_dir / "extension-client.rc").read_text().strip()),
            "path": str(stability_dir),
        }
        rows.append(extension_row)

    with outputs["runs"].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    request_fields = [
        "phase", "repeat", "concurrency", "request_id", "deployment_id", "status",
        "input_tokens", "cached_input_tokens", "output_tokens", "ttft_ms", "tpot_ms",
        "e2e_ms", "http_status", "finish_reason", "stream_done", "error_reason",
    ]
    request_count = 0
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    with outputs["requests"].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=request_fields, extrasaction="ignore")
        writer.writeheader()
        for phase, repeat, case_dir in cases:
            c = int(read_json(case_dir / "summary.json")["concurrency"])
            with (case_dir / "requests.jsonl").open(encoding="utf-8") as records:
                for line in records:
                    record = json.loads(line)
                    writer.writerow({"phase": phase, "repeat": repeat, "concurrency": c, **record})
                    grouped.setdefault((phase, repeat, c), []).append(record)
                    request_count += 1
        if extension_present:
            with (stability_dir / "extension-requests.jsonl").open(encoding="utf-8") as records:
                for line in records:
                    record = json.loads(line)
                    writer.writerow({"phase": "stability-extension", "repeat": "stability", "concurrency": 64, **record})
                    grouped.setdefault(("stability-extension", "stability", 64), []).append(record)
                    request_count += 1

    telemetry_fields = [
        "phase", "repeat", "concurrency", "timestamp", "index", "pci_bus_id",
        "serial_number", "memory_used_mib", "memory_total_mib", "gpu_util_pct",
        "memory_util_pct", "power_w", "temperature_c",
    ]
    telemetry_count = 0
    with outputs["telemetry"].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=telemetry_fields)
        writer.writeheader()
        for phase, repeat, case_dir in cases:
            c = read_json(case_dir / "summary.json")["concurrency"]
            path = case_dir / "telemetry.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8", errors="replace") as source:
                for raw in csv.reader(source):
                    values = [value.strip() for value in raw]
                    if len(values) < 10:
                        continue
                    try:
                        index = int(values[1])
                    except ValueError:
                        continue
                    if index not in (0, 2, 4, 6):
                        continue
                    writer.writerow(dict(zip(telemetry_fields, [phase, repeat, c, *values[:10]])))
                    telemetry_count += 1

    formal = {}
    for c in CONCURRENCIES:
        values = [row["validated_output_tok_s"] for row in rows if row["phase"] == "formal" and row["concurrency"] == c]
        formal[str(c)] = descriptive_stats(values)
    rapid = {str(row["concurrency"]): row["validated_output_tok_s"] for row in rows if row["phase"] == "rapid"}
    stability = next(row for row in rows if row["phase"] == "stability")
    pre = read_json(rapid_root / "correctness-pre" / "summary.json")
    post = read_json(stability_root / "correctness-post" / "summary.json")

    latency = {}
    for key, records in grouped.items():
        ok = [record for record in records if record.get("status") == "ok"]
        latency["/".join(map(str, key))] = {
            "ok": len(ok),
            "ttft_ms_p50": percentile([float(x["ttft_ms"]) for x in ok], 0.50),
            "ttft_ms_p95": percentile([float(x["ttft_ms"]) for x in ok], 0.95),
            "tpot_ms_p50": percentile([float(x["tpot_ms"]) for x in ok], 0.50),
            "tpot_ms_p95": percentile([float(x["tpot_ms"]) for x in ok], 0.95),
            "e2e_ms_p50": percentile([float(x["e2e_ms"]) for x in ok], 0.50),
            "e2e_ms_p95": percentile([float(x["e2e_ms"]) for x in ok], 0.95),
        }

    extension_required = stability["duration_s"] < 3600
    checks = {
        "all_clients_passed": all(case_passed(row) for row in rows),
        "formal_at_least_three_engine_cold_blocks": all(item["n"] >= 3 for item in formal.values()),
        "correctness_pre_12_of_12": pre.get("all_cases_pass") is True and pre.get("passed") == 12,
        "correctness_post_12_of_12": post.get("all_cases_pass") is True and post.get("passed") == 12,
        "stability_main_1024_ok": stability["requests"] == stability["ok"] == 1024,
        "conditional_extension_satisfied": (not extension_required) or (
            extension_row is not None and extension_row["requests"] == extension_row["ok"] == 512 and case_passed(extension_row)
        ),
        "request_rows_match": request_count == sum(row["requests"] for row in rows),
        "physical_telemetry_rows_present": telemetry_count > 0,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "framework": "FANT vendor vLLM Graph32",
        "model": "Qwen3.8-27B",
        "rapid_validated_output_tok_s": rapid,
        "formal_validated_output_tok_s": formal,
        "stability": {"main": stability, "extension": extension_row},
        "request_latency": latency,
        "caveats": [
            "C64 is outside the frozen graph capture buckets and is a saturation/stress point.",
            "Synthetic fixed-token ignore_eos benchmark; not a production SLA or model-quality score.",
            "Engine-cold means a fresh container and engine process; host page cache may remain warm.",
            "Formal intervals are descriptive because n=3.",
        ],
    }
    write_json_exclusive(outputs["summary"], summary)
    write_json_exclusive(outputs["validation"], {"schema_version": 1, "passed": passed, "checks": checks})

    lines = [
        "# GR308 FANT Graph vLLM Baseline 报告", "",
        f"结论：{'通过' if passed else '未通过'}。模型为 Qwen3.8-27B，厂商 FANT Graph32 模式。", "",
        "## 冻结配置", "",
        "- `fantasy/vllm:1.1.0.20-py310-ubuntu22.04`，BF16，TP=4，设备 0/2/4/6。",
        "- Graph bucket：1/2/4/8/16/32；C64 是 bucket 外 saturation/stress 点。",
        "- `max-model-len=32768`，`gpu-memory-utilization=0.40`，`block-size=800`。", "",
        "## Rapid（128→64）", "", "| 并发 | validated output tok/s |", "|---:|---:|",
    ]
    for c in CONCURRENCIES:
        lines.append(f"| {c} | {rapid[str(c)]:.4f} |")
    lines += ["", "## Formal（1024→256，3 个 engine-cold block）", "", "| 并发 | 三点 tok/s | mean | median | CV |", "|---:|---|---:|---:|---:|"]
    for c in CONCURRENCIES:
        item = formal[str(c)]
        values = ", ".join(f"{x:.4f}" for x in item["values"])
        lines.append(f"| {c} | {values} | {item['mean']:.4f} | {item['median']:.4f} | {item['cv_pct']:.3f}% |")
    lines += [
        "", "## 稳定性与正确性", "",
        f"- 稳定性主段：{stability['ok']}/{stability['requests']}，{stability['validated_output_tok_s']:.4f} output tok/s。",
        f"- 前后正确性哨兵：{pre.get('passed')}/12、{post.get('passed')}/12。", "",
        "## 解释边界", "",
        "C64 不在冻结 Graph capture bucket 中，因此不能将其与 eager C64 的差异解释为纯 Graph 单变量收益。",
        "Formal 仅有 3 个 engine-cold block，区间仅作描述；结果不代表生产 SLA 或业务质量。", "",
    ]
    with outputs["report"].open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(json.dumps({"passed": passed, "runs": len(rows), "requests": request_count, "telemetry_rows": telemetry_count}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
