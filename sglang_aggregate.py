#!/usr/bin/env python3
"""Validate and aggregate the GR308 SGLang baseline evidence."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
import sys
from pathlib import Path


T975_DF2 = 4.3026527299
CONCURRENCIES = (1, 8, 16, 32, 64)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_repeats(values: list[float]) -> dict:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    if len(values) == 3:
        margin = T975_DF2 * std / math.sqrt(3)
    else:
        margin = None
    return {
        "values": values,
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "sample_std": std,
        "cv_pct": std / mean * 100 if mean else None,
        "descriptive_t95_ci": [mean - margin, mean + margin] if margin is not None else None,
    }


def is_comparable_stability(summary: dict) -> bool:
    return all(summary.get(k) == v for k, v in {
        "input_tokens": 128,
        "output_tokens": 64,
        "concurrency": 64,
        "requests": 1024,
    }.items())


def case_entry(phase: str, repeat: str, case_dir: Path) -> dict:
    summary = read_json(case_dir / "summary.json")
    rc = int((case_dir / "client.rc").read_text().strip())
    requests = int(summary["requests"])
    ok = int(summary.get("statuses", {}).get("ok", 0))
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


def validate_records(case_dir: Path, expected: int, input_tokens: int, output_tokens: int) -> tuple[list[dict], list[str]]:
    rows = [json.loads(line) for line in (case_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = []
    if len(rows) != expected:
        errors.append(f"{case_dir}: rows={len(rows)} expected={expected}")
    for i, row in enumerate(rows):
        checks = {
            "status": row.get("status") == "ok",
            "http": row.get("http_status") == 200,
            "input": row.get("input_tokens") == input_tokens,
            "output": row.get("output_tokens") == output_tokens,
            "stream_done": row.get("stream_done") is True,
            "finish_reason": row.get("finish_reason") == "length",
        }
        if not all(checks.values()):
            errors.append(f"{case_dir}: row {i} failed {checks}")
    return rows, errors


def svg_line_chart(path: Path, title: str, x_values: list[int], series: list[dict], y_label: str, log_y: bool = False):
    width, height = 820, 480
    left, right, top, bottom = 90, 30, 45, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    all_y = [v for s in series for v in s["values"] if v is not None and v > 0]
    ymin = min(all_y) if log_y else 0.0
    ymax = max(all_y) * 1.12
    if log_y:
        ymin = max(ymin * 0.8, 0.1)
        transform = lambda y: (math.log(y) - math.log(ymin)) / (math.log(ymax) - math.log(ymin))
    else:
        transform = lambda y: y / ymax
    x_pos = {x: left + i * plot_w / (len(x_values) - 1) for i, x in enumerate(x_values)}
    y_pos = lambda y: top + plot_h * (1 - transform(y))
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>']
    for i in range(6):
        frac = i / 5
        value = math.exp(math.log(ymin) + frac * (math.log(ymax) - math.log(ymin))) if log_y else frac * ymax
        yy = y_pos(value)
        out += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#dddddd"/>',
                f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>']
    for x in x_values:
        xx = x_pos[x]
        out += [f'<text x="{xx:.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{x}</text>']
    out += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
            f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="13">Concurrency</text>',
            f'<text x="20" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 20 {top+plot_h/2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>']
    for idx, s in enumerate(series):
        color = colors[idx % len(colors)]
        pts = " ".join(f"{x_pos[x]:.1f},{y_pos(v):.1f}" for x, v in zip(x_values, s["values"]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="{s.get("dash", "none")}"/>')
        for x, v in zip(x_values, s["values"]):
            out.append(f'<circle cx="{x_pos[x]:.1f}" cy="{y_pos(v):.1f}" r="4" fill="{color}"/>')
        if s.get("ci"):
            for x, ci in zip(x_values, s["ci"]):
                xx, y1, y2 = x_pos[x], y_pos(ci[0]), y_pos(ci[1])
                out += [f'<line x1="{xx:.1f}" y1="{y1:.1f}" x2="{xx:.1f}" y2="{y2:.1f}" stroke="{color}"/>',
                        f'<line x1="{xx-5:.1f}" y1="{y1:.1f}" x2="{xx+5:.1f}" y2="{y1:.1f}" stroke="{color}"/>',
                        f'<line x1="{xx-5:.1f}" y1="{y2:.1f}" x2="{xx+5:.1f}" y2="{y2:.1f}" stroke="{color}"/>']
        lx, ly = left + 12 + idx * 190, top + 14
        out += [f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{lx+35}" y="{ly+4}" font-family="sans-serif" font-size="12">{html.escape(s["name"])}</text>']
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    sglang = root / "sglang"
    analysis = root / "analysis"
    bundle = root / "analysis-output"
    figures = bundle / "figures"
    analysis.mkdir(exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    outputs = [analysis / "sglang-run-level.csv", analysis / "sglang-request-level.csv",
               analysis / "sglang-telemetry-physical.csv", analysis / "sglang-baseline-summary.json",
               analysis / "sglang-validation.json", root / "SGLANG_REPORT_ZH.md",
               bundle / "analysis-report.md", bundle / "stats-appendix.md", bundle / "figure-catalog.md"]
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    cases = []
    rapid_expected = {1: 8, 8: 16, 16: 32, 32: 64, 64: 128}
    for c in CONCURRENCIES:
        cases.append(("rapid", "rapid", sglang / "rapid" / f"c{c}", rapid_expected[c], 128, 64))
    for repeat in ("r1", "r3", "r5"):
        for c in CONCURRENCIES:
            cases.append(("formal", repeat, sglang / "formal" / repeat / f"c{c}", {1:12,8:16,16:32,32:64,64:128}[c], 1024, 256))
    cases.append(("stability", "comparable", sglang / "stability" / "c64", 1024, 128, 64))
    cases.append(("supplemental-long-context", "1024x256", sglang / "stability-long-context-1024x256" / "c64", 1024, 1024, 256))

    run_rows, record_errors, request_rows = [], [], []
    for phase, repeat, case_dir, expected, inp, out in cases:
        entry = case_entry(phase, repeat, case_dir)
        rows, errors = validate_records(case_dir, expected, inp, out)
        run_rows.append(entry)
        record_errors.extend(errors)
        for row in rows:
            request_rows.append({"phase": phase, "repeat": repeat, "concurrency": entry["concurrency"], **row})

    with outputs[0].open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0]))
        writer.writeheader(); writer.writerows(run_rows)

    request_fields = ["phase","repeat","concurrency","request_id","deployment_id","status","input_tokens","cached_input_tokens","output_tokens","ttft_ms","tpot_ms","e2e_ms","scheduled_arrival_epoch_ms","actual_start_epoch_ms","client_schedule_delay_ms","http_status","finish_reason","stream_done","error_reason","synthetic_ignore_eos","warmup"]
    with outputs[1].open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=request_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(request_rows)

    telemetry_sources = []
    for phase, repeat, case_dir, *_ in cases:
        p = case_dir / "telemetry.csv"
        if p.exists(): telemetry_sources.append((phase, repeat, case_dir.name, p))
    telemetry_fields = ["phase","repeat","case","timestamp","index","pci_bus_id","serial_number","memory_used_mib","memory_total_mib","gpu_util_pct","memory_util_pct","power_w","temperature_c"]
    telemetry_count = 0
    telemetry_numeric = {"memory_used_mib": [], "gpu_util_pct": [], "memory_util_pct": [], "power_w": [], "temperature_c": []}
    with outputs[2].open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=telemetry_fields); writer.writeheader()
        for phase, repeat, case_name, source in telemetry_sources:
            with source.open(newline="", encoding="utf-8", errors="replace") as raw:
                for row in csv.reader(raw):
                    if len(row) < 10: continue
                    values = [x.strip() for x in row[:10]]
                    try: index = int(values[1])
                    except ValueError: continue
                    if index not in (0, 2, 4, 6): continue
                    item = dict(zip(telemetry_fields, [phase, repeat, case_name, *values]))
                    writer.writerow(item); telemetry_count += 1
                    for key in telemetry_numeric:
                        try: telemetry_numeric[key].append(float(item[key]))
                        except ValueError: pass

    formal_stats = {}
    for c in CONCURRENCIES:
        values = [r["validated_output_tok_s"] for r in run_rows if r["phase"] == "formal" and r["concurrency"] == c]
        formal_stats[str(c)] = summarize_repeats(values)
    rapid = {str(r["concurrency"]): r["validated_output_tok_s"] for r in run_rows if r["phase"] == "rapid"}
    stability = next(r for r in run_rows if r["phase"] == "stability")
    supplemental = next(r for r in run_rows if r["phase"] == "supplemental-long-context")

    latency = {}
    for c in CONCURRENCIES:
        rows = [r for r in request_rows if r["phase"] == "formal" and r["concurrency"] == c and r["status"] == "ok"]
        latency[str(c)] = {"n": len(rows)}
        for metric in ("ttft_ms", "tpot_ms", "e2e_ms", "client_schedule_delay_ms"):
            vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
            latency[str(c)][metric + "_p50"] = percentile(vals, .50)
            latency[str(c)][metric + "_p95"] = percentile(vals, .95)

    pre, post = read_json(sglang / "correctness-pre" / "summary.json"), read_json(sglang / "correctness-post" / "summary.json")
    long_post = read_json(sglang / "correctness-post-long-context-1024x256" / "summary.json")
    extension_skipped = (sglang / "stability" / "c64" / "extension-skipped.json").exists()

    vllm_path = Path("/data/infra/benchmarks/gr308-vllm-baseline-20260907T095637Z/analysis/baseline-summary.json")
    vllm = read_json(vllm_path) if vllm_path.exists() else None
    comparison = {}
    if vllm:
        for c in CONCURRENCIES:
            vm = float(vllm["formal_validated_output_tok_s"][str(c)]["mean"])
            sm = formal_stats[str(c)]["mean"]
            comparison[str(c)] = {"sglang_mean": sm, "vllm_mean": vm, "ratio": sm/vm, "relative_pct": (sm/vm-1)*100}

    telemetry_summary = {key: {"n": len(vals), "mean": statistics.mean(vals) if vals else None, "max": max(vals) if vals else None} for key, vals in telemetry_numeric.items()}
    checks = {
        "all_client_rc_zero": all(r["client_rc"] == 0 for r in run_rows),
        "all_requests_exact": not record_errors,
        "all_summaries_success": all(r["ok"] == r["requests"] for r in run_rows),
        "formal_three_blocks_each": all(x["n"] == 3 for x in formal_stats.values()),
        "formal_cv_le_5_pct": all(x["cv_pct"] <= 5 for x in formal_stats.values()),
        "comparable_stability_protocol": is_comparable_stability(read_json(sglang / "stability" / "c64" / "summary.json")),
        "comparable_stability_1024_of_1024": stability["ok"] == 1024,
        "extension_skipped_only_after_60m": extension_skipped and stability["duration_s"] >= 3600,
        "correctness_pre_12_of_12": pre.get("all_cases_pass") is True and pre.get("passed") == 12 and pre.get("cases_total") == 12,
        "correctness_post_12_of_12": post.get("all_cases_pass") is True and post.get("passed") == 12 and post.get("cases_total") == 12,
        "supplemental_long_context_1024_of_1024": supplemental["ok"] == 1024,
        "supplemental_post_12_of_12": long_post.get("all_cases_pass") is True and long_post.get("passed") == 12,
        "telemetry_present": telemetry_count > 0,
        "owned_container_stopped_cleanly": (sglang / "stability" / "docker-stop.txt").exists(),
    }
    passed = all(checks.values())

    summary = {
        "schema_version": 1, "status": "passed" if passed else "failed",
        "scope": "SGLang upstream with GR308 compatibility adaptation; same model and formal workload as vLLM baseline",
        "model": "Qwen3.8-27B", "served_model_name": "qwen3.8-27b",
        "primary_metric": "validated completed output tokens per second",
        "rapid_validated_output_tok_s": rapid, "formal_validated_output_tok_s": formal_stats,
        "formal_request_latency_ms": latency, "comparable_stability": stability,
        "supplemental_long_context_stress": supplemental, "telemetry": telemetry_summary,
        "correctness": {"pre": {"passed": pre.get("passed"), "total": pre.get("cases_total")}, "post": {"passed": post.get("passed"), "total": post.get("cases_total")}},
        "descriptive_vllm_comparison": comparison,
        "comparison_limit": "Framework runs were not randomized/interleaved; ratios are descriptive system-level comparisons, not causal scheduler estimates.",
        "excluded_from_comparable_stability": "sglang/stability-long-context-1024x256 (retained as supplemental evidence)",
    }
    outputs[3].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = {"schema_version": 1, "passed": passed, "checks": checks, "errors": record_errors}
    outputs[4].write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    svals = [formal_stats[str(c)]["mean"] for c in CONCURRENCIES]
    sci = [formal_stats[str(c)]["descriptive_t95_ci"] for c in CONCURRENCIES]
    throughput_series = [{"name": "SGLang formal mean", "values": svals, "ci": sci}]
    if comparison: throughput_series.append({"name": "vLLM formal mean", "values": [comparison[str(c)]["vllm_mean"] for c in CONCURRENCIES], "dash": "8 5"})
    svg_line_chart(figures / "figure-01-throughput-scaling.svg", "Formal throughput scaling", list(CONCURRENCIES), throughput_series, "Validated output tok/s")
    svg_line_chart(figures / "figure-02-latency.svg", "SGLang formal request latency", list(CONCURRENCIES), [
        {"name": "TTFT p50", "values": [latency[str(c)]["ttft_ms_p50"] for c in CONCURRENCIES]},
        {"name": "TTFT p95", "values": [latency[str(c)]["ttft_ms_p95"] for c in CONCURRENCIES], "dash": "8 5"},
        {"name": "E2E p50", "values": [latency[str(c)]["e2e_ms_p50"] for c in CONCURRENCIES]},
        {"name": "E2E p95", "values": [latency[str(c)]["e2e_ms_p95"] for c in CONCURRENCIES], "dash": "8 5"},
    ], "Latency (ms, log scale)", log_y=True)

    report = ["# GR308 SGLang Baseline 实验报告", "", f"结论：{'通过' if passed else '未通过'}。", "",
              "## 冻结配置", "", "- 模型：`/data/models/Qwen3.8-27B`，served name `qwen3.8-27b`", "- BF16，TP=4，设备 0/2/4/6，上下文 32768", "- SGLang upstream + GR308 兼容层，`torch_native` attention，关闭 radix cache", "- 接口仅绑定 `127.0.0.1:30000`", "",
              "## Formal（1024→256，三个 engine-cold block）", "", "| 并发 | 三次 tok/s | mean | SD | CV | 描述性 95% t-CI |", "|---:|---|---:|---:|---:|---:|"]
    for c in CONCURRENCIES:
        s=formal_stats[str(c)]; ci=s["descriptive_t95_ci"]
        report.append(f"| {c} | {', '.join(f'{v:.4f}' for v in s['values'])} | {s['mean']:.4f} | {s['sample_std']:.4f} | {s['cv_pct']:.3f}% | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    report += ["", "## 与 vLLM 的描述性系统对比", "", "| 并发 | SGLang mean | vLLM mean | SGLang/vLLM | 相对差异 |", "|---:|---:|---:|---:|---:|"]
    for c in CONCURRENCIES:
        d=comparison[str(c)]; report.append(f"| {c} | {d['sglang_mean']:.4f} | {d['vllm_mean']:.4f} | {d['ratio']:.3f}× | {d['relative_pct']:+.1f}% |")
    report += ["", "该比较是系统级描述：两套框架未随机交错运行，兼容层、算子和运行时均可能影响结果，不能把差异单独归因于调度器。", "",
               "## 稳定性与正确性", "", f"- 同口径 C64 128→64：{stability['ok']}/{stability['requests']} 成功，{stability['validated_output_tok_s']:.4f} output tok/s，持续 {stability['duration_s']/60:.1f} 分钟。", "- 主测超过 60 分钟，因此按预注册条件跳过 512 请求扩展段。", f"- 前置 sentinel：{pre.get('passed')}/12；后置 sentinel：{post.get('passed')}/12。", f"- 补充长上下文压力测试 1024→256：{supplemental['ok']}/{supplemental['requests']} 成功，{supplemental['validated_output_tok_s']:.4f} output tok/s，持续 {supplemental['duration_s']/3600:.2f} 小时。该结果不进入 vLLM 稳定性同口径比较。", "",
               "## 边界", "", "- 固定 token、ignore_eos 的合成诊断，不是生产 SLA 或模型质量分数。", "- n=3 的区间仅作描述，不支持细小差异的强推断。", "- 本报告保留协议偏差数据并明确隔离，没有覆盖或删除原始证据。", ""]
    outputs[5].write_text("\n".join(report), encoding="utf-8")

    (bundle / "analysis-report.md").write_text("\n".join(["# Analysis Report", "", "## Analysis question", "", "在同一 Qwen3.8-27B、BF16、TP4 和 Formal 工作负载下，SGLang 的吞吐扩展、延迟和长稳表现如何；与既有 vLLM baseline 的系统级差异有多大。", "", "## Key findings", "", f"- SGLang Formal C64 mean 为 {formal_stats['64']['mean']:.4f} output tok/s，三次 CV 为 {formal_stats['64']['cv_pct']:.3f}%。", f"- 同口径 C64 稳定性为 {stability['validated_output_tok_s']:.4f} output tok/s，1024/1024 成功，运行 {stability['duration_s']/60:.1f} 分钟。", "- 并发增大后 SGLang 吞吐在 C16–C64 附近明显趋于平台；这支持后续从算子、设备利用率和批处理执行路径排查瓶颈。", "- 与 vLLM 的差异仅能描述为当前系统栈差异，不能因果归因到框架调度器。", "", "## Claim Candidates", "", "- Claim:", "  - Source evidence: Formal 三个 engine-cold block 及同口径稳定性主测。", "  - Allowed wording: 当前 GR308 兼容栈下，SGLang 在 C16 以上吞吐扩展明显趋缓。", "  - Forbidden stronger wording: SGLang 调度器必然比 vLLM 差。", "  - Uncertainty: 框架兼容层和底层算子不同，运行未随机交错。", "  - Next check: 对齐 attention/MLP kernel、批处理 shape，并做随机交错 A/B blocks。", "  - Decision: keep", ""]), encoding="utf-8")
    (bundle / "stats-appendix.md").write_text("\n".join(["# Statistics Appendix", "", "- 重复单位：每个 Formal 并发下 3 个独立新容器/进程 block；host page cache 与驱动状态允许保持热。", "- 报告 mean、样本 SD、CV 和 df=2 的描述性 95% t 区间。", "- n=3 太小，未进行正态性检验或参数显著性检验；没有报告 p 值。", "- vLLM/SGLang 未随机交错，不能把三个 block 当作严格配对样本。", "- 框架对比效应以吞吐比和相对百分比呈现，属于描述性系统效应。", "- 多重比较：未做显著性检验，因此不涉及 p 值校正；五个并发点均完整披露。", ""]), encoding="utf-8")
    (bundle / "figure-catalog.md").write_text("\n".join(["# Figure Catalog", "", "## figure-01-throughput-scaling.svg", "", "- Purpose: 展示吞吐随并发扩展，以及当前 SGLang 与既有 vLLM 系统级曲线的差异。", "- Data: 两框架 Formal 三 block mean；SGLang 误差条为描述性 95% t-CI。", "- Observation: SGLang 在 C16 后接近平台，vLLM 继续随并发增长。", "- Interpretation: 下一步优先调查高并发执行路径；不能仅据此归因调度器。", "", "## figure-02-latency.svg", "", "- Purpose: 展示 SGLang Formal 的 TTFT/E2E p50 与 p95 随并发变化。", "- Data: 每个并发三个 block 的成功请求合并；纵轴为对数尺度。", "- Observation: 高并发下端到端延迟增长显著。", "- Interpretation: 吞吐平台与排队/执行时延增长共同指向高并发瓶颈。", ""]), encoding="utf-8")

    critical = []
    for base in (analysis, bundle, sglang / "correctness-pre", sglang / "correctness-post", sglang / "stability" / "c64"):
        critical.extend(p for p in base.rglob("*") if p.is_file())
    critical += [root / "SGLANG_REPORT_ZH.md", root / "manifest" / "host-after.txt"]
    hash_path = root / "FINAL_SHA256SUMS"
    with hash_path.open("x", encoding="utf-8") as f:
        for p in sorted(set(p for p in critical if p.exists())):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            f.write(f"{digest}  {p.relative_to(root)}\n")
    print(json.dumps({"passed": passed, "run_rows": len(run_rows), "request_rows": len(request_rows), "telemetry_rows": telemetry_count, "report": str(outputs[5])}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
