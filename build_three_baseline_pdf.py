#!/usr/bin/env python3
"""Build the verified GR308 three-baseline comparison report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

import pandas as pd
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/Users/USER/Desktop/Work/GR308")
OUT = ROOT / "docs/benchmarks/GR308_THREE_BASELINE_REPORT_20260910"
FIG = OUT / "analysis-output/figures"

EAGER_DIR = ROOT / "docs/benchmarks/gr308-vllm-baseline-20260907T095637Z"
SGLANG_DIR = ROOT / "docs/benchmarks/gr308-sglang-baseline-20260908T053931Z"
GRAPH_DIR = ROOT / "docs/benchmarks/gr308-vllm-fant-graph-20260909T141559Z"

EAGER_PATH = EAGER_DIR / "baseline-summary.json"
SGLANG_PATH = SGLANG_DIR / "analysis/sglang-baseline-summary.json"
GRAPH_PATH = GRAPH_DIR / "analysis/baseline-summary.json"

PDF_PATH = OUT / "GR308_Qwen3.8-27B_三Baseline对比报告.pdf"

FONT = "ArialUnicodeEmbedded"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
pdfmetrics.registerFont(TTFont(FONT, FONT_PATH, subfontIndex=0))

BLUE = HexColor("#0072B2")
ORANGE = HexColor("#E69F00")
GREEN = HexColor("#009E73")
INK = HexColor("#17212B")
MUTED = HexColor("#5C6773")
GRID = HexColor("#D9E0E6")
PALE = HexColor("#F3F6F8")
PALE_BLUE = HexColor("#EAF4FA")
PALE_GREEN = HexColor("#E7F4EF")
PALE_ORANGE = HexColor("#FFF5DF")
RED = HexColor("#D55E00")

METHODS = ["FANT vLLM Eager", "SGLang + GR308 兼容层", "FANT vLLM Graph32"]
METHOD_COLORS = [BLUE, ORANGE, GREEN]
CONCURRENCIES = [1, 8, 16, 32, 64]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


eager = load_json(EAGER_PATH)
sglang = load_json(SGLANG_PATH)
graph = load_json(GRAPH_PATH)


def validate_sources():
    assert eager["status"] == "passed"
    assert sglang["status"] == "passed"
    assert graph["status"] == "passed"
    for data in (eager, sglang, graph):
        assert data["model"] == "Qwen3.8-27B"
        assert set(map(int, data["formal_validated_output_tok_s"].keys())) == set(CONCURRENCIES)
        for c in CONCURRENCIES:
            point = data["formal_validated_output_tok_s"][str(c)]
            assert point["n"] == 3
            assert len(point["values"]) == 3
            assert point["cv_pct"] < 5


validate_sources()


def formal(data, c, field="mean"):
    return float(data["formal_validated_output_tok_s"][str(c)][field])


def rapid(data, c):
    return float(data["rapid_validated_output_tok_s"][str(c)])


def ratio(a, b):
    return a / b


def rel_pct(a, b):
    return (a / b - 1.0) * 100.0


formal_series = {
    METHODS[0]: [formal(eager, c) for c in CONCURRENCIES],
    METHODS[1]: [formal(sglang, c) for c in CONCURRENCIES],
    METHODS[2]: [formal(graph, c) for c in CONCURRENCIES],
}
formal_ci = {
    METHODS[0]: [eager["formal_validated_output_tok_s"][str(c)]["descriptive_t95_ci"] for c in CONCURRENCIES],
    METHODS[1]: [sglang["formal_validated_output_tok_s"][str(c)]["descriptive_t95_ci"] for c in CONCURRENCIES],
    METHODS[2]: [graph["formal_validated_output_tok_s"][str(c)]["descriptive_t95_ci"] for c in CONCURRENCIES],
}
rapid_series = {
    METHODS[0]: [rapid(eager, c) for c in CONCURRENCIES],
    METHODS[1]: [rapid(sglang, c) for c in CONCURRENCIES],
    METHODS[2]: [rapid(graph, c) for c in CONCURRENCIES],
}
stability_values = [
    float(eager["stability"]["main"]["validated_output_tok_s"]),
    float(sglang["comparable_stability"]["validated_output_tok_s"]),
    float(graph["stability"]["main"]["validated_output_tok_s"]),
]
stability_duration = [
    float(eager["stability"]["main"]["duration_s"]),
    float(sglang["comparable_stability"]["duration_s"]),
    float(graph["stability"]["main"]["duration_s"]),
]


def sglang_stability_latency():
    p = SGLANG_DIR / "analysis/sglang-request-level.csv"
    df = pd.read_csv(p)
    x = df[(df["phase"] == "stability") & (df["repeat"] == "comparable")]
    assert len(x) == 1024
    return {
        "ttft_p50": float(x["ttft_ms"].quantile(0.50)),
        "ttft_p95": float(x["ttft_ms"].quantile(0.95)),
        "tpot_p50": float(x["tpot_ms"].quantile(0.50)),
        "tpot_p95": float(x["tpot_ms"].quantile(0.95)),
        "e2e_p50": float(x["e2e_ms"].quantile(0.50)),
        "e2e_p95": float(x["e2e_ms"].quantile(0.95)),
    }


sg_stab_latency = sglang_stability_latency()
eager_stab_latency = eager["request_latency"]["stability/stability-retry02/64"]
graph_stab_latency = graph["request_latency"]["stability/stability/64"]
stability_latency = {
    "TTFT P50": [
        eager_stab_latency["ttft_ms_p50"] / 1000,
        sg_stab_latency["ttft_p50"] / 1000,
        graph_stab_latency["ttft_ms_p50"] / 1000,
    ],
    "E2E P50": [
        eager_stab_latency["e2e_ms_p50"] / 1000,
        sg_stab_latency["e2e_p50"] / 1000,
        graph_stab_latency["e2e_ms_p50"] / 1000,
    ],
}


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ZHBody", fontName=FONT, fontSize=9.2, leading=14, textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="ZHSmall", fontName=FONT, fontSize=7.5, leading=10.5, textColor=MUTED))
styles.add(ParagraphStyle(name="ZHCaption", fontName=FONT, fontSize=8, leading=11.5, textColor=MUTED, spaceBefore=4, spaceAfter=8))
styles.add(ParagraphStyle(name="ZHTitle", fontName=FONT, fontSize=25, leading=33, textColor=INK, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="ZHSubtitle", fontName=FONT, fontSize=11, leading=17, textColor=MUTED))
styles.add(ParagraphStyle(name="ZHSection", fontName=FONT, fontSize=16, leading=22, textColor=INK, spaceBefore=4, spaceAfter=9))
styles.add(ParagraphStyle(name="ZHSubsection", fontName=FONT, fontSize=11.5, leading=16, textColor=INK, spaceBefore=7, spaceAfter=5))
styles.add(ParagraphStyle(name="ZHMetric", fontName=FONT, fontSize=16, leading=20, textColor=INK, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="ZHMetricLabel", fontName=FONT, fontSize=7.5, leading=10, textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="ZHCenterSmall", fontName=FONT, fontSize=7.5, leading=10, textColor=INK, alignment=TA_CENTER))


def P(text, style="ZHBody"):
    return Paragraph(str(text), styles[style])


def section(title):
    return [P(title, "ZHSection")]


def bullet(text):
    return Paragraph(f"• {text}", styles["ZHBody"])


def make_table(rows, widths, header=True, font_size=7.6, aligns=None):
    cooked = []
    for ridx, row in enumerate(rows):
        cooked_row = []
        for cidx, cell in enumerate(row):
            align = TA_LEFT
            if aligns and cidx < len(aligns):
                align = aligns[cidx]
            style = ParagraphStyle(
                f"cell-{ridx}-{cidx}-{id(rows)}",
                parent=styles["ZHSmall"],
                fontName=FONT,
                fontSize=font_size,
                leading=font_size + 3,
                textColor=colors.white if header and ridx == 0 else INK,
                alignment=align,
            )
            cooked_row.append(Paragraph(str(cell), style))
        cooked.append(cooked_row)
    table = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), PALE))
    table.setStyle(TableStyle(commands))
    return table


def info_box(title, body, fill=PALE_BLUE, border=BLUE):
    box = Table([[P(title, "ZHSubsection")], [P(body, "ZHBody")]], colWidths=[168 * mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.8, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return box


def marker(d, x, y, color, kind):
    if kind == 0:
        d.add(Circle(x, y, 3.3, fillColor=color, strokeColor=colors.white, strokeWidth=0.7))
    elif kind == 1:
        d.add(Rect(x - 3.1, y - 3.1, 6.2, 6.2, fillColor=color, strokeColor=colors.white, strokeWidth=0.7))
    else:
        d.add(Polygon([x, y + 3.8, x - 3.6, y - 3.2, x + 3.6, y - 3.2], fillColor=color, strokeColor=colors.white, strokeWidth=0.7))


def line_chart(series, title, y_max, ci=None, y_label="output tok/s"):
    w, h = 475, 245
    left, right, bottom, top = 50, 15, 34, 31
    pw, ph = w - left - right, h - bottom - top
    d = Drawing(w, h)
    d.add(String(left, h - 14, title, fontName=FONT, fontSize=11, fillColor=INK))
    steps = 5
    for i in range(steps + 1):
        yv = y_max * i / steps
        y = bottom + ph * i / steps
        d.add(Line(left, y, left + pw, y, strokeColor=GRID, strokeWidth=0.45))
        d.add(String(left - 7, y - 2.5, f"{yv:.0f}", fontName=FONT, fontSize=7.2, fillColor=MUTED, textAnchor="end"))
    xs = [left + pw * i / (len(CONCURRENCIES) - 1) for i in range(len(CONCURRENCIES))]
    for x, c in zip(xs, CONCURRENCIES):
        d.add(String(x, bottom - 14, str(c), fontName=FONT, fontSize=7.8, fillColor=MUTED, textAnchor="middle"))
    d.add(String(left + pw / 2, 4, "并发数", fontName=FONT, fontSize=7.8, fillColor=MUTED, textAnchor="middle"))
    d.add(String(3, bottom + ph / 2, y_label, fontName=FONT, fontSize=7.3, fillColor=MUTED))
    for midx, (name, values) in enumerate(series.items()):
        color = METHOD_COLORS[midx]
        pts = []
        for i, (x, value) in enumerate(zip(xs, values)):
            y = bottom + ph * value / y_max
            pts.append((x, y))
            if ci:
                low, high = ci[name][i]
                y0 = bottom + ph * low / y_max
                y1 = bottom + ph * high / y_max
                d.add(Line(x, y0, x, y1, strokeColor=color, strokeWidth=0.8))
                d.add(Line(x - 3, y0, x + 3, y0, strokeColor=color, strokeWidth=0.8))
                d.add(Line(x - 3, y1, x + 3, y1, strokeColor=color, strokeWidth=0.8))
        for a, b in zip(pts, pts[1:]):
            d.add(Line(a[0], a[1], b[0], b[1], strokeColor=color, strokeWidth=1.7, strokeDashArray=[5, 2] if midx == 1 else None))
        for x, y in pts:
            marker(d, x, y, color, midx)
    legend_x = left + 7
    legend_y = h - 29
    for midx, name in enumerate(series):
        x = legend_x + midx * 137
        d.add(Line(x, legend_y, x + 16, legend_y, strokeColor=METHOD_COLORS[midx], strokeWidth=1.8, strokeDashArray=[5, 2] if midx == 1 else None))
        marker(d, x + 8, legend_y, METHOD_COLORS[midx], midx)
        d.add(String(x + 20, legend_y - 2.5, name, fontName=FONT, fontSize=6.8, fillColor=INK))
    return d


def bar_chart(values, title, y_max, y_label="output tok/s", annotations=None):
    w, h = 475, 235
    left, right, bottom, top = 52, 18, 47, 31
    pw, ph = w - left - right, h - bottom - top
    d = Drawing(w, h)
    d.add(String(left, h - 14, title, fontName=FONT, fontSize=11, fillColor=INK))
    for i in range(6):
        yv = y_max * i / 5
        y = bottom + ph * i / 5
        d.add(Line(left, y, left + pw, y, strokeColor=GRID, strokeWidth=0.45))
        d.add(String(left - 7, y - 2.5, f"{yv:.0f}", fontName=FONT, fontSize=7.2, fillColor=MUTED, textAnchor="end"))
    slot = pw / 3
    bw = slot * 0.45
    for i, (name, value, color) in enumerate(zip(METHODS, values, METHOD_COLORS)):
        x = left + slot * (i + 0.5) - bw / 2
        bh = ph * value / y_max
        d.add(Rect(x, bottom, bw, bh, fillColor=color, strokeColor=None))
        d.add(String(x + bw / 2, bottom + bh + 6, f"{value:.2f}", fontName=FONT, fontSize=8.2, fillColor=INK, textAnchor="middle"))
        d.add(String(x + bw / 2, bottom - 13, name, fontName=FONT, fontSize=6.9, fillColor=MUTED, textAnchor="middle"))
        if annotations:
            d.add(String(x + bw / 2, bottom - 25, annotations[i], fontName=FONT, fontSize=6.7, fillColor=MUTED, textAnchor="middle"))
    d.add(String(3, bottom + ph / 2, y_label, fontName=FONT, fontSize=7.3, fillColor=MUTED))
    return d


def grouped_latency_chart():
    w, h = 475, 235
    left, right, bottom, top = 52, 18, 47, 31
    pw, ph = w - left - right, h - bottom - top
    ymax = 320
    d = Drawing(w, h)
    d.add(String(left, h - 14, "同口径稳定性请求延迟（C64，128→64）", fontName=FONT, fontSize=11, fillColor=INK))
    for i in range(5):
        yv = ymax * i / 4
        y = bottom + ph * i / 4
        d.add(Line(left, y, left + pw, y, strokeColor=GRID, strokeWidth=0.45))
        d.add(String(left - 7, y - 2.5, f"{yv:.0f}", fontName=FONT, fontSize=7.2, fillColor=MUTED, textAnchor="end"))
    groups = list(stability_latency.keys())
    group_slot = pw / len(groups)
    bw = group_slot / 5
    for gidx, gname in enumerate(groups):
        center = left + group_slot * (gidx + 0.5)
        vals = stability_latency[gname]
        for midx, (value, color) in enumerate(zip(vals, METHOD_COLORS)):
            x = center + (midx - 1) * bw * 1.25 - bw / 2
            bh = ph * value / ymax
            d.add(Rect(x, bottom, bw, bh, fillColor=color, strokeColor=None))
            d.add(String(x + bw / 2, bottom + bh + 5, f"{value:.1f}", fontName=FONT, fontSize=6.8, fillColor=INK, textAnchor="middle"))
        d.add(String(center, bottom - 14, gname, fontName=FONT, fontSize=7.4, fillColor=MUTED, textAnchor="middle"))
    d.add(String(3, bottom + ph / 2, "秒", fontName=FONT, fontSize=7.3, fillColor=MUTED))
    for midx, name in enumerate(METHODS):
        x = left + midx * 137
        d.add(Rect(x, h - 31, 8, 8, fillColor=METHOD_COLORS[midx], strokeColor=None))
        d.add(String(x + 12, h - 30, name, fontName=FONT, fontSize=6.8, fillColor=INK))
    return d


formal_chart = line_chart(formal_series, "Formal 吞吐随并发变化（1024→256）", 45, formal_ci)
rapid_chart = line_chart(rapid_series, "Rapid 吞吐随并发变化（128→64）", 60)
stability_chart = bar_chart(
    stability_values,
    "稳定性主段吞吐（C64，128→64，1024 请求）",
    50,
    annotations=[f"{v/60:.1f} min" for v in stability_duration],
)
latency_chart = grouped_latency_chart()


def save_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    for name, drawing in [
        ("figure-01-formal-throughput.pdf", formal_chart),
        ("figure-02-rapid-throughput.pdf", rapid_chart),
        ("figure-03-stability-throughput.pdf", stability_chart),
        ("figure-04-stability-latency.pdf", latency_chart),
    ]:
        renderPDF.drawToFile(drawing, str(FIG / name))


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, height - 14 * mm, width - 20 * mm, height - 14 * mm)
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, height - 10.5 * mm, "GR308 · Qwen3.8-27B · 三 Baseline 对比")
        canvas.drawRightString(width - 20 * mm, 10 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_story():
    story = []
    story += [Spacer(1, 28 * mm), P("GR308 Qwen3.8-27B", "ZHSubtitle"), Spacer(1, 5 * mm)]
    story += [P("三套推理框架 Baseline<br/>数据报告", "ZHTitle"), Spacer(1, 12 * mm)]
    story += [P("FANT vLLM Eager · SGLang + GR308 兼容层 · FANT vLLM Graph32", "ZHSubtitle")]
    story += [Spacer(1, 26 * mm)]
    cover_metrics = [
        [P("37.50", "ZHMetric"), P("44.26", "ZHMetric"), P("12/12", "ZHMetric")],
        [P("Formal 峰值 output tok/s<br/>Graph32 · C32", "ZHMetricLabel"), P("稳定性 output tok/s<br/>Graph32 · C64", "ZHMetricLabel"), P("三方案前后哨兵<br/>全部通过", "ZHMetricLabel")],
    ]
    t = Table(cover_metrics, colWidths=[56 * mm] * 3, rowHeights=[15 * mm, 16 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.7, BLUE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [t, Spacer(1, 27 * mm)]
    story += [P("报告日期：2026-09-10", "ZHSmall"), P("证据状态：三套正式运行 validation 均为 passed；报告为系统级描述性比较。", "ZHSmall")]
    story += [PageBreak()]

    story += section("1. 执行摘要")
    story += [info_box(
        "主要结论",
        "FANT Graph32 是本轮中低并发性能上限：Formal 在 C1/C8/C16/C32 相对 FANT vLLM Eager 分别提高 "
        f"{rel_pct(formal(graph,1),formal(eager,1)):.1f}%、{rel_pct(formal(graph,8),formal(eager,8)):.1f}%、"
        f"{rel_pct(formal(graph,16),formal(eager,16)):.1f}%、{rel_pct(formal(graph,32),formal(eager,32)):.1f}%。"
        "但 C64 超出冻结的 Graph capture bucket，Formal 仅比 Eager 高约 "
        f"{rel_pct(formal(graph,64),formal(eager,64)):.1f}%，同口径稳定性吞吐也仅高 {rel_pct(stability_values[2],stability_values[0]):.1f}%。",
        PALE_GREEN,
        GREEN,
    ), Spacer(1, 5 * mm)]
    story += [bullet("FANT vLLM Eager 随并发增加持续扩展，Formal 从 C1 的 1.6428 tok/s 增至 C64 的 32.0792 tok/s。")]
    story += [bullet("SGLang 在 C1 的 Formal 吞吐比 Eager 高 32.1%，但 C8–C64 扩展明显受限；C64 仅为 Eager 的 19.8%。")]
    story += [bullet("Graph32 的 Formal 峰值出现在 C32（37.5037 tok/s）；C64 回落 13.5%，显示 capture bucket 内与 bucket 外饱和点的行为不同。")]
    story += [bullet("三套正式结果均通过请求完整性、稳定性和正确性门禁；所有 Formal 点均为 3 个 engine-cold block，CV 均低于 0.6%。")]
    story += [Spacer(1, 4 * mm), info_box(
        "结论边界",
        "三套运行没有随机交错；框架、兼容层、算子实现和运行时路径同时变化。本文不报告 p 值，不使用“显著优于”措辞，也不把差异单独归因于调度器或 Graph。",
        PALE_ORANGE,
        ORANGE,
    )]
    story += [PageBreak()]

    story += section("2. 测试对象与实验口径")
    hardware_rows = [
        ["项目", "冻结配置"],
        ["服务器", "Ubuntu 22.04；2× AMD EPYC 9654；224 逻辑 CPU；约 1 TiB 内存；2 个 NUMA node"],
        ["加速卡", "4× Fantasy GR308；每卡 57,344 MiB（约 56 GiB），物理总显存约 224 GiB"],
        ["设备映射", "逻辑设备 0/2/4/6，各对应一张物理卡的 .0 function；4 卡位于 NUMA0"],
        ["模型", "本地 /data/models/Qwen3.8-27B；BF16；TP=4；served name qwen3.8-27b"],
        ["公共限制", "max model length 32768；text-only；OpenAI-compatible 客户端；固定 token ID；ignore_eos=true"],
    ]
    story += [make_table(hardware_rows, [35 * mm, 133 * mm], aligns=[TA_LEFT, TA_LEFT]), Spacer(1, 6 * mm)]
    framework_rows = [
        ["方案", "执行路径", "定位"],
        ["FANT vLLM Eager", "厂商适配 vLLM；eager execution", "可运行基础性能基线"],
        ["SGLang + GR308 兼容层", "SGLang upstream + 兼容适配；torch_native attention；radix cache 关闭", "另一开源框架适配基线"],
        ["FANT vLLM Graph32", "厂商 Graph；capture bucket=1/2/4/8/16/32", "现有厂商栈性能上限参考"],
    ]
    story += [make_table(framework_rows, [38 * mm, 83 * mm, 47 * mm], aligns=[TA_LEFT, TA_LEFT, TA_LEFT]), Spacer(1, 6 * mm)]
    protocol_rows = [
        ["阶段", "输入→输出", "并发", "请求数", "重复/条件"],
        ["Rapid", "128→64", "1/8/16/32/64", "8/16/32/64/128", "单轮筛查"],
        ["Formal", "1024→256", "1/8/16/32/64", "12/16/32/64/128", "3 个 engine-cold block"],
        ["稳定性主段", "128→64", "64", "1024", "三方案同口径"],
        ["稳定性扩展", "128→64", "64", "512", "主段不足 60 分钟时执行"],
        ["正确性哨兵", "结构化任务", "—", "12 cases", "性能前后各一次"],
    ]
    story += [make_table(protocol_rows, [30 * mm, 31 * mm, 38 * mm, 30 * mm, 39 * mm], aligns=[TA_LEFT, TA_CENTER, TA_CENTER, TA_CENTER, TA_LEFT])]
    story += [Spacer(1, 5 * mm), P("主指标为客户端验证的 completed output tokens per second，越高越好。TTFT、TPOT、E2E 延迟以成功请求为单位；正确性哨兵只验证冻结的 12 个精确任务，不构成业务质量认证。", "ZHCaption")]
    story += [PageBreak()]

    story += section("3. Formal 核心吞吐结果")
    story += [formal_chart, P("图 1｜Formal（1024→256）吞吐扩展曲线。每点为 3 个 engine-cold block 的均值，误差条为描述性 95% t 区间（n=3）；不用于细小差异的显著性推断。", "ZHCaption")]
    rows = [["并发", "Eager mean±SD", "SGLang mean±SD", "Graph32 mean±SD", "Graph/Eager", "SGLang/Eager"]]
    for c in CONCURRENCIES:
        e = eager["formal_validated_output_tok_s"][str(c)]
        s = sglang["formal_validated_output_tok_s"][str(c)]
        g = graph["formal_validated_output_tok_s"][str(c)]
        rows.append([
            str(c), f"{e['mean']:.4f} ± {e['sample_std']:.4f}", f"{s['mean']:.4f} ± {s['sample_std']:.4f}",
            f"{g['mean']:.4f} ± {g['sample_std']:.4f}", f"{g['mean']/e['mean']:.3f}×", f"{s['mean']/e['mean']:.3f}×",
        ])
    story += [make_table(rows, [16 * mm, 34 * mm, 38 * mm, 38 * mm, 22 * mm, 24 * mm], aligns=[TA_CENTER] * 6, font_size=7.2)]
    story += [Spacer(1, 5 * mm), P("读图重点", "ZHSubsection")]
    story += [bullet("Graph32 在 C1–C32 显著改变了系统的吞吐扩展形状；最大均值为 C32 的 37.5037 tok/s。这里的“显著”仅指幅度明显，不是统计显著性术语。")]
    story += [bullet("Eager 在 C64 达到 32.0792 tok/s；Graph32 在 C64 为 32.4532 tok/s，两者仅相差 1.2%。")]
    story += [bullet("SGLang C8 后趋于平台化，C8→C64 仅由 5.2778 增至 6.3494 tok/s，表现出明显的并发扩展差距。")]
    story += [PageBreak()]

    story += section("4. Rapid 与短输出响应能力")
    story += [rapid_chart, P("图 2｜Rapid（128→64）吞吐扩展曲线。该阶段为单轮快速筛查，没有跨 block 误差估计；用于观察形状，不替代 Formal 结论。", "ZHCaption")]
    rapid_rows = [["并发", "FANT vLLM Eager", "SGLang", "FANT Graph32", "Graph 相对 Eager"]]
    for c in CONCURRENCIES:
        rapid_rows.append([
            str(c), f"{rapid(eager,c):.4f}", f"{rapid(sglang,c):.4f}", f"{rapid(graph,c):.4f}", f"{rel_pct(rapid(graph,c),rapid(eager,c)):+.1f}%",
        ])
    story += [make_table(rapid_rows, [22 * mm, 36 * mm, 32 * mm, 36 * mm, 42 * mm], aligns=[TA_CENTER] * 5)]
    story += [Spacer(1, 5 * mm), info_box(
        "关键观察",
        "Graph32 在 Rapid C32 达到 52.8004 tok/s，随后在 C64 回落到 47.0459 tok/s；Eager C64 为 47.0706 tok/s，几乎完全相同。短输出与 Formal 都指向同一个边界：Graph 优势集中在捕获桶覆盖的 C1–C32。",
        PALE_GREEN,
        GREEN,
    )]
    story += [PageBreak()]

    story += section("5. 稳定性、延迟与正确性")
    story += [stability_chart, P("图 3｜三方案同口径稳定性主段：C64、128→64、1024 请求。柱上为吞吐，柱下为主段持续时间。三组均为 1024/1024 成功。", "ZHCaption")]
    stability_rows = [
        ["方案", "成功请求", "吞吐 tok/s", "持续时间", "扩展段"],
        [METHODS[0], "1024/1024", f"{stability_values[0]:.4f}", f"{stability_duration[0]/60:.1f} min", "512/512，44.3500 tok/s"],
        [METHODS[1], "1024/1024", f"{stability_values[1]:.4f}", f"{stability_duration[1]/60:.1f} min", "按 ≥60 min 条件跳过"],
        [METHODS[2], "1024/1024", f"{stability_values[2]:.4f}", f"{stability_duration[2]/60:.1f} min", "512/512，46.2399 tok/s"],
    ]
    story += [make_table(stability_rows, [44 * mm, 25 * mm, 31 * mm, 26 * mm, 42 * mm], aligns=[TA_LEFT, TA_CENTER, TA_CENTER, TA_CENTER, TA_LEFT])]
    story += [PageBreak(), P("5.1 同口径稳定性请求延迟", "ZHSubsection"), latency_chart, P("图 4｜同口径稳定性请求的 P50 延迟。TTFT 为首 token 延迟，E2E 为整请求完成延迟；单位为秒。SGLang 数据从 1024 条请求级记录重新计算。", "ZHCaption")]
    latency_rows = [
        ["方案", "TTFT P50", "TTFT P95", "TPOT P50", "E2E P50", "E2E P95"],
        [METHODS[0], f"{eager_stab_latency['ttft_ms_p50']/1000:.2f}s", f"{eager_stab_latency['ttft_ms_p95']/1000:.2f}s", f"{eager_stab_latency['tpot_ms_p50']:.1f}ms", f"{eager_stab_latency['e2e_ms_p50']/1000:.2f}s", f"{eager_stab_latency['e2e_ms_p95']/1000:.2f}s"],
        [METHODS[1], f"{sg_stab_latency['ttft_p50']/1000:.2f}s", f"{sg_stab_latency['ttft_p95']/1000:.2f}s", f"{sg_stab_latency['tpot_p50']:.1f}ms", f"{sg_stab_latency['e2e_p50']/1000:.2f}s", f"{sg_stab_latency['e2e_p95']/1000:.2f}s"],
        [METHODS[2], f"{graph_stab_latency['ttft_ms_p50']/1000:.2f}s", f"{graph_stab_latency['ttft_ms_p95']/1000:.2f}s", f"{graph_stab_latency['tpot_ms_p50']:.1f}ms", f"{graph_stab_latency['e2e_ms_p50']/1000:.2f}s", f"{graph_stab_latency['e2e_ms_p95']/1000:.2f}s"],
    ]
    story += [make_table(latency_rows, [39 * mm, 25 * mm, 25 * mm, 27 * mm, 25 * mm, 27 * mm], aligns=[TA_LEFT] + [TA_CENTER] * 5, font_size=7.2)]
    story += [Spacer(1, 6 * mm)]
    correctness_rows = [
        ["方案", "性能前哨兵", "性能后哨兵", "结论"],
        [METHODS[0], "记录通过", "12/12", "通过"],
        [METHODS[1], "12/12", "12/12", "通过"],
        [METHODS[2], "12/12", "12/12", "通过"],
    ]
    story += [make_table(correctness_rows, [52 * mm, 35 * mm, 35 * mm, 46 * mm], aligns=[TA_LEFT, TA_CENTER, TA_CENTER, TA_CENTER])]
    story += [P("注：Eager 的正式 validation 明确记录 post sentinel 12/12；其报告与证据链记录 pre sentinel 已执行，但汇总 JSON 未单列 pre 数字，因此此处保持原证据措辞，不补造 12/12。", "ZHCaption")]
    story += [PageBreak()]

    story += section("6. 统计可信度与解释边界")
    cv_rows = [["并发", "Eager CV", "SGLang CV", "Graph32 CV", "最大 CV"]]
    for c in CONCURRENCIES:
        cvs = [formal(x, c, "cv_pct") for x in (eager, sglang, graph)]
        cv_rows.append([str(c), f"{cvs[0]:.3f}%", f"{cvs[1]:.3f}%", f"{cvs[2]:.3f}%", f"{max(cvs):.3f}%"])
    story += [make_table(cv_rows, [24 * mm, 34 * mm, 34 * mm, 34 * mm, 42 * mm], aligns=[TA_CENTER] * 5), Spacer(1, 6 * mm)]
    story += [bullet("单位：Formal 的 engine-cold block；每个框架、每个并发 n=3。请求行不是独立重复实验，不能把单请求数量当作统计样本量。")]
    story += [bullet("不确定性：表格与图 1 使用样本标准差和描述性 95% t 区间；n=3 时区间对分布假设敏感。")]
    story += [bullet("推断限制：运行顺序未随机化、未交错，宿主页缓存和驱动状态可能保持热态；因此不执行框架间 t 检验/ANOVA，也不报告 p 值。")]
    story += [bullet("机制限制：SGLang、Eager 与 Graph32 同时改变框架与底层执行路径。观察到的差异是系统级效应，不能直接等同于调度器、算子或 Graph 的单独贡献。")]
    story += [bullet("C64 限制：Graph capture bucket 冻结为 1/2/4/8/16/32；C64 是 bucket 外 saturation/stress 点。")]
    story += [bullet("质量限制：固定 token ID、ignore_eos=true 的合成 workload 适合性能诊断，但不代表真实提示分布、业务 goodput、生产 SLA 或模型质量。")]
    story += [Spacer(1, 6 * mm), info_box(
        "允许使用的结论",
        "“在本次冻结配置和合成负载下，Graph32 在 C1–C32 的描述性吞吐明显高于 Eager 与 SGLang；C64 与 Eager 接近。”",
        PALE_GREEN,
        GREEN,
    ), Spacer(1, 4 * mm), info_box(
        "禁止扩大的结论",
        "不能写成“Graph32 在所有并发下显著优于其他框架”“差异由调度器导致”或“这些吞吐可直接作为生产 SLA”。",
        PALE_ORANGE,
        ORANGE,
    )]
    story += [PageBreak()]

    story += section("附录 A｜完整 Formal 数值")
    full_rows = [["方案", "并发", "三个 engine-cold block（tok/s）", "mean", "SD", "CV", "描述性 95% CI"]]
    for label, data in zip(METHODS, (eager, sglang, graph)):
        for c in CONCURRENCIES:
            p = data["formal_validated_output_tok_s"][str(c)]
            full_rows.append([
                label, str(c), ", ".join(f"{x:.4f}" for x in p["values"]), f"{p['mean']:.4f}",
                f"{p['sample_std']:.4f}", f"{p['cv_pct']:.3f}%", f"[{p['descriptive_t95_ci'][0]:.4f}, {p['descriptive_t95_ci'][1]:.4f}]",
            ])
    story += [make_table(full_rows, [31 * mm, 13 * mm, 48 * mm, 20 * mm, 18 * mm, 17 * mm, 31 * mm], aligns=[TA_LEFT, TA_CENTER, TA_CENTER, TA_CENTER, TA_CENTER, TA_CENTER, TA_CENTER], font_size=6.45)]
    story += [Spacer(1, 7 * mm), P("附录 B｜证据位置", "ZHSection")]
    evidence_rows = [
        ["方案", "服务器正式结果目录", "本地机器可读摘要"],
        [METHODS[0], "/data/infra/benchmarks/gr308-vllm-baseline-20260907T095637Z", str(EAGER_PATH)],
        [METHODS[1], "/data/infra/benchmarks/gr308-sglang-baseline-20260908T053931Z", str(SGLANG_PATH)],
        [METHODS[2], "/data/infra/benchmarks/gr308-vllm-fant-graph-20260909T141559Z", str(GRAPH_PATH)],
    ]
    story += [make_table(evidence_rows, [39 * mm, 69 * mm, 70 * mm], aligns=[TA_LEFT, TA_LEFT, TA_LEFT], font_size=6.4)]
    story += [Spacer(1, 4 * mm), P("FANT Graph 的三次启动失败尝试分别保存在 20260909T135112Z、20260909T135742Z、20260909T140333Z 目录，并带 EXCLUDED_RUN.json；它们未进入正式统计。SGLang 的 1024→256 长上下文稳定性结果保留为补充压力证据，不进入同口径稳定性比较。", "ZHCaption")]
    return story


def write_analysis_bundle():
    aout = OUT / "analysis-output"
    aout.mkdir(parents=True, exist_ok=True)
    analysis = f"""# 三 Baseline 严格分析摘要

## 分析问题

在同一 GR308 服务器、同一 Qwen3.8-27B BF16 模型、TP=4 和冻结合成负载下，比较 FANT vLLM Eager、SGLang + GR308 兼容层、FANT vLLM Graph32 的吞吐扩展、同口径稳定性与请求延迟。

## 关键发现

- Graph32 Formal 在 C1/C8/C16/C32 相对 Eager 分别为 {ratio(formal(graph,1),formal(eager,1)):.3f}×、{ratio(formal(graph,8),formal(eager,8)):.3f}×、{ratio(formal(graph,16),formal(eager,16)):.3f}×、{ratio(formal(graph,32),formal(eager,32)):.3f}×。
- Graph32 C64 为 {formal(graph,64):.4f} tok/s，Eager 为 {formal(eager,64):.4f} tok/s，仅相差 {rel_pct(formal(graph,64),formal(eager,64)):+.1f}%。
- SGLang C1 相对 Eager为 {rel_pct(formal(sglang,1),formal(eager,1)):+.1f}%，但 C64 为 {rel_pct(formal(sglang,64),formal(eager,64)):+.1f}%。
- 同口径稳定性主段三者均 1024/1024 成功；吞吐依次为 {stability_values[0]:.4f}、{stability_values[1]:.4f}、{stability_values[2]:.4f} tok/s。

## Claim Candidates

- Claim: Graph32 在本轮冻结负载的 C1–C32 中提供最高描述性吞吐。
  - Source evidence: 三方案 Formal 各并发 3 个 engine-cold block。
  - Allowed wording: 在本次系统级对比中，Graph32 的 C1–C32 均值最高。
  - Forbidden stronger wording: Graph 调度器在所有业务负载上显著优于其他框架。
  - Uncertainty: 未随机交错；框架与底层执行路径同时变化；n=3。
  - Next check: 随机交错 5–10 个 block，并做组件级消融。
  - Decision: keep

- Claim: Graph32 的 C64 饱和吞吐没有形成相对 Eager 的明显优势。
  - Source evidence: Formal 32.4532 vs 32.0792 tok/s；稳定性 44.2578 vs 44.0619 tok/s。
  - Allowed wording: C64 的描述性差异约为 1.2% / 0.4%。
  - Forbidden stronger wording: Graph 对高并发完全无效。
  - Uncertainty: C64 在冻结 capture bucket 之外。
  - Next check: 增加 64 capture bucket 后重跑。
  - Decision: keep
"""
    stats_lines = [
        "# 统计附录", "", "- 分析单位：engine-cold block。每方案、每并发 n=3。", "- 主指标：validated completed output tok/s，越高越好。",
        "- 描述统计：均值、样本 SD、CV、描述性 95% t 区间。", "- 未进行显著性检验：n=3 且框架运行未随机交错，不满足强推断设计。",
        "- 效应表达：系统级吞吐比值与相对百分比，不解释为单组件因果效应。", "",
        "| 并发 | Eager mean±SD | SGLang mean±SD | Graph32 mean±SD | Graph/Eager | SGLang/Eager |", "|---:|---:|---:|---:|---:|---:|",
    ]
    for c in CONCURRENCIES:
        e = eager["formal_validated_output_tok_s"][str(c)]
        s = sglang["formal_validated_output_tok_s"][str(c)]
        g = graph["formal_validated_output_tok_s"][str(c)]
        stats_lines.append(f"| {c} | {e['mean']:.4f}±{e['sample_std']:.4f} | {s['mean']:.4f}±{s['sample_std']:.4f} | {g['mean']:.4f}±{g['sample_std']:.4f} | {g['mean']/e['mean']:.3f}× | {s['mean']/e['mean']:.3f}× |")
    catalog = """# 图表目录

## figure-01-formal-throughput.pdf
- Purpose: 比较正式长上下文负载下的吞吐扩展。
- Data source: 三份 baseline-summary.json 的 Formal 三 block 均值与描述性 95% t 区间。
- Key observation: Graph32 在 C1–C32 领先，C64 回落并接近 Eager。
- Caveat: n=3；区间只作描述；未随机交错。

## figure-02-rapid-throughput.pdf
- Purpose: 比较短输出快速筛查的扩展形状。
- Data source: 三份 summary 的 Rapid 单轮值。
- Key observation: Graph32 在 C32 达峰，C64 与 Eager 收敛。
- Caveat: 单轮，无跨 block 误差估计。

## figure-03-stability-throughput.pdf
- Purpose: 比较同口径 C64 稳定性主段。
- Data source: 各方案 128→64、C64、1024 请求正式稳定性主段。
- Key observation: Eager 与 Graph32 接近，SGLang 明显较低；三者均零请求失败。
- Caveat: 单个长运行，不是独立重复。

## figure-04-stability-latency.pdf
- Purpose: 对比同口径稳定性请求的 TTFT/E2E P50。
- Data source: Eager/Graph summary 与 SGLang 1024 条请求级记录。
- Key observation: Graph32 与 Eager 接近，SGLang E2E P50 更高。
- Caveat: 闭环并发延迟受排队与服务共同影响。
"""
    (aout / "analysis-report.md").write_text(analysis, encoding="utf-8")
    (aout / "stats-appendix.md").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    (aout / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def build_pdf():
    OUT.mkdir(parents=True, exist_ok=True)
    save_figures()
    write_analysis_bundle()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=17 * mm,
        title="GR308 Qwen3.8-27B 三套推理框架 Baseline 数据报告",
        author="Codex",
        subject="FANT vLLM Eager、SGLang GR308 兼容层与 FANT vLLM Graph32 的系统级对比",
    )
    doc.build(build_story(), onFirstPage=header_footer, onLaterPages=header_footer)
    print(PDF_PATH)


if __name__ == "__main__":
    build_pdf()
