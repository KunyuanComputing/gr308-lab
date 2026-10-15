#!/usr/bin/env python3
"""Generate charts for the 2026-09-17 comprehensive GR308 Qwen tuning report."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output" / "风华_GR308_Qwen系列并发性能测试与调优报告_20260917" / "data" / "report_data.json"
OUT = ROOT / "output" / "风华_GR308_Qwen系列并发性能测试与调优报告_20260917" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT_REG = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"
fp = FontProperties(fname=FONT_REG)
fpb = FontProperties(fname=FONT_BOLD)
data = json.loads(DATA.read_text())

models = ["35B", "9B", "30B", "80B"]
blue, green, orange, gray = "#2479B8", "#168A68", "#D77A00", "#AEBCC9"

def polish(ax, title, ylabel="tokens/s"):
    ax.set_title(title, fontproperties=fpb, fontsize=11, color="#0D2B4D", loc="left")
    ax.set_ylabel(ylabel, fontproperties=fp, fontsize=8, color="#626A73")
    ax.grid(axis="y", color="#DCE4EC", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BAC7D5")
    ax.tick_params(axis="both", labelsize=8, colors="#626A73")


# Figure 1: same-concurrency external reference comparison.
fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.0), dpi=180)
panels = [
    (axes[0, 0], "c1", "decode", "C1 Decode Avg TPS", "tokens/s"),
    (axes[0, 1], "c16", "decode", "C16 Decode Avg TPS", "tokens/s"),
    (axes[1, 0], "c1", "ttft", "C1 Median TTFT", "ms"),
    (axes[1, 1], "c16", "ttft", "C16 Median TTFT", "ms"),
]
x = np.arange(len(models))
for ax, conc, metric, title, ylabel in panels:
    ref = [data["reference"][m][conc][metric] for m in models]
    ours = [data["selected"][m][conc][metric] for m in models]
    b1 = ax.bar(x - 0.19, ref, .38, color=gray, label="对照表")
    b2 = ax.bar(x + 0.19, ours, .38, color=blue, label="选定方案")
    ax.set_xticks(x, models, fontproperties=fp)
    polish(ax, title, ylabel)
    ymax = max(ref + ours)
    ax.set_ylim(0, ymax * 1.18)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax*.02,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=6.5, color="#3C4650")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2,
           frameon=False, prop=fp, bbox_to_anchor=(.5, 1.01))
fig.suptitle("四模型 C1/C16 与外部对照表的同口径比较", fontproperties=fpb,
             fontsize=14, color="#0D2B4D", y=1.045)
plt.tight_layout(rect=(0, 0, 1, .97), pad=1.6)
fig.savefig(OUT / "figure-01-reference-comparison.png", bbox_inches="tight", facecolor="white")
plt.close(fig)


# Figure 2: aggregate throughput scaling and per-card normalization.
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), dpi=180)
c16 = [data["selected"][m]["c16"]["aggregate"] for m in models]
c32 = [data["selected"][m]["c32"]["aggregate"] for m in models]
pc32 = [data["selected"][m]["c32"]["aggregate"] / data["selected"][m]["cards"] for m in models]
b1 = axes[0].bar(x - .19, c16, .38, color=blue, label="C16")
b2 = axes[0].bar(x + .19, c32, .38, color=green, label="C32")
axes[0].set_xticks(x, models, fontproperties=fp)
polish(axes[0], "选定配置的聚合输出吞吐")
axes[0].legend(frameon=False, prop=fp, fontsize=8)
for bars in (b1, b2):
    axes[0].bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
b3 = axes[1].bar(x, pc32, .58, color=[blue, green, orange, "#7158A5"])
axes[1].set_xticks(x, models, fontproperties=fp)
polish(axes[1], "C32 每物理卡吞吐")
axes[1].bar_label(b3, fmt="%.1f", fontsize=7, padding=2)
fig.suptitle("高并发容量与资源归一化", fontproperties=fpb, fontsize=14,
             color="#0D2B4D", y=1.04)
plt.tight_layout(pad=1.5)
fig.savefig(OUT / "figure-02-c16-c32-throughput.png", bbox_inches="tight", facecolor="white")
plt.close(fig)


# Figure 3: C1 and batch roof efficiencies.
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), dpi=180,
                         gridspec_kw={"width_ratios": [1, 1.15]})
c1_system = [100 * data["selected"][m]["c1"]["aggregate"] / data["roof"][m]["c1"] for m in models]
c1_decode = [100 * data["selected"][m]["c1"]["decode"] / data["roof"][m]["c1"] for m in models]
c16_eff = [100 * data["selected"][m]["c16"]["aggregate"] / data["roof"][m]["ideal"] for m in models]
c32_eff = [100 * data["selected"][m]["c32"]["aggregate"] / data["roof"][m]["ideal"] for m in models]

b0 = axes[0].bar(x - .19, c1_system, .38, color=blue, label="系统效率")
b1 = axes[0].bar(x + .19, c1_decode, .38, color=orange, label="纯 Decode 效率")
axes[0].set_xticks(x, models, fontproperties=fp)
polish(axes[0], "C1 单请求屋顶效率", "%")
axes[0].set_ylim(0, max(c1_decode) * 1.18)
axes[0].legend(frameon=False, prop=fp, fontsize=7)
for bars in (b0, b1):
    axes[0].bar_label(bars, fmt="%.1f%%", fontsize=7, padding=3)

b2 = axes[1].bar(x - .19, c16_eff, .38, color=blue, label="C16")
b3 = axes[1].bar(x + .19, c32_eff, .38, color=green, label="C32")
axes[1].set_xticks(x, models, fontproperties=fp)
polish(axes[1], "C16/C32 批处理屋顶效率", "%")
axes[1].set_ylim(0, max(c32_eff) * 1.25)
axes[1].legend(frameon=False, prop=fp, fontsize=7)
for bars in (b2, b3):
    axes[1].bar_label(bars, fmt="%.1f%%", fontsize=7, padding=3)
fig.suptitle("实测吞吐 / 理论屋顶", fontproperties=fpb, fontsize=14,
             color="#0D2B4D", y=1.03)
plt.tight_layout(pad=1.5)
fig.savefig(OUT / "figure-03-roof-efficiency.png", bbox_inches="tight", facecolor="white")
plt.close(fig)


# Figure 4: tuning-path C16 aggregate TPS, four panels.
fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), dpi=180)
for ax, m in zip(axes.flat, models):
    rows = [r for r in data["tuning"][m] if r["c16"] is not None]
    labels = [r["group"] for r in rows]
    vals = [r["c16"] for r in rows]
    colors = [green if r["group"] == "Final" or r["group"] == "MTP3" else ("#C74444" if "EP" in r["group"] else blue) for r in rows]
    bars = ax.bar(np.arange(len(rows)), vals, color=colors)
    ax.set_xticks(np.arange(len(rows)), labels, rotation=32, ha="right", fontproperties=fp, fontsize=7)
    polish(ax, f"{m} C16 调优路径")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.bar_label(bars, fmt="%.0f", fontsize=6.2, padding=2)
fig.suptitle("参数与并行策略消融：C16 聚合输出吞吐", fontproperties=fpb,
             fontsize=14, color="#0D2B4D", y=1.02)
plt.tight_layout(rect=(0, 0, 1, .98), pad=1.4)
fig.savefig(OUT / "figure-04-tuning-paths.png", bbox_inches="tight", facecolor="white")
plt.close(fig)


# Figure 5: user-provided RTX 5090 comparison, limited to 35B and 9B.
fig, ax = plt.subplots(figsize=(9.4, 3.2), dpi=180)
labels = ["Qwen3.6-35B-A3B\nC16/卡", "Qwen3.5-9B\nC16"]
gr308 = [data["rtx5090"]["35B"]["gr308_tps"],
         data["rtx5090"]["9B"]["gr308_tps"]]
rtx5090 = [data["rtx5090"]["35B"]["rtx5090_per_card_tps"],
           data["rtx5090"]["9B"]["rtx5090_per_card_tps"]]
x2 = np.arange(len(labels))
b1 = ax.bar(x2 - .19, gr308, .38, color=blue, label="GR308 BF16（本报告）")
b2 = ax.bar(x2 + .19, rtx5090, .38, color="#343A40", label="RTX 5090（所给数据）")
ax.set_xticks(x2, labels, fontproperties=fp)
polish(ax, "GR308 与 RTX 5090：C16 每卡吞吐对照", "output tokens/s")
ax.set_ylim(0, max(rtx5090) * 1.18)
ax.legend(frameon=False, prop=fp, fontsize=8, loc="upper left")
for bars in (b1, b2):
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
ratios = [data["rtx5090"]["35B"]["ratio_pct"], data["rtx5090"]["9B"]["ratio_pct"]]
for i, (g, n, ratio) in enumerate(zip(gr308, rtx5090, ratios)):
    ax.text(i, max(g, n) * 1.08, f"GR/5090 = {ratio:.2f}%",
            ha="center", va="bottom", fontsize=8, color="#0D2B4D",
            fontproperties=fpb)
fig.text(.5, -.01,
         "35B 的 RTX 5090 数值为 8 卡总吞吐 3990.23 / 8 的每卡归一值，原测试采用 4×TP2，并非单卡独立运行。",
         ha="center", va="bottom", fontsize=7, color="#626A73", fontproperties=fp)
plt.tight_layout(rect=(0, .08, 1, 1), pad=1.4)
fig.savefig(OUT / "figure-05-rtx5090-comparison.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print(OUT)
