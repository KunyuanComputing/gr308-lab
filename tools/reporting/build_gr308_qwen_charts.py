#!/usr/bin/env python3
"""Generate chart assets for the GR308 Qwen benchmark report."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tmp" / "pdfs" / "gr308_qwen_report" / "performance_comparison.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
FONT_REG = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"

reference = {
    "35B": {1: (1731.87, 52.91), 16: (3487.11, 12.74)},
    "9B": {1: (1545.07, 45.09), 16: (2487.47, 11.97)},
    "30B": {1: (1245.71, 26.85), 16: (3384.84, 11.70)},
    "80B": {1: (1072.45, 33.23), 16: (6048.71, 15.02)},
}
measured = {
    "35B": {1: (1537.19, 50.75), 16: (3321.47, 13.93)},
    "9B": {1: (2335.93, 54.64), 16: (3501.62, 18.90)},
    "30B": {1: (1054.43, 26.43), 16: (4225.69, 15.22)},
    "80B": {1: (595.37, 28.86), 16: (3528.79, 16.20)},
}

models = ["35B", "9B", "30B", "80B"]
fp = FontProperties(fname=FONT_REG)
fpb = FontProperties(fname=FONT_BOLD)
fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), dpi=180)
panels = [
    (axes[0, 0], 1, 1, "C1 Decode Avg TPS", "tokens/s"),
    (axes[0, 1], 16, 1, "C16 Decode Avg TPS", "tokens/s"),
    (axes[1, 0], 1, 0, "C1 Median TTFT", "ms"),
    (axes[1, 1], 16, 0, "C16 Median TTFT", "ms"),
]
x = range(len(models))
for ax, conc, metric_idx, title, ylabel in panels:
    ours = [measured[m][conc][metric_idx] for m in models]
    refs = [reference[m][conc][metric_idx] for m in models]
    bars1 = ax.bar([v - 0.19 for v in x], refs, width=0.38, color="#B7C4D2", label="对照表")
    bars2 = ax.bar([v + 0.19 for v in x], ours, width=0.38, color="#2479B8", label="本次实测")
    ax.set_title(title, fontproperties=fpb, fontsize=10, color="#0D2B4D", loc="left")
    ax.set_ylabel(ylabel, fontproperties=fp, fontsize=7, color="#626A73")
    ax.set_xticks(list(x), models, fontproperties=fp, fontsize=8)
    ax.grid(axis="y", color="#DCE4EC", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BAC7D5")
    ax.tick_params(axis="y", labelsize=7, colors="#626A73")
    ymax = max(refs + ours)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax*0.025,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=6.4, color="#3C4650")
    ax.set_ylim(0, ymax * 1.19)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           prop=fp, bbox_to_anchor=(0.5, 1.01))
fig.suptitle("四模型 C1/C16 同口径对照", fontproperties=fpb, fontsize=13,
             color="#0D2B4D", y=1.04)
plt.tight_layout(rect=(0, 0, 1, 0.97), pad=1.7)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT)
