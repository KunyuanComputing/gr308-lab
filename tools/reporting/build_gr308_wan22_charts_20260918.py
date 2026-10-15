#!/usr/bin/env python3
"""Generate charts for the GR308 Wan2.2 tuning report."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "output" / "风华_GR308_Wan2.2视频生成性能测试与调优报告_20260918"
DATA = json.loads((REPORT / "data" / "report_data.json").read_text(encoding="utf-8"))
OUT = REPORT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "/System/Library/Fonts/STHeiti Light.ttc"
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = "Heiti TC"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 9})

BLUE = "#1675C1"
NAVY = "#0D2B4D"
GREEN = "#168A68"
ORANGE = "#D77A00"
RED = "#C74444"
GRAY = "#78838E"
LIGHT = "#E9EEF4"


def by_id(name):
    return next(row for row in DATA["short_results"] if row["id"] == name)


# Figure 1: short-test stage breakdown.
names = ["FP32 baseline", "Global FP16", "M3", "S2"]
rows = [by_id(name) for name in names]
text = np.array([row["text_seconds"] for row in rows])
denoise = np.array([row["denoise_seconds"] for row in rows])
decode = np.array([row["decode_seconds"] for row in rows])
fig, ax = plt.subplots(figsize=(8.2, 4.0))
y = np.arange(len(names))
ax.barh(y, text, color=GRAY, label="Text encode")
ax.barh(y, denoise, left=text, color=ORANGE, label="Denoise")
ax.barh(y, decode, left=text + denoise, color=BLUE, label="VAE decode")
ax.set_yticks(y, names)
ax.invert_yaxis()
ax.set_xlabel("Wall time (s)")
ax.set_title("短测阶段耗时：瓶颈始终是 VAE decode")
ax.grid(axis="x", alpha=0.22)
ax.legend(frameon=False, ncol=3, loc="lower right")
for i, row in enumerate(rows):
    ax.text(row["total_seconds"] + 12, i, f"{row['total_seconds']:.1f}s", va="center", color=NAVY)
fig.tight_layout()
fig.savefig(OUT / "figure-01-stage-breakdown.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# Figure 2: VAE mixed-precision progression.
names = ["M0", "M1", "M2", "M3", "S1", "S2"]
decode = np.array([by_id(name)["decode_seconds"] for name in names])
baseline = by_id("FP32 baseline")["decode_seconds"]
colors = [LIGHT, "#D9E7F3", "#BED8EC", "#8CBFE0", "#54A5D3", GREEN]
fig, ax = plt.subplots(figsize=(8.2, 4.0))
bars = ax.bar(names, decode, color=colors, edgecolor=NAVY, linewidth=0.6)
ax.axhline(baseline, color=RED, linestyle="--", linewidth=1.2, label=f"FP32 baseline {baseline:.1f}s")
ax.set_ylabel("VAE decode wall time (s)")
ax.set_title("逐层混合精度消融")
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False)
for bar, value in zip(bars, decode):
    gain = 100 * (1 - value / baseline)
    ax.text(bar.get_x() + bar.get_width()/2, value + 18, f"{value:.0f}s\n-{gain:.1f}%", ha="center", fontsize=8)
ax.set_ylim(0, max(baseline * 1.12, decode.max() * 1.15))
fig.tight_layout()
fig.savefig(OUT / "figure-02-mixed-precision-path.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# Figure 3: full-reference quality metrics.
quality = DATA["quality_results"]
names = ["Global FP16", "M3", "S2"]
psnr = [quality[name]["psnr_db"] for name in names]
ssim = [quality[name]["ssim_luma"] for name in names]
mae = [quality[name]["mae_0_255"] for name in names]
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.5))
palette = [RED, BLUE, GREEN]
for ax, values, title, ylabel in zip(
    axes,
    [psnr, ssim, mae],
    ["PSNR", "Luma SSIM", "MAE"],
    ["dB (higher is better)", "score (higher is better)", "0-255 (lower is better)"],
):
    bars = ax.bar(names, values, color=palette, edgecolor=NAVY, linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        label = f"{value:.3f}" if title == "Luma SSIM" else f"{value:.2f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02, label, ha="center", fontsize=8)
fig.suptitle("相对 FP32 工程参考的单帧质量")
fig.tight_layout()
fig.savefig(OUT / "figure-03-quality-metrics.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# Figure 4: visual comparison using the same fixed prompt and seed.
paths = [
    ROOT / "output/wan22_quality/frames/fp32_baseline.png",
    ROOT / "output/wan22_quality/frames/fp16_decode.png",
    ROOT / "output/wan22_quality/wave5/s2_frame.png",
]
labels = ["FP32 baseline", "Global FP16 (reject)", "S2 mixed precision"]
fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.15))
for ax, path, label in zip(axes, paths, labels):
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_title(label, color=RED if "reject" in label else NAVY)
    ax.axis("off")
fig.tight_layout(pad=0.8)
fig.savefig(OUT / "figure-04-quality-triptych.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Wrote charts to {OUT}")
