#!/usr/bin/env python3
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace


ROOT = Path(__file__).resolve().parent
FRAME_DIR = ROOT / "frames"
FIGURE_DIR = ROOT / "analysis-output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def rgb_to_luma(x: np.ndarray) -> np.ndarray:
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Standard single-scale SSIM: 11x11-equivalent Gaussian window, sigma=1.5.
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = gaussian_filter(a, sigma=1.5)
    mu_b = gaussian_filter(b, sigma=1.5)
    sigma_a = gaussian_filter(a * a, sigma=1.5) - mu_a * mu_a
    sigma_b = gaussian_filter(b * b, sigma=1.5) - mu_b * mu_b
    sigma_ab = gaussian_filter(a * b, sigma=1.5) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    )
    return float(np.mean(score))


def entropy_u8(x: np.ndarray) -> float:
    hist = np.bincount(np.clip(np.rint(x), 0, 255).astype(np.uint8).ravel(), minlength=256)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def image_stats(x: np.ndarray) -> dict:
    y = rgb_to_luma(x)
    row_mean = y.mean(axis=1)
    vertical_neighbor = np.abs(np.diff(y, axis=0)).mean()
    horizontal_neighbor = np.abs(np.diff(y, axis=1)).mean()
    return {
        "rgb_mean": [float(v) for v in x.mean(axis=(0, 1))],
        "rgb_std": [float(v) for v in x.std(axis=(0, 1))],
        "luma_mean": float(y.mean()),
        "luma_std": float(y.std()),
        "luma_entropy_bits": entropy_u8(y),
        "laplacian_variance": float(np.var(laplace(y))),
        "dark_clipping_pct": float(100 * np.mean(x <= 2)),
        "bright_clipping_pct": float(100 * np.mean(x >= 253)),
        "row_mean_step_abs": float(np.abs(np.diff(row_mean)).mean()),
        "vertical_neighbor_abs": float(vertical_neighbor),
        "horizontal_neighbor_abs": float(horizontal_neighbor),
        "vertical_horizontal_gradient_ratio": float(vertical_neighbor / horizontal_neighbor),
    }


baseline = load_rgb(FRAME_DIR / "fp32_baseline.png")
optimized = load_rgb(FRAME_DIR / "fp16_decode.png")
if baseline.shape != optimized.shape:
    raise ValueError(f"Shape mismatch: {baseline.shape} vs {optimized.shape}")

error = baseline - optimized
mse = float(np.mean(error**2))
mae = float(np.mean(np.abs(error)))
psnr = float("inf") if mse == 0 else float(10 * math.log10((255.0**2) / mse))
ssim_rgb = [ssim(baseline[..., c], optimized[..., c]) for c in range(3)]
ssim_luma = ssim(rgb_to_luma(baseline), rgb_to_luma(optimized))

metrics = {
    "comparison": {
        "unit": "single generated frame from one prompt and one seed",
        "shape_hwc": list(baseline.shape),
        "mae_0_255": mae,
        "mse_0_255": mse,
        "psnr_db": psnr,
        "ssim_rgb_mean": float(np.mean(ssim_rgb)),
        "ssim_rgb_channels": ssim_rgb,
        "ssim_luma": ssim_luma,
        "pixel_abs_error_p50": float(np.percentile(np.abs(error), 50)),
        "pixel_abs_error_p95": float(np.percentile(np.abs(error), 95)),
        "pixel_abs_error_max": float(np.max(np.abs(error))),
    },
    "fp32_baseline": image_stats(baseline),
    "fp16_decode": image_stats(optimized),
}
(ROOT / "analysis-output" / "quality-metrics.json").write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
)

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11})
names = ["PSNR\n(dB)", "SSIM\n(luma)", "MAE\n/ 255"]
values = [psnr / 40.0, ssim_luma, mae / 255.0]
labels = [f"{psnr:.2f}", f"{ssim_luma:.3f}", f"{mae:.1f}"]
fig, ax = plt.subplots(figsize=(6.8, 3.8))
bars = ax.bar(names, values, color=["#0072B2", "#E69F00", "#D55E00"], edgecolor="black")
ax.set_ylabel("Normalized display scale")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.25)
for bar, label in zip(bars, labels):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, label, ha="center")
ax.text(
    0.01,
    -0.22,
    "PSNR normalized by 40 dB; MAE normalized by 255. Values above bars are raw metrics.",
    transform=ax.transAxes,
    fontsize=8,
)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "figure-02-reference-quality-metrics.pdf", bbox_inches="tight")
plt.close(fig)

yb = rgb_to_luma(baseline).mean(axis=1)
yo = rgb_to_luma(optimized).mean(axis=1)
fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.plot(yb, np.arange(len(yb)), label="FP32 baseline", color="#0072B2", linewidth=1.8)
ax.plot(yo, np.arange(len(yo)), label="FP16 decode", color="#D55E00", linewidth=1.5)
ax.invert_yaxis()
ax.set_xlabel("Mean row luminance (0–255)")
ax.set_ylabel("Image row")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "figure-03-row-luminance-profile.pdf", bbox_inches="tight")
plt.close(fig)

print(json.dumps(metrics, indent=2, ensure_ascii=False))
