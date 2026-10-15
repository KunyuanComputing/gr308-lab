#!/usr/bin/env python3
"""Compare Wan2.2 output frames against a fixed FP32 engineering reference."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def luma(image: np.ndarray) -> np.ndarray:
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mean_a = gaussian_filter(a, sigma=1.5)
    mean_b = gaussian_filter(b, sigma=1.5)
    var_a = gaussian_filter(a * a, sigma=1.5) - mean_a * mean_a
    var_b = gaussian_filter(b * b, sigma=1.5) - mean_b * mean_b
    covariance = gaussian_filter(a * b, sigma=1.5) - mean_a * mean_b
    score = ((2 * mean_a * mean_b + c1) * (2 * covariance + c2)) / (
        (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2)
    )
    return float(np.mean(score))


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")
    absolute_error = np.abs(reference - candidate)
    mse = float(np.mean((reference - candidate) ** 2))
    return {
        "psnr_db": float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse),
        "ssim_luma": ssim(luma(reference), luma(candidate)),
        "ssim_rgb_mean": float(
            np.mean([ssim(reference[..., channel], candidate[..., channel]) for channel in range(3)])
        ),
        "mae_0_255": float(np.mean(absolute_error)),
        "pixel_abs_error_p95": float(np.percentile(absolute_error, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load_rgb(args.reference)
    results = {
        path.stem: compare(reference, load_rgb(path)) for path in args.candidates
    }
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
