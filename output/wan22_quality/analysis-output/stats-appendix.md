# Statistical and Metric Appendix

## Comparison design

- Primary comparison: FP16 VAE decode versus FP32-effective VAE decode.
- Repeated-measure unit: one matched prompt/seed/frame pair.
- Sample size: n=1 matched frame pair.
- No hypothesis test or confidence interval is reported because n=1 does not support inferential statistics.

## Full-reference metrics

| Metric | Value | Direction |
|---|---:|---|
| PSNR | 14.254 dB | Higher is more similar |
| SSIM, luma | 0.7019 | Higher is more similar |
| SSIM, RGB mean | 0.5996 | Higher is more similar |
| MAE | 38.504 / 255 | Lower is more similar |
| MSE | 2441.519 | Lower is more similar |
| Absolute error, p50 | 30 / 255 | Lower is more similar |
| Absolute error, p95 | 103 / 255 | Lower is more similar |
| Absolute error, max | 170 / 255 | Lower is more similar |

SSIM uses a Gaussian local window with sigma 1.5 and standard constants C1=(0.01L)^2 and C2=(0.03L)^2 for L=255.

## Per-image diagnostics

| Diagnostic | FP32 baseline | FP16 decode |
|---|---:|---:|
| Mean RGB | 64.44, 169.74, 218.19 | 109.56, 163.03, 179.95 |
| RGB standard deviation | 67.32, 58.56, 49.70 | 59.78, 45.62, 41.70 |
| Mean luma | 150.85 | 152.88 |
| Luma standard deviation | 56.59 | 45.83 |
| Luma entropy | 7.315 bits | 7.282 bits |
| Laplacian variance | 237.95 | 264.64 |
| Dark clipping | 11.65% | 1.09% |
| Bright clipping | 11.68% | 0.009% |

The higher FP16 Laplacian variance must not be interpreted as better detail: the side-by-side image shows that strong artificial band and block boundaries inflate this edge-sensitive metric.

## Reproducibility

The three successful FP16 output videos have the same SHA-256 digest:

`70622580c7298367f947aceafc535a1da74775435c2f4c7b9908d3c6a5fe12a9`

The FP32 baseline digest differs:

`5015fdee6e9dc90c23ed110bba44028260be5be8df3cdec8bbd93fd03fb2f008`

## Interpretation boundary

These are descriptive diagnostics for one matched sample. They support rejecting this exact FP16 candidate for the observed workload, but they do not establish a failure rate over prompts, seeds, resolutions, or frame counts.
