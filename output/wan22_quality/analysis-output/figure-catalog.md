# Figure Catalog

## Figure 01 — Side-by-side output

- File: `figures/figure-01-side-by-side.png`
- Purpose: visually compare the FP32 engineering baseline and FP16-decode output.
- Data source: first and only frame from the matched 704×1280 outputs.
- Caption requirements: identify left/right precision, fixed prompt/seed, and n=1 limitation.
- Key observation: FP16 introduces horizontal banding, block discontinuities, color shift, contrast loss, and scene-structure deformation.
- Interpretation: the 68% runtime gain is not quality-preserving for this sample.
- Caveat: one frame cannot test temporal quality.

## Figure 02 — Reference-quality metrics

- File: `figures/figure-02-reference-quality-metrics.pdf`
- Purpose: summarize PSNR, SSIM, and MAE for the matched frame pair.
- Data source: extracted lossless PNG frames.
- Caption requirements: state n=1, define metric directions, and explain display normalization.
- Key observation: PSNR 14.25 dB and SSIM 0.702/0.600 indicate large structural and color disagreement.
- Interpretation: numerical evidence supports the visual rejection decision.
- Caveat: similarity metrics do not independently measure prompt faithfulness or aesthetics.

## Figure 03 — Row-luminance profile

- File: `figures/figure-03-row-luminance-profile.pdf`
- Purpose: expose spatially structured luminance deviations across image height.
- Data source: per-row mean luminance from both extracted frames.
- Caption requirements: define axes and identify FP32/FP16 curves.
- Key observation: the FP16 luminance profile departs strongly across broad vertical regions, consistent with visible horizontal layers.
- Interpretation: the error is structured and global rather than sparse pixel noise.
- Caveat: row averaging suppresses within-row spatial detail.

## Figure 04 — Absolute-difference heatmap

- File: `figures/figure-04-difference-heatmap.png`
- Purpose: localize where FP16 diverges from FP32.
- Data source: normalized grayscale absolute RGB difference with a Turbo color map.
- Caption requirements: clarify that colors are normalized within this frame and are not absolute physical units.
- Key observation: large differences cover the mountains, horizon, boat, and water rather than a small isolated region.
- Interpretation: corruption affects the entire decoded scene.
- Caveat: normalization exaggerates low-error areas and should be read alongside raw metrics.
