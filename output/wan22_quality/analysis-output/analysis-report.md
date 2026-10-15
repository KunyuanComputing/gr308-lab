# Wan2.2 FP16 VAE Decode Quality Analysis

## Analysis question

Does changing Wan2.2 VAE decode precision from the FP32-effective baseline to FP16 preserve generated-image quality under the fixed prompt, seed, resolution, and checkpoint used in the performance benchmark?

## Evidence inventory

- FP32-effective TP2 baseline: one generated frame, 704×1280.
- FP16-decode TP2 candidate: one generated frame, 704×1280.
- Fixed checkpoint, prompt, negative prompt, seed 42, resolution, and denoise-step count.
- Three successful FP16 runs produced byte-identical MP4 files.
- Unit of analysis: one prompt × one seed × one generated frame.

## Key findings

1. The FP16 result is not quality-equivalent to the FP32 baseline. It contains severe horizontal bands, tiled/block-like discontinuities, strong color shifts, reduced contrast, and loss of coherent scene structure.
2. Full-reference metrics agree with visual inspection: PSNR is 14.25 dB, luma SSIM is 0.702, mean RGB SSIM is 0.600, and mean absolute pixel error is 38.50/255.
3. The 95th-percentile absolute pixel error is 103/255, so the mismatch is spatially extensive rather than limited to a few outliers.
4. The FP16 frame shifts the channel means by approximately +45.1 red, −6.7 green, and −38.2 blue levels relative to FP32, consistent with the visible color cast.
5. Luma standard deviation falls from 56.59 to 45.83 (−19.0%), indicating contrast compression. Apparent edge energy increases, but visual inspection shows that this comes from artificial band/block boundaries rather than recovered detail.

## Decision

Reject global `--vae-decode-precision fp16` as a quality-preserving production configuration, despite its approximately 68% E2E speedup. Keep the FP32-effective decode as the valid quality baseline.

The speed result remains useful as an upper bound: it shows that lower-precision VAE computation can be much faster on GR308. The next experiment should use selective mixed precision—retain normalization, residual accumulation, upsampling, cache/padding, and final output layers in FP32 while testing FP16 only for eligible convolution blocks.

## Claim Candidates

- Claim:
  - Source evidence: fixed-prompt frame comparison, PSNR 14.25 dB, luma SSIM 0.702, RGB SSIM 0.600, and side-by-side inspection.
  - Allowed wording: “Global FP16 VAE decoding substantially accelerates this benchmark but causes severe deterministic visual corruption for the evaluated sample.”
  - Forbidden stronger wording: “FP16 VAE decoding always fails for Wan2.2 on GR308.”
  - Uncertainty: only one prompt, one seed, and one frame were evaluated; temporal quality was not tested.
  - Next check: layer-wise mixed-precision ablation across multiple prompts and a multi-frame clip.
  - Decision: keep.

- Claim:
  - Source evidence: three successful FP16 configurations produced byte-identical MP4 hashes.
  - Allowed wording: “The observed corruption is deterministic across the three repeated FP16 outputs in this test.”
  - Forbidden stronger wording: “The corruption mechanism is fully identified.”
  - Uncertainty: deterministic output does not identify which VAE layer first introduces the error.
  - Next check: capture intermediate activations and bisect precision by decoder block.
  - Decision: keep.

## Limitations

- There is only one comparison sample; inferential statistics and population-level quality claims are invalid.
- The output contains one frame, so flicker, motion coherence, temporal consistency, and video-level perceptual metrics cannot be evaluated.
- PSNR and SSIM measure similarity to the FP32 output, not semantic fidelity to an external ground-truth image.
- The FP32 baseline itself is model-generated and is used as the engineering reference, not as natural-image ground truth.
