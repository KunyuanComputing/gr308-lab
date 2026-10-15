# Qwen-Image-2.1 Omni-style FANT adapter

This project ports the parts of the vLLM-Omni Qwen-Image-2.1 execution model
that the GR308 software stack can currently support:

- explicit prepare, denoise, scheduler and decode phase boundaries;
- FHCCL capability probing and feature gates;
- two-rank tensor parallel attention and MLP projections for the DiT;
- 16+16 layer DiT block model parallelism with a safe host transfer fallback;
- distributed overlapping VAE tile decode across FANT logical devices;
- deterministic BF16 short-test scripts for the existing patched Diffusers model.

The measured FHCCL backend supports broadcast, all-reduce and all-gather. It
does not implement `alltoall_base`, so vLLM-Omni's Ulysses attention path is
disabled. CUDA graphs, FlashAttention, CUDA fused all-to-all and FP8 are also
disabled on this port.

The DiT block split lowers the maximum memory required on one logical device,
but its two stages execute serially for one image. The TP2 prototype instead
shards attention heads and MLP intermediate channels and reduces both row
parallel outputs with FHCCL. On the measured 1024x1024 four-step workload it
reduced DiT time from 163.33 seconds to 89.60 seconds (1.82x) and reduced the
per-rank peak to 29,495,120,896 bytes.

The installed FANT runtime exposes `HIPGraph`, `torch.fant.graph`, graph pool
handles and graphed callables. This Diffusers path remains eager because it
still constructs segments and RoPE inputs in Python, mutates dynamic KV cache
objects, and uses different first-step and cached-step execution paths. A graph
port therefore needs fixed prefill and cached-decode graphs plus static cache,
mask and RoPE buffers; it cannot reuse the text vLLM graph runner unchanged.

Operator-only profiling found two usable vendor kernels under
`torch.ops.iFormer`: RMSNorm is about 19x faster and the Q/K RoPE pair about
38.5x faster at sequence length 4096. Together they reduced a real TP2 block
from 0.67934 to 0.63190 seconds (6.98%) within the BF16 numerical gate. The
adapter is wired into `scripts/tp2_full_dit.py`; set
`ENABLE_FANT_OPERATORS=1` to enable it. These kernels have only completed
operator and single-block validation, not an end-to-end image run.

## Server layout

The scripts assume:

- work directory: `/data/infra/qwen-image-2.1-omni-fant-20260922`
- checkpoint: `/data/infra/qwen-image-2.1-probe-20260921/model`
- patched dependencies: `/data/infra/qwen-image-2.1-probe-20260921/deps`

Run `scripts/tp2_full_dit.py` with two `torchrun` ranks on devices 0 and 1 for
TP2 DiT inference, then launch `scripts/distributed_vae_decode.py` for the
distributed VAE stage. The scripts write JSON-compatible evidence to stdout
and save their output under the work directory's `logs/` and `output/` folders.

## Local verification

```bash
python3 -m pytest -q
python3 -m compileall -q src scripts tests
```
