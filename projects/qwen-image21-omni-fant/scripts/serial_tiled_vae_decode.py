#!/usr/bin/env python3
"""Reference serial tiled decode used to validate the distributed stitch."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch_fant  # noqa: F401
from diffusers import AutoencoderKLQwenImage21
from PIL import Image


base = Path(os.environ.get("WORKDIR", "/work"))
width = int(os.environ.get("WIDTH", "1024"))
height = int(os.environ.get("HEIGHT", "1024"))
steps = int(os.environ.get("STEPS", "4"))
latent_path = base / "output" / f"qwen-image-2.1-bfloat16-{width}x{height}-{steps}steps-latents.pt"
output_path = base / "output" / f"qwen-image-2.1-bfloat16-{width}x{height}-{steps}steps-serial-tiled.png"
started = time.perf_counter()
vae = AutoencoderKLQwenImage21.from_pretrained(
    str(base / "model" / "vae"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
).to("fant:0")
vae.enable_tiling()
packed = torch.load(latent_path, map_location="cpu").to("fant:0")
batch, _, channels = packed.shape
latents = packed.transpose(1, 2).reshape(batch, channels, 1, height // 16, width // 16).to(vae.dtype)
mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
latents = latents * std + mean
torch.fant.reset_peak_memory_stats(0)
decode_started = time.perf_counter()
decoded = vae.decode(latents, return_dict=False)[0][:, :, 0]
torch.fant.synchronize()
decode_seconds = time.perf_counter() - decode_started
array = ((decoded[0].float().cpu().permute(1, 2, 0) / 2 + 0.5).clamp(0, 1) * 255).round().byte().numpy()
Image.fromarray(array).save(output_path)
print(
    json.dumps(
        {
            "ok": True,
            "decode_seconds": decode_seconds,
            "total_seconds": time.perf_counter() - started,
            "peak_memory_bytes": torch.fant.max_memory_allocated(0),
            "output": str(output_path),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        },
        sort_keys=True,
    ),
    flush=True,
)
