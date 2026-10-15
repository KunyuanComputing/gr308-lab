#!/usr/bin/env python3
"""Two-rank Qwen-Image-2.1 VAE tiled decode over FHCCL all-reduce."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch_fant  # noqa: F401
from diffusers import AutoencoderKLQwenImage21
from PIL import Image


def unpatchify_cpu(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return tensor
    batch, packed_channels, frames, height, width = tensor.shape
    channels = packed_channels // (patch_size * patch_size)
    tensor = tensor.view(batch, channels, patch_size, patch_size, frames, height, width)
    tensor = tensor.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return tensor.view(batch, channels, frames, height * patch_size, width * patch_size)


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    base = Path(os.environ.get("WORKDIR", "/work"))
    width = int(os.environ.get("WIDTH", "512"))
    height = int(os.environ.get("HEIGHT", "512"))
    steps = int(os.environ.get("STEPS", "2"))
    latent_path = base / "output" / f"qwen-image-2.1-bfloat16-{width}x{height}-{steps}steps-latents.pt"
    output_path = base / "output" / f"qwen-image-2.1-bfloat16-{width}x{height}-{steps}steps-omni-fant-tp2.png"
    metrics = {"rank": rank, "world_size": world, "latent": str(latent_path), "output": str(output_path)}
    started = time.perf_counter()
    try:
        torch.fant.set_device(local_rank)
        device = torch.device(f"fant:{local_rank}")
        dist.init_process_group("fhccl")
        load_started = time.perf_counter()
        vae = AutoencoderKLQwenImage21.from_pretrained(
            str(base / "model" / "vae"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
        metrics["load_seconds"] = time.perf_counter() - load_started

        packed = torch.load(latent_path, map_location="cpu").to(device)
        batch, _, channels = packed.shape
        latent_height = height // 16
        latent_width = width // 16
        latents = packed.transpose(1, 2).reshape(batch, channels, 1, latent_height, latent_width).to(vae.dtype)
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
        std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
        latents = latents * std + mean

        tile_h = vae.tile_sample_min_height // vae.spatial_compression_ratio
        tile_w = vae.tile_sample_min_width // vae.spatial_compression_ratio
        stride_h = vae.tile_sample_stride_height // vae.spatial_compression_ratio
        stride_w = vae.tile_sample_stride_width // vae.spatial_compression_ratio
        positions = [(y, x) for y in range(0, latent_height, stride_h) for x in range(0, latent_width, stride_w)]
        patch = int(vae.config.patch_size or 1)
        max_dec_h = vae.tile_sample_min_height // patch
        max_dec_w = vae.tile_sample_min_width // patch
        packed_out_channels = int(vae.config.out_channels) * patch * patch
        tile_buffer = torch.zeros(
            (len(positions), batch, packed_out_channels, 1, max_dec_h, max_dec_w), dtype=vae.dtype, device=device
        )
        shapes = torch.zeros((len(positions), 2), dtype=torch.int64, device=device)
        torch.fant.reset_peak_memory_stats(local_rank)
        decode_started = time.perf_counter()
        local_tiles = 0
        for tile_id, (y, x) in enumerate(positions):
            if (tile_id + 1) % world != rank:
                continue
            vae.clear_cache()
            vae._conv_idx = [0]
            tile = latents[:, :, :, y : y + tile_h, x : x + tile_w]
            tile = vae.post_quant_conv(tile)
            decoded = vae.decoder(tile, feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=True)
            decoded_h, decoded_w = decoded.shape[-2:]
            tile_buffer[tile_id, :, :, :, :decoded_h, :decoded_w].copy_(decoded)
            shapes[tile_id] = torch.tensor([decoded_h, decoded_w], dtype=torch.int64, device=device)
            local_tiles += 1
        vae.clear_cache()
        torch.fant.synchronize()
        metrics["local_decode_seconds"] = time.perf_counter() - decode_started
        metrics["local_tiles"] = local_tiles
        metrics["peak_memory_bytes"] = torch.fant.max_memory_allocated(local_rank)

        collective_started = time.perf_counter()
        dist.all_reduce(tile_buffer, op=dist.ReduceOp.SUM)
        dist.all_reduce(shapes, op=dist.ReduceOp.SUM)
        torch.fant.synchronize()
        metrics["collective_seconds"] = time.perf_counter() - collective_started

        if rank == 0:
            tiles = []
            for tile_id in range(len(positions)):
                decoded_h, decoded_w = [int(v) for v in shapes[tile_id].cpu().tolist()]
                tiles.append(tile_buffer[tile_id, :, :, :, :decoded_h, :decoded_w].cpu())
            columns = len(range(0, latent_width, stride_w))
            rows = [tiles[index : index + columns] for index in range(0, len(tiles), columns)]
            blend_h = vae.tile_sample_min_height // patch - vae.tile_sample_stride_height // patch
            blend_w = vae.tile_sample_min_width // patch - vae.tile_sample_stride_width // patch
            sample_stride_h = vae.tile_sample_stride_height // patch
            sample_stride_w = vae.tile_sample_stride_width // patch
            result_rows = []
            for row_index, row in enumerate(rows):
                result_row = []
                for col_index, tile in enumerate(row):
                    if row_index > 0:
                        tile = vae.blend_v(rows[row_index - 1][col_index], tile, blend_h)
                    if col_index > 0:
                        tile = vae.blend_h(row[col_index - 1], tile, blend_w)
                    result_row.append(tile[..., :sample_stride_h, :sample_stride_w])
                result_rows.append(torch.cat(result_row, dim=-1))
            decoded = torch.cat(result_rows, dim=3)[..., : height // patch, : width // patch]
            decoded = unpatchify_cpu(decoded, patch).clamp(-1, 1)[:, :, 0]
            array = ((decoded[0].float().permute(1, 2, 0) / 2 + 0.5).clamp(0, 1) * 255).round().byte().numpy()
            Image.fromarray(array).save(output_path)
            metrics["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
            metrics["stitch_seconds"] = time.perf_counter() - collective_started - metrics["collective_seconds"]
        metrics["ok"] = True
        metrics["total_seconds"] = time.perf_counter() - started
        print("VAE_DECODE " + json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
        dist.barrier()
        dist.destroy_process_group()
    except Exception as error:
        metrics.update(ok=False, error=repr(error), traceback=traceback.format_exc())
        print("VAE_DECODE " + json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
        raise
    finally:
        # Avoid a known torch_fant allocator destructor assertion after FHCCL.
        if metrics.get("ok"):
            os._exit(0)


if __name__ == "__main__":
    main()
