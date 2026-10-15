#!/usr/bin/env python3
"""End-to-end two-rank tensor parallel Qwen-Image-2.1 DiT short test."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_fant  # noqa: F401
from diffusers import QwenImage21Pipeline


_FANT_OPS = None
_ORIGINAL_ROTARY = None
_ROTARY_CACHE = None


class IFormerRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, epsilon: float):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.eps = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_3d = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])
        output_3d = torch.empty_like(input_3d)
        _FANT_OPS.rms_norm(output_3d, input_3d, self.weight, self.eps)
        return output_3d.view_as(hidden_states)


def iformer_rotary_emb_qwen(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
    global _ROTARY_CACHE
    if not use_real or use_real_unbind_dim != -1:
        return _ORIGINAL_ROTARY(x, freqs_cis, use_real, use_real_unbind_dim)
    cos, sin = freqs_cis
    cache_key = (cos.data_ptr(), sin.data_ptr(), x.device, x.dtype)
    if _ROTARY_CACHE is None or _ROTARY_CACHE[0] != cache_key:
        cos_half = cos[..., ::2].to(device=x.device, dtype=x.dtype).contiguous()
        sin_half = sin[..., ::2].to(device=x.device, dtype=x.dtype).contiguous()
        _ROTARY_CACHE = (cache_key, cos, sin, cos_half, sin_half)
    _, _, _, cos_half, sin_half = _ROTARY_CACHE
    return torch.ops.iFormer.apply_rotary_emb(
        x, cos_half, sin_half, True, False, 0, None, None
    )


def enable_fant_operator_optimizations(transformer) -> None:
    global _FANT_OPS, _ORIGINAL_ROTARY
    import iFormer._C  # noqa: F401
    from diffusers.models.transformers import transformer_qwenimage21
    from vllm_fant import _custom_ops as fant_ops

    _FANT_OPS = fant_ops
    if _ORIGINAL_ROTARY is None:
        _ORIGINAL_ROTARY = transformer_qwenimage21.apply_rotary_emb_qwen
        transformer_qwenimage21.apply_rotary_emb_qwen = iformer_rotary_emb_qwen
    for block in transformer.transformer_blocks:
        attention = block.attn
        attention.norm_q = IFormerRMSNorm(attention.norm_q.weight, attention.norm_q.eps)
        attention.norm_k = IFormerRMSNorm(attention.norm_k.weight, attention.norm_k.eps)


class RowParallelLinear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = F.linear(hidden_states, self.weight)
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output


def column_parallel(linear: nn.Linear, rank: int, world: int) -> nn.Linear:
    if linear.out_features % world:
        raise ValueError(f"out_features={linear.out_features} is not divisible by world_size={world}")
    local_out = linear.out_features // world
    start, end = rank * local_out, (rank + 1) * local_out
    replacement = nn.Linear(linear.in_features, local_out, bias=False, dtype=linear.weight.dtype)
    replacement.weight = nn.Parameter(linear.weight[start:end].detach().clone(), requires_grad=False)
    return replacement


def row_parallel(linear: nn.Linear, rank: int, world: int) -> RowParallelLinear:
    if linear.in_features % world:
        raise ValueError(f"in_features={linear.in_features} is not divisible by world_size={world}")
    local_in = linear.in_features // world
    start, end = rank * local_in, (rank + 1) * local_in
    return RowParallelLinear(linear.weight[:, start:end])


def shard_transformer(transformer, rank: int, world: int) -> dict[str, int]:
    if world != 2:
        raise ValueError("this validated prototype currently requires world_size=2")
    parameter_bytes_before = sum(p.numel() * p.element_size() for p in transformer.parameters())
    for block in transformer.transformer_blocks:
        attention = block.attn
        original_heads = attention.heads
        if original_heads % world:
            raise ValueError("attention head count must be divisible by world size")
        attention.to_q = column_parallel(attention.to_q, rank, world)
        attention.to_k = column_parallel(attention.to_k, rank, world)
        attention.to_v = column_parallel(attention.to_v, rank, world)
        attention.to_out[0] = row_parallel(attention.to_out[0], rank, world)
        attention.heads = original_heads // world
        attention.inner_dim //= world

        mlp = block.img_mlp
        mlp.gate_layer = column_parallel(mlp.gate_layer, rank, world)
        mlp.proj = column_parallel(mlp.proj, rank, world)
        mlp.out = row_parallel(mlp.out, rank, world)
    parameter_bytes_after = sum(p.numel() * p.element_size() for p in transformer.parameters())
    return {
        "transformer_parameter_bytes_before": parameter_bytes_before,
        "transformer_parameter_bytes_after": parameter_bytes_after,
    }


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device_ids = [int(value) for value in os.environ.get("DEVICE_IDS", "0,1").split(",")]
    if len(device_ids) != world:
        raise ValueError("DEVICE_IDS must contain one physical FANT device per rank")
    device_index = device_ids[local_rank]
    enable_fant_operators = os.environ.get("ENABLE_FANT_OPERATORS", "0") == "1"
    steps = int(os.environ.get("STEPS", "2"))
    width = int(os.environ.get("WIDTH", "512"))
    height = int(os.environ.get("HEIGHT", "512"))
    base = Path(os.environ.get("WORKDIR", "/work"))
    torch.fant.set_device(device_index)
    device = torch.device(f"fant:{device_index}")
    dist.init_process_group("fhccl")
    metrics = {
        "rank": rank,
        "device_index": device_index,
        "world_size": world,
        "mode": "tp2_attention_heads_and_mlp",
        "steps": steps,
        "width": width,
        "height": height,
        "fant_operator_optimizations": enable_fant_operators,
    }
    started = time.perf_counter()
    try:
        load_started = time.perf_counter()
        pipe = QwenImage21Pipeline.from_pretrained(
            str(base / "model"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        metrics["load_seconds"] = time.perf_counter() - load_started
        if enable_fant_operators:
            enable_fant_operator_optimizations(pipe.transformer)
        metrics.update(shard_transformer(pipe.transformer, rank, world))
        pipe.text_encoder.to(device)
        pipe.transformer.to(device)
        torch.fant.synchronize()
        metrics["memory_after_placement_bytes"] = torch.fant.memory_allocated(device_index)
        torch.fant.reset_peak_memory_stats(device_index)

        prompt = (
            "A clean product photograph of a red ceramic teapot on a light wooden table, soft studio lighting, "
            "with the Chinese text ‘风华三号’ clearly printed on the teapot."
        )
        encode_started = time.perf_counter()
        prompt_embeds, prompt_embeds_mask, _ = pipe.encode_prompt(prompt=prompt, image=None, device=device)
        torch.fant.synchronize()
        metrics["prompt_encode_seconds"] = time.perf_counter() - encode_started
        dist.barrier()

        denoise_started = time.perf_counter()
        step_events = []

        def on_step_end(_pipe, index, timestep, callback_kwargs):
            torch.fant.synchronize()
            step_events.append(
                {
                    "step": index,
                    "timestep": float(timestep.float().cpu()),
                    "elapsed_seconds": time.perf_counter() - denoise_started,
                }
            )
            return callback_kwargs

        latents = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(42),
            output_type="latent",
            callback_on_step_end=on_step_end,
        ).images
        torch.fant.synchronize()
        metrics["denoise_seconds"] = time.perf_counter() - denoise_started
        metrics["step_events"] = step_events
        metrics["peak_memory_bytes"] = torch.fant.max_memory_allocated(device_index)

        gathered = [torch.empty_like(latents) for _ in range(world)]
        dist.all_gather(gathered, latents)
        if rank == 0:
            cross_rank_difference = (gathered[0].float() - gathered[1].float()).abs()
            metrics["cross_rank_mae"] = cross_rank_difference.mean().item()
            metrics["cross_rank_max_abs"] = cross_rank_difference.max().item()
            output = base / "output" / f"qwen-image-2.1-bfloat16-{width}x{height}-{steps}steps-dit-tp2.pt"
            torch.save(latents.cpu(), output)
            metrics["output"] = str(output)
        metrics["ok"] = True
    except Exception as error:
        metrics.update(ok=False, error=repr(error), traceback=traceback.format_exc())
    finally:
        metrics["total_seconds"] = time.perf_counter() - started
        print("TP2_FULL " + json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if metrics.get("ok"):
            os._exit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
