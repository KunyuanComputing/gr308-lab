#!/usr/bin/env python3
"""Two-rank tensor-parallel probe for one Qwen-Image-2.1 transformer block.

The prototype shards attention heads and SwiGLU intermediate channels. Row
parallel outputs are summed with FHCCL all-reduce. It deliberately tests one
real checkpoint block before changing the 32-layer model.
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_fant  # noqa: F401
from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen


def sync() -> None:
    torch.fant.synchronize()


def select_modulation(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale, gate = params.chunk(2, dim=-1)
    return scale.unsqueeze(1), gate.unsqueeze(1)


class TP2Block:
    def __init__(
        self,
        block,
        rank: int,
        device: torch.device,
        rms_norm_mode: str = "eager",
        rotary_mode: str = "none",
    ):
        self.rank = rank
        self.device = device
        self.rms_norm_mode = rms_norm_mode
        self.rotary_mode = rotary_mode
        self.fant_ops = None
        if rms_norm_mode == "vllm_fant":
            from vllm_fant import _custom_ops as fant_ops

            self.fant_ops = fant_ops
        elif rms_norm_mode != "eager":
            raise ValueError(f"unsupported RMS_NORM_MODE={rms_norm_mode}")
        if rotary_mode == "iformer":
            import iFormer._C  # noqa: F401
        elif rotary_mode not in {"none", "eager"}:
            raise ValueError(f"unsupported ROTARY_MODE={rotary_mode}")
        self.dim = block.attn.inner_dim
        self.heads = block.attn.heads
        self.head_dim = self.dim // self.heads
        self.local_heads = self.heads // 2
        self.head_start = rank * self.local_heads * self.head_dim
        self.head_end = (rank + 1) * self.local_heads * self.head_dim
        mlp_dim = block.img_mlp.proj.out_features
        self.mlp_start = rank * (mlp_dim // 2)
        self.mlp_end = (rank + 1) * (mlp_dim // 2)

        def shard_rows(weight, start, end):
            return weight[start:end].detach().to(device).contiguous()

        def shard_cols(weight, start, end):
            return weight[:, start:end].detach().to(device).contiguous()

        self.wq = shard_rows(block.attn.to_q.weight, self.head_start, self.head_end)
        self.wk = shard_rows(block.attn.to_k.weight, self.head_start, self.head_end)
        self.wv = shard_rows(block.attn.to_v.weight, self.head_start, self.head_end)
        self.wo = shard_cols(block.attn.to_out[0].weight, self.head_start, self.head_end)
        self.norm_q_weight = block.attn.norm_q.weight.detach().to(device).contiguous()
        self.norm_k_weight = block.attn.norm_k.weight.detach().to(device).contiguous()
        self.norm_eps = block.attn.norm_q.eps
        self.w_gate = shard_rows(block.img_mlp.gate_layer.weight, self.mlp_start, self.mlp_end)
        self.w_proj = shard_rows(block.img_mlp.proj.weight, self.mlp_start, self.mlp_end)
        self.w_mlp_out = shard_cols(block.img_mlp.out.weight, self.mlp_start, self.mlp_end)
        self.norm1_eps = block.img_norm1.eps
        self.norm2_eps = block.img_norm2.eps

    def rms_norm(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.rms_norm_mode == "eager":
            return F.rms_norm(value, (self.head_dim,), weight, self.norm_eps).to(value.dtype)
        input_3d = value.reshape(-1, self.local_heads, self.head_dim)
        output_3d = torch.empty_like(input_3d)
        self.fant_ops.rms_norm(output_3d, input_3d, weight, self.norm_eps)
        return output_3d.view_as(value)

    def apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rotary_emb is None or self.rotary_mode == "none":
            return query, key
        cos, sin = rotary_emb
        if self.rotary_mode == "eager":
            return (
                apply_rotary_emb_qwen(query, (cos, sin), use_real=True),
                apply_rotary_emb_qwen(key, (cos, sin), use_real=True),
            )
        query = torch.ops.iFormer.apply_rotary_emb(
            query, cos, sin, True, False, 0, None, None
        )
        key = torch.ops.iFormer.apply_rotary_emb(
            key, cos, sin, True, False, 0, None, None
        )
        return query, key

    def __call__(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)
        scale1, gate1 = select_modulation(mod1)
        normalized = F.layer_norm(hidden_states, (self.dim,), eps=self.norm1_eps)
        normalized = normalized * (1 + scale1)

        query = F.linear(normalized, self.wq).unflatten(-1, (self.local_heads, self.head_dim))
        key = F.linear(normalized, self.wk).unflatten(-1, (self.local_heads, self.head_dim))
        value = F.linear(normalized, self.wv).unflatten(-1, (self.local_heads, self.head_dim))
        query = self.rms_norm(query, self.norm_q_weight)
        key = self.rms_norm(key, self.norm_k_weight)
        query, key = self.apply_rotary(query, key, rotary_emb)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0
        )
        attended = attended.transpose(1, 2).contiguous().flatten(2, 3).type_as(query)
        attention_output = F.linear(attended, self.wo)
        dist.all_reduce(attention_output, op=dist.ReduceOp.SUM)
        self.last_attention_output = attention_output.detach().clone()
        hidden_states = hidden_states + gate1.tanh() * attention_output
        self.last_hidden_after_attention = hidden_states.detach().clone()

        scale2, gate2 = select_modulation(mod2)
        normalized2 = F.layer_norm(hidden_states, (self.dim,), eps=self.norm2_eps)
        normalized2 = normalized2 * (1 + scale2)
        mlp_local = F.silu(F.linear(normalized2, self.w_gate)) * F.linear(normalized2, self.w_proj)
        mlp_output = F.linear(mlp_local, self.w_mlp_out)
        dist.all_reduce(mlp_output, op=dist.ReduceOp.SUM)
        self.last_mlp_output = mlp_output.detach().clone()
        return hidden_states + gate2.tanh() * mlp_output


def reference_stages(block, hidden_states: torch.Tensor, modulation: torch.Tensor, rotary_emb=None):
    mod1, mod2 = modulation.chunk(2, dim=-1)
    img_modulated, img_gate1 = block._modulate(block.img_norm1(hidden_states), mod1, None)
    attention_output = block.attn(hidden_states=img_modulated, rotary_emb=rotary_emb, segments=None)
    hidden_after_attention = hidden_states + img_gate1.tanh() * attention_output
    img_modulated2, _ = block._modulate(block.img_norm2(hidden_after_attention), mod2, None)
    mlp_output = block.img_mlp(img_modulated2)
    return attention_output, hidden_after_attention, mlp_output


def difference_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    difference = (reference.float() - candidate.float()).abs()
    return {
        "mae": difference.mean().item(),
        "max_abs": difference.max().item(),
        "reference_abs_max": reference.float().abs().max().item(),
        "candidate_abs_max": candidate.float().abs().max().item(),
        "reference_finite": bool(torch.isfinite(reference).all()),
        "candidate_finite": bool(torch.isfinite(candidate).all()),
    }


def timed(operation, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    sync()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    sync()
    dist.barrier()
    return (time.perf_counter() - start) / iterations


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device_ids = [int(value) for value in os.environ.get("DEVICE_IDS", "0,1").split(",")]
    if len(device_ids) != world:
        raise ValueError("DEVICE_IDS must contain one physical FANT device per rank")
    device_index = device_ids[local_rank]
    sequence = int(os.environ.get("SEQ_LEN", "1024"))
    iterations = int(os.environ.get("ITERATIONS", "3"))
    rms_norm_mode = os.environ.get("RMS_NORM_MODE", "eager")
    rotary_mode = os.environ.get("ROTARY_MODE", "none")
    model_path = os.environ.get("MODEL_PATH", "/work/model/transformer")
    torch.fant.set_device(device_index)
    device = torch.device(f"fant:{device_index}")
    dist.init_process_group("fhccl")

    transformer = QwenImage21Transformer2DModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    block = transformer.transformer_blocks[0]
    transformer.transformer_blocks = torch.nn.ModuleList()
    del transformer

    tp_block = TP2Block(
        block,
        rank,
        device,
        rms_norm_mode=rms_norm_mode,
        rotary_mode=rotary_mode,
    )
    eager_comparison_block = None
    if os.environ.get("COMPARE_EAGER", "0") == "1" and (
        rms_norm_mode != "eager" or rotary_mode not in {"none", "eager"}
    ):
        eager_comparison_block = TP2Block(
            block,
            rank,
            device,
            rms_norm_mode="eager",
            rotary_mode="eager" if rotary_mode != "none" else "none",
        )
    reference = block.to(device) if rank == 0 else None

    generator = torch.Generator(device="cpu").manual_seed(20260922)
    hidden_cpu = torch.randn((1, sequence, 4096), generator=generator, dtype=torch.bfloat16)
    modulation_cpu = torch.randn((1, 4 * 4096), generator=generator, dtype=torch.bfloat16) * 0.05
    hidden = hidden_cpu.to(device)
    modulation = modulation_cpu.to(device)
    reference_rotary = None
    candidate_rotary = None
    if rotary_mode != "none":
        angles = torch.randn(
            (sequence, tp_block.head_dim // 2), generator=generator, dtype=torch.float32
        )
        cos_half = angles.cos().to(torch.bfloat16).to(device)
        sin_half = angles.sin().to(torch.bfloat16).to(device)
        reference_rotary = (
            cos_half.repeat_interleave(2, dim=-1),
            sin_half.repeat_interleave(2, dim=-1),
        )
        candidate_rotary = (
            reference_rotary if rotary_mode == "eager" else (cos_half, sin_half)
        )

    reference_output = None
    reference_seconds = None
    if rank == 0:
        reference_output = reference(hidden, modulation, rotary_emb=reference_rotary, segments=None)
        sync()
        start = time.perf_counter()
        for _ in range(iterations):
            reference_output = reference(hidden, modulation, rotary_emb=reference_rotary, segments=None)
        sync()
        reference_seconds = (time.perf_counter() - start) / iterations
    dist.barrier()

    tp_output = tp_block(hidden, modulation, candidate_rotary)
    eager_tp_output = None
    if eager_comparison_block is not None:
        eager_tp_output = eager_comparison_block(hidden, modulation, reference_rotary)
        sync()
    tp_seconds = timed(
        lambda: tp_block(hidden, modulation, candidate_rotary), warmup=1, iterations=iterations
    )

    result = {
        "rank": rank,
        "device_index": device_index,
        "rms_norm_mode": rms_norm_mode,
        "rotary_mode": rotary_mode,
        "sequence_length": sequence,
        "iterations": iterations,
        "tp2_seconds_per_block": tp_seconds,
        "peak_memory_bytes": torch.fant.max_memory_allocated(local_rank),
    }
    if rank == 0:
        difference = (reference_output.float() - tp_output.float()).abs()
        ref_attention, ref_hidden_after_attention, ref_mlp = reference_stages(
            reference, hidden, modulation, reference_rotary
        )
        result.update(
            reference_seconds_per_block=reference_seconds,
            speedup=reference_seconds / tp_seconds,
            mae=difference.mean().item(),
            max_abs=difference.max().item(),
            exact_percent=100 * (difference == 0).float().mean().item(),
            reference_finite=bool(torch.isfinite(reference_output).all()),
            tp2_finite=bool(torch.isfinite(tp_output).all()),
            attention=difference_summary(ref_attention, tp_block.last_attention_output),
            hidden_after_attention=difference_summary(
                ref_hidden_after_attention, tp_block.last_hidden_after_attention
            ),
            mlp=difference_summary(ref_mlp, tp_block.last_mlp_output),
        )
        if eager_tp_output is not None:
            result["optimized_vs_eager_tp2"] = difference_summary(eager_tp_output, tp_output)
    print("TP2_BLOCK " + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
