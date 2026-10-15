#!/usr/bin/env python3
"""Profile one real TP2 Qwen-Image-2.1 block by synchronized stage."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_fant  # noqa: F401
from diffusers import QwenImage21Transformer2DModel

from tp2_block_probe import TP2Block, select_modulation


def synchronize() -> None:
    torch.fant.synchronize()


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    sequence = int(os.environ.get("SEQ_LEN", "4096"))
    iterations = int(os.environ.get("ITERATIONS", "3"))
    model_path = os.environ.get("MODEL_PATH", "/work/model/transformer")
    torch.fant.set_device(local_rank)
    device = torch.device(f"fant:{local_rank}")
    dist.init_process_group("fhccl")

    transformer = QwenImage21Transformer2DModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    block = transformer.transformer_blocks[0]
    transformer.transformer_blocks = torch.nn.ModuleList()
    del transformer
    tp = TP2Block(block, rank, device)
    del block

    generator = torch.Generator(device="cpu").manual_seed(20260922)
    hidden = torch.randn((1, sequence, 4096), generator=generator, dtype=torch.bfloat16).to(device)
    modulation = (
        torch.randn((1, 4 * 4096), generator=generator, dtype=torch.bfloat16) * 0.05
    ).to(device)

    totals: dict[str, float] = {}

    def measured(name, operation):
        synchronize()
        started = time.perf_counter()
        output = operation()
        synchronize()
        totals[name] = totals.get(name, 0.0) + time.perf_counter() - started
        return output

    def iteration() -> torch.Tensor:
        mod1, mod2 = measured("modulation_split", lambda: modulation.chunk(2, dim=-1))
        scale1, gate1 = measured("attention_modulation", lambda: select_modulation(mod1))
        normalized = measured(
            "attention_layer_norm",
            lambda: F.layer_norm(hidden, (tp.dim,), eps=tp.norm1_eps) * (1 + scale1),
        )
        query = measured("q_gemm", lambda: F.linear(normalized, tp.wq))
        key = measured("k_gemm", lambda: F.linear(normalized, tp.wk))
        value = measured("v_gemm", lambda: F.linear(normalized, tp.wv))
        query = measured(
            "q_rms_norm",
            lambda: F.rms_norm(
                query.unflatten(-1, (tp.local_heads, tp.head_dim)),
                (tp.head_dim,), tp.norm_q_weight, tp.norm_eps,
            ).to(value.dtype),
        )
        key = measured(
            "k_rms_norm",
            lambda: F.rms_norm(
                key.unflatten(-1, (tp.local_heads, tp.head_dim)),
                (tp.head_dim,), tp.norm_k_weight, tp.norm_eps,
            ).to(value.dtype),
        )
        value = value.unflatten(-1, (tp.local_heads, tp.head_dim))
        attended = measured(
            "sdpa",
            lambda: F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0
            ),
        )
        attended = measured(
            "attention_relayout",
            lambda: attended.transpose(1, 2).contiguous().flatten(2, 3).type_as(query),
        )
        attention_output = measured("attention_out_gemm", lambda: F.linear(attended, tp.wo))
        measured("attention_all_reduce", lambda: dist.all_reduce(attention_output))
        hidden_after_attention = measured(
            "attention_residual", lambda: hidden + gate1.tanh() * attention_output
        )

        scale2, gate2 = measured("mlp_modulation", lambda: select_modulation(mod2))
        normalized2 = measured(
            "mlp_layer_norm",
            lambda: F.layer_norm(hidden_after_attention, (tp.dim,), eps=tp.norm2_eps) * (1 + scale2),
        )
        gate = measured("mlp_gate_gemm", lambda: F.linear(normalized2, tp.w_gate))
        projection = measured("mlp_proj_gemm", lambda: F.linear(normalized2, tp.w_proj))
        mlp_local = measured("silu_multiply", lambda: F.silu(gate) * projection)
        mlp_output = measured("mlp_out_gemm", lambda: F.linear(mlp_local, tp.w_mlp_out))
        measured("mlp_all_reduce", lambda: dist.all_reduce(mlp_output))
        return measured("mlp_residual", lambda: hidden_after_attention + gate2.tanh() * mlp_output)

    with torch.no_grad():
        iteration()
        totals.clear()
        dist.barrier()
        wall_started = time.perf_counter()
        output = None
        for _ in range(iterations):
            output = iteration()
        synchronize()
        dist.barrier()
        wall_seconds = (time.perf_counter() - wall_started) / iterations

    averages = {name: seconds / iterations for name, seconds in totals.items()}
    categories = {
        "qkv_and_attention_out_gemm": sum(
            averages[name] for name in ("q_gemm", "k_gemm", "v_gemm", "attention_out_gemm")
        ),
        "mlp_gemm": sum(
            averages[name] for name in ("mlp_gate_gemm", "mlp_proj_gemm", "mlp_out_gemm")
        ),
        "sdpa": averages["sdpa"],
        "all_reduce": averages["attention_all_reduce"] + averages["mlp_all_reduce"],
    }
    categories["other"] = sum(averages.values()) - sum(categories.values())
    result = {
        "rank": rank,
        "sequence_length": sequence,
        "iterations": iterations,
        "wall_seconds_per_block": wall_seconds,
        "sum_synchronized_stage_seconds": sum(averages.values()),
        "categories_seconds": categories,
        "categories_percent": {
            name: 100 * seconds / sum(averages.values()) for name, seconds in categories.items()
        },
        "stages_seconds": averages,
        "finite": bool(torch.isfinite(output).all()),
    }
    print("TP2_PROFILE " + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
