#!/usr/bin/env python3
"""Benchmark fused Qwen-Image-2.1 TP2 MLP gate/up projection and SiLU on FANT."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_fant  # noqa: F401
from diffusers import QwenImage21Transformer2DModel
from vllm_fant import _custom_ops as fant_ops


def sync() -> None:
    torch.fant.synchronize()


def summarize(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "p10_seconds": ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)],
        "p90_seconds": ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)],
    }


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "finite": bool(torch.isfinite(candidate).all()),
        "mae": delta.mean().item(),
        "max_abs": delta.max().item(),
        "exact_fraction": (delta == 0).float().mean().item(),
    }


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device_ids = [int(value) for value in os.environ.get("DEVICE_IDS", "0,1").split(",")]
    if len(device_ids) != world:
        raise ValueError("DEVICE_IDS must contain one physical FANT device per rank")
    device_index = device_ids[local_rank]
    sequence = int(os.environ.get("SEQ_LEN", "4096"))
    warmup = max(3, int(os.environ.get("WARMUP", "3")))
    iterations = max(10, int(os.environ.get("ITERATIONS", "10")))
    model_path = os.environ.get("MODEL_PATH", "/work/model/transformer")

    torch.fant.set_device(device_index)
    device = torch.device(f"fant:{device_index}")
    dist.init_process_group("fhccl")
    result: dict[str, object] = {
        "rank": rank,
        "device_index": device_index,
        "sequence_length": sequence,
        "warmup": warmup,
        "iterations": iterations,
        "candidates": {},
    }
    try:
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        mlp = transformer.transformer_blocks[0].img_mlp
        local_out = mlp.proj.out_features // world
        start, end = rank * local_out, (rank + 1) * local_out
        gate_weight = mlp.gate_layer.weight[start:end].detach().to(device).contiguous()
        proj_weight = mlp.proj.weight[start:end].detach().to(device).contiguous()
        fused_weight = torch.cat((gate_weight, proj_weight), dim=0).contiguous()
        hidden_size = mlp.proj.in_features
        del transformer, mlp

        generator = torch.Generator(device="cpu").manual_seed(20260922)
        hidden = torch.randn(
            (1, sequence, hidden_size), generator=generator, dtype=torch.bfloat16
        ).to(device)

        def reference_operation():
            return F.silu(F.linear(hidden, gate_weight)) * F.linear(hidden, proj_weight)

        def fused_gemm_eager_activation():
            projected = F.linear(hidden, fused_weight)
            gate, projection = projected.chunk(2, dim=-1)
            return F.silu(gate) * projection

        fused_output = torch.empty((1, sequence, local_out), dtype=hidden.dtype, device=device)
        projected_once = F.linear(hidden, fused_weight)

        def fused_gemm_iformer_activation():
            projected = F.linear(hidden, fused_weight)
            fant_ops.silu_and_mul(fused_output, projected)
            return fused_output

        def eager_activation_only():
            gate, projection = projected_once.chunk(2, dim=-1)
            return F.silu(gate) * projection

        def iformer_activation_only():
            fant_ops.silu_and_mul(fused_output, projected_once)
            return fused_output

        reference = reference_operation()
        sync()
        candidates = {
            "separate_gemms_eager_silu": reference_operation,
            "fused_gemm_eager_silu": fused_gemm_eager_activation,
            "fused_gemm_iformer_silu": fused_gemm_iformer_activation,
            "eager_silu_mul_only": eager_activation_only,
            "iformer_silu_mul_only": iformer_activation_only,
        }
        prepared = {}
        for name, operation in candidates.items():
            try:
                output = None
                for _ in range(warmup):
                    output = operation()
                sync()
                prepared[name] = (operation, compare(reference, output))
            except Exception as error:
                result["candidates"][name] = {
                    "ok": False,
                    "phase": "prepare",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }

        for name, (operation, numerical) in prepared.items():
            samples = []
            dist.barrier()
            for _ in range(iterations):
                sync()
                started = time.perf_counter()
                output = operation()
                sync()
                samples.append(time.perf_counter() - started)
            candidate = {
                "ok": numerical["finite"] and numerical["mae"] <= 0.01 and numerical["max_abs"] <= 0.25,
                "numerical": numerical,
            }
            candidate.update(summarize(samples))
            result["candidates"][name] = candidate
        result["ok"] = True
    except Exception as error:
        result.update(ok=False, error=repr(error), traceback=traceback.format_exc())
    print("MLP_BENCH " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if result.get("ok"):
        os._exit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
