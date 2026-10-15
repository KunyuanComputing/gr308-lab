#!/usr/bin/env python3
"""Benchmark Qwen-Image-2.1 RMSNorm and fused elementwise candidates on FANT."""

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


def sync() -> None:
    torch.fant.synchronize()


def summary(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "p10_seconds": ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)],
        "p90_seconds": ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)],
    }


def numerical(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
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
        block = transformer.transformer_blocks[0]
        weight = block.attn.norm_q.weight.detach().to(device).contiguous()
        epsilon = float(block.attn.norm_q.eps)
        hidden_size = block.attn.inner_dim
        layer_norm_epsilon = float(block.img_norm1.eps)
        head_dim = block.attn.inner_dim // block.attn.heads
        local_heads = block.attn.heads // world
        del transformer, block

        generator = torch.Generator(device="cpu").manual_seed(20260922)
        input_tensor = torch.randn(
            (1, sequence, local_heads, head_dim), generator=generator, dtype=torch.bfloat16
        ).to(device)
        reference = F.rms_norm(input_tensor, (head_dim,), weight, epsilon)
        sync()

        candidates = {
            "eager_rms_norm": lambda: F.rms_norm(input_tensor, (head_dim,), weight, epsilon),
            "explicit_fp32": lambda: (
                input_tensor.float()
                * torch.rsqrt(input_tensor.float().square().mean(dim=-1, keepdim=True) + epsilon)
                * weight.float()
            ).to(input_tensor.dtype),
            "explicit_bf16": lambda: (
                input_tensor
                * torch.rsqrt(input_tensor.square().mean(dim=-1, keepdim=True) + epsilon)
                * weight
            ),
        }

        try:
            from vllm_fant import _custom_ops as ops

            vllm_input = input_tensor.reshape(-1, local_heads, head_dim)
            vllm_output = torch.empty_like(vllm_input)

            def vllm_rms_norm():
                ops.rms_norm(vllm_output, vllm_input, weight, epsilon)
                return vllm_output.view_as(input_tensor)

            candidates["vllm_fant_rms_norm"] = vllm_rms_norm
        except Exception as error:
            result["vllm_import_error"] = repr(error)

        try:
            compiled = torch.compile(
                lambda value: F.rms_norm(value, (head_dim,), weight, epsilon),
                dynamic=False,
                fullgraph=True,
            )
            candidates["compiled_rms_norm"] = lambda: compiled(input_tensor)
        except Exception as error:
            result["compile_setup_error"] = repr(error)

        prepared = {}
        for name, operation in candidates.items():
            try:
                output = None
                for _ in range(warmup):
                    output = operation()
                sync()
                check = numerical(reference, output)
                prepared[name] = (operation, check)
            except Exception as error:
                result["candidates"][name] = {
                    "ok": False,
                    "phase": "prepare",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }

        for name, (operation, check) in prepared.items():
            samples = []
            dist.barrier()
            for _ in range(iterations):
                sync()
                started = time.perf_counter()
                output = operation()
                sync()
                samples.append(time.perf_counter() - started)
            candidate = {
                "ok": check["finite"] and check["mae"] <= 0.01 and check["max_abs"] <= 0.25,
                "numerical": check,
            }
            candidate.update(summary(samples))
            result["candidates"][name] = candidate

        result["fusion_groups"] = {}
        hidden = torch.randn(
            (1, sequence, hidden_size), generator=generator, dtype=torch.bfloat16
        ).to(device)
        branch = torch.randn(
            (1, sequence, hidden_size), generator=generator, dtype=torch.bfloat16
        ).to(device)
        scale = torch.randn(
            (1, 1, hidden_size), generator=generator, dtype=torch.bfloat16
        ).to(device) * 0.05
        gate = torch.randn(
            (1, 1, hidden_size), generator=generator, dtype=torch.bfloat16
        ).to(device) * 0.05

        def run_group(name, eager_operation, compiled_operation):
            reference_output = eager_operation()
            sync()
            group_result = {}
            for candidate_name, operation in {
                "eager": eager_operation,
                "compiled": compiled_operation,
            }.items():
                try:
                    output = None
                    for _ in range(warmup):
                        output = operation()
                    sync()
                    check = numerical(reference_output, output)
                    samples = []
                    dist.barrier()
                    for _ in range(iterations):
                        sync()
                        started = time.perf_counter()
                        output = operation()
                        sync()
                        samples.append(time.perf_counter() - started)
                    candidate = {
                        "ok": check["finite"] and check["mae"] <= 0.01 and check["max_abs"] <= 0.25,
                        "numerical": check,
                    }
                    candidate.update(summary(samples))
                    group_result[candidate_name] = candidate
                except Exception as error:
                    group_result[candidate_name] = {
                        "ok": False,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
            result["fusion_groups"][name] = group_result

        eager_norm_modulation = lambda: F.layer_norm(
            hidden, (hidden_size,), eps=layer_norm_epsilon
        ) * (1 + scale)
        compiled_norm_modulation_fn = torch.compile(
            lambda value, modulation: F.layer_norm(
                value, (hidden_size,), eps=layer_norm_epsilon
            )
            * (1 + modulation),
            dynamic=False,
            fullgraph=True,
        )
        run_group(
            "layer_norm_modulation",
            eager_norm_modulation,
            lambda: compiled_norm_modulation_fn(hidden, scale),
        )

        eager_gated_residual = lambda: hidden + gate.tanh() * branch
        compiled_gated_residual_fn = torch.compile(
            lambda value, residual_gate, branch_value: value
            + residual_gate.tanh() * branch_value,
            dynamic=False,
            fullgraph=True,
        )
        run_group(
            "gated_residual",
            eager_gated_residual,
            lambda: compiled_gated_residual_fn(hidden, gate, branch),
        )
        result["ok"] = True
    except Exception as error:
        result.update(ok=False, error=repr(error), traceback=traceback.format_exc())
    print("ELEMENTWISE_BENCH " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if result.get("ok"):
        os._exit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
