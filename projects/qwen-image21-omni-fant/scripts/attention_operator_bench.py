#!/usr/bin/env python3
"""Benchmark exact Qwen-Image-2.1 TP2 attention operator candidates on FANT."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import statistics
import time
import traceback
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_fant  # noqa: F401
from diffusers import QwenImage21Transformer2DModel


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


def difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "finite": bool(torch.isfinite(candidate).all()),
        "mae": delta.mean().item(),
        "max_abs": delta.max().item(),
        "exact_fraction": (delta == 0).float().mean().item(),
    }


def explicit_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    return torch.matmul(scores.softmax(dim=-1), value.float()).to(query.dtype)


def chunked_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, chunk: int
) -> torch.Tensor:
    """Exact attention using query/key chunks and online FP32 softmax accumulation."""
    scale = query.shape[-1] ** -0.5
    outputs = []
    for q_start in range(0, query.shape[-2], chunk):
        q = query[..., q_start : q_start + chunk, :].float()
        running_max = torch.full(
            q.shape[:-1] + (1,), -torch.inf, dtype=torch.float32, device=query.device
        )
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros_like(q)
        for k_start in range(0, key.shape[-2], chunk):
            k = key[..., k_start : k_start + chunk, :].float()
            v = value[..., k_start : k_start + chunk, :].float()
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            block_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - new_max)
            probability = torch.exp(scores - new_max)
            accumulator = accumulator * old_scale + torch.matmul(probability, v)
            running_sum = running_sum * old_scale + probability.sum(dim=-1, keepdim=True)
            running_max = new_max
        outputs.append((accumulator / running_sum).to(query.dtype))
    return torch.cat(outputs, dim=-2)


def backend_attention(name: str):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    mapping = {
        "sdpa_math": SDPBackend.MATH,
        "sdpa_flash": SDPBackend.FLASH_ATTENTION,
        "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    backend = mapping[name]

    def operation(query, key, value):
        with sdpa_kernel(backends=[backend]):
            return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)

    return operation


def make_candidate(name: str):
    if name == "eager":
        return lambda q, k, v: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    if name.startswith("sdpa_"):
        return backend_attention(name)
    if name == "explicit":
        return explicit_attention
    if name.startswith("chunked_"):
        chunk = int(name.rsplit("_", 1)[1])
        return lambda q, k, v: chunked_attention(q, k, v, chunk)
    if name in {"flex_eager", "flex_compile"}:
        from torch.nn.attention.flex_attention import flex_attention

        operation = flex_attention
        if name == "flex_compile":
            operation = torch.compile(operation, dynamic=False, fullgraph=True)
        return lambda q, k, v: operation(q, k, v)
    raise ValueError(f"unknown candidate: {name}")


def runtime_inventory() -> dict[str, object]:
    interesting = lambda value: any(token in value.lower() for token in ("att", "flash", "sdpa"))
    try:
        import torch.nn.attention as attention

        torch_attention = tuple(filter(interesting, dir(attention)))
    except Exception as error:
        torch_attention = (f"error: {error!r}",)
    return {
        "flash_attn_module": importlib.util.find_spec("flash_attn") is not None,
        "torch_fant_symbols": tuple(filter(interesting, dir(torch_fant))),
        "torch_fant_ops": tuple(filter(interesting, dir(torch.ops.fant))),
        "torch_attention_symbols": torch_attention,
        "flex_attention_module": importlib.util.find_spec("torch.nn.attention.flex_attention") is not None,
    }


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device_ids = [int(value) for value in os.environ.get("DEVICE_IDS", "0,1").split(",")]
    if len(device_ids) != int(os.environ["WORLD_SIZE"]):
        raise ValueError("DEVICE_IDS must contain one physical FANT device per rank")
    device_index = device_ids[local_rank]
    sequence = int(os.environ.get("SEQ_LEN", "1024"))
    warmup = max(3, int(os.environ.get("WARMUP", "3")))
    iterations = max(10, int(os.environ.get("ITERATIONS", "10")))
    names = [name.strip() for name in os.environ.get("CANDIDATES", "eager").split(",") if name.strip()]
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
        "inventory": runtime_inventory(),
        "candidates": {},
    }
    try:
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        block = transformer.transformer_blocks[0]
        heads = block.attn.heads
        head_dim = block.attn.inner_dim // heads
        local_heads = heads // 2
        del transformer, block

        generator = torch.Generator(device="cpu").manual_seed(20260922)
        shape = (1, local_heads, sequence, head_dim)
        query = torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)
        key = torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)
        value = torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)
        reference = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        sync()

        prepared = {}
        for name in names:
            try:
                operation = make_candidate(name)
                for _ in range(warmup):
                    candidate_output = operation(query, key, value)
                sync()
                prepared[name] = (operation, difference(reference, candidate_output))
            except Exception as error:
                result["candidates"][name] = {
                    "ok": False,
                    "phase": "prepare",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }

        samples = {name: [] for name in prepared}
        order = [name for name in prepared for _ in range(iterations)]
        random.Random(20260922).shuffle(order)
        for name in order:
            operation, _ = prepared[name]
            dist.barrier()
            sync()
            started = time.perf_counter()
            candidate_output = operation(query, key, value)
            sync()
            samples[name].append(time.perf_counter() - started)

        for name, (operation, numerical) in prepared.items():
            torch.fant.reset_peak_memory_stats(local_rank)
            candidate_output = operation(query, key, value)
            sync()
            candidate = {
                "ok": numerical["finite"] and numerical["mae"] <= 0.01 and numerical["max_abs"] <= 0.25,
                "numerical": numerical,
                "peak_memory_bytes": torch.fant.max_memory_allocated(local_rank),
            }
            candidate.update(summarize(samples[name]))
            result["candidates"][name] = candidate
        result["ok"] = True
    except Exception as error:
        result.update(ok=False, error=repr(error), traceback=traceback.format_exc())
    print("ATTENTION_BENCH " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if result.get("ok"):
        os._exit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
