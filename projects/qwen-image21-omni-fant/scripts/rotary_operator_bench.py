#!/usr/bin/env python3
"""Benchmark Qwen-Image interleaved RoPE against the iFormer FANT operator."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
import traceback

import iFormer._C  # noqa: F401
import torch
import torch.distributed as dist
import torch_fant  # noqa: F401


def sync() -> None:
    torch.fant.synchronize()


def qwen_rope(value: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    cos = cos_half.repeat_interleave(2, dim=-1)[None, :, None, :]
    sin = sin_half.repeat_interleave(2, dim=-1)[None, :, None, :]
    real, imaginary = value.reshape(*value.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack((-imaginary, real), dim=-1).flatten(3)
    return (value.float() * cos + rotated.float() * sin).to(value.dtype)


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "finite": bool(torch.isfinite(candidate).all()),
        "mae": delta.mean().item(),
        "max_abs": delta.max().item(),
        "exact_fraction": (delta == 0).float().mean().item(),
    }


def summarize(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "p10_seconds": ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)],
        "p90_seconds": ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)],
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
    heads = int(os.environ.get("HEADS", "16"))
    head_dim = int(os.environ.get("HEAD_DIM", "128"))
    warmup = max(3, int(os.environ.get("WARMUP", "3")))
    iterations = max(10, int(os.environ.get("ITERATIONS", "10")))
    torch.fant.set_device(device_index)
    device = torch.device(f"fant:{device_index}")
    dist.init_process_group("fhccl")
    result: dict[str, object] = {
        "rank": rank,
        "device_index": device_index,
        "sequence_length": sequence,
        "heads": heads,
        "head_dim": head_dim,
        "candidates": {},
    }
    try:
        generator = torch.Generator(device="cpu").manual_seed(20260922)
        shape = (1, sequence, heads, head_dim)
        query = torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)
        key = torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)
        angles = torch.randn(
            (sequence, head_dim // 2), generator=generator, dtype=torch.float32
        )
        cos_half = angles.cos().to(torch.bfloat16).to(device)
        sin_half = angles.sin().to(torch.bfloat16).to(device)

        def eager_pair():
            return qwen_rope(query, cos_half, sin_half), qwen_rope(key, cos_half, sin_half)

        def iformer_pair():
            q = torch.ops.iFormer.apply_rotary_emb(
                query, cos_half, sin_half, True, False, 0, None, None
            )
            k = torch.ops.iFormer.apply_rotary_emb(
                key, cos_half, sin_half, True, False, 0, None, None
            )
            return q, k

        reference_q, reference_k = eager_pair()
        sync()
        for name, operation in {"eager_qwen_rope": eager_pair, "iformer_rope": iformer_pair}.items():
            try:
                output_q = output_k = None
                for _ in range(warmup):
                    output_q, output_k = operation()
                sync()
                q_check = compare(reference_q, output_q)
                k_check = compare(reference_k, output_k)
                samples = []
                dist.barrier()
                for _ in range(iterations):
                    sync()
                    started = time.perf_counter()
                    output_q, output_k = operation()
                    sync()
                    samples.append(time.perf_counter() - started)
                candidate = {
                    "ok": all(
                        check["finite"] and check["mae"] <= 0.01 and check["max_abs"] <= 0.25
                        for check in (q_check, k_check)
                    ),
                    "query_numerical": q_check,
                    "key_numerical": k_check,
                }
                candidate.update(summarize(samples))
                result["candidates"][name] = candidate
            except Exception as error:
                result["candidates"][name] = {
                    "ok": False,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
        result["ok"] = True
    except Exception as error:
        result.update(ok=False, error=repr(error), traceback=traceback.format_exc())
    print("ROTARY_BENCH " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if result.get("ok"):
        os._exit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
