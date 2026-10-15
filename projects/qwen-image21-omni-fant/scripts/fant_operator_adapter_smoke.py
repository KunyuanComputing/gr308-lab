#!/usr/bin/env python3
"""Smoke-test the deployment adapter's iFormer RMSNorm and RoPE wrappers."""

from __future__ import annotations

import json
import os

import iFormer._C  # noqa: F401
import torch
import torch.nn.functional as F
import torch_fant  # noqa: F401
from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen
from vllm_fant import _custom_ops as fant_ops

import tp2_full_dit as adapter


def error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "finite": bool(torch.isfinite(candidate).all()),
        "mae": delta.mean().item(),
        "max_abs": delta.max().item(),
    }


def main() -> None:
    device_index = int(os.environ.get("DEVICE_ID", "0"))
    sequence = int(os.environ.get("SEQ_LEN", "256"))
    torch.fant.set_device(device_index)
    device = torch.device(f"fant:{device_index}")
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    value = torch.randn(
        (1, sequence, 16, 128), generator=generator, dtype=torch.bfloat16
    ).to(device)
    weight = torch.randn((128,), generator=generator, dtype=torch.bfloat16).to(device)

    adapter._FANT_OPS = fant_ops
    rms_module = adapter.IFormerRMSNorm(weight, 1e-6).to(device)
    rms_reference = F.rms_norm(value, (128,), weight, 1e-6)
    rms_candidate = rms_module(value)

    angles = torch.randn((sequence, 64), generator=generator, dtype=torch.float32)
    cos_half = angles.cos().to(torch.bfloat16).to(device)
    sin_half = angles.sin().to(torch.bfloat16).to(device)
    rotary = (
        cos_half.repeat_interleave(2, dim=-1),
        sin_half.repeat_interleave(2, dim=-1),
    )
    adapter._ORIGINAL_ROTARY = apply_rotary_emb_qwen
    rotary_reference = apply_rotary_emb_qwen(value, rotary, use_real=True)
    rotary_candidate = adapter.iformer_rotary_emb_qwen(value, rotary, use_real=True)
    rotary_candidate_repeat = adapter.iformer_rotary_emb_qwen(value, rotary, use_real=True)
    torch.fant.synchronize()

    result = {
        "device_index": device_index,
        "sequence_length": sequence,
        "rms_norm": error(rms_reference, rms_candidate),
        "rotary": error(rotary_reference, rotary_candidate),
        "rotary_cache_repeat": error(rotary_candidate, rotary_candidate_repeat),
    }
    result["ok"] = (
        result["rms_norm"]["finite"]
        and result["rms_norm"]["mae"] <= 0.01
        and result["rms_norm"]["max_abs"] <= 0.25
        and result["rotary"]["finite"]
        and result["rotary"]["mae"] <= 0.01
        and result["rotary"]["max_abs"] <= 0.25
        and result["rotary_cache_repeat"]["max_abs"] == 0
    )
    print("FANT_OPERATOR_ADAPTER " + json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
