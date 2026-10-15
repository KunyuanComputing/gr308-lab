#!/usr/bin/env python3
"""Run Qwen-Image-2.1 DiT with transformer blocks split across two FANT devices."""

from __future__ import annotations

import json
import os
import time
import traceback
import weakref
from pathlib import Path
from typing import Any

import torch
import torch_fant  # noqa: F401
from diffusers import QwenImage21Pipeline


def move_nested(
    value: Any,
    device: torch.device,
    cache: dict[int, tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor]],
) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device == device:
            return value
        cached = cache.get(id(value))
        if cached is not None and cached[0]() is value:
            return cached[1]
        # Direct FANT0 -> FANT1 tensor copies crash PAL 0.18.0. Bounce through
        # pinned host-visible memory until the runtime exposes safe peer copy.
        if value.device.type == "fant" and device.type == "fant":
            moved = value.to("cpu").to(device)
        else:
            moved = value.to(device)
        key = id(value)

        def drop_entry(_reference, *, key=key):
            cache.pop(key, None)

        cache[key] = (weakref.ref(value, drop_entry), moved)
        return moved
    if isinstance(value, tuple):
        return tuple(move_nested(item, device, cache) for item in value)
    if isinstance(value, list):
        return [move_nested(item, device, cache) for item in value]
    if isinstance(value, dict):
        return {key: move_nested(item, device, cache) for key, item in value.items()}
    return value


def place_pipeline(pipe, split: int, first: torch.device, second: torch.device) -> None:
    """Place modules directly from CPU; torch_fant cannot migrate FANT0 -> FANT1."""
    pipe.text_encoder.to(first)
    transformer = pipe.transformer
    blocks = transformer.transformer_blocks
    if not 0 < split < len(blocks):
        raise ValueError(f"split must be between 1 and {len(blocks) - 1}")
    for name in ("time_text_embed", "txt_in", "img_in", "modulation"):
        getattr(transformer, name).to(first)
    print(f"DIT_SPLIT placing_blocks=0:{split} device={first}", flush=True)
    print(f"DIT_SPLIT placing_blocks={split}:{len(blocks)} device={second}", flush=True)
    for index, block in enumerate(blocks):
        device = first if index < split else second
        block.to(device)
    transformer.norm_out.to(second)
    transformer.proj_out.to(second)


def install_block_pipeline(transformer, split: int, first: torch.device, second: torch.device):
    blocks = transformer.transformer_blocks

    cache: dict[int, tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor]] = {}

    def second_device_pre_hook(_module, args, kwargs):
        return move_nested(args, second, cache), move_nested(kwargs, second, cache)

    handles = [
        block.register_forward_pre_hook(second_device_pre_hook, with_kwargs=True) for block in blocks[split:]
    ]
    handles.append(transformer.norm_out.register_forward_pre_hook(second_device_pre_hook, with_kwargs=True))

    def output_to_first(_module, _args, output):
        return move_nested(output, first, {})

    handles.append(transformer.register_forward_hook(output_to_first))
    return handles


def parameter_bytes(module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def sync_all() -> None:
    current = torch.fant.current_device()
    for index in (0, 1):
        torch.fant.set_device(index)
        torch.fant.synchronize()
    torch.fant.set_device(current)


def main() -> None:
    base = Path(os.environ.get("WORKDIR", "/work"))
    steps = int(os.environ.get("STEPS", "1"))
    width = int(os.environ.get("WIDTH", "1024"))
    height = int(os.environ.get("HEIGHT", "1024"))
    split = int(os.environ.get("SPLIT", "16"))
    first, second = torch.device("fant:0"), torch.device("fant:1")
    output = base / "output" / f"qwen-image-2.1-bfloat16-{width}x{height}-{steps}steps-dit-block16x16.pt"
    metrics = {
        "mode": "block_pipeline_16x16",
        "steps": steps,
        "width": width,
        "height": height,
        "split": split,
        "output": str(output),
    }
    started = time.perf_counter()
    handles = []
    try:
        pipe = QwenImage21Pipeline.from_pretrained(
            str(base / "model"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        print("DIT_SPLIT pipeline_loaded_cpu", flush=True)
        place_pipeline(pipe, split, first, second)
        print("DIT_SPLIT pipeline_placed", flush=True)
        handles = install_block_pipeline(pipe.transformer, split, first, second)
        print("DIT_SPLIT hooks_installed", flush=True)
        sync_all()
        metrics["first_block_parameter_bytes"] = sum(parameter_bytes(block) for block in pipe.transformer.transformer_blocks[:split])
        metrics["second_block_parameter_bytes"] = sum(parameter_bytes(block) for block in pipe.transformer.transformer_blocks[split:])
        metrics["memory_after_split_bytes"] = {
            "fant:0": torch.fant.memory_allocated(0),
            "fant:1": torch.fant.memory_allocated(1),
        }
        torch.fant.reset_peak_memory_stats(0)
        torch.fant.reset_peak_memory_stats(1)
        step_events = []
        denoise_started = time.perf_counter()

        def on_step_end(_pipe, index, timestep, callback_kwargs):
            sync_all()
            step_events.append(
                {
                    "step": index,
                    "timestep": float(timestep.float().cpu()),
                    "elapsed_seconds": time.perf_counter() - denoise_started,
                }
            )
            return callback_kwargs

        latents = pipe(
            prompt=(
                "A clean product photograph of a red ceramic teapot on a light wooden table, soft studio lighting, "
                "with the Chinese text ‘风华三号’ clearly printed on the teapot."
            ),
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(42),
            output_type="latent",
            callback_on_step_end=on_step_end,
        ).images
        sync_all()
        metrics["denoise_seconds"] = time.perf_counter() - denoise_started
        metrics["step_events"] = step_events
        metrics["peak_memory_bytes"] = {
            "fant:0": torch.fant.max_memory_allocated(0),
            "fant:1": torch.fant.max_memory_allocated(1),
        }
        torch.save(latents.cpu(), output)
        metrics["ok"] = True
    except Exception as error:
        metrics.update(ok=False, error=repr(error), traceback=traceback.format_exc())
    finally:
        for handle in handles:
            handle.remove()
        metrics["total_seconds"] = time.perf_counter() - started
        log = base / "logs" / f"dual-device-dit-{width}x{height}-{steps}steps.json"
        log.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if not metrics.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
