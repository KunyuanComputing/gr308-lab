#!/usr/bin/env python3
"""Run the prepare plus denoise phases and persist latents for parallel decode."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch_fant  # noqa: F401
from diffusers import QwenImage21Pipeline


def main() -> None:
    base = Path(os.environ.get("WORKDIR", "/work"))
    steps = int(os.environ.get("STEPS", "2"))
    width = int(os.environ.get("WIDTH", "512"))
    height = int(os.environ.get("HEIGHT", "512"))
    prompt = os.environ.get(
        "PROMPT",
        "A clean product photograph of a red ceramic teapot on a light wooden table, soft studio lighting, "
        "with the Chinese text ‘风华三号’ clearly printed on the teapot.",
    )
    output = base / "output" / f"qwen-image-2.1-bfloat16-{width}x{height}-{steps}steps-latents.pt"
    metrics = {
        "phase_protocol": ["prepare_encode", "denoise_step", "step_scheduler", "post_decode"],
        "steps": steps,
        "width": width,
        "height": height,
        "device": "fant:0",
        "output": str(output),
    }
    started = time.perf_counter()
    step_events = []
    try:
        load_started = time.perf_counter()
        pipe = QwenImage21Pipeline.from_pretrained(str(base / "model"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        pipe.to("fant:0")
        metrics["prepare_load_seconds"] = time.perf_counter() - load_started
        torch.fant.reset_peak_memory_stats(0)

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

        denoise_started = time.perf_counter()
        latents = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(42),
            output_type="latent",
            callback_on_step_end=on_step_end,
        ).images
        torch.fant.synchronize()
        metrics["denoise_seconds"] = time.perf_counter() - denoise_started
        metrics["denoise_peak_memory_bytes"] = torch.fant.max_memory_allocated(0)
        metrics["step_events"] = step_events
        torch.save(latents.cpu(), output)
        metrics["ok"] = True
    except Exception as error:
        metrics.update(ok=False, error=repr(error), traceback=traceback.format_exc())
    finally:
        metrics["total_seconds"] = time.perf_counter() - started
        metrics_path = base / "logs" / f"short-generate-{width}x{height}-{steps}steps.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if not metrics.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
