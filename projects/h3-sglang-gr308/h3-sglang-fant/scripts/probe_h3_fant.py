#!/usr/bin/env python3
"""Probe the real H3 W4A8 path and reject compatibility-only dispatch."""

import argparse
import json
import time
import traceback
from pathlib import Path

from h3_sglang_fant.w4a8_gate import evaluate_native_gate

STEM = "blocks.0.attn.out_proj"


def _operator_names(torch_module) -> list[str]:
    matches = []
    tokens = ("int8", "w4a8", "gemm", "matmul")
    for namespace_name in dir(torch_module.ops):
        if namespace_name.startswith("_"):
            continue
        try:
            namespace = getattr(torch_module.ops, namespace_name)
            for name in dir(namespace):
                qualified = f"{namespace_name}.{name}"
                if any(token in qualified.lower() for token in tokens):
                    matches.append(qualified)
        except Exception:
            continue
    try:
        import iFormer

        for name in dir(getattr(iFormer, "ops", object())):
            qualified = f"iFormer.ops.{name}"
            if any(token in qualified.lower() for token in tokens):
                matches.append(qualified)
    except Exception:
        pass
    return sorted(set(matches))


def _load_projection(path: Path):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        return {
            suffix: checkpoint.get_tensor(f"{STEM}.{suffix}")
            for suffix in (
                "weight",
                "weight_s_rel",
                "weight_s_channel",
                "weight_codebook",
            )
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="fant:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "device_type": args.device.split(":", 1)[0],
        "projection": STEM,
    }
    try:
        import torch
        import torch_fant  # noqa: F401
        import comfy_kitchen as kitchen
        from comfy_kitchen.registry import registry
        from comfy_kitchen.tensor.w4a8_int8 import w4a8_int8_linear

        if report["device_type"] != "fant":
            raise ValueError("An explicit fant:N device is required")
        torch.fant.set_device(int(args.device.split(":", 1)[1]))
        report["torch_version"] = torch.__version__
        report["privateuse_backend"] = torch._C._get_privateuse1_backend_name()
        report["privateuse1_int_mm_kernel"] = bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::_int_mm", "PrivateUse1"
            )
        )
        report["int_mm_dispatch_table"] = torch._C._dispatch_dump_table(
            "aten::_int_mm"
        )
        report["candidate_vendor_operators"] = _operator_names(torch)
        report["registered_backends"] = registry.list_backends()

        tensors = _load_projection(args.checkpoint)
        qdata = tensors["weight"]
        s_rel = tensors["weight_s_rel"]
        s_channel = tensors["weight_s_channel"]
        codebook = tensors["weight_codebook"]
        generator = torch.Generator(device="cpu").manual_seed(1101)
        x = torch.randn(
            (1, qdata.shape[1] * 2),
            generator=generator,
            dtype=torch.bfloat16,
        )
        common = {
            "group_size": 16,
            "convrot_groupsize": 256,
            "out_dtype": torch.bfloat16,
        }
        with kitchen.use_backend("eager"):
            reference = w4a8_int8_linear(
                x,
                qdata,
                s_rel,
                s_channel,
                codebook=codebook,
                **common,
            ).float()

        operands = {
            "x": x.to(args.device),
            "qdata": qdata.to(args.device),
            "s_rel": s_rel.float().to(args.device),
            "s_channel": s_channel.to(args.device),
            "codebook": codebook.to(args.device),
            **common,
        }
        selection_kwargs = dict(operands)
        selection_kwargs.update({"correction": None, "bias": None})
        selected = registry.get_capable_backend(
            "w4a8_int8_linear", selection_kwargs
        )
        implementation = registry.get_implementation(
            "w4a8_int8_linear", backend=selected, kwargs=selection_kwargs
        )
        report["selected_backend"] = selected
        report["implementation"] = (
            f"{implementation.__module__}.{implementation.__qualname__}"
        )

        started = time.perf_counter()
        output = implementation(**selection_kwargs)
        torch.fant.synchronize()
        report["wall_s"] = time.perf_counter() - started
        report["output_device"] = str(output.device)
        actual = output.detach().cpu().float()
        report["finite"] = bool(torch.isfinite(actual).all())
        report["relative_l2"] = float(
            torch.linalg.vector_norm(actual - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        )
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()

    report = evaluate_native_gate(report)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["native_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
