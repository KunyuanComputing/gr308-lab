"""Integration checks for the FANT decode-only dual-layout linear patch."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fant_dual_layout_linear import install_fant_dual_layout_linear_patch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def main() -> None:
    torch.fant.set_device(0)
    device = "fant:0"
    method = UnquantizedLinearMethod()

    original = torch.randn((2048, 1024), device=device, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(original.clone(), requires_grad=False)
    )

    install_fant_dual_layout_linear_patch(
        target_shapes={(2048, 1024)},
        max_decode_tokens=32,
        max_copies=1,
        defer_small_calls=1,
    )
    method.process_weights_after_loading(layer)

    # The original layout remains available for Prefill/mixed batches.
    assert tuple(layer.weight.shape) == (2048, 1024)
    assert layer.weight.is_contiguous()
    # Decode storage is allocated lazily, after vLLM's large Prefill profile
    # allocations have completed, to avoid fragmenting FANT device memory.
    assert not hasattr(layer, "_fant_decode_weight")

    prefill_x = torch.randn((64, 1024), device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(
        method.apply(layer, prefill_x),
        F.linear(prefill_x, original),
        rtol=1e-2,
        atol=1e-2,
    )
    assert not hasattr(layer, "_fant_decode_weight")

    decode_x = torch.randn((16, 1024), device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(
        method.apply(layer, decode_x),
        F.linear(decode_x, original),
        rtol=1e-2,
        atol=1e-2,
    )
    assert not hasattr(layer, "_fant_decode_weight")

    torch.testing.assert_close(
        method.apply(layer, decode_x),
        F.linear(decode_x, original),
        rtol=1e-2,
        atol=1e-2,
    )
    decode_weight = layer._fant_decode_weight
    assert tuple(decode_weight.shape) == (1024, 2048)
    assert decode_weight.is_contiguous()

    # Processing twice must reuse the same persistent Decode tensor.
    ptr = decode_weight.data_ptr()
    method.process_weights_after_loading(layer)
    assert layer._fant_decode_weight.data_ptr() == ptr

    second_target = SimpleNamespace(
        weight=torch.nn.Parameter(original.clone(), requires_grad=False)
    )
    method.process_weights_after_loading(second_target)
    assert not getattr(second_target, "_fant_dual_layout_linear", False)

    other_weight = torch.randn((1536, 1024), device=device, dtype=torch.bfloat16)
    other = SimpleNamespace(
        weight=torch.nn.Parameter(other_weight, requires_grad=False)
    )
    method.process_weights_after_loading(other)
    assert not hasattr(other, "_fant_decode_weight")

    print("PASS: dual-layout patch isolates Decode and preserves Prefill")


if __name__ == "__main__":
    main()
