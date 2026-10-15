"""GPU integration checks for slab-backed Decode linear weights."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fant_slab_linear import install_fant_slab_linear_patch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def make_layer(weight: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        weight=torch.nn.Parameter(weight.clone(), requires_grad=False)
    )


def main() -> None:
    torch.fant.set_device(0)
    device = "fant:0"
    method = UnquantizedLinearMethod()
    original = torch.randn((2048, 1024), device=device, dtype=torch.bfloat16)

    install_fant_slab_linear_patch(
        target_shapes={(2048, 1024)},
        max_decode_tokens=32,
        max_copies=2,
    )

    first = make_layer(original)
    second = make_layer(original)
    third = make_layer(original)
    method.process_weights_after_loading(first)
    method.process_weights_after_loading(second)
    method.process_weights_after_loading(third)

    first_decode = first._fant_decode_weight
    second_decode = second._fant_decode_weight
    assert tuple(first_decode.shape) == (1024, 2048)
    assert tuple(second_decode.shape) == (1024, 2048)
    assert first_decode.is_contiguous()
    assert second_decode.is_contiguous()
    assert first_decode.untyped_storage().data_ptr() == second_decode.untyped_storage().data_ptr()
    assert second_decode.data_ptr() - first_decode.data_ptr() == original.numel() * original.element_size()
    assert not getattr(third, "_fant_slab_linear", False)

    prefill_x = torch.randn((64, 1024), device=device, dtype=torch.bfloat16)
    decode_x = torch.randn((16, 1024), device=device, dtype=torch.bfloat16)
    for layer in (first, second):
        torch.testing.assert_close(
            method.apply(layer, prefill_x),
            F.linear(prefill_x, original),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            method.apply(layer, decode_x),
            F.linear(decode_x, original),
            rtol=1e-2,
            atol=1e-2,
        )

    first_ptr = first_decode.data_ptr()
    method.process_weights_after_loading(first)
    assert first._fant_decode_weight.data_ptr() == first_ptr
    print("PASS: one slab backs Decode layouts while Prefill keeps original weights")


if __name__ == "__main__":
    main()
