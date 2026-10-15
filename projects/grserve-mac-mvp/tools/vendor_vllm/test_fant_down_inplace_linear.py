"""GPU integration checks for the in-place MLP down-projection layout."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fant_down_inplace_linear import install_fant_down_inplace_linear_patch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def main() -> None:
    torch.fant.set_device(0)
    device = "fant:0"
    method = UnquantizedLinearMethod()
    original = torch.randn((2048, 1024), device=device, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(original.clone(), requires_grad=False)
    )
    other = SimpleNamespace(
        weight=torch.nn.Parameter(
            torch.randn((2048, 2048), device=device, dtype=torch.bfloat16),
            requires_grad=False,
        )
    )

    install_fant_down_inplace_linear_patch(target_shapes={(2048, 1024)})
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(other)

    assert tuple(layer.weight.shape) == (1024, 2048)
    assert layer.weight.is_contiguous()
    assert getattr(layer, "_fant_down_inplace_linear", False)
    assert tuple(other.weight.shape) == (2048, 2048)
    assert not getattr(other, "_fant_down_inplace_linear", False)

    for tokens in (16, 256):
        x = torch.randn((tokens, 1024), device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(
            method.apply(layer, x),
            F.linear(x, original),
            rtol=1e-2,
            atol=1e-2,
        )

    first_ptr = layer.weight.data_ptr()
    method.process_weights_after_loading(layer)
    assert layer.weight.data_ptr() == first_ptr
    assert tuple(layer.weight.shape) == (1024, 2048)
    print("PASS: selected down projection uses one in-place Decode layout")


if __name__ == "__main__":
    main()
