"""Integration checks for the FANT contiguous-linear vLLM patch.

Run this inside the vendor vLLM container with both this file and
``fant_contiguous_linear.py`` mounted in the same directory.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fant_contiguous_linear import (
    _shape_is_eligible,
    install_fant_contiguous_linear_patch,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def main() -> None:
    torch.fant.set_device(0)
    device = "fant:0"
    method = UnquantizedLinearMethod()

    original_weight = torch.randn(
        (2048, 1024), device=device, dtype=torch.bfloat16
    )
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(original_weight.clone(), requires_grad=False)
    )
    x = torch.randn((16, 1024), device=device, dtype=torch.bfloat16)
    expected = F.linear(x, original_weight)

    install_fant_contiguous_linear_patch(
        min_dimension=1024,
        max_dimension=65536,
    )
    assert _shape_is_eligible((17408, 5120))
    assert not _shape_is_eligible((65537, 5120))
    method.process_weights_after_loading(layer)

    assert tuple(layer.weight.shape) == (1024, 2048)
    assert layer.weight.is_contiguous()
    assert getattr(layer, "_fant_contiguous_linear", False)
    actual = method.apply(layer, x)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    # Installing twice and processing the same layer twice must not transpose
    # the weight back to its original layout.
    install_fant_contiguous_linear_patch(
        min_dimension=1024,
        max_dimension=65536,
    )
    method.process_weights_after_loading(layer)
    assert tuple(layer.weight.shape) == (1024, 2048)

    # Small projections stay on the vendor's existing path.
    small_weight = torch.randn((48, 1024), device=device, dtype=torch.bfloat16)
    small = SimpleNamespace(
        weight=torch.nn.Parameter(small_weight, requires_grad=False)
    )
    method.process_weights_after_loading(small)
    assert tuple(small.weight.shape) == (48, 1024)
    assert not getattr(small, "_fant_contiguous_linear", False)

    # The vocabulary projection is deliberately excluded.  On the vendor
    # stack its huge contiguous GEMM can race the first TP logits all-gather
    # and crash inside RCCL topology initialization.
    vocab_weight = torch.empty(
        (65537, 1024), device="meta", dtype=torch.bfloat16
    )
    vocab = SimpleNamespace(
        weight=torch.nn.Parameter(vocab_weight, requires_grad=False)
    )
    method.process_weights_after_loading(vocab)
    assert tuple(vocab.weight.shape) == (65537, 1024)
    assert not getattr(vocab, "_fant_contiguous_linear", False)

    print("PASS: contiguous linear patch preserves output and is idempotent")


if __name__ == "__main__":
    main()
