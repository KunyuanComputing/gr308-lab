"""FANT-specific contiguous-weight path for unquantized vLLM linear layers.

The vendor BF16 GEMM is substantially faster when its right-hand operand is a
contiguous ``[in_features, out_features]`` tensor.  vLLM normally stores the
weight as ``[out_features, in_features]`` and passes a transposed view to
``F.linear``.  This patch converts sufficiently large weights once, after
loading, and dispatches inference through ``torch.mm``.
"""

from __future__ import annotations

from typing import Any

import torch

_INSTALLED = False
_MIN_DIMENSION = 1024
_MAX_DIMENSION = 65536
_ORIGINAL_PROCESS: Any = None
_ORIGINAL_APPLY: Any = None


def _shape_is_eligible(shape: torch.Size | tuple[int, ...]) -> bool:
    return (
        len(shape) == 2
        and min(shape) >= _MIN_DIMENSION
        and max(shape) <= _MAX_DIMENSION
    )


def _eligible(weight: torch.Tensor) -> bool:
    return (
        weight.device.type == "fant"
        and _shape_is_eligible(weight.shape)
        and weight.dtype in (torch.bfloat16, torch.float16)
    )


def install_fant_contiguous_linear_patch(
    *,
    min_dimension: int = 1024,
    max_dimension: int = 65536,
) -> None:
    """Install an idempotent patch on vLLM's unquantized linear method."""

    global _INSTALLED, _MIN_DIMENSION, _MAX_DIMENSION
    global _ORIGINAL_PROCESS, _ORIGINAL_APPLY
    if min_dimension <= 0:
        raise ValueError("min_dimension must be positive")
    if max_dimension < min_dimension:
        raise ValueError("max_dimension must be at least min_dimension")
    _MIN_DIMENSION = min_dimension
    _MAX_DIMENSION = max_dimension
    if _INSTALLED:
        return

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    _ORIGINAL_PROCESS = UnquantizedLinearMethod.process_weights_after_loading
    _ORIGINAL_APPLY = UnquantizedLinearMethod.apply

    def process_weights_after_loading(self: Any, layer: torch.nn.Module) -> None:
        _ORIGINAL_PROCESS(self, layer)
        weight = layer.weight
        if getattr(weight, "_fant_contiguous_in_out", False):
            layer._fant_contiguous_linear = True
            return
        if not _eligible(weight):
            return

        # Rebind the Parameter storage so only one copy remains resident after
        # this hook.  Parameter metadata used by checkpoint loading stays on
        # the original object, but loading is already complete at this point.
        weight.data = weight.detach().transpose(0, 1).contiguous()
        weight._fant_contiguous_in_out = True
        layer._fant_contiguous_linear = True

    def apply(
        self: Any,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not getattr(layer, "_fant_contiguous_linear", False):
            return _ORIGINAL_APPLY(self, layer, x, bias)

        output_shape = (*x.shape[:-1], layer.weight.shape[1])
        x_2d = x.reshape(-1, x.shape[-1])
        if bias is None:
            output = torch.mm(x_2d, layer.weight)
        else:
            output = torch.addmm(bias, x_2d, layer.weight)
        return output.reshape(output_shape)

    UnquantizedLinearMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    UnquantizedLinearMethod.apply = apply
    _INSTALLED = True
