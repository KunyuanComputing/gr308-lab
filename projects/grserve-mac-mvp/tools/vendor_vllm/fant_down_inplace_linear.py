"""FANT Decode-optimized in-place layout for selected MLP down projections.

Only explicitly listed ``[out_features, in_features]`` shapes are converted.
The conversion happens after checkpoint loading, replaces the original tensor
storage, and therefore does not retain a second copy of model weights.
"""

from __future__ import annotations

from typing import Any

import torch


_INSTALLED = False
_TARGET_SHAPES: set[tuple[int, int]] = {(5120, 8704)}
_ORIGINAL_PROCESS: Any = None
_ORIGINAL_APPLY: Any = None


def install_fant_down_inplace_linear_patch(
    *, target_shapes: set[tuple[int, int]] | None = None
) -> None:
    """Install an idempotent patch for exact FANT linear weight shapes."""

    global _INSTALLED, _TARGET_SHAPES, _ORIGINAL_PROCESS, _ORIGINAL_APPLY
    if target_shapes is not None:
        if not target_shapes or any(
            len(shape) != 2 or min(shape) <= 0 for shape in target_shapes
        ):
            raise ValueError("target_shapes must contain positive 2D shapes")
        _TARGET_SHAPES = set(target_shapes)
    if _INSTALLED:
        return

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    _ORIGINAL_PROCESS = UnquantizedLinearMethod.process_weights_after_loading
    _ORIGINAL_APPLY = UnquantizedLinearMethod.apply

    def process_weights_after_loading(self: Any, layer: torch.nn.Module) -> None:
        if getattr(layer, "_fant_down_inplace_linear", False):
            return
        _ORIGINAL_PROCESS(self, layer)
        weight = layer.weight
        if (
            weight.device.type != "fant"
            or weight.dtype not in (torch.bfloat16, torch.float16)
            or tuple(weight.shape) not in _TARGET_SHAPES
        ):
            return
        weight.data = weight.detach().transpose(0, 1).contiguous()
        layer._fant_down_inplace_linear = True

    def apply(
        self: Any,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not getattr(layer, "_fant_down_inplace_linear", False):
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
