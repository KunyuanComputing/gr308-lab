"""Slab-backed, Decode-only linear layouts for the FANT vLLM backend."""

from __future__ import annotations

from collections.abc import Set
from typing import Any

import torch

_INSTALLED = False
_TARGET_SHAPES: frozenset[tuple[int, int]] = frozenset({(5120, 8704)})
_MAX_DECODE_TOKENS = 32
_MAX_COPIES = 8
_REGISTERED_COPIES = 0
_SLABS: dict[tuple[torch.device, torch.dtype, tuple[int, int]], torch.Tensor] = {}
_ORIGINAL_PROCESS: Any = None
_ORIGINAL_APPLY: Any = None


def _eligible(weight: torch.Tensor) -> bool:
    return (
        weight.device.type == "fant"
        and weight.ndim == 2
        and tuple(weight.shape) in _TARGET_SHAPES
        and weight.dtype in (torch.bfloat16, torch.float16)
    )


def install_fant_slab_linear_patch(
    *,
    target_shapes: Set[tuple[int, int]] | None = None,
    max_decode_tokens: int = 32,
    max_copies: int = 8,
) -> None:
    """Install an idempotent small-batch path backed by one device slab."""

    global _INSTALLED, _TARGET_SHAPES, _MAX_DECODE_TOKENS, _MAX_COPIES
    global _REGISTERED_COPIES, _ORIGINAL_PROCESS, _ORIGINAL_APPLY

    if max_decode_tokens <= 0:
        raise ValueError("max_decode_tokens must be positive")
    if max_copies <= 0:
        raise ValueError("max_copies must be positive")
    if target_shapes is not None:
        normalized = frozenset(tuple(map(int, shape)) for shape in target_shapes)
        if not normalized or any(len(shape) != 2 for shape in normalized):
            raise ValueError("target_shapes must contain 2-D shapes")
        _TARGET_SHAPES = normalized
    _MAX_DECODE_TOKENS = max_decode_tokens
    _MAX_COPIES = max_copies
    if _INSTALLED:
        return

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    _ORIGINAL_PROCESS = UnquantizedLinearMethod.process_weights_after_loading
    _ORIGINAL_APPLY = UnquantizedLinearMethod.apply

    def process_weights_after_loading(self: Any, layer: torch.nn.Module) -> None:
        global _REGISTERED_COPIES
        _ORIGINAL_PROCESS(self, layer)
        if getattr(layer, "_fant_slab_linear", False):
            return
        weight = layer.weight
        if not _eligible(weight) or _REGISTERED_COPIES >= _MAX_COPIES:
            return

        out_features, in_features = weight.shape
        key = (weight.device, weight.dtype, tuple(weight.shape))
        slab = _SLABS.get(key)
        if slab is None:
            slab = torch.empty(
                (_MAX_COPIES, in_features, out_features),
                device=weight.device,
                dtype=weight.dtype,
            )
            _SLABS[key] = slab

        decode_weight = slab[_REGISTERED_COPIES]
        decode_weight.copy_(weight.detach().transpose(0, 1))
        layer._fant_decode_weight = decode_weight
        layer._fant_slab_linear = True
        _REGISTERED_COPIES += 1

    def apply(
        self: Any,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        if (
            not getattr(layer, "_fant_slab_linear", False)
            or x_2d.shape[0] > _MAX_DECODE_TOKENS
        ):
            return _ORIGINAL_APPLY(self, layer, x, bias)

        decode_weight = layer._fant_decode_weight
        output_shape = (*x.shape[:-1], decode_weight.shape[1])
        if bias is None:
            output = torch.mm(x_2d, decode_weight)
        else:
            output = torch.addmm(bias, x_2d, decode_weight)
        return output.reshape(output_shape)

    UnquantizedLinearMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    UnquantizedLinearMethod.apply = apply
    _INSTALLED = True
