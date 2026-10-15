# SPDX-License-Identifier: Apache-2.0
"""Fantasy-specific adapters for SGLang's FLA/GDN kernels."""

from __future__ import annotations

import os
from typing import Optional

import torch

_debug_emitted = False


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Correctness fallback for the vendor Triton short-sequence store bug."""
    del chunk_indices, kwargs
    token_dim = 2 if head_first else 1
    work = g.movedim(token_dim, 1) if token_dim != 1 else g
    out = torch.empty_like(work, dtype=output_dtype or g.dtype)

    if cu_seqlens is None:
        ranges = [(batch_idx, 0, work.shape[1]) for batch_idx in range(work.shape[0])]
    else:
        starts = cu_seqlens.detach().cpu().tolist()
        ranges = [(0, starts[i], starts[i + 1]) for i in range(len(starts) - 1)]

    for batch_idx, seq_start, seq_end in ranges:
        for start in range(seq_start, seq_end, chunk_size):
            end = min(start + chunk_size, seq_end)
            segment = work[batch_idx : batch_idx + 1, start:end].to(out.dtype)
            if reverse:
                segment = torch.flip(
                    torch.cumsum(torch.flip(segment, dims=[1]), dim=1), dims=[1]
                )
            else:
                segment = torch.cumsum(segment, dim=1)
            if scale is not None:
                segment = segment * scale
            out[batch_idx : batch_idx + 1, start:end] = segment

    return out.movedim(1, token_dim) if token_dim != 1 else out


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the vendor GDN state-update op while preserving SGLang semantics.

    SGLang stores recurrent state as ``[N, H, V, K]`` and addresses a shared
    cache through ``initial_state_indices``.  The GR308 iFormer operator uses
    ``[N, H, K, V]`` and takes a dense per-request state tensor.  This adapter
    performs only those boundary conversions; the recurrence itself remains
    the vendor-optimized kernel used by the factory vLLM image.
    """
    global _debug_emitted

    if gk is not None or use_exp2:
        raise NotImplementedError("Fantasy vendor GDN supports scalar g only")

    from iFormer.ops.triton.fla.chunk_gated_delta_rule_fwd_h import (
        chunk_gated_delta_rule_fwd_h_impl,
    )
    from sglang.kernels.ops.attention.fla.index import prepare_chunk_offsets

    del chunk_indices
    batch, tokens, key_heads, key_dim = k.shape
    value_heads, value_dim = u.shape[-2:]
    chunk_size = 64

    if cu_seqlens is None:
        sequence_count = batch
        sequence_starts = torch.arange(
            0,
            (batch + 1) * tokens,
            tokens,
            dtype=torch.int32,
            device=k.device,
        )
        total_chunks = batch * ((tokens + chunk_size - 1) // chunk_size)
    else:
        sequence_count = len(cu_seqlens) - 1
        sequence_starts = cu_seqlens.to(dtype=torch.int32)
        total_chunks = sum(
            (int(length) + chunk_size - 1) // chunk_size
            for length in (sequence_starts[1:] - sequence_starts[:-1]).cpu()
        )
    chunk_offsets = prepare_chunk_offsets(sequence_starts, chunk_size).to(
        dtype=torch.int32
    )

    if os.getenv("SGLANG_FANT_GDN_DIAGNOSTIC_BYPASS") == "1":
        return k.new_zeros(
            batch, total_chunks, value_heads, value_dim, key_dim
        ), torch.zeros_like(u)

    selected_state = None
    safe_indices = None
    valid_state = None
    if initial_state is not None:
        if initial_state_indices is None:
            safe_indices = torch.arange(
                sequence_count, dtype=torch.int64, device=k.device
            )
            valid_state = torch.ones(
                sequence_count, dtype=torch.bool, device=k.device
            )
        else:
            state_indices = initial_state_indices[:sequence_count].to(torch.int64)
            valid_state = state_indices >= 0
            safe_indices = state_indices.clamp_min(0)
        selected_state = (
            initial_state.index_select(0, safe_indices)
            .transpose(-1, -2)
            .contiguous()
        )
        selected_state.mul_(valid_state[:, None, None, None])

    h_vendor = k.new_empty(
        batch, total_chunks, value_heads, key_dim, value_dim
    )
    v_new = torch.empty_like(u) if save_new_value else None
    final_state = torch.empty(
        sequence_count,
        value_heads,
        key_dim,
        value_dim,
        dtype=torch.float32,
        device=k.device,
    )

    if not _debug_emitted:
        print(
            "FANT_GDN_ADAPTER_METADATA",
            f"k={tuple(k.shape)}:{k.dtype}",
            f"w={tuple(w.shape)}:{w.dtype}",
            f"u={tuple(u.shape)}:{u.dtype}",
            f"g={None if g is None else (tuple(g.shape), g.dtype)}",
            f"initial_state={None if initial_state is None else (tuple(initial_state.shape), initial_state.dtype)}",
            f"indices={None if initial_state_indices is None else initial_state_indices.detach().cpu().tolist()}",
            f"cu={sequence_starts.detach().cpu().tolist()}",
            flush=True,
        )
        _debug_emitted = True

    chunk_gated_delta_rule_fwd_h_impl(
        k,
        u,
        w,
        v_new,
        g,
        None,
        h_vendor,
        selected_state,
        final_state,
        sequence_starts,
        chunk_offsets,
        tokens,
        value_heads,
        key_heads,
        key_dim,
        value_dim,
        chunk_size,
        sequence_count,
        g is not None,
        False,
        selected_state is not None,
        True,
        v_new is not None,
    )

    if initial_state is not None:
        valid_positions = torch.where(valid_state)[0]
        updated_state = final_state.transpose(-1, -2).contiguous()
        initial_state.index_copy_(
            0,
            safe_indices.index_select(0, valid_positions),
            updated_state.index_select(0, valid_positions),
        )

    # Restore SGLang's public [B, NT, H, V, K] state layout.
    return h_vendor.transpose(-1, -2).contiguous(), v_new
