"""Compile the vendor's Triton GDN kernel instead of its precompiled operator."""

from __future__ import annotations

import importlib

import torch
import torch_fant  # noqa: F401


def main() -> None:
    module = importlib.import_module(
        "iFormer.ops.triton.fla.chunk_gated_delta_rule_fwd_h"
    )

    torch.manual_seed(0)
    torch.fant.set_device(0)
    batch, tokens, key_heads, value_heads, dim = 1, 64, 4, 12, 128
    dtype, device = torch.bfloat16, "fant:0"

    k = torch.randn(batch, tokens, key_heads, dim, dtype=dtype, device=device)
    u = torch.randn(batch, tokens, value_heads, dim, dtype=dtype, device=device)
    w = torch.randn(batch, tokens, value_heads, dim, dtype=dtype, device=device)
    v_new = torch.empty_like(u)
    g = torch.randn(
        batch, tokens, value_heads, dtype=torch.float32, device=device
    ).clamp_max_(0)
    h = torch.empty(batch, 1, value_heads, dim, dim, dtype=dtype, device=device)
    initial_state = torch.zeros(
        batch, value_heads, dim, dim, dtype=torch.float32, device=device
    )
    final_state = torch.empty_like(initial_state)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    chunk_offsets = torch.tensor([0, 1], dtype=torch.int32, device=device)

    module.chunk_gated_delta_rule_fwd_h_impl(
        k,
        u,
        w,
        v_new,
        g,
        None,
        h,
        initial_state,
        final_state,
        cu_seqlens,
        chunk_offsets,
        tokens,
        value_heads,
        key_heads,
        dim,
        dim,
        64,
        batch,
        True,
        False,
        True,
        True,
        True,
    )
    torch.fant.synchronize()

    assert torch.isfinite(h).all().item()
    assert torch.isfinite(v_new).all().item()
    assert torch.isfinite(final_state).all().item()
    print("VENDOR_TRITON_FANT_CHUNK_DELTA_H_OK")


if __name__ == "__main__":
    main()
