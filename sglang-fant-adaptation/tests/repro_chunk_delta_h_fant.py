"""Reproduce the SGLang Qwen3.5/3.8 GDN state-update kernel on GR308.

This is intentionally a subprocess-level test: the current vendor Triton
compiler aborts the interpreter while lowering this kernel, so a test runner
cannot catch the failure as a Python exception.
"""

from __future__ import annotations

import os
import sys


SOURCE_ROOT = os.environ.get("SGLANG_SOURCE_ROOT", "/sglang/python")
PLUGIN_ROOT = os.environ.get("SGLANG_FANT_PLUGIN_ROOT", "/plugin")
DEPENDENCY_ROOT = os.environ.get("SGLANG_DEPENDENCY_ROOT", "/sgdeps")

# Preserve the vendor PyTorch build.  The dependency overlay deliberately does
# not contain torch/triton and is made visible only after torch is imported.
import torch

sys.path[:0] = [SOURCE_ROOT, PLUGIN_ROOT, DEPENDENCY_ROOT]
import torch_fant  # noqa: F401,E402

from sglang.kernels.ops.attention.fla.chunk_delta_h import (  # noqa: E402
    chunk_gated_delta_rule_fwd_h,
)


def main() -> None:
    torch.manual_seed(0)
    torch.fant.set_device(0)

    # Qwen3.8 GDN dimensions after TP=4: 16 key heads / 48 value heads,
    # each sharded to Hg=4 and H=12; both key/value head dimensions are 128.
    batch, tokens, key_heads, value_heads, dim = 1, 64, 4, 12, 128
    dtype = torch.bfloat16
    device = "fant:0"

    k = torch.randn(batch, tokens, key_heads, dim, dtype=dtype, device=device)
    w = torch.randn(batch, tokens, value_heads, dim, dtype=dtype, device=device)
    u = torch.randn(batch, tokens, value_heads, dim, dtype=dtype, device=device)
    g = torch.randn(
        batch, tokens, value_heads, dtype=torch.float32, device=device
    ).clamp_max_(0)
    initial_state = torch.zeros(
        1, value_heads, dim, dim, dtype=torch.float32, device=device
    )
    initial_state_indices = torch.zeros(1, dtype=torch.int64, device=device)

    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
    )
    torch.fant.synchronize()

    assert h.shape == (batch, 1, value_heads, dim, dim)
    assert v_new.shape == u.shape
    assert torch.isfinite(h).all().item()
    assert torch.isfinite(v_new).all().item()
    print("SGLANG_FANT_CHUNK_DELTA_H_OK")


if __name__ == "__main__":
    main()
