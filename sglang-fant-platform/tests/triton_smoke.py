"""Minimal vendor Triton execution probe; not part of the regular unit suite."""

import torch
import torch_fant  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets) + tl.load(y + offsets))


x = torch.arange(16, dtype=torch.float32, device="fant")
y = torch.ones(16, dtype=torch.float32, device="fant")
out = torch.empty_like(x)
add_kernel[(1,)](x, y, out, n=16)
torch.fant.synchronize()
assert torch.equal(out.cpu(), torch.arange(16, dtype=torch.float32) + 1)
print("TRITON_FANT_OK", triton.__version__)
