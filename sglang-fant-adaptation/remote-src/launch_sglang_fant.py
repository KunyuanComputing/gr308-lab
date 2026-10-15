#!/usr/bin/env python3
"""Launch upstream SGLang with vendor torch and isolated Python dependencies."""

from __future__ import annotations

import os
import importlib
import runpy
import sys


def main() -> None:
    source_root = os.environ.get("SGLANG_SOURCE_ROOT", "/sglang/python")
    plugin_root = os.environ.get("SGLANG_FANT_PLUGIN_ROOT", "/plugin")
    dependency_root = os.environ.get("SGLANG_DEPENDENCY_ROOT", "/sgdeps")

    # Load vendor torch before exposing the dependency overlay, which excludes
    # torch itself. This preserves torch.fant and all vendor compiled operators.
    import torch  # noqa: F401

    sys.path[:0] = [source_root, plugin_root, dependency_root]

    # torch imports the vendor image's older typing_extensions before the
    # isolated SGLang dependency overlay is visible.  Rebind that one pure
    # Python module so FastAPI/AnyIO can use typing_extensions.sentinel.
    sys.modules.pop("typing_extensions", None)
    importlib.import_module("typing_extensions")

    import torch_fant  # noqa: F401

    # The vendor Triton is older than current SGLang.  Some kernels mention
    # CUDA programmatic-dependent-launch helpers inside compile-time-disabled
    # branches; Triton's dependency parser still resolves those names.  Supply
    # inert JIT helpers on hardware where PDL is unavailable.
    import triton
    import triton.language as tl

    if not hasattr(tl.extra.cuda, "gdc_wait"):
        @triton.jit
        def _fant_gdc_wait():
            pass

        @triton.jit
        def _fant_gdc_launch_dependents():
            pass

        tl.extra.cuda.gdc_wait = _fant_gdc_wait
        tl.extra.cuda.gdc_launch_dependents = _fant_gdc_launch_dependents

    os.environ.setdefault("SGLANG_PLATFORM", "gr308_fant")
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
