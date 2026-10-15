#!/usr/bin/env python3
"""Launch upstream SGLang with vendor torch and isolated Python dependencies."""

from __future__ import annotations

import os
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
    import torch_fant  # noqa: F401

    os.environ.setdefault("SGLANG_PLATFORM", "gr308_fant")
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
