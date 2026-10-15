#!/usr/bin/env python3
"""Report the FTUCA SDK inputs used by the standalone FANT SDPA extension."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


SDK_PATHS = (
    "/usr/local/fant/bin/hipcc",
    "/usr/local/fant/include/hip/hip_bfloat16.h",
    "/usr/local/fant/include/hip/devicelib/bfloat16/bfloat162_arithemtic.h",
    "/usr/local/fant/include/fhblas/fhblas.h",
    "/usr/local/fant/lib/libfhblas.so.4.0",
)


def sdk_path_status(path_exists: Callable[[str], bool] = os.path.exists) -> dict[str, bool]:
    """Return existence flags in the fixed, reproducible SDK path order."""
    return {path: bool(path_exists(path)) for path in SDK_PATHS}


def read_compiler_version(compiler: str) -> str:
    """Return the first nonempty line from ``hipcc --version``."""
    completed = subprocess.run(
        [compiler, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout or completed.stderr
    return next((line.strip() for line in output.splitlines() if line.strip()), "")


def collect_record(
    *,
    torch_module: Any,
    torch_fant_module: Any,
    library_path: Path | None,
    path_exists: Callable[[str], bool] = os.path.exists,
    compiler_version: Callable[[str], str] = read_compiler_version,
) -> dict[str, Any]:
    """Collect probe facts, loading the custom operator library when supplied."""
    if library_path is not None:
        torch_module.ops.load_library(str(library_path))
    namespace = getattr(torch_module.ops, "fant_sdpa", None)
    return {
        "paths": sdk_path_status(path_exists),
        "torch_version": torch_module.__version__,
        "torch_fant_file": torch_fant_module.__file__,
        "compiler_version": compiler_version(SDK_PATHS[0]),
        "copy_smoke_available": bool(
            namespace is not None and hasattr(namespace, "copy_smoke")
        ),
    }


def format_record(record: dict[str, Any]) -> str:
    """Format the machine readable single line probe record."""
    return "FANT_SDPA_SDK " + json.dumps(record, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    arguments = parser.parse_args(argv)
    try:
        import torch
        import torch_fant
    except ModuleNotFoundError as error:
        raise RuntimeError("FANT SDPA SDK probe requires torch and torch_fant") from error

    record = collect_record(
        torch_module=torch,
        torch_fant_module=torch_fant,
        library_path=arguments.library,
    )
    print(format_record(record))


if __name__ == "__main__":
    main()
