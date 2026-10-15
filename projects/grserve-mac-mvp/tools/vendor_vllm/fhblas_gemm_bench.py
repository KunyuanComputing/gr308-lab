"""Benchmark the native fhBLAS BF16 GEMM path on FANT GPUs.

The benchmark keeps vLLM's original ``[out_features, in_features]`` weight
layout and maps PyTorch's row-major ``X @ W.T`` operation onto fhBLAS's
column-major GEMM ABI.  This lets us test vendor algorithm flags without
allocating a second transposed copy of the model weights.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


FHBLAS_OP_N = 111
FHBLAS_OP_T = 112
FHBLAS_R_32F = 151
FHBLAS_R_16B = 168
FHBLAS_GEMM_DEFAULT = 160
FANTGEMM_ROW_MAJOR = 0
FANTGEMM_COL_MAJOR = 1
FANTGEMM_BF16 = 6


@dataclass
class Measurement:
    m: int
    implementation: str
    flags: int | None
    milliseconds: float
    tflops: float
    max_abs_error: float


class Fhblas:
    def __init__(self, library: str) -> None:
        self.lib = ctypes.CDLL(library)
        self.handle = ctypes.c_void_p()

        self.lib.fhblasCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.fhblasCreate.restype = ctypes.c_int
        self.lib.fhblasDestroy.argtypes = [ctypes.c_void_p]
        self.lib.fhblasDestroy.restype = ctypes.c_int
        self.lib.fhblasGemmExWithFlags.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.fhblasGemmExWithFlags.restype = ctypes.c_int

        self._check(self.lib.fhblasCreate(ctypes.byref(self.handle)), "create")

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"fhBLAS {operation} failed with status {status}")

    def close(self) -> None:
        if self.handle:
            self._check(self.lib.fhblasDestroy(self.handle), "destroy")
            self.handle = ctypes.c_void_p()

    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        *,
        flags: int,
    ) -> None:
        """Compute row-major ``output = x @ weight.T`` without a weight copy."""

        m, k = x.shape
        n, weight_k = weight.shape
        if weight_k != k or output.shape != (m, n):
            raise ValueError("incompatible matrix shapes")
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)

        # Row-major buffers are the transpose of their column-major views:
        # Y.T[N,M] = op(W_col[K,N], T) @ op(X_col[K,M], N).
        status = self.lib.fhblasGemmExWithFlags(
            self.handle,
            FHBLAS_OP_T,
            FHBLAS_OP_N,
            n,
            m,
            k,
            ctypes.byref(alpha),
            ctypes.c_void_p(weight.data_ptr()),
            FHBLAS_R_16B,
            k,
            ctypes.c_void_p(x.data_ptr()),
            FHBLAS_R_16B,
            k,
            ctypes.byref(beta),
            ctypes.c_void_p(output.data_ptr()),
            FHBLAS_R_16B,
            n,
            FHBLAS_R_32F,
            FHBLAS_GEMM_DEFAULT,
            flags,
        )
        self._check(status, f"gemm(flags={flags})")


class Fantgemm:
    """Thin binding for the lower-level layout-aware fantGEMM entry point."""

    SYMBOL = "_Z16launch_fant_gemmPvPKvS1_S1_S_8format_tS2_8layout_tS3_S3_jjjjjjjjmmmmff"

    def __init__(self, library: str) -> None:
        self.lib = ctypes.CDLL(library)
        self.launch = getattr(self.lib, self.SYMBOL)
        self.launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.launch.restype = ctypes.c_int

    def linear(
        self, x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor
    ) -> None:
        m, k = x.shape
        n, weight_k = weight.shape
        if weight_k != k or output.shape != (m, n):
            raise ValueError("incompatible matrix shapes")
        status = self.launch(
            ctypes.c_void_p(torch.fant.current_stream().fant_stream),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            FANTGEMM_BF16,
            FANTGEMM_BF16,
            FANTGEMM_ROW_MAJOR,
            FANTGEMM_COL_MAJOR,
            FANTGEMM_ROW_MAJOR,
            m,
            n,
            k,
            1,
            k,
            k,
            n,
            n,
            0,
            0,
            0,
            0,
            1.0,
            0.0,
        )
        if status != 0:
            raise RuntimeError(f"fantGEMM launch failed with status {status}")


def elapsed_ms(callable_, repetitions: int) -> float:
    for _ in range(5):
        callable_()
    torch.fant.synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        callable_()
    torch.fant.synchronize()
    return (time.perf_counter() - started) * 1000.0 / repetitions


def repetitions_for(m: int) -> int:
    if m <= 16:
        return 200
    if m <= 256:
        return 80
    return 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--in-features", type=int, default=8704)
    parser.add_argument("--out-features", type=int, default=5120)
    parser.add_argument("--m", type=int, nargs="+", default=[16, 256, 2048])
    parser.add_argument("--flags", type=int, nargs="+", default=[0, 2])
    parser.add_argument(
        "--library", default="/usr/local/fant/lib/libfhblas.so"
    )
    parser.add_argument(
        "--fantgemm-library", default="/usr/local/fant/lib/libfantgemm.so"
    )
    args = parser.parse_args()

    torch.fant.set_device(args.device)
    device = f"fant:{args.device}"
    weight = torch.randn(
        (args.out_features, args.in_features),
        dtype=torch.bfloat16,
        device=device,
    )
    fhblas = Fhblas(args.library)
    fantgemm = Fantgemm(args.fantgemm_library)
    results: list[Measurement] = []

    try:
        for m in args.m:
            x = torch.randn(
                (m, args.in_features), dtype=torch.bfloat16, device=device
            )
            expected = F.linear(x, weight)
            repetitions = repetitions_for(m)
            baseline_ms = elapsed_ms(lambda: F.linear(x, weight), repetitions)
            flop = 2.0 * m * args.in_features * args.out_features
            results.append(
                Measurement(
                    m=m,
                    implementation="torch.nn.functional.linear",
                    flags=None,
                    milliseconds=baseline_ms,
                    tflops=flop / (baseline_ms * 1e9),
                    max_abs_error=0.0,
                )
            )

            transposed_weight = weight.transpose(0, 1).contiguous()
            transposed_ms = elapsed_ms(
                lambda: torch.mm(x, transposed_weight), repetitions
            )
            transposed_output = torch.mm(x, transposed_weight)
            torch.fant.synchronize()
            transposed_error = (
                (transposed_output.float() - expected.float()).abs().max().item()
            )
            results.append(
                Measurement(
                    m=m,
                    implementation="torch.mm_transposed_storage",
                    flags=None,
                    milliseconds=transposed_ms,
                    tflops=flop / (transposed_ms * 1e9),
                    max_abs_error=transposed_error,
                )
            )

            for flags in args.flags:
                output = torch.empty(
                    (m, args.out_features), dtype=torch.bfloat16, device=device
                )
                fhblas.linear(x, weight, output, flags=flags)
                torch.fant.synchronize()
                max_abs_error = (output.float() - expected.float()).abs().max().item()
                native_ms = elapsed_ms(
                    lambda: fhblas.linear(x, weight, output, flags=flags),
                    repetitions,
                )
                results.append(
                    Measurement(
                        m=m,
                        implementation="fhblasGemmExWithFlags",
                        flags=flags,
                        milliseconds=native_ms,
                        tflops=flop / (native_ms * 1e9),
                        max_abs_error=max_abs_error,
                    )
                )

            direct_output = torch.empty(
                (m, args.out_features), dtype=torch.bfloat16, device=device
            )
            fantgemm.linear(x, weight, direct_output)
            torch.fant.synchronize()
            direct_error = (
                (direct_output.float() - expected.float()).abs().max().item()
            )
            direct_ms = elapsed_ms(
                lambda: fantgemm.linear(x, weight, direct_output), repetitions
            )
            results.append(
                Measurement(
                    m=m,
                    implementation="launch_fant_gemm",
                    flags=None,
                    milliseconds=direct_ms,
                    tflops=flop / (direct_ms * 1e9),
                    max_abs_error=direct_error,
                )
            )
    finally:
        fhblas.close()

    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
