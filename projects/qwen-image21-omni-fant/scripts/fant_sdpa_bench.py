#!/usr/bin/env python3
"""Forward-only FTUCA SDPA benchmark gates and hardware entry point."""

from __future__ import annotations

import argparse
import math
import os
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


SUPPORTED_SEQUENCE_LENGTHS = frozenset({16, 128, 1024, 4096})


@dataclass(frozen=True)
class BenchmarkConfig:
    seq_len: int
    heads: int
    head_dim: int


def _compare_without_torch(
    reference: Iterable[float], candidate: Iterable[float]
) -> dict[str, float | bool]:
    reference_values = [float(value) for value in reference]
    candidate_values = [float(value) for value in candidate]
    if len(reference_values) != len(candidate_values):
        raise ValueError("reference and candidate must contain the same number of values")
    if not reference_values:
        raise ValueError("reference and candidate must not be empty")

    deltas = [
        abs(reference_value - candidate_value)
        for reference_value, candidate_value in zip(reference_values, candidate_values)
    ]
    return {
        "finite": all(
            math.isfinite(value)
            for value in reference_values + candidate_values + deltas
        ),
        "mae": sum(deltas) / len(deltas),
        "max_abs": max(deltas),
        "exact_fraction": sum(delta == 0.0 for delta in deltas) / len(deltas),
    }


def compare(reference: Any, candidate: Any) -> dict[str, float | bool]:
    """Compare outputs with vectorized on-device reductions when Torch is present."""
    try:
        import torch
    except ModuleNotFoundError:
        return _compare_without_torch(reference, candidate)

    reference_values = torch.as_tensor(reference).float()
    candidate_values = torch.as_tensor(candidate).float()
    if reference_values.shape != candidate_values.shape:
        raise ValueError("reference and candidate must have the same shape")
    if reference_values.numel() == 0:
        raise ValueError("reference and candidate must not be empty")

    deltas = torch.sub(reference_values, candidate_values).abs()
    all_finite = torch.isfinite(reference_values).all() & torch.isfinite(
        candidate_values
    ).all()
    return {
        "finite": bool(all_finite.item()),
        "mae": deltas.mean().item(),
        "max_abs": deltas.max().item(),
        "exact_fraction": deltas.eq(0).float().mean().item(),
    }


def passes_gates(
    metrics: dict[str, float | bool], candidate_median: float, reference_median: float
) -> bool:
    """Return whether numerical and five-percent speed improvement gates pass."""
    return bool(
        metrics["finite"]
        and metrics["mae"] <= 0.01
        and metrics["max_abs"] <= 0.25
        and candidate_median <= 0.95 * reference_median
    )


def percentile_summary(samples: Iterable[float]) -> dict[str, object]:
    """Summarize samples with median and nearest-rank p10 and p90."""
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("samples must not be empty")
    ordered = sorted(values)

    def nearest_rank(percentile: float) -> float:
        return ordered[math.ceil(percentile * len(ordered)) - 1]

    return {
        "samples": values,
        "median": statistics.median(values),
        "p10": nearest_rank(0.10),
        "p90": nearest_rank(0.90),
    }


def parse_config(
    argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> BenchmarkConfig:
    """Parse CLI overrides and environment defaults, then validate the fixed shape."""
    environment = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=environment.get("SEQ_LEN", "1024"))
    parser.add_argument("--heads", type=int, default=environment.get("HEADS", "16"))
    parser.add_argument("--head-dim", type=int, default=environment.get("HEAD_DIM", "128"))
    arguments = parser.parse_args(argv)

    if arguments.seq_len not in SUPPORTED_SEQUENCE_LENGTHS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_SEQUENCE_LENGTHS))
        raise ValueError(f"SEQ_LEN must be one of {supported}")
    if arguments.heads != 16:
        raise ValueError("HEADS must equal 16")
    if arguments.head_dim != 128:
        raise ValueError("HEAD_DIM must equal 128")
    return BenchmarkConfig(arguments.seq_len, arguments.heads, arguments.head_dim)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Validate the requested shape before a later task supplies hardware execution."""
    parse_config(argv)
    raise RuntimeError("hardware benchmark execution is not implemented yet")


if __name__ == "__main__":
    main()
