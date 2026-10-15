#!/usr/bin/env python3
"""Resume the unrun Qwen3.6-35B-A3B H--J short tuning stages."""

from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import gr308_adaptive_short_matrix as matrix
import gr308_parallel_goal as base


OUTPUT_ROOT = Path("/data/infra/benchmarks/gr308-35b-followup-resume-20260915")
G1_RESULT = Path(
    "/data/infra/benchmarks/gr308-35b-followup-short-20260915/"
    "g1-qwen3-6-35b-a3b/result.json"
)
BACKGROUND_CONTAINERS = (
    "glm53-int4-download",
    "dsv4-gptq-download",
    "glm53-gptq-convert",
)


def background_job(name: str) -> base.Job:
    return base.Job("background-no-gpu", (6, 7), 0, "0", OUTPUT_ROOT / name,
                    name, 0, "background")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    allowed = matrix._allowed_jobs(OUTPUT_ROOT)
    allowed.extend(background_job(name) for name in BACKGROUND_CONTAINERS)

    g1_spec = matrix.Spec(
        "G1-reference", "Qwen3.6-35B-A3B", (0, 1), 22100, "0-7",
        3, (4, 8, 16, 32, 64), 2048, 1088, None,
    )
    g1 = matrix._row_from_result("G1-reference", G1_RESULT, g1_spec)
    rows = [g1]

    try:
        for mtp, graph in (
            (2, (4, 8, 16, 32, 64)),
            (4, (4, 8, 16, 32, 64, 128)),
        ):
            rows.append(matrix.run_spec(
                matrix._spec_like(g1_spec, f"H-mtp{mtp}", mtp=mtp, graph=graph),
                OUTPUT_ROOT, allowed,
            ))

        best_mtp = matrix.choose_best(rows)
        mtp_source = matrix.Spec(**best_mtp["spec"])
        i_spec = matrix._spec_like(mtp_source, "I-triton", moe_backend="triton")
        i_row = matrix.run_spec(i_spec, OUTPUT_ROOT, allowed)
        rows.append(i_row)

        best_backend = matrix.choose_best([best_mtp, i_row])
        backend_source = matrix.Spec(**best_backend["spec"])
        for name, gemm, rms in (
            ("J-gemm", True, False),
            ("J-allreduce-rms", False, True),
            ("J-both", True, True),
        ):
            rows.append(matrix.run_spec(
                matrix._spec_like(
                    backend_source, name,
                    fuse_gemm_comms=gemm,
                    fuse_allreduce_rms=rms,
                ),
                OUTPUT_ROOT, allowed,
            ))

        payload = {
            "status": "complete",
            "rows": rows,
            "best": matrix.choose_best(rows),
            "finished": time.time(),
        }
        exit_code = 0
    except Exception as exc:
        payload = {
            "status": "failed",
            "rows": rows,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "finished": time.time(),
        }
        exit_code = 1

    (OUTPUT_ROOT / "matrix-summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
