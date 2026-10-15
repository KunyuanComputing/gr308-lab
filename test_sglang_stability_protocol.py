#!/usr/bin/env python3
"""Static contract test for the vLLM-comparable SGLang stability runner."""

import pathlib
import sys


script = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")

required = {
    "rapid-shape input": "--input-tokens 128 --output-tokens 64",
    "main sample count": "--concurrency 64 --requests 1024",
    "conditional extension": "measurement_duration_s",
    "extension sample count": "--concurrency 64 --requests 512",
    "post sentinel": "status correctness_post running",
    "exact main validation": 'validate_case "$case_dir/summary.json" 1024',
}

missing = [name for name, needle in required.items() if needle not in script]
if missing:
    raise AssertionError("missing stability protocol clauses: " + ", ".join(missing))

for forbidden in ("--input-tokens 1024 --output-tokens 256",):
    if forbidden in script:
        raise AssertionError(f"formal workload leaked into stability protocol: {forbidden}")

print("stability protocol contract passed")
