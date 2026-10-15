#!/usr/bin/env python3
"""Generate the opt-in Fantasy diffusion registry overlay."""

import argparse
import json
from pathlib import Path

from h3_sglang_fant.registry_builder import build_registry_overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    digest = build_registry_overlay(args.source, args.output)
    print(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "output": str(args.output.resolve()),
                "sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
