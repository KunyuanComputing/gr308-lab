#!/usr/bin/env bash
set -Eeuo pipefail
exec uv run --locked python tools/verify.py --profile full
