#!/usr/bin/env python3
"""Fail-closed validation for one GRServe full verification summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REQUIRED_PASS_GATES = frozenset(
    {
        "preflight",
        "Debug",
        "Release",
        "C11 ABI",
        "C++",
        "Python",
        "proto",
        "manifest",
        "golden",
        "fault matrix",
        "ASan/UBSan",
        "Linux amd64",
        "API E2E",
    }
)
TSAN_GATE = "TSan"
TSAN_ALLOWLIST_REASON = "TSAN_RUNTIME_UNSUPPORTED_APPLE_CLANG"
EXACT_GATES = REQUIRED_PASS_GATES | {TSAN_GATE}


class AcceptanceError(ValueError):
    """The supplied summary cannot certify the Mac MVP."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(_value: str) -> object:
    raise AcceptanceError("NON_FINITE_NUMBER")


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise AcceptanceError("NON_FINITE_NUMBER")
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise AcceptanceError("INVALID_JSON_TYPE")
        for item in value.values():
            _validate_json_value(item)
        return
    raise AcceptanceError("INVALID_JSON_TYPE")


def load_full_summary(path: Path) -> dict[str, object]:
    """Load JSON without accepting duplicate keys or non-JSON numeric values."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except AcceptanceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AcceptanceError("INVALID_SUMMARY_JSON") from error
    if type(value) is not dict:
        raise AcceptanceError("INVALID_SUMMARY_TYPE")
    _validate_json_value(value)
    return value


def _validate_gate(name: str, value: object) -> None:
    if type(value) is not dict:
        raise AcceptanceError(f"INVALID_GATE:{name}")
    gate = value
    if set(gate) != {"status"} or gate.get("status") != "PASS":
        raise AcceptanceError(f"GATE_NOT_PASS:{name}")


def validate_full_summary(summary: Mapping[str, object]) -> bool:
    """Require exactly the full-profile gate contract and no critical skips."""
    if type(summary) is not dict:
        raise AcceptanceError("INVALID_SUMMARY_TYPE")
    _validate_json_value(summary)
    if summary.get("profile") != "full" or type(summary.get("profile")) is not str:
        raise AcceptanceError("NOT_FULL_PROFILE")
    if summary.get("status") != "PASS" or type(summary.get("status")) is not str:
        raise AcceptanceError("SUMMARY_NOT_PASS")
    critical = summary.get("critical_skip_count")
    if type(critical) is not int or critical != 0:
        raise AcceptanceError("CRITICAL_SKIP_COUNT_NOT_ZERO")
    gates = summary.get("gates")
    if type(gates) is not dict:
        raise AcceptanceError("INVALID_GATES_TYPE")
    if set(gates) != EXACT_GATES:
        raise AcceptanceError("GATE_SET_MISMATCH")
    for name in REQUIRED_PASS_GATES:
        _validate_gate(name, gates[name])
    tsan = gates[TSAN_GATE]
    if type(tsan) is not dict:
        raise AcceptanceError("INVALID_GATE:TSan")
    if tsan == {"status": "PASS"}:
        return True
    if tsan == {"status": "SKIP", "reason": TSAN_ALLOWLIST_REASON}:
        return True
    raise AcceptanceError("GATE_NOT_PASS:TSan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        validate_full_summary(load_full_summary(arguments.summary))
    except AcceptanceError as error:
        print(f"NOT_ACCEPTED:{error}", file=sys.stderr)
        return 1
    print("ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
