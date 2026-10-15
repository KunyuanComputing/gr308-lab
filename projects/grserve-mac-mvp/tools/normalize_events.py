#!/usr/bin/env python3
"""Strictly normalize public GRServe event JSONL and compute its digest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


MAX_LINE_BYTES = 1024 * 1024
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_EVENTS = 100_000
DIAGNOSTIC_FIELDS = frozenset({"monotonicTick", "runId"})
SEMANTIC_FIELDS = frozenset(
    {
        "schemaVersion",
        "requestId",
        "transactionId",
        "rank",
        "epoch",
        "state",
        "error",
        "sequenceNumber",
        "delivery",
    }
)
ALL_FIELDS = SEMANTIC_FIELDS | DIAGNOSTIC_FIELDS
GATE_FIELDS = frozenset(
    {
        "schemaVersion",
        "monotonicTick",
        "runId",
        "gateId",
        "status",
        "reason",
        "commandId",
        "faultPlanSeed",
    }
)
GATE_STATUSES = frozenset(
    {
        "VERIFICATION_GATE_STATUS_PASS",
        "VERIFICATION_GATE_STATUS_FAIL",
        "VERIFICATION_GATE_STATUS_SKIP",
        "VERIFICATION_GATE_STATUS_NOT_RUN",
        "VERIFICATION_GATE_STATUS_INCOMPLETE",
    }
)
STATES = frozenset(
    {
        "EVENT_STATE_CREATED",
        "EVENT_STATE_RESERVED",
        "EVENT_STATE_PREPARED",
        "EVENT_STATE_EXECUTED",
        "EVENT_STATE_COMMIT_DECIDED",
        "EVENT_STATE_LOCAL_COMMITTED",
        "EVENT_STATE_PUBLISHED",
        "EVENT_STATE_ABORTED",
        "EVENT_STATE_FATAL",
    }
)
ERRORS = frozenset(
    {
        "EVENT_ERROR_CODE_NONE",
        "EVENT_ERROR_CODE_INVALID_ARGUMENT",
        "EVENT_ERROR_CODE_INCOMPATIBLE_VERSION",
        "EVENT_ERROR_CODE_UNSUPPORTED_CAPABILITY",
        "EVENT_ERROR_CODE_RESOURCE_EXHAUSTED",
        "EVENT_ERROR_CODE_DEADLINE_EXCEEDED",
        "EVENT_ERROR_CODE_RANK_FAILURE",
        "EVENT_ERROR_CODE_REPLICA_FATAL",
        "EVENT_ERROR_CODE_INTERNAL",
    }
)
DELIVERIES = frozenset(
    {
        "DELIVERY_OUTCOME_NONE",
        "DELIVERY_OUTCOME_DELIVERED",
        "DELIVERY_OUTCOME_DROPPED_CLIENT_DISCONNECT",
    }
)
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
DECIMAL = re.compile(r"^(0|[1-9][0-9]*)$")


class NormalizationError(ValueError):
    """The event stream is not valid canonical-input JSONL."""


@dataclass(frozen=True)
class NormalizedEvents:
    canonical: bytes
    sha256: str
    event_count: int


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NormalizationError("duplicate JSON object key")
        result[key] = value
    return result


def _bounded_json_int(value: str) -> int:
    if len(value) > 128:
        raise NormalizationError("JSON integer is too large")
    try:
        return int(value)
    except ValueError as error:
        raise NormalizationError("invalid JSON integer") from error


def _bounded_json_float(value: str) -> float:
    if len(value) > 128:
        raise NormalizationError("JSON number is too large")
    try:
        parsed = float(value)
    except ValueError as error:
        raise NormalizationError("invalid JSON number") from error
    if not math.isfinite(parsed):
        raise NormalizationError("non-finite JSON number")
    return parsed


def _uint64_string(value: object, *, nonzero: bool = False) -> bool:
    if (
        not isinstance(value, str)
        or len(value) > 20
        or DECIMAL.fullmatch(value) is None
        or (len(value) == 20 and value > "18446744073709551615")
    ):
        return False
    integer = int(value)
    return integer <= (1 << 64) - 1 and (not nonzero or integer != 0)


def _validate(event: object) -> dict[str, object]:
    if isinstance(event, dict) and set(event) == GATE_FIELDS:
        return _validate_gate(event)
    if not isinstance(event, dict) or set(event) != ALL_FIELDS:
        raise NormalizationError("event has unknown, missing, or non-object structure")
    if type(event["schemaVersion"]) is not int or event["schemaVersion"] != 1:
        raise NormalizationError("unsupported event schema version")
    if not _uint64_string(event["monotonicTick"]):
        raise NormalizationError("invalid monotonic tick")
    if not isinstance(event["runId"], str) or RUN_ID.fullmatch(event["runId"]) is None:
        raise NormalizationError("invalid run id")
    for field in ("requestId", "transactionId", "epoch", "sequenceNumber"):
        if not _uint64_string(event[field], nonzero=True):
            raise NormalizationError(f"invalid {field}")
    rank = event["rank"]
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank <= (1 << 32) - 1
    ):
        raise NormalizationError("invalid rank")
    if event["state"] not in STATES:
        raise NormalizationError("invalid state")
    if event["error"] not in ERRORS:
        raise NormalizationError("invalid error")
    if event["delivery"] not in DELIVERIES:
        raise NormalizationError("invalid delivery")
    return {key: value for key, value in event.items() if key in SEMANTIC_FIELDS}


def _validate_gate(event: dict[str, object]) -> dict[str, object]:
    if type(event["schemaVersion"]) is not int or event["schemaVersion"] != 1:
        raise NormalizationError("unsupported event schema version")
    if not _uint64_string(event["monotonicTick"]):
        raise NormalizationError("invalid monotonic tick")
    if not isinstance(event["runId"], str) or RUN_ID.fullmatch(event["runId"]) is None:
        raise NormalizationError("invalid run id")
    if not isinstance(event["gateId"], str) or not event["gateId"]:
        raise NormalizationError("invalid gate id")
    if event["status"] not in GATE_STATUSES:
        raise NormalizationError("invalid gate status")
    if not isinstance(event["reason"], str):
        raise NormalizationError("invalid gate reason")
    command_ids = event["commandId"]
    if (
        not isinstance(command_ids, list)
        or not all(isinstance(item, str) and item for item in command_ids)
        or len(command_ids) != len(set(command_ids))
    ):
        raise NormalizationError("invalid gate command ids")
    if not _uint64_string(event["faultPlanSeed"]):
        raise NormalizationError("invalid fault plan seed")
    return {
        key: value
        for key, value in event.items()
        if key not in {"monotonicTick", "runId"}
    }


def normalize_events(content: bytes) -> NormalizedEvents:
    if not content or len(content) > MAX_INPUT_BYTES or not content.endswith(b"\n"):
        raise NormalizationError("event input is empty, too large, or unterminated")
    lines = content.splitlines()
    if not lines or len(lines) > MAX_EVENTS:
        raise NormalizationError("invalid event count")
    normalized: list[dict[str, object]] = []
    previous_tick = -1
    for line in lines:
        if not line or len(line) > MAX_LINE_BYTES:
            raise NormalizationError("event line is empty or too large")
        try:
            decoded = line.decode("utf-8")
            event = json.loads(
                decoded,
                object_pairs_hook=_unique_object,
                parse_int=_bounded_json_int,
                parse_float=_bounded_json_float,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    NormalizationError("non-finite JSON number")
                ),
            )
        except (
            UnicodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
        ) as error:
            raise NormalizationError("invalid event JSON") from error
        validated = _validate(event)
        assert isinstance(event, dict)
        tick = int(str(event["monotonicTick"]))
        if tick <= previous_tick:
            raise NormalizationError("monotonic ticks are not strictly increasing")
        previous_tick = tick
        normalized.append(validated)
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return NormalizedEvents(
        canonical=canonical,
        sha256=hashlib.sha256(canonical).hexdigest(),
        event_count=len(normalized),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        result = normalize_events(arguments.input.read_bytes())
        if arguments.output is not None:
            arguments.output.write_bytes(result.canonical + b"\n")
        print(result.sha256)
        return 0
    except (OSError, NormalizationError) as error:
        print(f"EVENT_NORMALIZATION_FAILED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
