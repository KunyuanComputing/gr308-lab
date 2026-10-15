#!/usr/bin/env python3
"""Independent standard-library oracle for the frozen MiniQwen model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any


VOCAB_SIZE = 16
HIDDEN_SIZE = 8
GDN_LAYERS = 3
WEIGHT_COUNT = 456
INITIAL_RNG_STATE = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
PROMPT = [1, 4, 7]
GENERATED_STEPS = 8


def _round_shift(value: int) -> int:
    magnitude = (abs(value) + (1 << 15)) >> 16
    return magnitude if value >= 0 else -magnitude


def _i32(value: int) -> int:
    return max(-(1 << 31), min((1 << 31) - 1, value))


def _mean(values: list[int]) -> int:
    total = sum(values)
    magnitude = abs(total) // len(values)
    return magnitude if total >= 0 else -magnitude


def _rng(value: int) -> int:
    value ^= (value << 13) & MASK64
    value ^= value >> 7
    value ^= (value << 17) & MASK64
    return value & MASK64


def _matrix_vector(matrix: tuple[int, ...], vector: list[int]) -> list[int]:
    result: list[int] = []
    for row in range(HIDDEN_SIZE):
        accumulator = sum(
            matrix[row * HIDDEN_SIZE + column] * vector[column]
            for column in range(HIDDEN_SIZE)
        )
        if not -(1 << 63) <= accumulator < (1 << 63):
            raise OverflowError("int64 accumulator overflow")
        result.append(_i32(_round_shift(accumulator)))
    return result


def _initial_state() -> dict[str, Any]:
    return {
        "full_keys": [],
        "full_values": [],
        "gdn_states": [[0] * HIDDEN_SIZE for _ in range(GDN_LAYERS)],
        "rng_state": INITIAL_RNG_STATE,
    }


def _process_token(
    weights: tuple[int, ...], token_id: int, state: dict[str, Any]
) -> tuple[dict[str, Any], list[int]]:
    if not 0 <= token_id < VOCAB_SIZE:
        raise ValueError("token out of range")
    cursor = 0
    embedding_count = VOCAB_SIZE * HIDDEN_SIZE
    hidden = list(
        weights[token_id * HIDDEN_SIZE : (token_id + 1) * HIDDEN_SIZE]
    )
    cursor += embedding_count
    next_gdn: list[list[int]] = []
    for layer in range(GDN_LAYERS):
        gate = weights[cursor : cursor + HIDDEN_SIZE]
        input_diagonal = weights[cursor + HIDDEN_SIZE : cursor + 2 * HIDDEN_SIZE]
        state_diagonal = weights[
            cursor + 2 * HIDDEN_SIZE : cursor + 3 * HIDDEN_SIZE
        ]
        cursor += 3 * HIDDEN_SIZE
        layer_state: list[int] = []
        layer_hidden: list[int] = []
        for dimension in range(HIDDEN_SIZE):
            accumulator = (
                hidden[dimension] * input_diagonal[dimension]
                + state["gdn_states"][layer][dimension]
                * state_diagonal[dimension]
            )
            if not -(1 << 63) <= accumulator < (1 << 63):
                raise OverflowError("int64 accumulator overflow")
            recurrent = _i32(_round_shift(accumulator))
            gated = recurrent * gate[dimension]
            if not -(1 << 63) <= gated < (1 << 63):
                raise OverflowError("int64 accumulator overflow")
            layer_state.append(recurrent)
            layer_hidden.append(_i32(_round_shift(gated)))
        next_gdn.append(layer_state)
        hidden = layer_hidden

    matrix_count = HIDDEN_SIZE * HIDDEN_SIZE
    key = _matrix_vector(weights[cursor : cursor + matrix_count], hidden)
    cursor += matrix_count
    value = _matrix_vector(weights[cursor : cursor + matrix_count], hidden)
    cursor += matrix_count
    full_keys = [*state["full_keys"], key]
    full_values = [*state["full_values"], value]
    hidden = [
        _mean([entry[dimension] for entry in full_values])
        for dimension in range(HIDDEN_SIZE)
    ]
    logits: list[int] = []
    for token in range(VOCAB_SIZE):
        accumulator = sum(
            weights[cursor + token * HIDDEN_SIZE + dimension] * hidden[dimension]
            for dimension in range(HIDDEN_SIZE)
        )
        if not -(1 << 63) <= accumulator < (1 << 63):
            raise OverflowError("int64 accumulator overflow")
        logits.append(_i32(_round_shift(accumulator)))
    return {
        "full_keys": full_keys,
        "full_values": full_values,
        "gdn_states": next_gdn,
        "rng_state": state["rng_state"],
    }, logits


def _append_vector(encoded: bytearray, values: list[int]) -> None:
    encoded.extend(struct.pack("<I", len(values)))
    for value in values:
        encoded.extend(struct.pack("<i", value))


def _encode_state(state: dict[str, Any]) -> bytes:
    encoded = bytearray(struct.pack("<I", 1))
    encoded.extend(struct.pack("<I", len(state["full_keys"])))
    for vector in state["full_keys"]:
        _append_vector(encoded, vector)
    encoded.extend(struct.pack("<I", len(state["full_values"])))
    for vector in state["full_values"]:
        _append_vector(encoded, vector)
    encoded.extend(struct.pack("<I", len(state["gdn_states"])))
    for vector in state["gdn_states"]:
        _append_vector(encoded, vector)
    encoded.extend(struct.pack("<Q", state["rng_state"]))
    return bytes(encoded)


def build_document(weights_bytes: bytes) -> dict[str, Any]:
    if len(weights_bytes) != WEIGHT_COUNT * 4:
        raise ValueError("invalid weight byte count")
    weights = struct.unpack(f"<{WEIGHT_COUNT}i", weights_bytes)
    state = _initial_state()
    context = list(PROMPT)
    steps: list[dict[str, Any]] = []
    for _ in range(GENERATED_STEPS):
        processed = len(state["full_values"])
        if processed > len(context):
            raise ValueError("state is ahead of context")
        logits: list[int] | None = None
        for token_id in context[processed:]:
            state, logits = _process_token(weights, token_id, state)
        if logits is None:
            raise ValueError("context has no unprocessed token")
        token_id = max(range(VOCAB_SIZE), key=logits.__getitem__)
        state["rng_state"] = _rng(state["rng_state"])
        encoded_state = _encode_state(state)
        steps.append(
            {
                "canonical_digest": hashlib.sha256(encoded_state).hexdigest(),
                "canonical_state_hex": encoded_state.hex(),
                "full_keys": state["full_keys"],
                "full_values": state["full_values"],
                "gdn_states": state["gdn_states"],
                "rng_state": str(state["rng_state"]),
                "token_id": token_id,
            }
        )
        context.append(token_id)
    return {
        "arithmetic": {
            "accumulator": "signed int64",
            "format": "Q16.16",
            "round_shift": "sign(value) * ((abs(value) + 2^(bits-1)) >> bits)",
            "saturation": "signed int32",
            "gdn": "state=round(input*input_diag+state*state_diag); hidden=round(state*gate_diag)",
            "full": "append projected key/value; hidden=per-dimension integer mean of all values",
            "output": "frozen matrix logits; first token wins argmax ties",
        },
        "golden_version": 1,
        "model_id": "mini-qwen38-hybrid",
        "model_format_version": 1,
        "prompt_token_ids": PROMPT,
        "rng": {
            "advance": "once before each committed token",
            "algorithm": "xorshift64",
            "greedy_argmax_uses_rng": False,
            "initial_state": str(INITIAL_RNG_STATE),
            "operations": [
                "x ^= (x << 13) & 0xffffffffffffffff",
                "x ^= x >> 7",
                "x ^= (x << 17) & 0xffffffffffffffff",
                "x &= 0xffffffffffffffff",
            ],
        },
        "state_digest_encoding": {
            "algorithm": "sha256",
            "byte_order": "little-endian",
            "fields": [
                "uint32 model_format_version",
                "vector<vector<int32>> full_keys",
                "vector<vector<int32>> full_values",
                "vector<vector<int32>> gdn_states",
                "uint64 rng_state",
            ],
            "vector_length": "uint32 before every vector",
        },
        "steps": steps,
        "topology": ["GDN", "GDN", "GDN", "FULL"],
        "weight_layout": [
            {"count": 128, "name": "token_embedding"},
            {"count": 72, "name": "gdn_diagonal_gate_input_state"},
            {"count": 128, "name": "full_key_value_matrices"},
            {"count": 128, "name": "output_logits_matrix"},
        ],
        "weight_sha256": hashlib.sha256(weights_bytes).hexdigest(),
    }


def _canonical_json(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _manifest_is_valid(path: Path, weights_bytes: bytes) -> bool:
    fields: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if not separator or name in fields:
                return False
            fields[name] = value.strip().strip('"')
    except (OSError, UnicodeError):
        return False
    return fields == {
        "schema_version": "1",
        "model_id": "mini-qwen38-hybrid",
        "model_format_version": "1",
        "topology": "GDN,GDN,GDN,FULL",
        "dtype": "Q16_16",
        "vocab_size": "16",
        "hidden_size": "8",
        "max_context_tokens": "64",
        "weight_sha256": hashlib.sha256(weights_bytes).hexdigest(),
        "golden_version": "1",
        "hal_abi_major": "1",
        "hal_abi_minor": "0",
        "required_capability": "EXECUTABLE_LAUNCH",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--emit", action="store_true")
    args = parser.parse_args()
    golden_path = args.check
    weights_path = args.weights
    if golden_path is not None and weights_path is None:
        weights_path = golden_path.with_name("weights.i32le")
    manifest_path = args.manifest
    if golden_path is not None and manifest_path is None:
        manifest_path = golden_path.with_name("manifest.textproto")
    if weights_path is None or (golden_path is None and not args.emit):
        parser.error("use --check GOLDEN or --emit --weights WEIGHTS")
    weights_bytes = weights_path.read_bytes()
    if manifest_path is not None and not _manifest_is_valid(
        manifest_path, weights_bytes
    ):
        print("manifest does not match independent oracle inputs", file=sys.stderr)
        return 1
    actual = build_document(weights_bytes)
    if args.emit:
        sys.stdout.write(_canonical_json(actual))
    if golden_path is not None:
        expected = json.loads(golden_path.read_text(encoding="utf-8"))
        if expected != actual:
            print("golden.json does not match independent oracle", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
