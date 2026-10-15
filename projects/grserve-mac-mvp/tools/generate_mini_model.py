#!/usr/bin/env python3
"""Generate deterministic MiniQwen assets under an existing real parent directory."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import struct
import sys
from typing import Callable, cast, NamedTuple


VOCAB_SIZE = 16
HIDDEN_SIZE = 8
GDN_LAYERS = 3
EMBEDDING_WEIGHTS = VOCAB_SIZE * HIDDEN_SIZE
GDN_WEIGHTS = GDN_LAYERS * 3 * HIDDEN_SIZE
FULL_WEIGHTS = 2 * HIDDEN_SIZE * HIDDEN_SIZE
LOGIT_WEIGHTS = VOCAB_SIZE * HIDDEN_SIZE
WEIGHT_COUNT = EMBEDDING_WEIGHTS + GDN_WEIGHTS + FULL_WEIGHTS + LOGIT_WEIGHTS
MASK64 = (1 << 64) - 1
INITIAL_RNG_STATE = 0x9E3779B97F4A7C15
ASSET_NAMES = ("weights.i32le", "golden.json", "manifest.textproto")
RENAME_EXCHANGE = 0x00000002
RENAME_SWAP = 0x00000002


class OwnedDirectory(NamedTuple):
    path: Path
    name: str
    device: int
    inode: int


def coefficient(index: int) -> int:
    return (((index * 17 + 3) % 23) - 11) * 1024


def round_shift(value: int, bits: int) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")
    magnitude = (abs(value) + (1 << (bits - 1))) >> bits
    return magnitude if value >= 0 else -magnitude


def saturate_i32(value: int) -> int:
    return max(-(1 << 31), min((1 << 31) - 1, value))


def argmax_first(values: list[int]) -> int:
    if not values:
        raise ValueError("argmax requires at least one value")
    return max(range(len(values)), key=values.__getitem__)


def xorshift64(state: int) -> int:
    state &= MASK64
    state ^= (state << 13) & MASK64
    state ^= state >> 7
    state ^= (state << 17) & MASK64
    return state & MASK64


def integer_mean(values: list[int]) -> int:
    if not values:
        raise ValueError("mean requires at least one value")
    total = sum(values)
    magnitude = abs(total) // len(values)
    return magnitude if total >= 0 else -magnitude


def decode_weights(contents: bytes) -> tuple[int, ...]:
    if len(contents) != WEIGHT_COUNT * 4:
        raise ValueError("invalid weight byte count")
    return struct.unpack(f"<{WEIGHT_COUNT}i", contents)


def _matrix_vector(matrix: tuple[int, ...], vector: list[int]) -> list[int]:
    if len(matrix) != HIDDEN_SIZE * HIDDEN_SIZE or len(vector) != HIDDEN_SIZE:
        raise ValueError("invalid matrix or vector size")
    result = []
    for row in range(HIDDEN_SIZE):
        accumulator = sum(
            matrix[row * HIDDEN_SIZE + column] * vector[column]
            for column in range(HIDDEN_SIZE)
        )
        if not -(1 << 63) <= accumulator < (1 << 63):
            raise OverflowError("int64 accumulator overflow")
        result.append(saturate_i32(round_shift(accumulator, 16)))
    return result


def initial_state() -> dict[str, object]:
    return {
        "full_keys": [],
        "full_values": [],
        "gdn_states": [[0] * HIDDEN_SIZE for _ in range(GDN_LAYERS)],
        "rng_state": INITIAL_RNG_STATE,
    }


def execute_token(
    weights: tuple[int, ...], token_id: int, state: dict[str, object]
) -> dict[str, object]:
    """Execute one deterministic candidate step using only integer arithmetic.

    A caller publishes this returned state only when its transaction commits.
    The RNG advances once in the candidate but does not influence greedy argmax.
    """
    if len(weights) != WEIGHT_COUNT or not 0 <= token_id < VOCAB_SIZE:
        raise ValueError("invalid weights or token")
    cursor = 0
    embeddings = weights[cursor : cursor + EMBEDDING_WEIGHTS]
    cursor += EMBEDDING_WEIGHTS
    hidden = list(
        embeddings[token_id * HIDDEN_SIZE : (token_id + 1) * HIDDEN_SIZE]
    )

    old_gdn_states = state["gdn_states"]
    if not isinstance(old_gdn_states, list) or len(old_gdn_states) != GDN_LAYERS:
        raise ValueError("invalid GDN state")
    next_gdn_states: list[list[int]] = []
    for layer in range(GDN_LAYERS):
        gate = weights[cursor : cursor + HIDDEN_SIZE]
        input_diagonal = weights[cursor + HIDDEN_SIZE : cursor + 2 * HIDDEN_SIZE]
        state_diagonal = weights[
            cursor + 2 * HIDDEN_SIZE : cursor + 3 * HIDDEN_SIZE
        ]
        cursor += 3 * HIDDEN_SIZE
        old_state = old_gdn_states[layer]
        if not isinstance(old_state, list) or len(old_state) != HIDDEN_SIZE:
            raise ValueError("invalid GDN layer state")
        next_state = []
        next_hidden = []
        for dimension in range(HIDDEN_SIZE):
            accumulator = (
                hidden[dimension] * input_diagonal[dimension]
                + old_state[dimension] * state_diagonal[dimension]
            )
            if not -(1 << 63) <= accumulator < (1 << 63):
                raise OverflowError("int64 accumulator overflow")
            recurrent = saturate_i32(round_shift(accumulator, 16))
            gated = recurrent * gate[dimension]
            if not -(1 << 63) <= gated < (1 << 63):
                raise OverflowError("int64 accumulator overflow")
            next_state.append(recurrent)
            next_hidden.append(saturate_i32(round_shift(gated, 16)))
        next_gdn_states.append(next_state)
        hidden = next_hidden

    key_matrix = weights[cursor : cursor + HIDDEN_SIZE * HIDDEN_SIZE]
    cursor += HIDDEN_SIZE * HIDDEN_SIZE
    value_matrix = weights[cursor : cursor + HIDDEN_SIZE * HIDDEN_SIZE]
    cursor += HIDDEN_SIZE * HIDDEN_SIZE
    key = _matrix_vector(key_matrix, hidden)
    value = _matrix_vector(value_matrix, hidden)
    old_keys = state["full_keys"]
    old_values = state["full_values"]
    if not isinstance(old_keys, list) or not isinstance(old_values, list):
        raise ValueError("invalid FullKV state")
    full_keys = [*old_keys, key]
    full_values = [*old_values, value]
    hidden = [
        integer_mean([entry[dimension] for entry in full_values])
        for dimension in range(HIDDEN_SIZE)
    ]

    logits_matrix = weights[cursor : cursor + LOGIT_WEIGHTS]
    logits = []
    for token in range(VOCAB_SIZE):
        accumulator = sum(
            logits_matrix[token * HIDDEN_SIZE + dimension] * hidden[dimension]
            for dimension in range(HIDDEN_SIZE)
        )
        if not -(1 << 63) <= accumulator < (1 << 63):
            raise OverflowError("int64 accumulator overflow")
        logits.append(saturate_i32(round_shift(accumulator, 16)))
    return {
        "full_keys": full_keys,
        "full_values": full_values,
        "gdn_states": next_gdn_states,
        "logits": logits,
        "rng_state": xorshift64(cast(int, state["rng_state"])),
        "token_id": argmax_first(logits),
    }


def build_weights() -> bytes:
    return b"".join(struct.pack("<i", coefficient(i)) for i in range(WEIGHT_COUNT))


def build_manifest(weight_sha256: str) -> bytes:
    text = f'''schema_version: 1
model_id: "mini-qwen38-hybrid"
model_format_version: 1
topology: "GDN,GDN,GDN,FULL"
dtype: "Q16_16"
vocab_size: 16
hidden_size: 8
max_context_tokens: 64
weight_sha256: "{weight_sha256}"
golden_version: 1
hal_abi_major: 1
hal_abi_minor: 0
required_capability: "EXECUTABLE_LAUNCH"
'''
    return text.encode("utf-8")


def encode_state(state: dict[str, object]) -> bytes:
    encoded = bytearray(struct.pack("<I", 1))
    for field in ("full_keys", "full_values", "gdn_states"):
        vectors = cast(list[list[int]], state[field])
        encoded.extend(struct.pack("<I", len(vectors)))
        for vector in vectors:
            encoded.extend(struct.pack("<I", len(vector)))
            encoded.extend(struct.pack(f"<{len(vector)}i", *vector))
    encoded.extend(struct.pack("<Q", cast(int, state["rng_state"])))
    return bytes(encoded)


def build_golden(weights_bytes: bytes, weight_sha256: str) -> bytes:
    """Compute Task12 golden data through this generator's Task11 model path."""
    weights = decode_weights(weights_bytes)
    state = initial_state()
    context = [1, 4, 7]
    steps: list[dict[str, object]] = []
    for _ in range(8):
        processed = len(cast(list[list[int]], state["full_values"]))
        if processed >= len(context):
            raise ValueError("context has no unprocessed token")
        result: dict[str, object] | None = None
        for token_id in context[processed:]:
            rng_state = cast(int, state["rng_state"])
            result = execute_token(weights, token_id, state)
            result["rng_state"] = rng_state
            state = result
        if result is None:
            raise ValueError("missing model result")
        state["rng_state"] = xorshift64(cast(int, state["rng_state"]))
        token_id = cast(int, result["token_id"])
        state_bytes = encode_state(state)
        steps.append(
            {
                "canonical_digest": hashlib.sha256(state_bytes).hexdigest(),
                "canonical_state_hex": state_bytes.hex(),
                "full_keys": state["full_keys"],
                "full_values": state["full_values"],
                "gdn_states": state["gdn_states"],
                "rng_state": str(state["rng_state"]),
                "token_id": token_id,
            }
        )
        context.append(token_id)
    document = {
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
        "prompt_token_ids": [1, 4, 7],
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
            {"count": EMBEDDING_WEIGHTS, "name": "token_embedding"},
            {"count": GDN_WEIGHTS, "name": "gdn_diagonal_gate_input_state"},
            {"count": FULL_WEIGHTS, "name": "full_key_value_matrices"},
            {"count": LOGIT_WEIGHTS, "name": "output_logits_matrix"},
        ],
        "weight_sha256": weight_sha256,
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_asset_descriptor(descriptor: int, contents: bytes) -> None:
    offset = 0
    while offset < len(contents):
        written = os.write(descriptor, contents[offset:])
        if written <= 0:
            raise OSError("asset write made no progress")
        offset += written
    os.fchmod(descriptor, 0o644)
    os.fsync(descriptor)


def write_asset_file(directory_fd: int, name: str, contents: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        _write_asset_descriptor(descriptor, contents)
    except BaseException:
        os.close(descriptor)
        os.unlink(name, dir_fd=directory_fd)
        raise
    os.close(descriptor)


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exchange_backend_name(
    platform_name: str | None = None, library: ctypes.CDLL | None = None
) -> str:
    platform_name = sys.platform if platform_name is None else platform_name
    library = ctypes.CDLL(None, use_errno=True) if library is None else library
    if platform_name == "darwin" and hasattr(library, "renameatx_np"):
        return "renameatx_np"
    if platform_name == "linux" and hasattr(library, "renameat2"):
        return "renameat2"
    raise RuntimeError("atomic directory exchange unsupported")


def _validate_exchange_name(name: str) -> None:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
        or Path(name).name != name
    ):
        raise ValueError("exchange names must be a single path component")


def exchange_directories(parent_fd: int, left_name: str, right_name: str) -> None:
    _validate_exchange_name(left_name)
    _validate_exchange_name(right_name)
    if left_name == right_name:
        raise ValueError("exchange names must be distinct")
    library = ctypes.CDLL(None, use_errno=True)
    backend = exchange_backend_name(library=library)
    function = getattr(library, backend)
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    flags = RENAME_SWAP if backend == "renameatx_np" else RENAME_EXCHANGE
    result = function(
        parent_fd,
        os.fsencode(left_name),
        parent_fd,
        os.fsencode(right_name),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "atomic directory exchange failed")


def _capture_owned_directory(
    parent_fd: int, parent: Path, name: str
) -> OwnedDirectory:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("staging ownership lost") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("staging ownership lost")
    return OwnedDirectory(parent / name, name, metadata.st_dev, metadata.st_ino)


def _verify_owned_directory(parent_fd: int, owned: OwnedDirectory) -> None:
    try:
        metadata = os.stat(owned.name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("staging ownership lost") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_dev != owned.device
        or metadata.st_ino != owned.inode
    ):
        raise RuntimeError("staging ownership lost")


def _owned_directory_matches(parent_fd: int, owned: OwnedDirectory) -> bool:
    try:
        _verify_owned_directory(parent_fd, owned)
    except RuntimeError:
        return False
    return True


def _open_owned_directory(parent_fd: int, owned: OwnedDirectory) -> int:
    _verify_owned_directory(parent_fd, owned)
    directory_fd = os.open(
        owned.name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    opened = os.fstat(directory_fd)
    if opened.st_dev != owned.device or opened.st_ino != owned.inode:
        os.close(directory_fd)
        raise RuntimeError("staging ownership lost")
    return directory_fd


def _remove_directory_contents(directory_fd: int) -> None:
    with os.scandir(directory_fd) as entries:
        names = [entry.name for entry in entries]
    for name in names:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(child_fd)
                if opened.st_dev != metadata.st_dev or opened.st_ino != metadata.st_ino:
                    raise RuntimeError("staging ownership lost")
                _remove_directory_contents(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def remove_owned_directory(parent_fd: int, owned: OwnedDirectory) -> None:
    directory_fd = _open_owned_directory(parent_fd, owned)
    try:
        _remove_directory_contents(directory_fd)
    finally:
        os.close(directory_fd)
    _verify_owned_directory(parent_fd, owned)
    os.rmdir(owned.name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _create_staging_directory(
    parent_fd: int, parent: Path, output_name: str
) -> OwnedDirectory:
    prefix = f".{output_name}.staging-"
    for _ in range(100):
        name = prefix + secrets.token_hex(8)
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        owned = _capture_owned_directory(parent_fd, parent, name)
        try:
            directory_fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            try:
                os.fchmod(directory_fd, 0o755)
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException as setup_error:
            try:
                remove_owned_directory(parent_fd, owned)
            except BaseException as cleanup_error:
                raise RuntimeError("staging cleanup failed") from cleanup_error
            raise setup_error
        return owned
    raise RuntimeError("could not allocate staging directory")


def _output_directory(
    parent_fd: int, parent: Path, output_name: str
) -> OwnedDirectory | None:
    try:
        metadata = os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("output must be a directory")
    return OwnedDirectory(
        parent / output_name, output_name, metadata.st_dev, metadata.st_ino
    )


def _rollback_exchange(
    parent_fd: int,
    output_name: str,
    staging: OwnedDirectory,
    new_output: OwnedDirectory,
) -> OwnedDirectory:
    _verify_owned_directory(parent_fd, staging)
    _verify_owned_directory(parent_fd, new_output)
    exchange_directories(parent_fd, output_name, staging.name)
    os.fsync(parent_fd)
    return OwnedDirectory(
        staging.path, staging.name, new_output.device, new_output.inode
    )


def _publish_update(
    parent_fd: int,
    output_name: str,
    old_output: OwnedDirectory,
    prepared: OwnedDirectory,
    exchange_hook: Callable[[str, Path], None] | None,
) -> None:
    try:
        if exchange_hook is not None:
            exchange_hook("before_exchange", prepared.path)
        _verify_owned_directory(parent_fd, old_output)
        _verify_owned_directory(parent_fd, prepared)
        exchange_directories(parent_fd, output_name, prepared.name)
    except BaseException:
        if _owned_directory_matches(parent_fd, prepared):
            remove_owned_directory(parent_fd, prepared)
        raise
    old_snapshot = OwnedDirectory(
        prepared.path, prepared.name, old_output.device, old_output.inode
    )
    new_output = OwnedDirectory(
        old_output.path, output_name, prepared.device, prepared.inode
    )
    try:
        if exchange_hook is not None:
            exchange_hook("before_commit", old_snapshot.path)
        os.fsync(parent_fd)
    except BaseException as precommit_error:
        try:
            restored_staging = _rollback_exchange(
                parent_fd, output_name, old_snapshot, new_output
            )
        except BaseException as rollback_error:
            raise RuntimeError("model asset recovery failed") from rollback_error
        try:
            remove_owned_directory(parent_fd, restored_staging)
        except BaseException as cleanup_error:
            raise RuntimeError("staging cleanup failed after recovery") from cleanup_error
        raise precommit_error

    # Commit point: the directory exchange is durable after the first parent
    # fsync succeeds. No later failure may expose the old snapshot as output.
    try:
        if exchange_hook is not None:
            exchange_hook("after_exchange", old_snapshot.path)
        remove_owned_directory(parent_fd, old_snapshot)
    except BaseException as cleanup_error:
        raise RuntimeError("post-commit cleanup failed") from cleanup_error


def generate(
    output: Path,
    exchange_hook: Callable[[str, Path], None] | None = None,
) -> None:
    output = output.absolute()
    try:
        parent_metadata = output.parent.lstat()
    except FileNotFoundError:
        raise FileNotFoundError("output parent must already exist") from None
    if output.parent.is_symlink() or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("output parent must be a real directory")
    parent_fd = os.open(
        output.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    prepared: OwnedDirectory | None = None
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        old_output = _output_directory(parent_fd, output.parent, output.name)
        prepared = _create_staging_directory(parent_fd, output.parent, output.name)

        weights = build_weights()
        digest = hashlib.sha256(weights).hexdigest()
        prepared_fd = _open_owned_directory(parent_fd, prepared)
        try:
            write_asset_file(prepared_fd, "weights.i32le", weights)
            write_asset_file(prepared_fd, "manifest.textproto", build_manifest(digest))
            write_asset_file(
                prepared_fd,
                "golden.json",
                build_golden(weights, digest),
            )
            os.fsync(prepared_fd)
        finally:
            os.close(prepared_fd)

        if old_output is not None:
            publishing = prepared
            prepared = None
            _publish_update(
                parent_fd, output.name, old_output, publishing, exchange_hook
            )
        else:
            _verify_owned_directory(parent_fd, prepared)
            os.rename(
                prepared.name,
                output.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            prepared = None
            os.fsync(parent_fd)
    except BaseException:
        if prepared is not None and _owned_directory_matches(parent_fd, prepared):
            remove_owned_directory(parent_fd, prepared)
            prepared = None
        raise
    finally:
        fcntl.flock(parent_fd, fcntl.LOCK_UN)
        os.close(parent_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="model directory; its parent must already exist and must not be a symlink",
    )
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
