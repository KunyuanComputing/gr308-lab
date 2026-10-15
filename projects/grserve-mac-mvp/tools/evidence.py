#!/usr/bin/env python3
"""Create one immutable, private local GRServe evidence directory."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.normalize_events import NormalizationError, normalize_events
from tools.acceptance import (
    AcceptanceError,
    load_full_summary,
    validate_full_summary,
)


DOCUMENT_FILES = (
    "preflight.json",
    "build.json",
    "cpp-tests.json",
    "python-tests.json",
    "sanitizer.json",
    "linux-amd64.json",
)
ARTIFACT_FILES = DOCUMENT_FILES + (
    "events.jsonl",
    "normalized-events.sha256",
    "assets.sha256",
    "summary.json",
)
RUN_ID_PATTERN = re.compile(r"^([0-9]{8}T[0-9]{6}Z)-([0-9a-f]{8})$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
# A POSIX absolute path begins at a token boundary.  Relative identifiers such
# as ASan/UBSan and tracked asset names are public evidence identities.
POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_.-])/(?!/)[^\s\"']+")
WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\[^\s\"']+")
PRIVATE_TEXT = re.compile(
    r"(?i)(prompt[ _-]*tokens?|socket[ _-]*path|"
    r"dynamic[ _-]*library[ _-]*path|backend[ _-]*private[ _-]*message|"
    r"fault[ _-]*plan)"
)
SAFE_STRUCTURED_KEYS = frozenset({"prompt_token_count", "fault_plan_seed"})
MAX_CONTEXT_TOKENS = 64
UINT64_MAX = (1 << 64) - 1
PENDING_DIRECTORY = ".pending"
DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class EvidenceError(RuntimeError):
    """Stable evidence creation failure code."""


REPORT_GATE_ORDER = (
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
    "TSan",
    "Linux amd64",
    "API E2E",
)
REPORT_ASSETS = (
    "models/mini-qwen38-hybrid/manifest.textproto",
    "models/mini-qwen38-hybrid/golden.json",
)


def _validate_run_id(run_id: str) -> None:
    match = RUN_ID_PATTERN.fullmatch(run_id)
    if match is None:
        raise EvidenceError("INVALID_RUN_ID")
    try:
        datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    except ValueError as error:
        raise EvidenceError("INVALID_RUN_ID") from error


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise EvidenceError("INVALID_ARTIFACT_DOCUMENT") from error


def _redact_public_string(value: str) -> str:
    value = POSIX_PATH.sub("<redacted-path>", value)
    value = WINDOWS_PATH.sub("<redacted-path>", value)
    if PRIVATE_TEXT.search(value) is not None:
        return "<redacted-private>"
    return value


def _redact_public_key(value: str) -> str:
    if value in SAFE_STRUCTURED_KEYS:
        return value
    return _redact_public_string(value)


def _redact_public_value(value: object) -> object:
    if isinstance(value, str):
        return _redact_public_string(value)
    if isinstance(value, list):
        return [_redact_public_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise EvidenceError("INVALID_ARTIFACT_DOCUMENT")
        redacted: dict[str, object] = {}
        for key, item in value.items():
            if key == "prompt_token_count" and (
                type(item) is not int or not 1 <= item <= MAX_CONTEXT_TOKENS
            ):
                raise EvidenceError("INVALID_ARTIFACT_DOCUMENT")
            if key == "fault_plan_seed" and (
                type(item) is not int or not 0 <= item <= UINT64_MAX
            ):
                raise EvidenceError("INVALID_ARTIFACT_DOCUMENT")
            public_key = _redact_public_key(key)
            if public_key in redacted:
                raise EvidenceError("INVALID_ARTIFACT_DOCUMENT")
            redacted[public_key] = _redact_public_value(item)
        return redacted
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError("INVALID_ARTIFACT_DOCUMENT")
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    raise EvidenceError("INVALID_ARTIFACT_DOCUMENT")


def _open_or_create_directory(parent_fd: int, name: str) -> int:
    created = False
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    descriptor = -1
    try:
        descriptor = os.open(name, DIRECTORY_FLAGS, dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o500 != 0o500
            or (not created and stat.S_IMODE(metadata.st_mode) & 0o022 != 0)
        ):
            raise OSError("not a directory")
        if created:
            os.fchmod(descriptor, 0o700)
        return descriptor
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise EvidenceError("UNSAFE_ARTIFACT_PATH") from error


def _write_regular_file(
    directory_fd: int, name: str, content: bytes
) -> tuple[int, int]:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size != len(content)
        ):
            raise OSError("unsafe artifact file identity")
        return (metadata.st_dev, metadata.st_ino)
    finally:
        os.close(descriptor)


def _rename_directory_exclusive(parent_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        result = library.renameatx_np(
            parent_fd, source_bytes, parent_fd, destination_bytes, 0x00000004
        )
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        result = library.renameat2(
            parent_fd, source_bytes, parent_fd, destination_bytes, 0x00000001
        )
    else:
        try:
            os.stat(destination, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(destination)
        os.rename(source, destination, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        return
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (17, 39):
            raise FileExistsError(destination)
        raise OSError(error_number, os.strerror(error_number))


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 64 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _pending_identity(
    parent_fd: int,
    descriptor: int,
    expected: tuple[int, int] | None = None,
    *,
    expected_mode: int = 0o700,
) -> tuple[int, int]:
    """Validate the current .pending pathname and opened directory together."""
    try:
        opened = os.fstat(descriptor)
        pathname = os.stat(
            PENDING_DIRECTORY, dir_fd=parent_fd, follow_symlinks=False
        )
    except OSError as error:
        raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED") from error
    identity = (opened.st_dev, opened.st_ino)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(pathname.st_mode)
        or opened.st_uid != os.geteuid()
        or pathname.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != expected_mode
        or stat.S_IMODE(pathname.st_mode) != expected_mode
        or opened.st_nlink < 2
        or pathname.st_nlink != opened.st_nlink
        or (pathname.st_dev, pathname.st_ino) != identity
        or (expected is not None and identity != expected)
    ):
        raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED")
    return identity


def _directory_entries(descriptor: int) -> set[str]:
    try:
        return set(os.listdir(descriptor))
    except OSError as error:
        raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED") from error


def _open_directory_matches_contents(
    descriptor: int,
    contents: Mapping[str, bytes],
    identities: Mapping[str, tuple[int, int]] | None = None,
) -> bool:
    files: dict[str, int] = {}
    try:
        if _directory_entries(descriptor) != set(ARTIFACT_FILES):
            return False
        for name in ARTIFACT_FILES:
            files[name] = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
        return _open_files_match_contents(files, contents, identities)
    except (OSError, EvidenceError):
        return False
    finally:
        for file_fd in files.values():
            try:
                os.close(file_fd)
            except OSError:
                pass


def _open_files_match_contents(
    files: Mapping[str, int],
    contents: Mapping[str, bytes],
    identities: Mapping[str, tuple[int, int]] | None = None,
) -> bool:
    if set(files) != set(ARTIFACT_FILES):
        return False
    try:
        for name in ARTIFACT_FILES:
            file_fd = files[name]
            file_metadata = os.fstat(file_fd)
            os.lseek(file_fd, 0, os.SEEK_SET)
            if (
                not stat.S_ISREG(file_metadata.st_mode)
                or file_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(file_metadata.st_mode) != 0o600
                or file_metadata.st_nlink != 1
                or file_metadata.st_size != len(contents[name])
                or (
                    identities is not None
                    and (file_metadata.st_dev, file_metadata.st_ino)
                    != identities[name]
                )
                or _read_all(file_fd) != contents[name]
            ):
                return False
        return True
    except OSError:
        return False


def _published_identity(
    parent_fd: int,
    run_id: str,
    descriptor: int,
    expected: tuple[int, int],
) -> None:
    try:
        opened = os.fstat(descriptor)
        pathname = os.stat(run_id, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise EvidenceError("ARTIFACT_DURABILITY_UNKNOWN") from error
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(pathname.st_mode)
        or opened.st_uid != os.geteuid()
        or pathname.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o500
        or stat.S_IMODE(pathname.st_mode) != 0o500
        or opened.st_nlink < 2
        or pathname.st_nlink != opened.st_nlink
        or (opened.st_dev, opened.st_ino) != expected
        or (pathname.st_dev, pathname.st_ino) != expected
    ):
        raise EvidenceError("ARTIFACT_DURABILITY_UNKNOWN")


def _matches_artifact_mode(
    parent_fd: int,
    run_id: str,
    contents: Mapping[str, bytes],
    expected_mode: int,
) -> bool:
    try:
        descriptor = os.open(run_id, DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError:
        return False
    try:
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != expected_mode
            or not _open_directory_matches_contents(descriptor, contents)
        ):
            return False
        final_metadata = os.fstat(descriptor)
        try:
            pathname_metadata = os.stat(
                run_id, dir_fd=parent_fd, follow_symlinks=False
            )
        except OSError:
            return False
        if (
            not stat.S_ISDIR(final_metadata.st_mode)
            or not stat.S_ISDIR(pathname_metadata.st_mode)
            or final_metadata.st_uid != os.geteuid()
            or pathname_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(final_metadata.st_mode) != expected_mode
            or stat.S_IMODE(pathname_metadata.st_mode) != expected_mode
            or (final_metadata.st_dev, final_metadata.st_ino) != identity
            or (pathname_metadata.st_dev, pathname_metadata.st_ino) != identity
        ):
            return False
        return True
    except OSError:
        return False
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _is_durability_unknown_artifact(parent_fd: int, run_id: str) -> bool:
    """Recognize the non-searchable state left by a failed final chmod.

    Mode 0600 deliberately prevents opening children, so this state cannot be
    reclassified as complete. Exact names and stable directory identity are
    sufficient only to preserve the dedicated durability-unknown outcome.
    """
    try:
        descriptor = os.open(run_id, DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError:
        return False
    try:
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or _directory_entries(descriptor) != set(ARTIFACT_FILES)
        ):
            return False
        final_metadata = os.fstat(descriptor)
        pathname = os.stat(run_id, dir_fd=parent_fd, follow_symlinks=False)
        return (
            stat.S_ISDIR(final_metadata.st_mode)
            and stat.S_ISDIR(pathname.st_mode)
            and final_metadata.st_uid == os.geteuid()
            and pathname.st_uid == os.geteuid()
            and stat.S_IMODE(final_metadata.st_mode) == 0o600
            and stat.S_IMODE(pathname.st_mode) == 0o600
            and (final_metadata.st_dev, final_metadata.st_ino) == identity
            and (pathname.st_dev, pathname.st_ino) == identity
        )
    except (OSError, EvidenceError):
        return False
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def write_evidence(
    repository_root: Path,
    run_id: str,
    *,
    documents: Mapping[str, object],
    events: bytes,
    assets_sha256: str,
    summary: Mapping[str, object] | None = None,
) -> Path:
    """Publish one immutable run directory.

    The exclusive parent-directory lock serializes cooperating publishers.
    Rename is the commit point. A failure syncing the parent after rename is
    reported as ARTIFACT_DURABILITY_UNKNOWN without rollback; a byte-for-byte
    complete retry is reported as ARTIFACT_ALREADY_COMMITTED without mutation.
    If the post-rename 0500 freeze itself fails, the committed 0600 tree is
    recognized on retry and continues to report ARTIFACT_DURABILITY_UNKNOWN.
    Because POSIX has no portable operation to remove an opened directory by
    descriptor, any pre-commit failure preserves the single .pending slot and
    reports ARTIFACT_RECOVERY_REQUIRED for explicit operator recovery.

    POSIX also cannot atomically bind mkdir to the subsequent open. An empty,
    owner-controlled directory substituted in that interval is inherently
    indistinguishable. The held directory descriptor and repeated pathname
    identity checks establish the strongest portable continuity evidence; no
    absolute atomicity is claimed. Before the final check and rename, the
    pending directory is frozen on Darwin to mode 0600: the missing search bit
    blocks directory-entry mutation while retaining the write bit Darwin needs
    to rename a directory. Fixed file descriptors and a directory stream opened
    before the freeze support the final inspection. The published mode is 0500.
    This closes ordinary late-write races, but an actively hostile same-UID
    process can retain a directory descriptor and chmod the directory again,
    or retain a writable artifact descriptor; that process remains outside the
    portable guarantee.
    """
    _validate_run_id(run_id)
    if set(documents) != set(DOCUMENT_FILES):
        raise EvidenceError("INVALID_ARTIFACT_DOCUMENTS")
    if SHA256_PATTERN.fullmatch(assets_sha256) is None:
        raise EvidenceError("INVALID_ASSETS_SHA256")
    reserved_summary_fields = {
        "schema_version",
        "run_id",
        "normalized_events_sha256",
        "assets_sha256",
        "event_count",
        "artifact_sha256",
    }
    supplied_summary = dict(summary or {})
    if reserved_summary_fields.intersection(supplied_summary):
        raise EvidenceError("INVALID_ARTIFACT_SUMMARY")
    try:
        normalized = normalize_events(events)
    except NormalizationError as error:
        raise EvidenceError("INVALID_EVENTS") from error
    for line in events.splitlines():
        event = json.loads(line)
        if event.get("runId") != run_id:
            raise EvidenceError("EVENT_RUN_ID_MISMATCH")

    contents: dict[str, bytes] = {
        name: _canonical_json(_redact_public_value(documents[name]))
        for name in DOCUMENT_FILES
    }
    contents["events.jsonl"] = events
    contents["normalized-events.sha256"] = (normalized.sha256 + "\n").encode(
        "ascii"
    )
    contents["assets.sha256"] = (assets_sha256 + "\n").encode("ascii")
    artifact_hashes = {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
    }
    public_summary = _redact_public_value(supplied_summary)
    if not isinstance(public_summary, dict):
        raise EvidenceError("INVALID_ARTIFACT_SUMMARY")
    contents["summary.json"] = _canonical_json(
        public_summary
        | {
            "schema_version": 1,
            "run_id": run_id,
            "normalized_events_sha256": normalized.sha256,
            "assets_sha256": assets_sha256,
            "event_count": normalized.event_count,
            "artifact_sha256": artifact_hashes,
        }
    )

    root = Path(repository_root)
    if not root.is_absolute():
        raise EvidenceError("UNSAFE_ARTIFACT_PATH")
    descriptors: list[int] = []
    parent_fd: int | None = None
    staging = PENDING_DIRECTORY
    staging_created = False
    parent_locked = False
    published = False
    try:
        descriptors.append(os.open(root, DIRECTORY_FLAGS))
        descriptors.append(_open_or_create_directory(descriptors[-1], "artifacts"))
        descriptors.append(_open_or_create_directory(descriptors[-1], "mac-mvp"))
        parent_fd = descriptors[-1]
        # Every GRServe publisher cooperates on this directory lock. It spans
        # existence inspection through rename and parent fsync, so a same-run
        # loser never creates a staging directory and therefore needs no
        # conflict cleanup.
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        parent_locked = True
        try:
            os.stat(run_id, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if _matches_artifact_mode(parent_fd, run_id, contents, 0o500):
                raise EvidenceError("ARTIFACT_ALREADY_COMMITTED")
            if _is_durability_unknown_artifact(parent_fd, run_id):
                raise EvidenceError("ARTIFACT_DURABILITY_UNKNOWN")
            raise EvidenceError("ARTIFACT_DIR_EXISTS")
        try:
            os.stat(staging, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED")
        os.mkdir(staging, 0o700, dir_fd=parent_fd)
        staging_created = True
        staging_fd = os.open(staging, DIRECTORY_FLAGS, dir_fd=parent_fd)
        descriptors.append(staging_fd)
        staging_identity = _pending_identity(parent_fd, staging_fd)
        if _directory_entries(staging_fd):
            raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED")

        written: set[str] = set()
        written_identities: dict[str, tuple[int, int]] = {}
        for name in ARTIFACT_FILES:
            _pending_identity(parent_fd, staging_fd, staging_identity)
            if _directory_entries(staging_fd) != written:
                raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED")
            written_identities[name] = _write_regular_file(
                staging_fd, name, contents[name]
            )
            written.add(name)
        _pending_identity(parent_fd, staging_fd, staging_identity)
        if not _open_directory_matches_contents(
            staging_fd, contents, written_identities
        ):
            raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED")
        os.fsync(staging_fd)
        _pending_identity(parent_fd, staging_fd, staging_identity)
        frozen_files: dict[str, int] = {}
        for name in ARTIFACT_FILES:
            frozen_files[name] = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=staging_fd,
            )
            descriptors.append(frozen_files[name])
        frozen_directory_fd = os.open(".", DIRECTORY_FLAGS, dir_fd=staging_fd)
        descriptors.append(frozen_directory_fd)
        os.fchmod(staging_fd, 0o600)
        os.fsync(staging_fd)
        _pending_identity(
            parent_fd, staging_fd, staging_identity, expected_mode=0o600
        )
        frozen_inodes: dict[str, int] = {}
        with os.scandir(frozen_directory_fd) as frozen_entries:
            for entry in frozen_entries:
                if entry.name in frozen_inodes:
                    raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED")
                frozen_inodes[entry.name] = entry.inode()
        if set(frozen_inodes) != set(ARTIFACT_FILES):
            raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED")
        for name, file_fd in frozen_files.items():
            if os.fstat(file_fd).st_ino != frozen_inodes[name]:
                raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED")
        if not _open_files_match_contents(
            frozen_files, contents, written_identities
        ):
            raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED")
        _pending_identity(
            parent_fd, staging_fd, staging_identity, expected_mode=0o600
        )
        _rename_directory_exclusive(parent_fd, staging, run_id)
        # Rename is the commit point: after this assignment the complete final
        # tree is immutable and must never be rolled back by an error path.
        published = True
        staging_created = False
        os.fchmod(staging_fd, 0o500)
        os.fsync(staging_fd)
        _published_identity(parent_fd, run_id, staging_fd, staging_identity)
        try:
            os.fsync(parent_fd)
        except OSError as error:
            raise EvidenceError("ARTIFACT_DURABILITY_UNKNOWN") from error
    except EvidenceError as error:
        if staging_created and not published:
            raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED") from error
        raise
    except FileExistsError as error:
        if parent_locked and parent_fd is not None and not published:
            try:
                os.stat(staging, dir_fd=parent_fd, follow_symlinks=False)
            except OSError:
                pass
            else:
                raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED") from error
        raise EvidenceError("ARTIFACT_DIR_EXISTS") from error
    except (OSError, TypeError, ValueError) as error:
        if published:
            raise EvidenceError("ARTIFACT_DURABILITY_UNKNOWN") from error
        if staging_created:
            raise EvidenceError("ARTIFACT_RECOVERY_REQUIRED") from error
        raise EvidenceError("ARTIFACT_WRITE_FAILED") from error
    finally:
        if parent_locked and parent_fd is not None:
            try:
                fcntl.flock(parent_fd, fcntl.LOCK_UN)
            except OSError:
                pass
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
    return root / "artifacts" / "mac-mvp" / run_id


def _report_summary_path(path: Path) -> Path:
    supplied = Path(path).absolute()
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise AcceptanceError("SUMMARY_PATH_INVALID") from error
    if supplied != resolved or resolved.name != "summary.json" or not resolved.is_file():
        raise AcceptanceError("SUMMARY_PATH_MUST_BE_EXACT")
    return resolved


def _report_field(mapping: object, key: str, expected_type: type) -> object:
    if type(mapping) is not dict or key not in mapping:
        raise AcceptanceError(f"REPORT_FIELD_MISSING:{key}")
    value = mapping[key]
    if type(value) is not expected_type:
        raise AcceptanceError(f"REPORT_FIELD_INVALID:{key}")
    return value


def _report_gate_rows(summary: dict[str, object]) -> tuple[list[str], list[str]]:
    gates = _report_field(summary, "gates", dict)
    assert isinstance(gates, dict)
    rows: list[str] = []
    skips: list[str] = []
    ordered = [name for name in REPORT_GATE_ORDER if name in gates]
    ordered.extend(sorted(set(gates) - set(ordered)))
    for name in ordered:
        if type(name) is not str or type(gates[name]) is not dict:
            raise AcceptanceError("REPORT_GATE_INVALID")
        gate = gates[name]
        if set(gate) not in ({"status"}, {"status", "reason"}):
            raise AcceptanceError(f"REPORT_GATE_INVALID:{name}")
        status = gate.get("status")
        reason = gate.get("reason", "")
        if type(status) is not str or type(reason) is not str:
            raise AcceptanceError(f"REPORT_GATE_INVALID:{name}")
        display_reason = reason if reason else "—"
        rows.append(f"| {name} | {status} | {display_reason} |")
        if status == "SKIP":
            skips.append(f"- `{name}`: `{reason or 'MISSING_REASON'}`")
    return rows, skips


def _report_submodule_rows(summary: dict[str, object]) -> list[str]:
    git = _report_field(summary, "git", dict)
    assert isinstance(git, dict)
    head = _report_field(git, "head", str)
    dirty = _report_field(git, "dirty", bool)
    if GIT_SHA_PATTERN.fullmatch(str(head)) is None or dirty:
        raise AcceptanceError("REPORT_GIT_IDENTITY_INVALID")
    submodules = _report_field(git, "submodules", list)
    assert isinstance(submodules, list)
    rows: list[str] = []
    names: set[str] = set()
    for item in submodules:
        name = _report_field(item, "name", str)
        gitlink = _report_field(item, "gitlink", str)
        resolved = _report_field(item, "head", str)
        clean = _report_field(item, "clean", bool)
        if (
            not name
            or name in names
            or GIT_SHA_PATTERN.fullmatch(str(gitlink)) is None
            or GIT_SHA_PATTERN.fullmatch(str(resolved)) is None
            or not clean
        ):
            raise AcceptanceError("REPORT_DEPENDENCY_IDENTITY_INVALID")
        names.add(str(name))
        rows.append(f"| {name} | `{gitlink}` | `{resolved}` |")
    if not rows:
        raise AcceptanceError("REPORT_DEPENDENCY_IDENTITY_MISSING")
    return rows


def _render_report_text(summary_path: Path) -> str:
    exact_path = _report_summary_path(summary_path)
    summary_bytes = exact_path.read_bytes()
    summary = load_full_summary(exact_path)
    if summary.get("profile") != "full":
        raise AcceptanceError("REPORT_REQUIRES_FULL_PROFILE")
    status = summary.get("status")
    if type(status) is not str or status not in {"PASS", "FAIL", "INCOMPLETE"}:
        raise AcceptanceError("REPORT_SUMMARY_STATUS_INVALID")
    critical = summary.get("critical_skip_count")
    if type(critical) is not int or critical < 0:
        raise AcceptanceError("REPORT_CRITICAL_SKIP_COUNT_INVALID")
    accepted = False
    try:
        accepted = validate_full_summary(summary)
    except AcceptanceError:
        accepted = False
    conclusion = "ACCEPTED" if accepted else "NOT ACCEPTED / BLOCKED"
    gate_rows, skip_rows = _report_gate_rows(summary)
    submodule_rows = _report_submodule_rows(summary)
    git = summary["git"]
    assert isinstance(git, dict)
    assets = _report_field(summary, "assets", dict)
    assert isinstance(assets, dict)
    asset_rows: list[str] = []
    for name in REPORT_ASSETS:
        digest = _report_field(assets, name, str)
        if SHA256_PATTERN.fullmatch(str(digest)) is None:
            raise AcceptanceError(f"REPORT_ASSET_IDENTITY_INVALID:{name}")
        asset_rows.append(f"| `{name}` | `{digest}` |")
    dependencies = _report_field(summary, "dependencies", dict)
    dependency_json = json.dumps(
        dependencies, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    summary_digest = hashlib.sha256(summary_bytes).hexdigest()
    run_id = _report_field(summary, "run_id", str)
    if RUN_ID_PATTERN.fullmatch(str(run_id)) is None:
        raise AcceptanceError("REPORT_RUN_ID_INVALID")
    caveat = (
        "本报告不构成 Mac MVP 通过认证；必须修复上表中的失败或不完整 gate，"
        "并用一次新的 full summary 重新生成报告。"
        if not accepted
        else "该结论只认证本次 full summary 所覆盖的本地 Mac MVP 契约。"
    )
    skips = "\n".join(skip_rows) if skip_rows else "- 无"
    return (
        "# GRServe Mac MVP 验收报告\n\n"
        f"**结论：{conclusion}**\n\n"
        f"{caveat}\n\n"
        "## 证据绑定\n\n"
        f"- Run ID：`{run_id}`\n"
        f"- Git SHA：`{git['head']}`\n"
        f"- Summary 精确路径：`{exact_path}`\n"
        f"- Summary SHA256：`{summary_digest}`\n"
        f"- Summary 状态：`{status}`\n"
        f"- Critical skip count：`{critical}`\n\n"
        "本报告不使用 `latest` 链接或目录修改时间定位证据；校验命令会依据上述"
        "精确 summary 路径重新渲染，并逐字节比较报告。\n\n"
        "## Gate 结果\n\n"
        "| Gate | 状态 | 原因 |\n"
        "|---|---|---|\n"
        + "\n".join(gate_rows)
        + "\n\n## 跳过项\n\n"
        + skips
        + "\n\n## 依赖 gitlink\n\n"
        "| 依赖 | Gitlink | Resolved HEAD |\n"
        "|---|---|---|\n"
        + "\n".join(submodule_rows)
        + "\n\n"
        f"- 其他锁定依赖（机器可读）：`{dependency_json}`\n\n"
        "## Manifest 与 Golden\n\n"
        "| 资产 | SHA256 |\n"
        "|---|---|\n"
        + "\n".join(asset_rows)
        + "\n\n## 范围边界与服务器后续\n\n"
        "即使结论为 ACCEPTED，也只说明本地协议、状态机、故障边界以及 CPU/模拟"
        "接口在该次证据中通过；不证明 GR308 性能或硬件正确性，也不代表服务器上线资格。\n\n"
        "服务器实验前仍必须完成独立的 Fant HAL/P0 资格验证：设备身份、内存、"
        "stream/event、kernel launch、copy、错误传播、资源回收和 FHCCL 冷启动。\n"
    )


def render_report(summary_path: Path, output_path: Path) -> Path:
    """Render a deterministic Chinese report from one explicitly named summary."""
    report = _render_report_text(Path(summary_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


def validate_report(summary_path: Path, report_path: Path) -> bool:
    """Require the report to be the exact deterministic rendering of the summary."""
    expected = _render_report_text(Path(summary_path)).encode("utf-8")
    try:
        actual = Path(report_path).read_bytes()
    except OSError as error:
        raise AcceptanceError("REPORT_READ_FAILED") from error
    if actual != expected:
        raise AcceptanceError("REPORT_SUMMARY_BINDING_MISMATCH")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--repo-root", required=True, type=Path)
    create.add_argument("--run-id", required=True)
    create.add_argument("--documents", required=True, type=Path)
    create.add_argument("--events", required=True, type=Path)
    create.add_argument("--assets-sha256", required=True)
    create.add_argument("--summary", type=Path)
    render = subparsers.add_parser("render-report")
    render.add_argument("--summary", required=True, type=Path)
    render.add_argument("--output", required=True, type=Path)
    validate = subparsers.add_parser("validate-report")
    validate.add_argument("--summary", required=True, type=Path)
    validate.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "render-report":
            render_report(arguments.summary, arguments.output)
            print(arguments.output)
            return 0
        if arguments.command == "validate-report":
            validate_report(arguments.summary, arguments.report)
            print("REPORT_VALID")
            return 0
        document_map: dict[str, Any] = {
            name: json.loads((arguments.documents / name).read_bytes())
            for name in DOCUMENT_FILES
        }
        destination = write_evidence(
            arguments.repo_root,
            arguments.run_id,
            documents=document_map,
            events=arguments.events.read_bytes(),
            assets_sha256=arguments.assets_sha256,
            summary=(
                json.loads(arguments.summary.read_bytes())
                if arguments.summary is not None
                else None
            ),
        )
        print(destination / "summary.json")
        return 0
    except (OSError, json.JSONDecodeError, EvidenceError, AcceptanceError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
