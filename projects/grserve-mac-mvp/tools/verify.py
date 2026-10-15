#!/usr/bin/env python3
"""Run reproducible GRServe verification profiles and publish safe evidence."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import tempfile
import io
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator
from google.protobuf import json_format  # type: ignore[import-untyped]

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evidence import DOCUMENT_FILES, EvidenceError, write_evidence
from grserve_api._generated import event_pb2

UV_VERSION = "0.11.7"
BASE_IMAGE_PATTERN = re.compile(r"^python@sha256:[0-9a-f]{64}$")
TSAN_UNSUPPORTED_PATTERN = re.compile(
    r"(?i)threadsanitizer:.*(?:unsupported.*(?:darwin|runtime)|darwin.*unsupported)"
)
GIT_HASH = re.compile(r"^[0-9a-f]{40}$")
SUBMODULES = {
    "third_party/googletest": "6910c9d9165801d8827d628cb72eb7ea9dd538c5",
    "third_party/protobuf": "43e1626812c1b543e56a7bec59dc09eb18248bd2",
}
ASSETS = (
    "uv.lock",
    "docker/base-image.lock",
    "docker/apt-packages.lock",
    "proto/engine.proto",
    "proto/event.proto",
    "proto/model_manifest.proto",
    "python/grserve_api/_generated/event_pb2.py",
    "models/mini-qwen38-hybrid/manifest.textproto",
    "models/mini-qwen38-hybrid/golden.json",
    "models/mini-qwen38-hybrid/weights.i32le",
)
DOCUMENT_GATES = {
    "preflight.json": ("preflight",),
    "build.json": ("Debug", "Release"),
    "cpp-tests.json": ("C11 ABI", "C++", "manifest", "fault matrix"),
    "python-tests.json": ("Python", "proto", "golden", "API E2E"),
    "sanitizer.json": ("ASan/UBSan", "TSan"),
    "linux-amd64.json": ("Linux amd64",),
}
MAC_GATES = (
    "preflight",
    "Debug",
    "Release",
    "C11 ABI",
    "C++",
    "manifest",
    "fault matrix",
    "ASan/UBSan",
    "TSan",
    "Python",
    "proto",
    "golden",
    "API E2E",
)
Runner = Callable[..., "CommandResult"]
ABSEIL_COMMIT = "9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f"
FAULT_PLAN_SEED = 17
APT_LOCK_PATTERN = re.compile(
    r"snapshot=(\d{8}T\d{6}Z)\n"
    r"build-essential=([^\s]+)\n"
    r"git=([^\s]+)\n?"
)
TRUSTED_GIT = "/usr/bin/git"
TRUSTED_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
HERMETIC_GIT_ENV = {
    "PATH": TRUSTED_PATH,
    "LANG": "C",
    "LC_ALL": "C",
    "HOME": "/var/empty",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_NO_REPLACE_OBJECTS": "1",
}
MAC_PRESETS = (
    "mac-arm64-debug",
    "mac-arm64-release",
    "mac-arm64-asan-ubsan",
    "mac-arm64-tsan",
)
EXPECTED_CTESTS = {
    "c.hal_header",
    "cpp.hal_abi",
    "cpp.smoke",
    "cpp.sha256",
    "cpp.event_log",
    "cpp.manifest",
    "cpp.mini_qwen",
    "cpp.backend_parity",
    "cpp.hal_loader",
    "cpp.hal_raii",
    "cpp.types",
    "cpp.error",
    "cpp.mock_async",
    "cpp.fault_plan",
    "cpp.transaction",
    "cpp.rank_worker",
    "cpp.admission",
    "cpp.controller_success",
    "cpp.controller_faults",
    "cpp.controller_cancel",
    "cpp.rpc.proto_contract",
    "cpp.rpc.frame_codec",
    "cpp.rpc.unix_socket",
    "cpp.rpc.session",
    "cpp.rpc.disconnect",
    "cpp.rpc.engine_help",
    "cpp.rpc.engine_process",
    "cmake.rpc_build_testing_off",
}


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    stdout_sha256: str = ""
    stdout_byte_count: int = 0
    stdout_truncated: bool = False
    stderr_sha256: str = ""
    stderr_byte_count: int = 0
    stderr_truncated: bool = False


@dataclass(frozen=True)
class VerificationOutcome:
    exit_code: int
    evidence_dir: Path | None
    reason: str | None = None


class VerificationInterrupted(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def subprocess_runner(
    command: tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout_seconds: float = 1800.0,
) -> CommandResult:
    class Capture:
        limit = 1024 * 1024

        def __init__(self) -> None:
            self.digest = hashlib.sha256()
            self.count = 0
            self.head = bytearray()
            self.tail = bytearray()

        def add(self, chunk: bytes) -> None:
            self.digest.update(chunk)
            self.count += len(chunk)
            half = self.limit // 2
            needed = half - len(self.head)
            if needed > 0:
                self.head.extend(chunk[:needed])
                chunk = chunk[needed:]
            if chunk:
                self.tail.extend(chunk)
                if len(self.tail) > half:
                    del self.tail[:-half]

        def text(self) -> str:
            value = bytes(self.head)
            if self.count > self.limit:
                marker = b"\n<...truncated...>\n"
                half = (self.limit - len(marker)) // 2
                value = bytes(self.head[:half]) + marker + bytes(self.tail[-half:])
            else:
                value += bytes(self.tail)
            return value.decode("utf-8", errors="replace")

    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            start_new_session=True,
        )
    except OSError:
        return CommandResult(
            127, "", "COMMAND_NOT_FOUND", round(time.monotonic() - started, 6)
        )
    captures = (Capture(), Capture())

    def consume(stream: Any, capture: Capture) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    return
                capture.add(chunk)
        except (OSError, ValueError):
            return

    threads = [
        threading.Thread(
            target=consume, args=(process.stdout, captures[0]), daemon=True
        ),
        threading.Thread(
            target=consume, args=(process.stderr, captures[1]), daemon=True
        ),
    ]
    for thread in threads:
        thread.start()
    if input_text is not None and process.stdin is not None:
        try:
            process.stdin.write(input_text.encode())
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def process_table() -> dict[int, tuple[int, str]]:
        try:
            output = subprocess.run(
                ("/bin/ps", "-axo", "pid=,ppid=,lstart="),
                check=False,
                capture_output=True,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                timeout=0.25,
            ).stdout.decode("utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired):
            return {}
        table: dict[int, tuple[int, str]] = {}
        pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+(.+)$")
        for line in output.splitlines():
            match = pattern.match(line)
            if match:
                table[int(match.group(1))] = (int(match.group(2)), match.group(3))
        return table

    descendants: dict[int, str] = {}

    def sample_descendants() -> None:
        table = process_table()
        frontier = {process.pid}
        while frontier:
            children = {
                pid
                for pid, (parent, _identity) in table.items()
                if parent in frontier and pid not in descendants
            }
            for pid in children:
                descendants[pid] = table[pid][1]
            frontier = children

    def matching_descendants() -> list[int]:
        table = process_table()
        return [
            pid
            for pid, identity in descendants.items()
            if pid in table and table[pid][1] == identity
        ]

    def terminate_tree() -> None:
        sample_descendants()
        for pid in matching_descendants():
            try:
                os.kill(pid, signal.SIGSTOP)
            except (ProcessLookupError, PermissionError):
                pass
        for signum in (signal.SIGTERM,):
            try:
                os.killpg(process.pid, signum)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline and matching_descendants():
            time.sleep(0.01)
        for pid in matching_descendants():
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + 0.75
        while time.monotonic() < deadline and matching_descendants():
            time.sleep(0.01)

    timed_out = False
    try:
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None and time.monotonic() < deadline:
            sample_descendants()
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        if process.poll() is None:
            timed_out = True
            terminate_tree()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
    except BaseException:
        terminate_tree()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    os.close(stream.fileno())
                except OSError:
                    pass
        for thread in threads:
            thread.join(timeout=0.5)
        raise
    if timed_out:
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    os.close(stream.fileno())
                except OSError:
                    pass
    for thread in threads:
        thread.join(timeout=0.5)
    stdout, stderr = captures[0], captures[1]
    stderr_text = "COMMAND_TIMEOUT" if timed_out else stderr.text()
    return CommandResult(
        124 if timed_out else process.returncode,
        stdout.text(),
        stderr_text,
        round(time.monotonic() - started, 6),
        stdout.digest.hexdigest(),
        stdout.count,
        stdout.count > stdout.limit,
        stderr.digest.hexdigest(),
        stderr.count,
        stderr.count > stderr.limit,
    )


def _run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def _record(cid: str, argv: tuple[str, ...], r: CommandResult) -> dict[str, object]:
    return {
        "command_id": cid,
        "argv": list(argv),
        "stdout": r.stdout,
        "stderr": r.stderr,
        "exit_code": r.exit_code,
        "duration_seconds": r.duration_seconds,
        "status": "PASS" if r.exit_code == 0 else "FAIL",
        "stdout_sha256": r.stdout_sha256
        or hashlib.sha256(r.stdout.encode()).hexdigest(),
        "stdout_byte_count": r.stdout_byte_count or len(r.stdout.encode()),
        "stdout_truncated": r.stdout_truncated,
        "stderr_sha256": r.stderr_sha256
        or hashlib.sha256(r.stderr.encode()).hexdigest(),
        "stderr_byte_count": r.stderr_byte_count or len(r.stderr.encode()),
        "stderr_truncated": r.stderr_truncated,
    }


def _hash_assets(root: Path) -> tuple[dict[str, str], str, list[str]]:
    files: dict[str, str] = {}
    missing = []
    combined = hashlib.sha256()
    for name in ASSETS:
        try:
            content = (root / name).read_bytes()
        except OSError:
            missing.append(name)
            continue
        files[name] = hashlib.sha256(content).hexdigest()
        label = name.encode()
        combined.update(len(label).to_bytes(4, "big"))
        combined.update(label)
        combined.update(len(content).to_bytes(8, "big"))
        combined.update(content)
    return files, combined.hexdigest(), missing


def _trusted_git_binary() -> bool:
    try:
        metadata = Path(TRUSTED_GIT).lstat()
    except OSError:
        return False
    return (
        metadata.st_uid == 0
        and metadata.st_mode & 0o022 == 0
        and stat.S_ISREG(metadata.st_mode)
    )


def _git_text(root: Path, *args: str) -> str:
    if not _trusted_git_binary():
        raise ValueError("TRUSTED_GIT_INVALID")
    completed = subprocess.run(
        (TRUSTED_GIT, "-C", str(root), *args),
        check=False,
        capture_output=True,
        env=HERMETIC_GIT_ENV,
    )
    if completed.returncode != 0:
        raise ValueError("SOURCE_PROVENANCE_GIT_FAILED")
    return completed.stdout.decode("utf-8", errors="strict")


def _reject_unsafe_git_mechanisms(root: Path) -> None:
    if _git_text(root, "for-each-ref", "--format=%(refname)", "refs/replace").strip():
        raise ValueError("SOURCE_CONTEXT_UNSAFE_GIT_MECHANISM")
    for name in ("info/grafts", "objects/info/alternates", "shallow"):
        raw = _git_text(root, "rev-parse", "--git-path", name).strip()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                raise ValueError("SOURCE_CONTEXT_UNSAFE_GIT_MECHANISM")
        except OSError as error:
            raise ValueError("SOURCE_CONTEXT_UNSAFE_GIT_MECHANISM") from error


def _tree_digest(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(entries):
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _create_docker_context(
    root: Path,
    destination: Path,
    expected_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a sealed context exclusively from committed Git objects."""
    if destination.exists():
        raise ValueError("SOURCE_CONTEXT_ALREADY_EXISTS")
    destination.mkdir(mode=0o700, parents=True)
    _reject_unsafe_git_mechanisms(root)
    expected_modules: dict[str, dict[str, object]] = {}
    if expected_identity is not None:
        raw_modules = expected_identity.get("submodules", [])
        if not isinstance(raw_modules, list):
            raise ValueError("SOURCE_CONTEXT_IDENTITY_MISMATCH")
        expected_modules = {
            str(item.get("name")): item
            for item in raw_modules
            if isinstance(item, dict)
        }
        root_head = str(expected_identity.get("head", ""))
        root_identity = expected_identity.get("identity", {})
        root_tree = (
            str(root_identity.get("head_tree", ""))
            if isinstance(root_identity, dict)
            else ""
        )
        if (
            GIT_HASH.fullmatch(root_head) is None
            or GIT_HASH.fullmatch(root_tree) is None
        ):
            raise ValueError("SOURCE_CONTEXT_IDENTITY_MISMATCH")
    else:
        root_head = _git_text(root, "rev-parse", "HEAD").strip()
        root_tree = _git_text(root, "rev-parse", f"{root_head}^{{tree}}").strip()
    resolved_root_tree = _git_text(root, "rev-parse", f"{root_head}^{{tree}}").strip()
    if resolved_root_tree != root_tree:
        raise ValueError("SOURCE_CONTEXT_IDENTITY_MISMATCH")
    groups: dict[str, list[tuple[str, str]]] = {"root": [], "protobuf": []}
    count = 0
    total = 0

    def archive(
        repository: Path, revision: str, prefix: str, group: str | None
    ) -> list[tuple[str, str]]:
        nonlocal count, total
        completed = subprocess.run(
            (
                TRUSTED_GIT,
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                revision,
            ),
            check=False,
            capture_output=True,
            env=HERMETIC_GIT_ENV,
        )
        if completed.returncode != 0:
            raise ValueError("SOURCE_CONTEXT_ARCHIVE_FAILED")
        entries: list[tuple[str, str]] = []
        with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as bundle:
            for member in bundle:
                parts = Path(member.name).parts
                if not parts or member.name.startswith("/") or ".." in parts:
                    raise ValueError("SOURCE_CONTEXT_UNSAFE_FILE")
                relative = str(Path(prefix) / member.name) if prefix else member.name
                target = destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.size > 64 * 1024 * 1024:
                    raise ValueError("SOURCE_CONTEXT_UNSAFE_FILE")
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError("SOURCE_CONTEXT_ARCHIVE_FAILED")
                content = source.read()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                target.chmod(member.mode & 0o777)
                digest = hashlib.sha256(content).hexdigest()
                entries.append((member.name, digest))
                if group is not None:
                    groups[group].append((member.name, digest))
                count += 1
                total += len(content)
        return entries

    archive(root, root_head, "", "root")
    submodules: dict[str, dict[str, str]] = {}
    for module, gitlink in SUBMODULES.items():
        module_root = root / module
        _reject_unsafe_git_mechanisms(module_root)
        expected = expected_modules.get(module.rsplit("/", 1)[-1], {})
        head = (
            str(expected.get("head", ""))
            if expected_identity is not None
            else _git_text(module_root, "rev-parse", "HEAD").strip()
        )
        head_tree = _git_text(module_root, "rev-parse", f"{head}^{{tree}}").strip()
        if (
            GIT_HASH.fullmatch(head) is None
            or GIT_HASH.fullmatch(head_tree) is None
            or head != gitlink
        ):
            raise ValueError("SOURCE_CONTEXT_SUBMODULE_MISMATCH")
        if expected_identity is not None:
            identity = expected.get("identity", {})
            if (
                expected.get("gitlink") != gitlink
                or expected.get("head") != head
                or not isinstance(identity, dict)
                or identity.get("head_tree") != head_tree
            ):
                raise ValueError("SOURCE_CONTEXT_IDENTITY_MISMATCH")
        group = "protobuf" if module == "third_party/protobuf" else None
        module_entries = archive(module_root, head, module, group)
        submodules[module] = {
            "gitlink": gitlink,
            "head": head,
            "tree_sha256": _tree_digest(module_entries),
        }
    if total > 1024 * 1024 * 1024:
        raise ValueError("SOURCE_CONTEXT_TOO_LARGE")
    protobuf = submodules.get("third_party/protobuf", {})
    manifest: dict[str, object] = {
        "format": 1,
        "root_head": root_head,
        "root_tree_sha256": _tree_digest(groups["root"]),
        "protobuf_gitlink": SUBMODULES.get("third_party/protobuf", ""),
        "protobuf_head": protobuf.get("head", ""),
        "protobuf_tree_sha256": protobuf.get("tree_sha256", _tree_digest([])),
        "submodules": submodules,
        "file_count": count,
        "total_bytes": total,
    }
    sealed_manifest = dict(manifest)
    sealed_manifest["files"] = {
        "root": dict(groups["root"]),
        "protobuf": dict(groups["protobuf"]),
    }
    (destination / "source-provenance.json").write_text(
        json.dumps(sealed_manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest


class Verification:
    def __init__(self, profile: str, root: Path, runner: Runner, run_id: str):
        self.profile = profile
        self.root = root.resolve()
        self.runner = runner
        self.run_id = run_id
        self.documents: dict[str, dict[str, object]] = {
            name: {"commands": []} for name in DOCUMENT_FILES
        }
        self.gates: dict[str, dict[str, str]] = {}
        self.apple_clang = False
        self.apple_clang_version = ""
        self.git: dict[str, object] = {"head": "", "dirty": True, "submodules": []}
        self.dependencies: dict[str, object] = {
            "abseil": {"expected": ABSEIL_COMMIT, "resolved": ""}
        }
        self.gate_events: list[Any] = []
        self.event_tick = 0
        self.source_provenance: dict[str, object] | None = None
        self.assets, self.asset_digest, self.missing_assets = _hash_assets(self.root)
        runtime_key = hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        runtime = Path("/tmp") / f"grserve-verify-{runtime_key}"
        (runtime / "home").mkdir(mode=0o700, parents=True, exist_ok=True)
        (runtime / "tmp").mkdir(mode=0o700, parents=True, exist_ok=True)
        docker_config = runtime / "home/.docker"
        docker_config.mkdir(mode=0o700, exist_ok=True)
        docker_config.chmod(0o700)
        docker_config_file = docker_config / "config.json"
        docker_config_file.write_text(
            json.dumps(
                {
                    "cliPluginsExtraDirs": [
                        "/Applications/Docker.app/Contents/Resources/cli-plugins"
                    ]
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        docker_config_file.chmod(0o600)
        self.base_env = {
            "PATH": os.pathsep.join(
                dict.fromkeys(
                    (
                        str(Path(sys.executable).parent),
                        "/opt/homebrew/bin",
                        "/usr/local/bin",
                        "/usr/bin",
                        "/bin",
                        "/usr/sbin",
                        "/sbin",
                    )
                )
            ),
            "LANG": "C",
            "LC_ALL": "C",
            "HOME": str(runtime / "home"),
            "TMPDIR": str(runtime / "tmp"),
            "PYTHONNOUSERSITE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_PLUGINS": "pytest_asyncio.plugin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        self.runtime = runtime

    def checkout_identity(
        self, doc: str, prefix: str, path: str | None = None
    ) -> dict[str, object]:
        base = (TRUSTED_GIT,) if path is None else (TRUSTED_GIT, "-C", path)
        slug = re.sub(r"[^a-zA-Z0-9_.-]", "-", prefix)
        temporary_index = self.runtime / "tmp" / f"{self.run_id}-{slug}.index"
        temporary_index.unlink(missing_ok=True)
        isolated = {"GIT_INDEX_FILE": str(temporary_index)}
        flags_v = self.command(
            doc, f"{prefix}.index-flags-v", base + ("ls-files", "-v")
        )
        flags_t = self.command(
            doc, f"{prefix}.index-flags-t", base + ("ls-files", "-t")
        )
        untracked = self.command(
            doc,
            f"{prefix}.untracked",
            base + ("ls-files", "--others", "--exclude-standard", "-z"),
        )
        replacements = self.command(
            doc,
            f"{prefix}.replace-refs",
            base + ("for-each-ref", "--format=%(refname)", "refs/replace"),
        )
        shallow = self.command(
            doc,
            f"{prefix}.shallow",
            base + ("rev-parse", "--is-shallow-repository"),
        )
        mechanism_paths = [
            self.command(
                doc,
                f"{prefix}.{cid}",
                base + ("rev-parse", "--git-path", name),
            )
            for cid, name in (
                ("grafts-path", "info/grafts"),
                ("alternates-path", "objects/info/alternates"),
                ("shallow-path", "shallow"),
            )
        ]
        repository = self.root if path is None else self.root / path

        def populated_mechanism(result: CommandResult) -> bool:
            candidate = Path(result.stdout.strip())
            if not candidate.is_absolute():
                candidate = repository / candidate
            try:
                return candidate.is_file() and candidate.stat().st_size > 0
            except OSError:
                return True

        head_tree = self.command(
            doc, f"{prefix}.head-tree", base + ("rev-parse", "HEAD^{tree}")
        )
        index_tree = self.command(doc, f"{prefix}.index-tree", base + ("write-tree",))
        read_tree = self.command(
            doc,
            f"{prefix}.isolated-read-tree",
            base + ("read-tree", "HEAD"),
            env=isolated,
        )
        add_worktree = self.command(
            doc,
            f"{prefix}.isolated-add-worktree",
            base + ("add", "-u", "--", "."),
            env=isolated,
        )
        worktree_tree = self.command(
            doc, f"{prefix}.worktree-tree", base + ("write-tree",), env=isolated
        )
        temporary_index.unlink(missing_ok=True)
        head_value = head_tree.stdout.strip()
        bad_assume = any(
            line and line[0].islower() for line in flags_v.stdout.splitlines()
        )
        bad_skip = any(line.startswith("S ") for line in flags_t.stdout.splitlines())
        valid = (
            all(
                item.exit_code == 0
                for item in (
                    flags_v,
                    flags_t,
                    untracked,
                    replacements,
                    shallow,
                    *mechanism_paths,
                    head_tree,
                    index_tree,
                    read_tree,
                    add_worktree,
                    worktree_tree,
                )
            )
            and not bad_assume
            and not bad_skip
            and not untracked.stdout
            and not replacements.stdout
            and shallow.stdout.strip() == "false"
            and not any(populated_mechanism(item) for item in mechanism_paths)
            and GIT_HASH.fullmatch(head_value) is not None
            and index_tree.stdout.strip() == head_value
            and worktree_tree.stdout.strip() == head_value
        )
        return {
            "clean": valid,
            "head_tree": head_value,
            "index_tree": index_tree.stdout.strip(),
            "worktree_tree": worktree_tree.stdout.strip(),
            "index_flags_clean": not bad_assume and not bad_skip,
        }

    def command(
        self,
        doc: str,
        cid: str,
        argv: tuple[str, ...],
        *,
        env: dict[str, str] | None = None,
        input_text: str | None = None,
    ) -> CommandResult:
        command_env = dict(self.base_env)
        if env:
            command_env.update(env)
        result = self.runner(
            argv, cwd=self.root, env=command_env, input_text=input_text
        )
        commands = self.documents[doc]["commands"]
        assert isinstance(commands, list)
        commands.append(_record(cid, argv, result))
        return result

    def gate(
        self,
        name: str,
        doc: str,
        commands: list[tuple[str, tuple[str, ...]]],
        *,
        env: dict[str, str] | None = None,
        record: bool = True,
    ) -> bool:
        passed = True
        self.gates[name] = {"status": "INCOMPLETE"}
        try:
            for cid, argv in commands:
                response = self.command(doc, cid, argv, env=env)
                if response.exit_code != 0:
                    passed = False
                    break
                if cid.endswith(".test-manifest") and not self._valid_test_manifest(
                    response
                ):
                    passed = False
                    break
                if (
                    argv
                    and argv[0] == "ctest"
                    and "-N" not in argv
                    and re.search(r"out of ([1-9][0-9]*)", response.stdout) is None
                ):
                    passed = False
                    break
                if (
                    argv
                    and argv[0] == "pytest"
                    and re.search(
                        r"(?:^|\s)([1-9][0-9]*) passed(?:\s|,)", response.stdout
                    )
                    is None
                ):
                    passed = False
                    break
        except BaseException:
            self.gates[name] = {"status": "INCOMPLETE", "reason": "COMMAND_INTERRUPTED"}
            self.record_gate(name, doc)
            raise
        self.gates[name] = {"status": "PASS" if passed else "FAIL"}
        if not passed:
            self.gates[name]["reason"] = "COMMAND_FAILED"
        if record:
            self.record_gate(name, doc)
        return passed

    @staticmethod
    def _valid_test_manifest(response: CommandResult) -> bool:
        try:
            document = json.loads(response.stdout)
            names = {item["name"] for item in document["tests"]}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return names == EXPECTED_CTESTS

    def attest_abseil(self, preset: str, gate: str, doc: str) -> bool:
        source = f"build/{preset}/_deps/absl-src"
        prefix = {
            "mac-arm64-debug": "mac.debug",
            "mac-arm64-release": "mac.release",
            "mac-arm64-asan-ubsan": "mac.asan",
            "mac-arm64-tsan": "mac.tsan",
        }[preset]
        results = {
            "head": self.command(
                doc,
                f"{prefix}.abseil-head",
                (TRUSTED_GIT, "-C", source, "rev-parse", "HEAD"),
            ),
            "status": self.command(
                doc,
                f"{prefix}.abseil-status",
                (
                    TRUSTED_GIT,
                    "-C",
                    source,
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ),
            ),
            "staged": self.command(
                doc,
                f"{prefix}.abseil-staged",
                (TRUSTED_GIT, "-C", source, "diff", "--cached", "--quiet"),
            ),
            "tree": self.command(
                doc,
                f"{prefix}.abseil-tree",
                (TRUSTED_GIT, "-C", source, "ls-tree", "-r", "HEAD"),
            ),
        }
        head = results["head"].stdout.strip()
        tree = hashlib.sha256(results["tree"].stdout.encode()).hexdigest()
        clean = (
            results["status"].exit_code == 0
            and not results["status"].stdout
            and results["staged"].exit_code == 0
            and results["tree"].exit_code == 0
        )
        identity = self.checkout_identity(doc, f"{prefix}.abseil-identity", source)
        clean = clean and bool(identity["clean"])
        expected_tree = (
            self.dependencies["abseil"].get("tree_sha256")
            if isinstance(self.dependencies["abseil"], dict)
            else None
        )
        valid = head == ABSEIL_COMMIT and clean and (expected_tree in (None, tree))
        abseil = self.dependencies["abseil"]
        assert isinstance(abseil, dict)
        abseil.setdefault("tree_sha256", tree)
        presets = abseil.setdefault("presets", {})
        assert isinstance(presets, dict)
        presets[preset] = {
            "head": head,
            "clean": clean,
            "tree_sha256": tree,
            "identity": identity,
        }
        if preset == "mac-arm64-debug":
            abseil["resolved"] = head
        if not valid:
            self.gates[gate] = {"status": "FAIL", "reason": "ABSEIL_PIN_MISMATCH"}
        return valid

    def preset_gate(
        self,
        name: str,
        doc: str,
        preset: str,
        commands: list[tuple[str, tuple[str, ...]]],
        *,
        env: dict[str, str] | None = None,
    ) -> bool:
        """Keep a preset gate incomplete through all post-build attestations."""
        self.gates[name] = {"status": "INCOMPLETE"}
        try:
            passed = self.gate(name, doc, commands, env=env, record=False)
            self.gates[name] = {"status": "INCOMPLETE"}
            toolchain = self.record_cmake_toolchain(preset, name, doc)
            abseil = self.attest_abseil(preset, name, doc)
            passed = passed and toolchain and abseil
            if passed:
                self.gates[name] = {"status": "PASS"}
            elif self.gates[name].get("status") != "FAIL":
                self.gates[name] = {"status": "FAIL", "reason": "COMMAND_FAILED"}
            self.record_gate(name, doc)
            return passed
        except BaseException:
            self.gates[name] = {"status": "INCOMPLETE", "reason": "COMMAND_INTERRUPTED"}
            self.record_gate(name, doc)
            raise

    def record_cmake_toolchain(self, preset: str, gate: str, doc: str) -> bool:
        build = self.root / "build" / preset
        try:
            cache_text = (build / "CMakeCache.txt").read_text(encoding="utf-8")
            compiler_files = list((build / "CMakeFiles").glob("*/CMake*Compiler.cmake"))
            compiler_text = "\n".join(
                path.read_text(encoding="utf-8") for path in compiler_files
            )
        except OSError:
            cache_text = compiler_text = ""

        def cache(name: str) -> str:
            match = re.search(
                rf"^{re.escape(name)}:[^=]*=(.*)$", cache_text, re.MULTILINE
            )
            return match.group(1) if match else ""

        def compiler(language: str, field: str) -> str:
            match = re.search(
                rf'set\(CMAKE_{language}_COMPILER_{field} "([^"]*)"\)', compiler_text
            )
            return match.group(1) if match else ""

        identity = {
            "c_compiler": Path(cache("CMAKE_C_COMPILER")).name,
            "cxx_compiler": Path(cache("CMAKE_CXX_COMPILER")).name,
            "c_id": compiler("C", "ID"),
            "cxx_id": compiler("CXX", "ID"),
            "c_version": compiler("C", "VERSION"),
            "cxx_version": compiler("CXX", "VERSION"),
            "sysroot": cache("CMAKE_OSX_SYSROOT"),
            "deployment_target": cache("CMAKE_OSX_DEPLOYMENT_TARGET"),
        }
        toolchains = self.documents["build.json"].setdefault("toolchains", {})
        assert isinstance(toolchains, dict)
        toolchains[preset] = identity
        valid = (
            identity["c_id"] == "AppleClang"
            and identity["cxx_id"] == "AppleClang"
            and bool(identity["c_compiler"])
            and bool(identity["cxx_compiler"])
            and str(identity["c_version"]).startswith(self.apple_clang_version)
            and str(identity["cxx_version"]).startswith(self.apple_clang_version)
            and identity["sysroot"] == "macosx"
            and identity["deployment_target"] == "14.0"
        )
        if not valid:
            self.gates[gate] = {
                "status": "FAIL",
                "reason": "TOOLCHAIN_IDENTITY_MISMATCH",
            }
        return valid

    def record_gate(
        self, name: str, doc: str, command_ids: list[str] | None = None
    ) -> None:
        if command_ids is None:
            commands = self.documents[doc]["commands"]
            assert isinstance(commands, list)
            prefixes = {
                "Debug": ("mac.debug.",),
                "Release": ("mac.release.",),
                "C11 ABI": ("mac.cpp.c11-abi",),
                "C++": ("mac.cpp.all",),
                "manifest": ("mac.cpp.manifest",),
                "fault matrix": ("mac.cpp.fault-matrix",),
                "ASan/UBSan": ("mac.asan.",),
                "TSan": ("mac.tsan.",),
                "Python": ("mac.python.",),
                "proto": ("mac.proto.",),
                "golden": ("mac.golden.",),
                "API E2E": ("mac.api-e2e.",),
                "Linux amd64": ("linux.",),
                "preflight": ("preflight.", "final."),
            }.get(name, ("preflight.",))
            command_ids = [
                str(item["command_id"])
                for item in commands
                if isinstance(item, dict)
                and str(item.get("command_id", "")).startswith(prefixes)
            ]
        self.gate_events = [
            event for event in self.gate_events if event.gate_id != name
        ]
        gate = self.gates[name]
        status_number = getattr(event_pb2, "VERIFICATION_GATE_STATUS_" + gate["status"])
        self.gate_events.append(
            getattr(event_pb2, "VerificationGateEvent")(
                schema_version=1,
                monotonic_tick=self.event_tick,
                run_id=self.run_id,
                gate_id=name,
                status=status_number,
                reason=gate.get("reason", ""),
                command_id=command_ids,
                fault_plan_seed=FAULT_PLAN_SEED if name == "fault matrix" else 0,
            )
        )
        self.event_tick += 1

    def revalidate_identity(self) -> bool:
        original = dict(self.gates.get("preflight", {"status": "INCOMPLETE"}))
        self.gates["preflight"] = {"status": "INCOMPLETE"}
        self.record_gate("preflight", "preflight.json")
        try:
            valid = self._revalidate_identity_impl()
            if original.get("status") == "FAIL":
                self.gates["preflight"] = original
            elif valid:
                self.gates["preflight"] = original
            else:
                self.gates["preflight"] = {
                    "status": "FAIL",
                    "reason": "IDENTITY_CHANGED_DURING_VERIFY",
                }
            self.record_gate("preflight", "preflight.json")
            return valid
        except BaseException:
            self.gates["preflight"] = {
                "status": "INCOMPLETE",
                "reason": "COMMAND_INTERRUPTED",
            }
            self.record_gate("preflight", "preflight.json")
            raise

    def _revalidate_identity_impl(self) -> bool:
        head = self.command(
            "preflight.json", "final.git-head", (TRUSTED_GIT, "rev-parse", "HEAD")
        )
        status = self.command(
            "preflight.json",
            "final.git-status",
            (TRUSTED_GIT, "status", "--porcelain", "--untracked-files=all"),
        )
        staged = self.command(
            "preflight.json",
            "final.git-staged",
            (TRUSTED_GIT, "diff", "--cached", "--quiet"),
        )
        valid = (
            head.exit_code == 0
            and head.stdout.strip() == self.git.get("head")
            and status.exit_code == 0
            and not status.stdout
            and staged.exit_code == 0
        )
        final_root_identity = self.checkout_identity(
            "preflight.json", "final.root-identity"
        )
        initial_root_identity = self.git.get("identity", {})
        valid &= bool(final_root_identity["clean"]) and (
            isinstance(initial_root_identity, dict)
            and final_root_identity["head_tree"]
            == initial_root_identity.get("head_tree")
        )
        raw_modules = self.git.get("submodules", [])
        assert isinstance(raw_modules, list)
        original_modules = {
            item["name"]: item for item in raw_modules if isinstance(item, dict)
        }
        for path in SUBMODULES:
            key = path.rsplit("/", 1)[-1]
            mh = self.command(
                "preflight.json",
                f"final.submodule-{key}-head",
                (TRUSTED_GIT, "-C", path, "rev-parse", "HEAD"),
            )
            ms = self.command(
                "preflight.json",
                f"final.submodule-{key}-status",
                (
                    TRUSTED_GIT,
                    "-C",
                    path,
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ),
            )
            mc = self.command(
                "preflight.json",
                f"final.submodule-{key}-staged",
                (TRUSTED_GIT, "-C", path, "diff", "--cached", "--quiet"),
            )
            original = original_modules.get(key, {})
            final_identity = self.checkout_identity(
                "preflight.json", f"final.submodule-{key}-identity", path
            )
            original_identity = original.get("identity", {})
            valid &= (
                mh.exit_code == 0
                and mh.stdout.strip() == original.get("head")
                and ms.exit_code == 0
                and not ms.stdout
                and mc.exit_code == 0
                and bool(final_identity["clean"])
                and isinstance(original_identity, dict)
                and final_identity["head_tree"] == original_identity.get("head_tree")
            )
        assets, digest, missing = _hash_assets(self.root)
        valid &= not missing and assets == self.assets and digest == self.asset_digest
        return bool(valid)

    def events_jsonl(self) -> bytes:
        lines = [
            json_format.MessageToJson(
                event,
                preserving_proto_field_name=False,
                indent=None,
                sort_keys=True,
                always_print_fields_with_no_presence=True,
            )
            for event in self.gate_events
        ]
        return ("\n".join(lines) + "\n").encode()

    def preflight(self, need_mac: bool, need_docker: bool) -> bool:
        passed = True
        reason = None
        uv = self.command("preflight.json", "preflight.uv-version", ("uv", "--version"))
        lock = self.command(
            "preflight.json", "preflight.uv-lock", ("uv", "lock", "--check")
        )
        passed &= (
            uv.exit_code == 0
            and re.fullmatch(
                rf"uv {re.escape(UV_VERSION)}(?: \([^\r\n]+\))?", uv.stdout.strip()
            )
            is not None
        )
        lock_output = lock.stdout + "\n" + lock.stderr
        lock_semantic_error = re.search(
            r"(?i)(lock mismatch|lockfile.*out.of.date|would change the lock|invalid lock)",
            lock_output,
        )
        passed &= lock.exit_code == 0 and lock_semantic_error is None
        trusted_git = self.command(
            "preflight.json", "preflight.trusted-git", (TRUSTED_GIT, "--version")
        )
        passed &= (
            trusted_git.exit_code == 0
            and trusted_git.stdout.startswith("git version ")
            and _trusted_git_binary()
        )
        if need_mac:
            xcode = self.command(
                "preflight.json", "preflight.xcode", ("xcodebuild", "-version")
            )
            clang = self.command(
                "preflight.json", "preflight.clang", ("clang", "--version")
            )
            cmake = self.command(
                "preflight.json", "preflight.cmake", ("cmake", "--version")
            )
            ninja = self.command(
                "preflight.json", "preflight.ninja", ("ninja", "--version")
            )
            sdk = self.command(
                "preflight.json",
                "preflight.macos-sdk",
                ("xcrun", "--sdk", "macosx", "--show-sdk-path"),
            )
            macos = self.command(
                "preflight.json",
                "preflight.macos-version",
                ("sw_vers", "-productVersion"),
            )
            self.apple_clang = clang.exit_code == 0 and "Apple clang" in clang.stdout
            version_match = re.search(r"Apple clang version ([^\s]+)", clang.stdout)
            self.apple_clang_version = version_match.group(1) if version_match else ""
            clt = (
                xcode.exit_code != 0
                and "CommandLineTools" in xcode.stderr
                and "command line tools instance" in xcode.stderr
            )
            self.documents["preflight.json"]["toolchainKind"] = (
                "CLT" if clt else "XCODE"
            )
            self.documents["build.json"]["compiler"] = (
                clang.stdout.splitlines()[0] if clang.stdout else "unavailable"
            )
            self.documents["build.json"]["presets"] = [
                "mac-arm64-debug",
                "mac-arm64-release",
                "mac-arm64-asan-ubsan",
                "mac-arm64-tsan",
            ]
            passed &= (
                (xcode.exit_code == 0 or clt)
                and self.apple_clang
                and cmake.exit_code == 0
                and ninja.exit_code == 0
                and sdk.exit_code == 0
                and sdk.stdout.strip().endswith(".sdk")
                and macos.exit_code == 0
                and bool(macos.stdout.strip())
            )
        head = self.command(
            "preflight.json",
            "preflight.git-head",
            (TRUSTED_GIT, "rev-parse", "HEAD"),
        )
        status = self.command(
            "preflight.json",
            "preflight.git-status",
            (TRUSTED_GIT, "status", "--porcelain", "--untracked-files=all"),
        )
        staged = self.command(
            "preflight.json",
            "preflight.git-staged",
            (TRUSTED_GIT, "diff", "--cached", "--quiet"),
        )
        root_clean = (
            status.exit_code == 0 and not status.stdout and staged.exit_code == 0
        )
        root_identity = self.checkout_identity(
            "preflight.json", "preflight.root-identity"
        )
        root_clean &= bool(root_identity["clean"])
        head_ok = (
            head.exit_code == 0 and GIT_HASH.fullmatch(head.stdout.strip()) is not None
        )
        self.git["head"] = head.stdout.strip() if head_ok else ""
        self.git["dirty"] = not root_clean
        self.git["identity"] = root_identity
        passed &= root_clean and head_ok
        listed = self.command(
            "preflight.json",
            "preflight.submodule-status",
            (TRUSTED_GIT, "submodule", "status", "--recursive"),
        )
        parsed = {}
        for line in listed.stdout.splitlines():
            match = re.fullmatch(
                r" ([0-9a-f]{40}) (third_party/(?:googletest|protobuf))(?: \([^\r\n]+\))?",
                line,
            )
            if match:
                parsed[match.group(2)] = match.group(1)
        modules = []
        modules_ok = listed.exit_code == 0 and parsed == SUBMODULES
        for path, gitlink in SUBMODULES.items():
            key = path.rsplit("/", 1)[-1]
            mh = self.command(
                "preflight.json",
                f"preflight.submodule-{key}-head",
                (TRUSTED_GIT, "-C", path, "rev-parse", "HEAD"),
            )
            ms = self.command(
                "preflight.json",
                f"preflight.submodule-{key}-status",
                (
                    TRUSTED_GIT,
                    "-C",
                    path,
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ),
            )
            mc = self.command(
                "preflight.json",
                f"preflight.submodule-{key}-staged",
                (TRUSTED_GIT, "-C", path, "diff", "--cached", "--quiet"),
            )
            actual = mh.stdout.strip()
            clean = ms.exit_code == 0 and not ms.stdout and mc.exit_code == 0
            identity = self.checkout_identity(
                "preflight.json", f"preflight.submodule-{key}-identity", path
            )
            clean &= bool(identity["clean"])
            pinned = (
                mh.exit_code == 0 and actual == gitlink and parsed.get(path) == gitlink
            )
            modules.append(
                {
                    "name": key,
                    "gitlink": gitlink,
                    "head": actual,
                    "clean": clean,
                    "identity": identity,
                }
            )
            modules_ok &= clean and pinned
        self.git["submodules"] = modules
        passed &= modules_ok
        if self.missing_assets:
            passed = False
            reason = "REQUIRED_ASSET_MISSING"
        try:
            apt_lock = (self.root / "docker/apt-packages.lock").read_text(
                encoding="utf-8"
            )
            dockerfile = (self.root / "docker/Dockerfile.linux-amd64").read_text(
                encoding="utf-8"
            )
            apt_match = APT_LOCK_PATTERN.fullmatch(apt_lock)
        except OSError:
            apt_match = None
            dockerfile = ""
        if apt_match is None or any(
            value not in dockerfile for value in apt_match.groups()
        ):
            passed = False
            reason = "APT_LOCK_INVALID"
        else:
            self.dependencies["apt"] = {
                "snapshot": apt_match.group(1),
                "build-essential": apt_match.group(2),
                "git": apt_match.group(3),
                "lock_sha256": self.assets.get("docker/apt-packages.lock", ""),
            }
        if need_docker:
            client = self.command(
                "preflight.json", "preflight.docker-client", ("docker", "--version")
            )
            if client.exit_code != 0:
                self.gates["Linux amd64"] = {
                    "status": "FAIL",
                    "reason": "DOCKER_DAEMON_UNAVAILABLE",
                }
                passed = False
            else:
                context = self.command(
                    "preflight.json",
                    "preflight.docker-context",
                    (
                        "docker",
                        "context",
                        "inspect",
                        "--format",
                        "{{json .Endpoints.docker.Host}}",
                    ),
                )
                try:
                    endpoint = json.loads(context.stdout)
                except (TypeError, ValueError, json.JSONDecodeError):
                    endpoint = ""
                local_endpoint = isinstance(endpoint, str) and (
                    (endpoint.startswith("unix:///") and ".." not in endpoint)
                    or endpoint.startswith("npipe:////./pipe/")
                )
                if context.exit_code != 0 or not local_endpoint:
                    self.gates["Linux amd64"] = {
                        "status": "FAIL",
                        "reason": "DOCKER_REMOTE_CONTEXT_FORBIDDEN",
                    }
                    passed = False
                else:
                    daemon = self.command(
                        "preflight.json",
                        "preflight.docker-daemon",
                        ("docker", "info", "--format", "{{.ServerVersion}}"),
                    )
                    if daemon.exit_code != 0:
                        self.gates["Linux amd64"] = {
                            "status": "FAIL",
                            "reason": "DOCKER_DAEMON_UNAVAILABLE",
                        }
                        passed = False
                    else:
                        platforms = self.command(
                            "preflight.json",
                            "preflight.docker-platform",
                            ("docker", "buildx", "inspect", "--bootstrap"),
                        )
                        if (
                            platforms.exit_code != 0
                            or not _buildx_supports_linux_amd64(platforms.stdout)
                        ):
                            self.gates["Linux amd64"] = {
                                "status": "FAIL",
                                "reason": "DOCKER_PLATFORM_UNAVAILABLE",
                            }
                            passed = False
        self.gates["preflight"] = {"status": "PASS" if passed else "FAIL"}
        if not passed and reason is None:
            reason = "PREFLIGHT_FAILED"
        if reason:
            self.gates["preflight"]["reason"] = reason
        if "Linux amd64" in self.gates:
            preflight_commands = self.documents["preflight.json"]["commands"]
            assert isinstance(preflight_commands, list)
            self.record_gate(
                "Linux amd64",
                "linux-amd64.json",
                [
                    str(item["command_id"])
                    for item in preflight_commands
                    if isinstance(item, dict)
                    and str(item.get("command_id", "")).startswith("preflight.docker-")
                ],
            )
        self.record_gate("preflight", "preflight.json")
        return passed

    def mac(self) -> None:
        self.preset_gate(
            "Debug",
            "build.json",
            "mac-arm64-debug",
            [
                (
                    "mac.debug.configure",
                    ("cmake", "--preset", "mac-arm64-debug", "--fresh"),
                ),
                (
                    "mac.debug.build",
                    ("cmake", "--build", "--preset", "mac-arm64-debug"),
                ),
                (
                    "mac.debug.test-manifest",
                    (
                        "ctest",
                        "-N",
                        "--show-only=json-v1",
                        "--preset",
                        "mac-arm64-debug",
                    ),
                ),
                (
                    "mac.debug.ctest",
                    ("ctest", "--preset", "mac-arm64-debug", "--no-tests=error"),
                ),
            ],
        )
        self.preset_gate(
            "Release",
            "build.json",
            "mac-arm64-release",
            [
                (
                    "mac.release.configure",
                    ("cmake", "--preset", "mac-arm64-release", "--fresh"),
                ),
                (
                    "mac.release.build",
                    ("cmake", "--build", "--preset", "mac-arm64-release"),
                ),
                (
                    "mac.release.test-manifest",
                    (
                        "ctest",
                        "-N",
                        "--show-only=json-v1",
                        "--preset",
                        "mac-arm64-release",
                    ),
                ),
                (
                    "mac.release.ctest",
                    ("ctest", "--preset", "mac-arm64-release", "--no-tests=error"),
                ),
            ],
        )
        self.gate(
            "C11 ABI",
            "cpp-tests.json",
            [
                (
                    "mac.cpp.c11-abi",
                    (
                        "ctest",
                        "--preset",
                        "mac-arm64-debug",
                        "--no-tests=error",
                        "-R",
                        "^c\\.hal_header$",
                    ),
                )
            ],
        )
        self.gate(
            "C++",
            "cpp-tests.json",
            [
                (
                    "mac.cpp.all",
                    (
                        "ctest",
                        "--preset",
                        "mac-arm64-debug",
                        "--no-tests=error",
                        "-R",
                        "^cpp\\.",
                    ),
                )
            ],
        )
        self.gate(
            "manifest",
            "cpp-tests.json",
            [
                (
                    "mac.cpp.manifest",
                    (
                        "ctest",
                        "--preset",
                        "mac-arm64-debug",
                        "--no-tests=error",
                        "-R",
                        "^cpp\\.manifest$",
                    ),
                )
            ],
        )
        self.gate(
            "fault matrix",
            "cpp-tests.json",
            [
                (
                    "mac.cpp.fault-matrix",
                    (
                        "ctest",
                        "--preset",
                        "mac-arm64-debug",
                        "--no-tests=error",
                        "-R",
                        "^(cpp\\.backend_parity|cpp\\.fault_plan|cpp\\.controller_faults)$",
                    ),
                )
            ],
            env={"GRSERVE_FAULT_PLAN_SEED": str(FAULT_PLAN_SEED)},
        )
        sanenv = {
            "ASAN_OPTIONS": "abort_on_error=1:detect_leaks=0",
            "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
        }
        self.preset_gate(
            "ASan/UBSan",
            "sanitizer.json",
            "mac-arm64-asan-ubsan",
            [
                (
                    "mac.asan.configure",
                    ("cmake", "--preset", "mac-arm64-asan-ubsan", "--fresh"),
                ),
                (
                    "mac.asan.build",
                    ("cmake", "--build", "--preset", "mac-arm64-asan-ubsan"),
                ),
                (
                    "mac.asan.test-manifest",
                    (
                        "ctest",
                        "-N",
                        "--show-only=json-v1",
                        "--preset",
                        "mac-arm64-asan-ubsan",
                    ),
                ),
                (
                    "mac.asan.ctest",
                    ("ctest", "--preset", "mac-arm64-asan-ubsan", "--no-tests=error"),
                ),
            ],
            env=sanenv,
        )
        self.tsan()
        self.gate(
            "Python",
            "python-tests.json",
            [
                (
                    "mac.python.ruff",
                    ("ruff", "check", "python", "tools", "tests/python"),
                ),
                (
                    "mac.python.mypy",
                    (
                        "mypy",
                        "--explicit-package-bases",
                        "python/grserve_api",
                        "tools/verify.py",
                        "tools/evidence.py",
                        "tools/acceptance.py",
                        "tools/normalize_events.py",
                        "tools/proto_codegen.py",
                        "tools/generate_mini_model.py",
                    ),
                ),
                (
                    "mac.python.pytest",
                    (
                        "pytest",
                        "-q",
                        "tests/python",
                        "--ignore=tests/python/test_verify_tool.py",
                        "--ignore=tests/python/test_proto_codegen.py",
                        "--ignore=tests/python/test_model_assets.py",
                        "--ignore=tests/python/test_api_e2e.py",
                    ),
                ),
            ],
            env={"MYPYPATH": "python"},
        )
        self.gate(
            "proto",
            "python-tests.json",
            [
                (
                    "mac.proto.generated",
                    (
                        "cmake",
                        "--build",
                        "--preset",
                        "mac-arm64-debug",
                        "--target",
                        "check_proto_generated",
                    ),
                ),
                (
                    "mac.proto.pytest",
                    ("pytest", "-q", "tests/python/test_proto_codegen.py"),
                ),
            ],
        )
        self.gate(
            "golden",
            "python-tests.json",
            [
                (
                    "mac.golden.pytest",
                    ("pytest", "-q", "tests/python/test_model_assets.py"),
                )
            ],
        )
        self.gate(
            "API E2E",
            "python-tests.json",
            [("mac.api-e2e.pytest", ("pytest", "-q", "tests/python/test_api_e2e.py"))],
            env={
                "GRSERVE_ENGINE_BIN": str(
                    self.root / "build/mac-arm64-debug/bin/grserve-engine"
                )
            },
        )

    def tsan(self) -> None:
        self.gates["TSan"] = {"status": "INCOMPLETE"}
        (self.root / "build/tsan-capability").mkdir(parents=True, exist_ok=True)
        source = "#include <thread>\nint main(){int x=0; std::thread t([&]{x=1;}); t.join(); return x!=1;}\n"
        compiled = self.command(
            "sanitizer.json",
            "mac.tsan.probe-compile",
            (
                "clang++",
                "-std=c++20",
                "-fsanitize=thread",
                "-x",
                "c++",
                "-",
                "-o",
                "build/tsan-capability/probe",
            ),
            input_text=source,
            env={"TSAN_OPTIONS": "halt_on_error=1"},
        )
        if compiled.exit_code != 0:
            self.gates["TSan"] = {"status": "FAIL", "reason": "COMMAND_FAILED"}
            self.record_gate("TSan", "sanitizer.json")
            return
        runtime = self.command(
            "sanitizer.json",
            "mac.tsan.probe-run",
            ("build/tsan-capability/probe",),
            env={"TSAN_OPTIONS": "halt_on_error=1"},
        )
        if runtime.exit_code != 0:
            unsupported = self.apple_clang and TSAN_UNSUPPORTED_PATTERN.search(
                runtime.stderr
            )
            if unsupported:
                try:
                    configured = self.command(
                        "sanitizer.json",
                        "mac.tsan.configure",
                        ("cmake", "--preset", "mac-arm64-tsan", "--fresh"),
                        env={"TSAN_OPTIONS": "halt_on_error=1"},
                    )
                    toolchain = self.record_cmake_toolchain(
                        "mac-arm64-tsan", "TSan", "sanitizer.json"
                    )
                    abseil = self.attest_abseil(
                        "mac-arm64-tsan", "TSan", "sanitizer.json"
                    )
                    if configured.exit_code == 0 and toolchain and abseil:
                        self.gates["TSan"] = {
                            "status": "SKIP",
                            "reason": "TSAN_RUNTIME_UNSUPPORTED_APPLE_CLANG",
                        }
                    elif self.gates["TSan"].get("status") != "FAIL":
                        self.gates["TSan"] = {
                            "status": "FAIL",
                            "reason": "COMMAND_FAILED",
                        }
                except BaseException:
                    self.gates["TSan"] = {
                        "status": "INCOMPLETE",
                        "reason": "COMMAND_INTERRUPTED",
                    }
                    self.record_gate("TSan", "sanitizer.json")
                    raise
            else:
                self.gates["TSan"] = {"status": "FAIL", "reason": "COMMAND_FAILED"}
            self.record_gate("TSan", "sanitizer.json")
            return
        self.preset_gate(
            "TSan",
            "sanitizer.json",
            "mac-arm64-tsan",
            [
                (
                    "mac.tsan.configure",
                    ("cmake", "--preset", "mac-arm64-tsan", "--fresh"),
                ),
                ("mac.tsan.build", ("cmake", "--build", "--preset", "mac-arm64-tsan")),
                (
                    "mac.tsan.test-manifest",
                    (
                        "ctest",
                        "-N",
                        "--show-only=json-v1",
                        "--preset",
                        "mac-arm64-tsan",
                    ),
                ),
                (
                    "mac.tsan.ctest",
                    ("ctest", "--preset", "mac-arm64-tsan", "--no-tests=error"),
                ),
            ],
            env={"TSAN_OPTIONS": "halt_on_error=1"},
        )

    def linux(self) -> None:
        if "Linux amd64" in self.gates:
            return
        try:
            base = (self.root / "docker/base-image.lock").read_text().strip()
        except OSError:
            base = ""
        if BASE_IMAGE_PATTERN.fullmatch(base) is None:
            self.gates["Linux amd64"] = {
                "status": "FAIL",
                "reason": "BASE_IMAGE_LOCK_INVALID",
            }
            self.record_gate("Linux amd64", "linux-amd64.json")
            return
        tag = f"grserve-linux-amd64:{uuid.uuid4().hex[:12]}"
        private_parent = Path(
            tempfile.mkdtemp(prefix="verify-context-", dir=self.root / "build")
        )
        context = private_parent / "context"
        try:
            self.source_provenance = _create_docker_context(
                self.root, context, expected_identity=self.git
            )
        except (OSError, ValueError):
            self.gates["Linux amd64"] = {
                "status": "FAIL",
                "reason": "SOURCE_CONTEXT_INVALID",
            }
            self.record_gate("Linux amd64", "linux-amd64.json")
            shutil.rmtree(private_parent, ignore_errors=True)
            return
        try:
            self.gate(
                "Linux amd64",
                "linux-amd64.json",
                [
                    (
                        "linux.docker-build",
                        (
                            "docker",
                            "build",
                            "--platform",
                            "linux/amd64",
                            "--build-arg",
                            f"BASE_IMAGE={base}",
                            "-f",
                            str(context / "docker/Dockerfile.linux-amd64"),
                            "-t",
                            tag,
                            str(context),
                        ),
                    ),
                    (
                        "linux.docker-run",
                        ("docker", "run", "--rm", "--platform", "linux/amd64", tag),
                    ),
                ],
            )
        finally:
            shutil.rmtree(private_parent, ignore_errors=True)

    def execute(self) -> None:
        mac = self.profile in {"mac", "full"}
        docker = self.profile in {"linux-amd64", "full"}
        preflight_ok = self.preflight(mac, docker)
        if mac:
            self.mac()
        if docker and not preflight_ok and "Linux amd64" not in self.gates:
            self.gates["Linux amd64"] = {
                "status": "FAIL",
                "reason": "SOURCE_IDENTITY_INVALID",
            }
            self.record_gate("Linux amd64", "linux-amd64.json")
        if docker:
            self.linux()


def _finalize(v: Verification, incomplete: bool) -> None:
    for docname, doc in v.documents.items():
        relevant = [name for name in DOCUMENT_GATES[docname] if name in v.gates]
        doc["gates"] = {name: v.gates[name] for name in relevant}
        if not relevant:
            doc["status"] = (
                "INCOMPLETE" if incomplete and doc["commands"] else "NOT_RUN"
            )
        elif any(v.gates[n]["status"] == "FAIL" for n in relevant):
            doc["status"] = "FAIL"
        elif any(v.gates[n]["status"] == "INCOMPLETE" for n in relevant):
            doc["status"] = "INCOMPLETE"
        elif all(v.gates[n]["status"] == "SKIP" for n in relevant):
            doc["status"] = "SKIP"
        else:
            doc["status"] = "PASS"
        failures = [
            v.gates[n].get("reason")
            for n in relevant
            if v.gates[n]["status"] == "FAIL" and v.gates[n].get("reason")
        ]
        if len(relevant) == 1 and failures:
            doc["reason"] = failures[0]


def _buildx_supports_linux_amd64(output: str) -> bool:
    platforms: set[str] = set()
    for line in output.splitlines():
        label, separator, values = line.partition(":")
        if separator and label.strip() == "Platforms":
            platforms.update(value.strip() for value in values.split(",") if value.strip())
    return "linux/amd64" in platforms


@contextmanager
def _publication_critical_region() -> Iterator[None]:
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        try:
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
        except ValueError:
            previous = {}
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _validate_published_identity(destination: Path, run_id: str, profile: str) -> None:
    try:
        published = json.loads((destination / "summary.json").read_text())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise EvidenceError("EVIDENCE_IDENTITY_MISMATCH") from error
    if (
        destination.name != run_id
        or published.get("profile") != profile
        or published.get("run_id") != run_id
    ):
        raise EvidenceError("EVIDENCE_IDENTITY_MISMATCH")


def _run_verification_unlocked(
    profile: str,
    repository_root: Path,
    *,
    runner: Runner = subprocess_runner,
    run_id: str | None = None,
) -> VerificationOutcome:
    if profile not in {"mac", "linux-amd64", "full"}:
        raise ValueError("invalid profile")
    root = repository_root.resolve()
    selected = run_id or _run_id()
    v = Verification(profile, root, runner, selected)
    exit_code = 0
    status = "PASS"
    reason = None
    try:
        v.execute()
        v.revalidate_identity()
        if any(g["status"] == "FAIL" for g in v.gates.values()):
            exit_code, status = 1, "FAIL"
    except VerificationInterrupted as error:
        exit_code, status, reason = (
            128 + error.signum,
            "INCOMPLETE",
            f"INTERRUPTED_SIGNAL_{error.signum}",
        )
    except BaseException:
        exit_code, status, reason = 2, "INCOMPLETE", "INTERNAL_ERROR"
    try:
        with _publication_critical_region():
            if status == "INCOMPLETE" and not v.gate_events:
                v.gates["preflight"] = {"status": "INCOMPLETE"}
                v.record_gate("preflight", "preflight.json")
            if status == "INCOMPLETE":
                recorded = {event.gate_id for event in v.gate_events}
                for gate_name, gate in list(v.gates.items()):
                    if gate["status"] != "INCOMPLETE" or gate_name in recorded:
                        continue
                    document = next(
                        name
                        for name, names in DOCUMENT_GATES.items()
                        if gate_name in names
                    )
                    v.record_gate(gate_name, document)
            _finalize(v, status == "INCOMPLETE")
            critical = sum(
                1
                for name, gate in v.gates.items()
                if gate["status"] == "SKIP" and name != "TSan"
            )
            if critical:
                exit_code, status = 1, "FAIL"
            summary: dict[str, object] = {
                "profile": profile,
                "status": status,
                "gates": v.gates,
                "critical_skip_count": critical,
                "git": v.git,
                "assets": v.assets,
                "dependencies": v.dependencies,
                "fault_plan_seed": FAULT_PLAN_SEED,
            }
            if v.source_provenance is not None:
                summary["source_provenance"] = v.source_provenance
            if reason:
                summary["reason"] = reason
            destination = write_evidence(
                root,
                selected,
                documents=v.documents,
                events=v.events_jsonl(),
                assets_sha256=v.asset_digest,
                summary=summary,
            )
            _validate_published_identity(destination, selected, profile)
    except EvidenceError as error:
        stable = str(error) or "EVIDENCE_PUBLICATION_FAILED"
        return VerificationOutcome(2, None, stable)
    except BaseException:
        return VerificationOutcome(2, None, "EVIDENCE_PUBLICATION_INTERRUPTED")
    return VerificationOutcome(exit_code, destination, reason)


def run_verification(
    profile: str,
    repository_root: Path,
    *,
    runner: Runner = subprocess_runner,
    run_id: str | None = None,
) -> VerificationOutcome:
    root = repository_root.resolve()
    lock_path = root / "build/.grserve-verify.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return VerificationOutcome(2, None, "VERIFICATION_ALREADY_RUNNING")
        return _run_verification_unlocked(profile, root, runner=runner, run_id=run_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("mac", "linux-amd64", "full"), required=True
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def interrupt(signum: int, _frame: object) -> None:
        raise VerificationInterrupted(signum)

    previous = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        outcome = run_verification(args.profile, root)
        if (
            outcome.evidence_dir is not None
            and (outcome.evidence_dir / "summary.json").is_file()
        ):
            print(f"GRSERVE_SUMMARY={outcome.evidence_dir / 'summary.json'}")
        elif outcome.reason:
            print(f"GRSERVE_VERIFY_ERROR={outcome.reason}", file=sys.stderr)
        return outcome.exit_code
    except VerificationInterrupted as error:
        return 128 + error.signum
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
