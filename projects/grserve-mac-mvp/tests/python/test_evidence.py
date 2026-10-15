from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.evidence import ARTIFACT_FILES, EvidenceError, write_evidence
from tools.normalize_events import NormalizationError, normalize_events


REPO_ROOT = Path(__file__).resolve().parents[2]


def _event(
    *,
    tick: str = "10",
    run_id: str = "20260907T101112Z-a1b2c3d4",
    state: str = "EVENT_STATE_PUBLISHED",
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "monotonicTick": tick,
        "runId": run_id,
        "requestId": "41",
        "transactionId": "52",
        "rank": 1,
        "epoch": "63",
        "state": state,
        "error": "EVENT_ERROR_CODE_NONE",
        "sequenceNumber": "7",
        "delivery": "DELIVERY_OUTCOME_DELIVERED",
    }


def _jsonl(*events: dict[str, object]) -> bytes:
    return b"".join(
        json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events
    )


def _gate_event() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "monotonicTick": "3",
        "runId": "20260907T101112Z-a1b2c3d4",
        "gateId": "fault matrix",
        "status": "VERIFICATION_GATE_STATUS_PASS",
        "reason": "",
        "commandId": ["mac.cpp.fault-matrix"],
        "faultPlanSeed": "17",
    }


def _documents() -> dict[str, object]:
    return {
        "preflight.json": {"status": "PASS"},
        "build.json": {"status": "PASS"},
        "cpp-tests.json": {"status": "PASS"},
        "python-tests.json": {"status": "PASS"},
        "sanitizer.json": {"status": "PASS"},
        "linux-amd64.json": {"status": "PASS"},
    }


def test_normalization_ignores_only_tick_and_run_id() -> None:
    first = normalize_events(_jsonl(_event()))
    second = normalize_events(
        _jsonl(_event(tick="999999", run_id="20260908T111213Z-deadbeef"))
    )
    assert first.canonical == second.canonical
    assert first.sha256 == second.sha256
    assert first.canonical == (
        b'[{"delivery":"DELIVERY_OUTCOME_DELIVERED","epoch":"63",'
        b'"error":"EVENT_ERROR_CODE_NONE","rank":1,"requestId":"41",'
        b'"schemaVersion":1,"sequenceNumber":"7",'
        b'"state":"EVENT_STATE_PUBLISHED","transactionId":"52"}]'
    )


def test_normalization_preserves_event_order_and_semantics() -> None:
    published = _event(tick="10")
    fatal = _event(tick="11", state="EVENT_STATE_FATAL")
    fatal_first = _event(tick="10", state="EVENT_STATE_FATAL")
    published_second = _event(tick="11")
    assert (
        normalize_events(_jsonl(published, fatal)).sha256
        != normalize_events(_jsonl(fatal_first, published_second)).sha256
    )
    assert (
        normalize_events(_jsonl(published)).sha256
        != normalize_events(_jsonl(fatal)).sha256
    )


@pytest.mark.parametrize("ticks", [("1", "1"), ("2", "1")])
def test_normalization_rejects_duplicate_or_decreasing_ticks(
    ticks: tuple[str, str],
) -> None:
    with pytest.raises(NormalizationError, match="strictly increasing"):
        normalize_events(_jsonl(_event(tick=ticks[0]), _event(tick=ticks[1])))


@pytest.mark.parametrize("schema_version", [True, 1.0, "1", 2**100])
def test_normalization_requires_exact_integer_schema_version(
    schema_version: object,
) -> None:
    with pytest.raises(NormalizationError):
        normalize_events(_jsonl({**_event(), "schemaVersion": schema_version}))


def test_normalization_accepts_canonical_integer_schema_version() -> None:
    normalized = normalize_events(_jsonl(_event()))
    assert b'"schemaVersion":1' in normalized.canonical


def test_normalization_accepts_typed_verification_gate_events() -> None:
    normalized = normalize_events(_jsonl(_gate_event()))
    assert normalized.event_count == 1
    assert b'"gateId":"fault matrix"' in normalized.canonical
    assert b'"faultPlanSeed":"17"' in normalized.canonical


@pytest.mark.parametrize(
    "content",
    [
        b'{"schemaVersion":1,"schemaVersion":1}\n',
        b"{bad json}\n",
        _jsonl({**_event(), "unknown": "private"}),
        _jsonl({**_event(), "rank": -1}),
        _jsonl({**_event(), "requestId": 41}),
        b"\n",
        b"{" + b"x" * (1024 * 1024) + b"}\n",
        b"[" * 2000 + b"0" + b"]" * 2000 + b"\n",
    ],
)
def test_normalization_rejects_invalid_or_unbounded_input(content: bytes) -> None:
    with pytest.raises(NormalizationError):
        normalize_events(content)


@pytest.mark.parametrize(
    "field",
    ["monotonicTick", "requestId", "transactionId", "epoch", "sequenceNumber"],
)
def test_normalization_rejects_huge_string_encoded_uint64(field: str) -> None:
    with pytest.raises(NormalizationError):
        normalize_events(_jsonl({**_event(), field: "9" * 5000}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_writer_rejects_non_finite_numbers_recursively(
    tmp_path: Path, value: float
) -> None:
    documents = _documents()
    documents["build.json"] = {"nested": [{"value": value}]}
    with pytest.raises(EvidenceError, match="^INVALID_ARTIFACT_DOCUMENT$"):
        write_evidence(
            tmp_path,
            "20260907T101112Z-a1b2c3d4",
            documents=documents,
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )


def test_writer_preserves_boolean_type_semantics(tmp_path: Path) -> None:
    documents = _documents()
    documents["build.json"] = {
        "nested": {"flag": True},
        "prompt_token_count": 3,
        "fault_plan_seed": 17,
    }
    directory = write_evidence(
        tmp_path,
        "20260907T101112Z-a1b2c3d4",
        documents=documents,
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
    )
    assert json.loads((directory / "build.json").read_bytes()) == {
        "fault_plan_seed": 17,
        "nested": {"flag": True},
        "prompt_token_count": 3,
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("prompt_token_count", True),
        ("prompt_token_count", 0),
        ("prompt_token_count", 65),
        ("prompt_token_count", "3"),
        ("prompt_token_count", [3]),
        ("prompt_token_count", {"value": 3}),
        ("prompt_token_count", 3.0),
        ("fault_plan_seed", True),
        ("fault_plan_seed", -1),
        ("fault_plan_seed", 2**64),
        ("fault_plan_seed", "17"),
        ("fault_plan_seed", [17]),
        ("fault_plan_seed", {"value": 17}),
        ("fault_plan_seed", 17.0),
    ],
)
def test_structured_redaction_whitelist_requires_bounded_integer_scalar(
    tmp_path: Path, key: str, value: object
) -> None:
    documents = _documents()
    documents["build.json"] = {key: value}
    with pytest.raises(EvidenceError, match="^INVALID_ARTIFACT_DOCUMENT$"):
        write_evidence(
            tmp_path,
            "20260907T101112Z-a1b2c3d4",
            documents=documents,
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )


def test_writer_redacts_sensitive_nested_keys_without_collisions(
    tmp_path: Path,
) -> None:
    documents = _documents()
    documents["build.json"] = {
        "nested": {
            "/tmp/private.sock": "socket value",
            "backend private message": "diagnostic",
        }
    }
    directory = write_evidence(
        tmp_path,
        "20260907T101112Z-a1b2c3d4",
        documents=documents,
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
    )
    content = (directory / "build.json").read_text()
    assert "/tmp/private.sock" not in content
    assert "backend private message" not in content

    collision_documents = _documents()
    collision_documents["build.json"] = {
        "/tmp/one": 1,
        "/tmp/two": 2,
    }
    with pytest.raises(EvidenceError, match="^INVALID_ARTIFACT_DOCUMENT$"):
        write_evidence(
            tmp_path,
            "20260907T101113Z-a1b2c3d4",
            documents=collision_documents,
            events=_jsonl(_event(run_id="20260907T101113Z-a1b2c3d4")),
            assets_sha256="a" * 64,
        )


def test_writer_does_not_exempt_sensitive_strings_containing_safe_words(
    tmp_path: Path,
) -> None:
    documents = _documents()
    documents["build.json"] = {
        "message": "prompt tokens count=3 values=101,102,103",
        "detail": "fault plan seed=17 rank=1 after=prepare",
        "nested_prompt": {"prompt tokens count=3 values=101": "private key"},
        "nested_fault": {"fault plan seed=17 rank=1": "private key"},
    }
    directory = write_evidence(
        tmp_path,
        "20260907T101112Z-a1b2c3d4",
        documents=documents,
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
    )
    content = (directory / "build.json").read_text()
    for secret in (
        "prompt tokens count=3 values=101,102,103",
        "fault plan seed=17 rank=1 after=prepare",
        "prompt tokens count=3 values=101",
        "fault plan seed=17 rank=1",
    ):
        assert secret not in content


def test_writer_creates_exact_secure_manifest(tmp_path: Path) -> None:
    run_id = "20260907T101112Z-a1b2c3d4"
    events = _jsonl(_event(run_id=run_id))
    directory = write_evidence(
        tmp_path,
        run_id,
        documents=_documents(),
        events=events,
        assets_sha256="a" * 64,
    )

    assert directory == tmp_path / "artifacts" / "mac-mvp" / run_id
    assert {path.name for path in directory.iterdir()} == set(ARTIFACT_FILES)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o500
    for path in directory.iterdir():
        assert stat.S_ISREG(path.lstat().st_mode)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    digest = normalize_events(events).sha256
    assert (directory / "normalized-events.sha256").read_text() == digest + "\n"
    assert (directory / "assets.sha256").read_text() == "a" * 64 + "\n"
    summary = json.loads((directory / "summary.json").read_bytes())
    assert summary["run_id"] == run_id
    assert summary["normalized_events_sha256"] == digest
    assert summary["assets_sha256"] == "a" * 64
    for name, expected in summary["artifact_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected


def test_writer_preserves_caller_summary_for_later_acceptance(tmp_path: Path) -> None:
    directory = write_evidence(
        tmp_path,
        "20260907T101112Z-a1b2c3d4",
        documents=_documents(),
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
        summary={"profile": "mac", "gates": {"Debug": {"status": "PASS"}}},
    )
    summary = json.loads((directory / "summary.json").read_bytes())
    assert summary["profile"] == "mac"
    assert summary["gates"] == {"Debug": {"status": "PASS"}}


def test_writer_redacts_private_paths_from_public_documents(tmp_path: Path) -> None:
    documents = _documents()
    documents["build.json"] = {
        "command": ["compiler", "--socket=/tmp/private.sock"],
        "backend": "/Users/USER/private/backend.dylib",
    }
    directory = write_evidence(
        tmp_path,
        "20260907T101112Z-a1b2c3d4",
        documents=documents,
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
    )
    content = (directory / "build.json").read_text()
    assert "/tmp/private.sock" not in content
    assert "/Users/USER/private/backend.dylib" not in content
    assert "<redacted-path>" in content


def test_writer_never_overwrites_existing_run(tmp_path: Path) -> None:
    run_id = "20260907T101112Z-a1b2c3d4"
    existing = tmp_path / "artifacts" / "mac-mvp" / run_id
    existing.mkdir(parents=True)
    marker = existing / "keep"
    marker.write_text("untouched")

    with pytest.raises(EvidenceError, match="^ARTIFACT_DIR_EXISTS$"):
        write_evidence(
            tmp_path,
            run_id,
            documents=_documents(),
            events=_jsonl(_event(run_id=run_id)),
            assets_sha256="a" * 64,
        )
    assert marker.read_text() == "untouched"
    assert list(existing.iterdir()) == [marker]


def test_existing_parent_directories_are_not_modified_on_existing_run(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    parent = artifacts / "mac-mvp"
    existing = parent / "20260907T101112Z-a1b2c3d4"
    existing.mkdir(parents=True)
    (artifacts / "artifacts-marker").write_text("artifacts")
    (parent / "parent-marker").write_text("parent")
    (existing / "run-marker").write_text("run")
    artifacts.chmod(0o755)
    parent.chmod(0o755)
    before = {
        path: (path.stat().st_ino, stat.S_IMODE(path.stat().st_mode))
        for path in (artifacts, parent, existing)
    }

    with pytest.raises(EvidenceError, match="^ARTIFACT_DIR_EXISTS$"):
        write_evidence(
            tmp_path,
            existing.name,
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )

    after = {
        path: (path.stat().st_ino, stat.S_IMODE(path.stat().st_mode))
        for path in (artifacts, parent, existing)
    }
    assert after == before
    assert (artifacts / "artifacts-marker").read_text() == "artifacts"
    assert (parent / "parent-marker").read_text() == "parent"
    assert (existing / "run-marker").read_text() == "run"


def test_existing_world_writable_parent_fails_without_chmod(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    artifacts.chmod(0o777)

    with pytest.raises(EvidenceError, match="^UNSAFE_ARTIFACT_PATH$"):
        write_evidence(
            tmp_path,
            "20260907T101112Z-a1b2c3d4",
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )
    assert stat.S_IMODE(artifacts.stat().st_mode) == 0o777
    assert list(artifacts.iterdir()) == []


def test_concurrent_same_run_has_one_winner_and_no_staging_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    lock_attempts = context.Queue()
    results = context.Queue()
    release = context.Event()
    staging_creations = context.Value("i", 0)
    original_rename = evidence._rename_directory_exclusive
    original_mkdir = evidence.os.mkdir
    original_flock = evidence.fcntl.flock

    first_pid: int | None = None

    def synchronized_rename(parent_fd: int, source: str, destination: str) -> None:
        nonlocal first_pid
        if first_pid is None:
            first_pid = os.getpid()
            ready.put(first_pid)
            if not release.wait(timeout=10):
                raise RuntimeError("rename barrier timed out")
        original_rename(parent_fd, source, destination)

    def count_staging(name: str, *args: object, **kwargs: object) -> None:
        if name == ".pending":
            with staging_creations.get_lock():
                staging_creations.value += 1
        original_mkdir(name, *args, **kwargs)

    def observe_lock(descriptor: int, operation: int) -> None:
        if operation == evidence.fcntl.LOCK_EX:
            lock_attempts.put(os.getpid())
        original_flock(descriptor, operation)

    monkeypatch.setattr(evidence, "_rename_directory_exclusive", synchronized_rename)
    monkeypatch.setattr(evidence.os, "mkdir", count_staging)
    monkeypatch.setattr(evidence.fcntl, "flock", observe_lock)
    run_id = "20260907T101112Z-a1b2c3d4"

    def create() -> None:
        try:
            write_evidence(
                tmp_path,
                run_id,
                documents=_documents(),
                events=_jsonl(_event()),
                assets_sha256="a" * 64,
            )
        except EvidenceError as error:
            results.put(str(error))
        else:
            results.put("SUCCESS")

    processes = [context.Process(target=create) for _ in range(2)]
    processes[0].start()
    assert lock_attempts.get(timeout=10) == processes[0].pid
    assert ready.get(timeout=10) == processes[0].pid
    processes[1].start()
    assert lock_attempts.get(timeout=10) == processes[1].pid
    release.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sorted(results.get(timeout=2) for _ in processes) == [
        "ARTIFACT_ALREADY_COMMITTED",
        "SUCCESS",
    ]
    assert staging_creations.value == 1
    parent = tmp_path / "artifacts" / "mac-mvp"
    assert [path.name for path in parent.iterdir()] == [run_id]


def test_post_rename_fsync_failure_has_stable_commit_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    original_fsync = evidence.os.fsync
    original_rename = evidence._rename_directory_exclusive
    committed = False
    failed = False

    def track_commit(parent_fd: int, source: str, destination: str) -> None:
        nonlocal committed
        original_rename(parent_fd, source, destination)
        committed = True

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal failed
        if committed and not failed:
            failed = True
            raise OSError("injected parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(evidence.os, "fsync", fail_parent_fsync)
    monkeypatch.setattr(evidence, "_rename_directory_exclusive", track_commit)
    run_id = "20260907T101112Z-a1b2c3d4"
    arguments = dict(
        documents=_documents(),
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
    )
    with pytest.raises(EvidenceError, match="^ARTIFACT_DURABILITY_UNKNOWN$"):
        write_evidence(tmp_path, run_id, **arguments)
    directory = tmp_path / "artifacts" / "mac-mvp" / run_id
    assert {path.name for path in directory.iterdir()} == set(ARTIFACT_FILES)

    monkeypatch.setattr(evidence.os, "fsync", original_fsync)
    with pytest.raises(EvidenceError, match="^ARTIFACT_ALREADY_COMMITTED$"):
        write_evidence(tmp_path, run_id, **arguments)


def test_post_rename_freeze_failure_has_stable_commit_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    original_fchmod = evidence.os.fchmod
    original_rename = evidence._rename_directory_exclusive
    committed = False

    def track_commit(parent_fd: int, source: str, destination: str) -> None:
        nonlocal committed
        original_rename(parent_fd, source, destination)
        committed = True

    def fail_final_freeze(descriptor: int, mode: int) -> None:
        if committed and mode == 0o500:
            raise OSError("injected final freeze failure")
        original_fchmod(descriptor, mode)

    monkeypatch.setattr(evidence, "_rename_directory_exclusive", track_commit)
    monkeypatch.setattr(evidence.os, "fchmod", fail_final_freeze)
    run_id = "20260907T101112Z-a1b2c3d4"
    arguments = dict(
        documents=_documents(),
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
    )
    with pytest.raises(EvidenceError, match="^ARTIFACT_DURABILITY_UNKNOWN$"):
        write_evidence(tmp_path, run_id, **arguments)
    directory = tmp_path / "artifacts" / "mac-mvp" / run_id
    assert stat.S_IMODE(directory.stat().st_mode) == 0o600
    assert {path.name for path in directory.iterdir()} == set(ARTIFACT_FILES)

    monkeypatch.setattr(evidence.os, "fchmod", original_fchmod)
    with pytest.raises(EvidenceError, match="^ARTIFACT_DURABILITY_UNKNOWN$"):
        write_evidence(tmp_path, run_id, **arguments)


def test_writer_never_creates_or_unlinks_a_named_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    original_write = evidence._write_regular_file

    def fixed_artifact_only(
        directory_fd: int, name: str, content: bytes
    ) -> tuple[int, int]:
        assert name in ARTIFACT_FILES
        return original_write(directory_fd, name, content)

    def forbidden_unlink(*args: object, **kwargs: object) -> None:
        raise AssertionError("publisher must not unlink by name")

    monkeypatch.setattr(evidence, "_write_regular_file", fixed_artifact_only)
    monkeypatch.setattr(evidence.os, "unlink", forbidden_unlink)
    directory = write_evidence(
        tmp_path,
        "20260907T101112Z-a1b2c3d4",
        documents=_documents(),
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
    )
    assert {path.name for path in directory.iterdir()} == set(ARTIFACT_FILES)


def test_pending_is_frozen_before_final_check_and_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    original_rename = evidence._rename_directory_exclusive
    original_write = evidence._write_regular_file
    attempted = False
    written_identities: dict[str, tuple[int, int]] = {}

    def record_identity(
        directory_fd: int, name: str, content: bytes
    ) -> tuple[int, int]:
        identity = original_write(directory_fd, name, content)
        written_identities[name] = identity
        return identity

    def attempt_late_write(parent_fd: int, source: str, destination: str) -> None:
        nonlocal attempted
        attempted = True
        staging_fd = os.open(source, evidence.DIRECTORY_FLAGS, dir_fd=parent_fd)
        try:
            with pytest.raises(PermissionError):
                os.open(
                    "foreign-marker",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=staging_fd,
                )
        finally:
            os.close(staging_fd)
        original_rename(parent_fd, source, destination)

    monkeypatch.setattr(evidence, "_write_regular_file", record_identity)
    monkeypatch.setattr(evidence, "_rename_directory_exclusive", attempt_late_write)
    directory = write_evidence(
        tmp_path,
        "20260907T101112Z-a1b2c3d4",
        documents=_documents(),
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
    )
    assert attempted
    assert stat.S_IMODE(directory.stat().st_mode) == 0o500
    assert {path.name for path in directory.iterdir()} == set(ARTIFACT_FILES)
    assert {
        path.name: (path.stat().st_dev, path.stat().st_ino)
        for path in directory.iterdir()
    } == written_identities


def test_freeze_binds_each_directory_entry_to_its_preopened_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    original_fchmod = evidence.os.fchmod
    replacement_identity: tuple[int, int] | None = None
    replaced = False

    def replace_summary_before_freeze(descriptor: int, mode: int) -> None:
        nonlocal replaced, replacement_identity
        if (
            mode == 0o600
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and not replaced
        ):
            replaced = True
            parent = tmp_path / "artifacts" / "mac-mvp"
            pending = parent / ".pending"
            (pending / "summary.json").rename(parent / "preserved-summary.json")
            foreign_fd = os.open(
                "summary.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=descriptor,
            )
            try:
                os.write(foreign_fd, b"foreign-summary")
                metadata = os.fstat(foreign_fd)
                replacement_identity = (metadata.st_dev, metadata.st_ino)
            finally:
                os.close(foreign_fd)
        original_fchmod(descriptor, mode)

    monkeypatch.setattr(evidence.os, "fchmod", replace_summary_before_freeze)
    run_id = "20260907T101112Z-a1b2c3d4"
    with pytest.raises(EvidenceError, match="^ARTIFACT_RECOVERY_REQUIRED$"):
        write_evidence(
            tmp_path,
            run_id,
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )

    parent = tmp_path / "artifacts" / "mac-mvp"
    pending = parent / ".pending"
    pending.chmod(0o700)
    foreign = pending / "summary.json"
    assert replaced
    assert (foreign.stat().st_dev, foreign.stat().st_ino) == replacement_identity
    assert foreign.read_bytes() == b"foreign-summary"
    assert (parent / "preserved-summary.json").is_file()
    assert not (parent / run_id).exists()


def test_existing_pending_requires_recovery_without_growth(tmp_path: Path) -> None:
    parent = tmp_path / "artifacts" / "mac-mvp"
    pending = parent / ".pending"
    pending.mkdir(parents=True, mode=0o700)
    marker = pending / "foreign-marker"
    marker.write_text("foreign", encoding="utf-8")
    before = (pending.stat().st_dev, pending.stat().st_ino, marker.read_bytes())

    for _ in range(3):
        with pytest.raises(EvidenceError, match="^ARTIFACT_RECOVERY_REQUIRED$"):
            write_evidence(
                tmp_path,
                "20260907T101112Z-a1b2c3d4",
                documents=_documents(),
                events=_jsonl(_event()),
                assets_sha256="a" * 64,
            )

    after = (pending.stat().st_dev, pending.stat().st_ino, marker.read_bytes())
    assert after == before
    assert [path.name for path in parent.iterdir()] == [".pending"]


def test_pending_created_between_check_and_mkdir_requires_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    original_mkdir = evidence.os.mkdir

    def race_mkdir(name: str, *args: object, **kwargs: object) -> None:
        if name == ".pending":
            original_mkdir(name, *args, **kwargs)
            raise FileExistsError("injected pending race")
        original_mkdir(name, *args, **kwargs)

    monkeypatch.setattr(evidence.os, "mkdir", race_mkdir)
    with pytest.raises(EvidenceError, match="^ARTIFACT_RECOVERY_REQUIRED$"):
        write_evidence(
            tmp_path,
            "20260907T101112Z-a1b2c3d4",
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )
    parent = tmp_path / "artifacts" / "mac-mvp"
    assert [path.name for path in parent.iterdir()] == [".pending"]


def test_pending_swap_between_mkdir_and_open_is_never_written_or_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    original_open = evidence.os.open
    swapped = False
    foreign_identity: tuple[int, int] | None = None

    def swap_after_open(name: str | Path, *args: object, **kwargs: object) -> int:
        nonlocal swapped, foreign_identity
        descriptor = original_open(name, *args, **kwargs)
        if name == ".pending" and not swapped:
            swapped = True
            parent_fd = kwargs["dir_fd"]
            assert isinstance(parent_fd, int)
            os.rename(
                ".pending",
                "preserved-owned",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.mkdir(".pending", 0o700, dir_fd=parent_fd)
            foreign_fd = original_open(
                ".pending", evidence.DIRECTORY_FLAGS, dir_fd=parent_fd
            )
            marker_fd = original_open(
                "foreign-marker",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=foreign_fd,
            )
            try:
                os.write(marker_fd, b"foreign-content")
                metadata = os.fstat(foreign_fd)
                foreign_identity = (metadata.st_dev, metadata.st_ino)
            finally:
                os.close(marker_fd)
                os.close(foreign_fd)
        return descriptor

    monkeypatch.setattr(evidence.os, "open", swap_after_open)
    run_id = "20260907T101112Z-a1b2c3d4"
    with pytest.raises(EvidenceError, match="^ARTIFACT_RECOVERY_REQUIRED$"):
        write_evidence(
            tmp_path,
            run_id,
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )

    parent = tmp_path / "artifacts" / "mac-mvp"
    foreign = parent / ".pending"
    assert (foreign.stat().st_dev, foreign.stat().st_ino) == foreign_identity
    assert (foreign / "foreign-marker").read_bytes() == b"foreign-content"
    assert list((parent / "preserved-owned").iterdir()) == []
    assert not (parent / run_id).exists()


def test_pending_with_extra_entry_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    original_write = evidence._write_regular_file

    def add_extra(directory_fd: int, name: str, content: bytes) -> tuple[int, int]:
        identity = original_write(directory_fd, name, content)
        if name == ARTIFACT_FILES[-1]:
            extra_fd = os.open(
                "foreign-marker",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(extra_fd, b"foreign-content")
            finally:
                os.close(extra_fd)
        return identity

    monkeypatch.setattr(evidence, "_write_regular_file", add_extra)
    run_id = "20260907T101112Z-a1b2c3d4"
    with pytest.raises(EvidenceError, match="^ARTIFACT_RECOVERY_REQUIRED$"):
        write_evidence(
            tmp_path,
            run_id,
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )
    parent = tmp_path / "artifacts" / "mac-mvp"
    assert ((parent / ".pending") / "foreign-marker").read_bytes() == b"foreign-content"
    assert not (parent / run_id).exists()


def test_write_failure_preserves_single_pending_for_manual_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    def fail_write(_directory_fd: int, _name: str, _content: bytes) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(evidence, "_write_regular_file", fail_write)
    with pytest.raises(EvidenceError, match="^ARTIFACT_RECOVERY_REQUIRED$"):
        write_evidence(
            tmp_path,
            "20260907T101112Z-a1b2c3d4",
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )
    parent = tmp_path / "artifacts" / "mac-mvp"
    pending = parent / ".pending"
    identity = (pending.stat().st_dev, pending.stat().st_ino)

    with pytest.raises(EvidenceError, match="^ARTIFACT_RECOVERY_REQUIRED$"):
        write_evidence(
            tmp_path,
            "20260907T101113Z-a1b2c3d4",
            documents=_documents(),
            events=_jsonl(_event(run_id="20260907T101113Z-a1b2c3d4")),
            assets_sha256="a" * 64,
        )
    assert (pending.stat().st_dev, pending.stat().st_ino) == identity
    assert [path.name for path in parent.iterdir()] == [".pending"]


def test_complete_artifact_swap_during_read_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    run_id = "20260907T101112Z-a1b2c3d4"
    arguments = dict(
        documents=_documents(),
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
    )
    directory = write_evidence(tmp_path, run_id, **arguments)
    parent = directory.parent
    preserved = parent / "preserved-complete"
    original_read_all = evidence._read_all
    swapped = False

    def swap_after_read(descriptor: int) -> bytes:
        nonlocal swapped
        content = original_read_all(descriptor)
        if not swapped:
            swapped = True
            directory.chmod(0o700)
            directory.rename(preserved)
            directory.mkdir(mode=0o700)
            (directory / "foreign-marker").write_text("foreign", encoding="utf-8")
        return content

    monkeypatch.setattr(evidence, "_read_all", swap_after_read)
    with pytest.raises(EvidenceError, match="^ARTIFACT_DIR_EXISTS$"):
        write_evidence(tmp_path, run_id, **arguments)
    assert (directory / "foreign-marker").read_text() == "foreign"
    assert {path.name for path in preserved.iterdir()} == set(ARTIFACT_FILES)


def test_complete_artifact_read_error_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    run_id = "20260907T101112Z-a1b2c3d4"
    arguments = dict(
        documents=_documents(),
        events=_jsonl(_event()),
        assets_sha256="a" * 64,
    )
    directory = write_evidence(tmp_path, run_id, **arguments)

    def fail_read(_descriptor: int) -> bytes:
        raise OSError("injected artifact read failure")

    monkeypatch.setattr(evidence, "_read_all", fail_read)
    with pytest.raises(EvidenceError, match="^ARTIFACT_DIR_EXISTS$"):
        write_evidence(tmp_path, run_id, **arguments)
    assert {path.name for path in directory.iterdir()} == set(ARTIFACT_FILES)


@pytest.mark.parametrize(
    "needle,replacement",
    [
        (b'"schemaVersion":1', b'"schemaVersion":' + b"9" * 5000),
        (b'"runId":"20260907T101112Z-a1b2c3d4"', b'"runId":' + b"9" * 5000),
    ],
)
def test_normalizer_cli_rejects_huge_integer_without_traceback(
    tmp_path: Path, needle: bytes, replacement: bytes
) -> None:
    path = tmp_path / "events.jsonl"
    content = _jsonl(_event()).replace(needle, replacement)
    path.write_bytes(content)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "normalize_events.py"), str(path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.startswith("EVENT_NORMALIZATION_FAILED:")
    assert "Traceback" not in result.stderr


def test_publication_race_preserves_pending_and_foreign_without_directory_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    foreign_identity: tuple[int, int] | None = None

    def lose_publication(parent_fd: int, _source: str, destination: str) -> None:
        nonlocal foreign_identity
        os.mkdir(destination, 0o700, dir_fd=parent_fd)
        foreign_fd = os.open(destination, evidence.DIRECTORY_FLAGS, dir_fd=parent_fd)
        marker_fd = -1
        try:
            marker_fd = os.open(
                "foreign-marker",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=foreign_fd,
            )
            os.write(marker_fd, b"foreign-content")
            metadata = os.fstat(foreign_fd)
            foreign_identity = (metadata.st_dev, metadata.st_ino)
        finally:
            if marker_fd >= 0:
                os.close(marker_fd)
            os.close(foreign_fd)
        raise FileExistsError("injected publication race")

    def forbidden_delete(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("evidence writer attempted destructive cleanup")

    monkeypatch.setattr(evidence, "_rename_directory_exclusive", lose_publication)
    monkeypatch.setattr(evidence.os, "rmdir", forbidden_delete)
    run_id = "20260907T101112Z-a1b2c3d4"
    with pytest.raises(EvidenceError, match="^ARTIFACT_RECOVERY_REQUIRED$"):
        write_evidence(
            tmp_path,
            run_id,
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )

    parent = tmp_path / "artifacts" / "mac-mvp"
    foreign = parent / run_id
    metadata = foreign.stat()
    assert (metadata.st_dev, metadata.st_ino) == foreign_identity
    assert (foreign / "foreign-marker").read_bytes() == b"foreign-content"
    assert {path.name for path in (parent / ".pending").iterdir()} == set(
        ARTIFACT_FILES
    )


@pytest.mark.parametrize(
    "run_id", ["../escape", "latest", "20260907/foo", "", ".hidden"]
)
def test_writer_rejects_invalid_run_ids(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(EvidenceError, match="^INVALID_RUN_ID$"):
        write_evidence(
            tmp_path,
            run_id,
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )


def test_writer_rejects_symlinked_artifact_path(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, tmp_path / "artifacts")
    with pytest.raises(EvidenceError, match="^UNSAFE_ARTIFACT_PATH$"):
        write_evidence(
            tmp_path,
            "20260907T101112Z-a1b2c3d4",
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )
    assert list(outside.iterdir()) == []


def test_writer_preserves_pending_and_does_not_claim_success_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    original = evidence._write_regular_file

    def fail_on_build(directory_fd: int, name: str, content: bytes) -> tuple[int, int]:
        if name == "build.json":
            raise OSError("disk full")
        return original(directory_fd, name, content)

    monkeypatch.setattr(evidence, "_write_regular_file", fail_on_build)
    run_id = "20260907T101112Z-a1b2c3d4"
    with pytest.raises(EvidenceError, match="^ARTIFACT_RECOVERY_REQUIRED$"):
        write_evidence(
            tmp_path,
            run_id,
            documents=_documents(),
            events=_jsonl(_event()),
            assets_sha256="a" * 64,
        )
    parent = tmp_path / "artifacts" / "mac-mvp"
    assert not (parent / run_id).exists()
    assert [path.name for path in parent.iterdir()] == [".pending"]
    assert [path.name for path in (parent / ".pending").iterdir()] == ["preflight.json"]


def test_repeated_write_failures_close_all_file_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.evidence as evidence

    def fail_write(_directory_fd: int, _name: str, _content: bytes) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(evidence, "_write_regular_file", fail_write)
    before = len(os.listdir("/dev/fd"))
    for ordinal in range(20):
        run_id = f"20260907T101112Z-{ordinal:08x}"
        with pytest.raises(EvidenceError, match="^ARTIFACT_RECOVERY_REQUIRED$"):
            write_evidence(
                tmp_path,
                run_id,
                documents=_documents(),
                events=_jsonl(_event(run_id=run_id)),
                assets_sha256="a" * 64,
            )
    assert len(os.listdir("/dev/fd")) == before
    parent = tmp_path / "artifacts" / "mac-mvp"
    assert [path.name for path in parent.iterdir()] == [".pending"]


def test_command_line_tools_are_directly_invocable() -> None:
    for tool in ("normalize_events.py", "evidence.py"):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / tool), "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
