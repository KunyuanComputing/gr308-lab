import fcntl
import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tools" / "generate_mini_model.py"
REFERENCE_MODEL = ROOT / "tests" / "reference_model.py"
ASSET_NAMES = ("manifest.textproto", "weights.i32le", "golden.json")


def run_generator(output: Path) -> None:
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(output)],
        cwd=ROOT,
        check=True,
    )


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_mini_model", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_is_byte_for_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_generator(first)
    run_generator(second)
    assert {name: (first / name).read_bytes() for name in ASSET_NAMES} == {
        name: (second / name).read_bytes() for name in ASSET_NAMES
    }


def test_concurrent_generators_share_output_without_partial_assets(
    tmp_path: Path,
) -> None:
    output = tmp_path / "model"
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_generator, output) for _ in range(16)]
        for future in futures:
            future.result()
    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == set(ASSET_NAMES)
    assert manifest_matches_weights(output)


def manifest_matches_weights(output: Path) -> bool:
    manifest_path = output / "manifest.textproto"
    weights_path = output / "weights.i32le"
    if not manifest_path.exists() or not weights_path.exists():
        return False
    if weights_path.stat().st_size != 456 * 4:
        return False
    manifest = manifest_path.read_text(encoding="utf-8")
    digest_line = next(
        line for line in manifest.splitlines() if line.startswith("weight_sha256:")
    )
    declared = digest_line.split('"', 2)[1]
    return declared == hashlib.sha256(weights_path.read_bytes()).hexdigest()


def staging_directories(parent: Path, output_name: str) -> list[Path]:
    prefix = f".{output_name}.staging-"
    return [entry for entry in parent.iterdir() if entry.name.startswith(prefix)]


def test_update_atomically_exchanges_complete_directories(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    original_build_weights = generator.build_weights
    monkeypatch.setattr(
        generator,
        "build_weights",
        lambda: bytes([original_build_weights()[0] ^ 1]) + original_build_weights()[1:],
    )
    observations: list[tuple[str, bool, bool]] = []

    def observe(phase: str, staging: Path) -> None:
        assert output.is_dir()
        observations.append(
            (phase, manifest_matches_weights(output), manifest_matches_weights(staging))
        )

    generator.generate(output, exchange_hook=observe)
    assert observations == [
        ("before_exchange", True, True),
        ("before_commit", True, True),
        ("after_exchange", True, True),
    ]
    assert {path.name for path in output.iterdir()} == set(ASSET_NAMES)


@pytest.mark.parametrize(
    ("crash_phase", "new_generation_visible"),
    [("before_exchange", False), ("after_exchange", True)],
)
def test_crash_around_exchange_keeps_complete_snapshot_and_retry_preserves_orphan(
    tmp_path: Path, monkeypatch, crash_phase: str, new_generation_visible: bool
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    old_weights = (output / "weights.i32le").read_bytes()
    original_build_weights = generator.build_weights
    new_weights = (
        original_build_weights()[:-4]
        + bytes([original_build_weights()[-4] ^ 1])
        + original_build_weights()[-3:]
    )
    monkeypatch.setattr(
        generator,
        "build_weights",
        lambda: new_weights,
    )

    context = multiprocessing.get_context("fork")

    def crash(phase: str, _staging: Path) -> None:
        assert output.is_dir()
        assert manifest_matches_weights(output)
        if phase == crash_phase:
            os._exit(73)

    process = context.Process(target=lambda: generator.generate(output, exchange_hook=crash))
    process.start()
    process.join()
    assert process.exitcode == 73
    assert output.is_dir()
    assert manifest_matches_weights(output)
    assert (output / "weights.i32le").read_bytes() == (
        new_weights if new_generation_visible else old_weights
    )
    generator.generate(output)
    assert manifest_matches_weights(output)
    assert (output / "weights.i32le").read_bytes() == new_weights
    assert len(staging_directories(tmp_path, output.name)) == 1


def test_different_payload_processes_cannot_publish_a_mixed_generation(
    tmp_path: Path,
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    original = generator.build_weights()
    payloads = [
        bytes([original[0] ^ 1]) + original[1:],
        original[:-1] + bytes([original[-1] ^ 1]),
    ]
    context = multiprocessing.get_context("fork")
    start = context.Event()

    def publish(payload: bytes) -> None:
        generator.build_weights = lambda: payload
        start.wait()
        generator.generate(output)

    processes = [context.Process(target=publish, args=(payload,)) for payload in payloads]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join()
        assert process.exitcode == 0
    assert (output / "weights.i32le").read_bytes() in payloads
    assert manifest_matches_weights(output)
    golden = json.loads((output / "golden.json").read_text(encoding="utf-8"))
    assert golden["weight_sha256"] == hashlib.sha256(
        (output / "weights.i32le").read_bytes()
    ).hexdigest()
    assert not staging_directories(tmp_path, output.name)


def test_unknown_staging_and_glob_metacharacters_are_never_cleaned(
    tmp_path: Path,
) -> None:
    generator = load_generator()
    ordinary_precious = tmp_path / ".model.staging-precious"
    ordinary_precious.mkdir()
    (ordinary_precious / "keep").write_text("ordinary", encoding="utf-8")
    generator.generate(tmp_path / "model")
    assert (ordinary_precious / "keep").read_text(encoding="utf-8") == "ordinary"

    output = tmp_path / "model[precious]*?"
    precious = tmp_path / f".{output.name}.staging-precious"
    precious.mkdir()
    (precious / "keep").write_text("do not delete", encoding="utf-8")

    generator.generate(output)
    generator.generate(output)
    assert (precious / "keep").read_text(encoding="utf-8") == "do not delete"
    assert manifest_matches_weights(output)


def test_replaced_staging_path_and_symlink_are_not_deleted(
    tmp_path: Path,
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    precious = tmp_path / "precious"
    precious.mkdir()
    keep = precious / "keep"
    keep.write_text("preserve", encoding="utf-8")
    moved: Path | None = None
    replacement_link: Path | None = None

    def replace_staging(phase: str, staging: Path) -> None:
        nonlocal moved, replacement_link
        if phase != "before_exchange":
            return
        moved = tmp_path / "moved-owned-staging"
        staging.rename(moved)
        staging.symlink_to(precious, target_is_directory=True)
        replacement_link = staging

    with pytest.raises(RuntimeError, match="staging ownership lost"):
        generator.generate(output, exchange_hook=replace_staging)
    assert keep.read_text(encoding="utf-8") == "preserve"
    assert moved is not None and manifest_matches_weights(moved)
    assert replacement_link is not None and replacement_link.is_symlink()
    assert output.is_dir() and manifest_matches_weights(output)


def test_missing_or_symlink_parent_is_rejected_without_creation(
    tmp_path: Path,
) -> None:
    generator = load_generator()
    missing_parent = tmp_path / "missing" / "nested"
    with pytest.raises(FileNotFoundError, match="parent must already exist"):
        generator.generate(missing_parent / "model")
    assert not (tmp_path / "missing").exists()

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="parent must be a real directory"):
        generator.generate(linked_parent / "model")
    assert not (real_parent / "model").exists()


def test_after_commit_hook_failure_keeps_new_output_and_old_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    old_weights = (output / "weights.i32le").read_bytes()
    new_weights = bytes([old_weights[0] ^ 1]) + old_weights[1:]
    monkeypatch.setattr(generator, "build_weights", lambda: new_weights)

    def fail_after_exchange(phase: str, _staging: Path) -> None:
        if phase == "after_exchange":
            raise RuntimeError("injected after-exchange failure")

    with pytest.raises(RuntimeError, match="post-commit cleanup failed"):
        generator.generate(output, exchange_hook=fail_after_exchange)
    assert (output / "weights.i32le").read_bytes() == new_weights
    assert manifest_matches_weights(output)
    old_snapshots = [
        path
        for path in staging_directories(tmp_path, output.name)
        if manifest_matches_weights(path)
        and (path / "weights.i32le").read_bytes() == old_weights
    ]
    assert len(old_snapshots) == 1


def test_before_exchange_hook_failure_keeps_old_output(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    old_weights = (output / "weights.i32le").read_bytes()
    monkeypatch.setattr(
        generator, "build_weights", lambda: bytes([old_weights[0] ^ 1]) + old_weights[1:]
    )

    def fail_before_exchange(phase: str, _staging: Path) -> None:
        if phase == "before_exchange":
            raise RuntimeError("injected before-exchange failure")

    with pytest.raises(RuntimeError, match="injected before-exchange failure"):
        generator.generate(output, exchange_hook=fail_before_exchange)
    assert (output / "weights.i32le").read_bytes() == old_weights
    assert manifest_matches_weights(output)
    assert not staging_directories(tmp_path, output.name)


def test_before_commit_hook_failure_rolls_exchange_back_to_old_output(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    old_weights = (output / "weights.i32le").read_bytes()
    monkeypatch.setattr(
        generator, "build_weights", lambda: bytes([old_weights[0] ^ 1]) + old_weights[1:]
    )
    exchange_count = 0
    real_exchange = generator.exchange_directories

    def count_exchange(parent_fd: int, left: str, right: str) -> None:
        nonlocal exchange_count
        exchange_count += 1
        real_exchange(parent_fd, left, right)

    def fail_before_commit(phase: str, _staging: Path) -> None:
        if phase == "before_commit":
            raise RuntimeError("injected before-commit failure")

    monkeypatch.setattr(generator, "exchange_directories", count_exchange)
    with pytest.raises(RuntimeError, match="injected before-commit failure"):
        generator.generate(output, exchange_hook=fail_before_commit)
    assert exchange_count == 2
    assert (output / "weights.i32le").read_bytes() == old_weights
    assert manifest_matches_weights(output)


def test_parent_fsync_failure_after_exchange_rolls_back(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    old_weights = (output / "weights.i32le").read_bytes()
    monkeypatch.setattr(
        generator, "build_weights", lambda: bytes([old_weights[0] ^ 1]) + old_weights[1:]
    )
    real_exchange = generator.exchange_directories
    real_fsync = generator.os.fsync
    exchanged = False
    failed = False

    def observe_exchange(parent_fd: int, left: str, right: str) -> None:
        nonlocal exchanged
        real_exchange(parent_fd, left, right)
        exchanged = True

    def fail_first_parent_fsync(fd: int) -> None:
        nonlocal failed
        if exchanged and not failed and os.fstat(fd).st_ino == tmp_path.stat().st_ino:
            failed = True
            raise OSError("injected parent fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(generator, "exchange_directories", observe_exchange)
    monkeypatch.setattr(generator.os, "fsync", fail_first_parent_fsync)
    with pytest.raises(OSError, match="injected parent fsync failure"):
        generator.generate(output)
    assert failed
    assert (output / "weights.i32le").read_bytes() == old_weights
    assert manifest_matches_weights(output)


def test_cleanup_failure_after_commit_keeps_new_output(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    old_weights = (output / "weights.i32le").read_bytes()
    monkeypatch.setattr(
        generator, "build_weights", lambda: bytes([old_weights[0] ^ 1]) + old_weights[1:]
    )
    def fail_cleanup(_parent_fd: int, _owned) -> None:
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(generator, "remove_owned_directory", fail_cleanup)
    with pytest.raises(RuntimeError, match="post-commit cleanup failed"):
        generator.generate(output)
    new_weights = bytes([old_weights[0] ^ 1]) + old_weights[1:]
    assert (output / "weights.i32le").read_bytes() == new_weights
    assert manifest_matches_weights(output)
    old_snapshots = [
        path
        for path in staging_directories(tmp_path, output.name)
        if manifest_matches_weights(path)
        and (path / "weights.i32le").read_bytes() == old_weights
    ]
    assert len(old_snapshots) == 1


def test_rollback_failure_preserves_both_complete_snapshots(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    old_weights = (output / "weights.i32le").read_bytes()
    new_weights = bytes([old_weights[0] ^ 1]) + old_weights[1:]
    monkeypatch.setattr(generator, "build_weights", lambda: new_weights)
    real_exchange = generator.exchange_directories
    exchange_count = 0

    def fail_rollback(parent_fd: int, left: str, right: str) -> None:
        nonlocal exchange_count
        exchange_count += 1
        if exchange_count == 2:
            raise OSError("injected rollback failure")
        real_exchange(parent_fd, left, right)

    real_fsync = generator.os.fsync
    exchanged = False

    def fail_precommit_fsync(fd: int) -> None:
        if exchanged and os.fstat(fd).st_ino == tmp_path.stat().st_ino:
            raise OSError("injected precommit fsync failure")
        real_fsync(fd)

    def track_exchange(parent_fd: int, left: str, right: str) -> None:
        nonlocal exchanged
        fail_rollback(parent_fd, left, right)
        exchanged = True

    monkeypatch.setattr(generator, "exchange_directories", track_exchange)
    monkeypatch.setattr(generator.os, "fsync", fail_precommit_fsync)
    with pytest.raises(RuntimeError, match="model asset recovery failed"):
        generator.generate(output)
    assert (output / "weights.i32le").read_bytes() == new_weights
    old_snapshots = [
        path
        for path in staging_directories(tmp_path, output.name)
        if manifest_matches_weights(path)
        and (path / "weights.i32le").read_bytes() == old_weights
    ]
    assert len(old_snapshots) == 1


def test_second_unlink_failure_never_rolls_partial_old_snapshot_into_output(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    old_weights = (output / "weights.i32le").read_bytes()
    new_weights = bytes([old_weights[0] ^ 1]) + old_weights[1:]
    monkeypatch.setattr(generator, "build_weights", lambda: new_weights)
    real_unlink = generator.os.unlink
    unlink_count = 0

    def fail_second_unlink(path, *args, **kwargs) -> None:
        nonlocal unlink_count
        unlink_count += 1
        if unlink_count == 2:
            raise OSError("injected mid-cleanup unlink failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(generator.os, "unlink", fail_second_unlink)
    with pytest.raises(RuntimeError, match="post-commit cleanup failed"):
        generator.generate(output)
    assert unlink_count == 2
    assert {path.name for path in output.iterdir()} == set(ASSET_NAMES)
    assert (output / "weights.i32le").read_bytes() == new_weights
    assert manifest_matches_weights(output)
    golden = json.loads((output / "golden.json").read_text(encoding="utf-8"))
    assert golden["weight_sha256"] == hashlib.sha256(new_weights).hexdigest()
    partial = staging_directories(tmp_path, output.name)
    assert len(partial) == 1
    partial_entries = {path.name for path in partial[0].iterdir()}
    assert 0 < len(partial_entries) < len(ASSET_NAMES)

    monkeypatch.setattr(generator.os, "unlink", real_unlink)
    generator.generate(output)
    assert {path.name for path in output.iterdir()} == set(ASSET_NAMES)
    assert manifest_matches_weights(output)
    assert partial[0].is_dir()
    assert {path.name for path in partial[0].iterdir()} == partial_entries


def test_rollback_fsync_failure_reports_recovery_and_preserves_snapshots(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    old_weights = (output / "weights.i32le").read_bytes()
    new_weights = bytes([old_weights[0] ^ 1]) + old_weights[1:]
    monkeypatch.setattr(generator, "build_weights", lambda: new_weights)
    real_fsync = generator.os.fsync
    exchange_happened = False
    parent_fsync_count = 0
    real_exchange = generator.exchange_directories

    def observe_exchange(parent_fd: int, left: str, right: str) -> None:
        nonlocal exchange_happened
        real_exchange(parent_fd, left, right)
        exchange_happened = True

    def fail_publication_and_recovery_fsync(fd: int) -> None:
        nonlocal parent_fsync_count
        if exchange_happened and os.fstat(fd).st_ino == tmp_path.stat().st_ino:
            parent_fsync_count += 1
            if parent_fsync_count <= 2:
                raise OSError("injected parent fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(generator, "exchange_directories", observe_exchange)
    monkeypatch.setattr(generator.os, "fsync", fail_publication_and_recovery_fsync)
    with pytest.raises(RuntimeError, match="model asset recovery failed"):
        generator.generate(output)
    assert parent_fsync_count == 2
    assert (output / "weights.i32le").read_bytes() == old_weights
    new_snapshots = [
        path
        for path in staging_directories(tmp_path, output.name)
        if manifest_matches_weights(path)
        and (path / "weights.i32le").read_bytes() == new_weights
    ]
    assert len(new_snapshots) == 1


def test_generator_holds_a_cross_process_parent_directory_lock(
    tmp_path: Path,
) -> None:
    generator = load_generator()
    output = tmp_path / "model"
    generator.generate(output)
    context = multiprocessing.get_context("fork")
    at_exchange = context.Event()
    release = context.Event()

    def hold_lock(_phase: str, _staging: Path) -> None:
        at_exchange.set()
        release.wait()

    process = context.Process(
        target=lambda: generator.generate(output, exchange_hook=hold_lock)
    )
    process.start()
    at_exchange.wait()
    parent_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(parent_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(parent_fd)
        release.set()
        process.join()
    assert process.exitcode == 0


def test_exchange_backend_and_parameters_are_strict(tmp_path: Path) -> None:
    generator = load_generator()
    expected = "renameatx_np" if sys.platform == "darwin" else "renameat2"
    if sys.platform in {"darwin", "linux"}:
        assert generator.exchange_backend_name() == expected
    else:
        with pytest.raises(RuntimeError, match="unsupported"):
            generator.exchange_backend_name()
    with pytest.raises(RuntimeError, match="unsupported"):
        generator.exchange_backend_name("win32")
    with pytest.raises(ValueError, match="single path component"):
        generator.exchange_directories(-1, "left/child", "right")
    with pytest.raises(ValueError, match="distinct"):
        generator.exchange_directories(-1, "same", "same")

    parent_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(OSError):
            generator.exchange_directories(parent_fd, "missing-left", "missing-right")
    finally:
        os.close(parent_fd)


def test_asset_permissions_are_set_before_fsync_and_close(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    events: list[str] = []
    real_fchmod = os.fchmod
    real_fsync = os.fsync
    real_close = os.close

    def record_fchmod(fd: int, mode: int) -> None:
        events.append("fchmod")
        real_fchmod(fd, mode)

    def record_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def record_close(fd: int) -> None:
        events.append("close")
        real_close(fd)

    monkeypatch.setattr(generator.os, "fchmod", record_fchmod)
    monkeypatch.setattr(generator.os, "fsync", record_fsync)
    monkeypatch.setattr(generator.os, "close", record_close)
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        generator.write_asset_file(directory_fd, "asset", b"payload")
    finally:
        os.close(directory_fd)
    assert events.index("fchmod") < events.index("fsync") < events.index("close")


def test_staging_setup_failure_only_cleans_the_exact_new_directory(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    precious = tmp_path / ".model.staging-precious"
    precious.mkdir()
    keep = precious / "keep"
    keep.write_text("preserve", encoding="utf-8")
    real_fchmod = generator.os.fchmod

    def fail_directory_permissions(fd: int, mode: int) -> None:
        if mode == 0o755:
            raise OSError("injected directory setup failure")
        real_fchmod(fd, mode)

    monkeypatch.setattr(generator.os, "fchmod", fail_directory_permissions)
    with pytest.raises(OSError, match="injected directory setup failure"):
        generator.generate(tmp_path / "model")
    assert keep.read_text(encoding="utf-8") == "preserve"
    assert staging_directories(tmp_path, "model") == [precious]


def test_manifest_digest_matches_weights(tmp_path: Path) -> None:
    output = tmp_path / "model"
    run_generator(output)
    manifest = (output / "manifest.textproto").read_text(encoding="utf-8")
    digest_line = next(
        line for line in manifest.splitlines() if line.startswith("weight_sha256:")
    )
    declared = digest_line.split('"', 2)[1]
    assert declared == hashlib.sha256((output / "weights.i32le").read_bytes()).hexdigest()


def test_frozen_arithmetic_helpers_and_layout() -> None:
    generator = load_generator()
    assert [generator.coefficient(i) for i in range(4)] == [-8192, 9216, 3072, -3072]
    assert generator.round_shift(32768, 16) == 1
    assert generator.round_shift(-32768, 16) == -1
    assert generator.saturate_i32(2**40) == 2**31 - 1
    assert generator.argmax_first([7, 9, 9, 1]) == 1
    assert generator.xorshift64(1) == 1082269761
    assert generator.WEIGHT_COUNT == 456


def test_reference_step_uses_frozen_gdn_full_and_greedy_rules() -> None:
    generator = load_generator()
    weights = generator.decode_weights(generator.build_weights())
    state = generator.initial_state()
    candidate = generator.execute_token(weights, 1, state)
    assert len(candidate["gdn_states"]) == 3
    assert all(len(layer) == generator.HIDDEN_SIZE for layer in candidate["gdn_states"])
    assert len(candidate["full_keys"]) == 1
    assert len(candidate["full_values"]) == 1
    assert 0 <= candidate["token_id"] < generator.VOCAB_SIZE
    assert candidate["rng_state"] == generator.xorshift64(state["rng_state"])

    assert generator.integer_mean([-3, -2]) == -2
    assert generator.integer_mean([3, 2]) == 2


def test_golden_contains_deterministic_task12_steps(tmp_path: Path) -> None:
    output = tmp_path / "model"
    run_generator(output)
    golden = json.loads((output / "golden.json").read_text(encoding="utf-8"))
    assert golden["model_id"] == "mini-qwen38-hybrid"
    assert golden["prompt_token_ids"] == [1, 4, 7]
    assert len(golden["steps"]) == 8
    assert all(
        set(step)
        == {
            "canonical_digest",
            "canonical_state_hex",
            "full_keys",
            "full_values",
            "gdn_states",
            "rng_state",
            "token_id",
        }
        for step in golden["steps"]
    )
    assert golden["steps"][0]["token_id"] == 0
    assert (
        golden["steps"][0]["canonical_digest"]
        == "2378a37ce02cb0d7018a1ff56d0157f9ea7dde7ca5803341cc4da9c485345579"
    )
    assert golden["steps"][0]["full_keys"][0] == [0] * 8
    assert golden["steps"][0]["gdn_states"][0][0] == 77
    assert golden["arithmetic"]["format"] == "Q16.16"
    assert golden["rng"] == {
        "algorithm": "xorshift64",
        "initial_state": str(0x9E3779B97F4A7C15),
        "operations": [
            "x ^= (x << 13) & 0xffffffffffffffff",
            "x ^= x >> 7",
            "x ^= (x << 17) & 0xffffffffffffffff",
            "x &= 0xffffffffffffffff",
        ],
        "greedy_argmax_uses_rng": False,
        "advance": "once before each committed token",
    }


def test_generator_is_independent_of_reference_oracle(
    tmp_path: Path, monkeypatch
) -> None:
    generator = load_generator()
    source = GENERATOR.read_text(encoding="utf-8")
    assert "reference_model" not in source

    def forbidden_subprocess(*_args, **_kwargs):
        raise AssertionError("generator must not execute an external oracle")

    monkeypatch.setattr(subprocess, "run", forbidden_subprocess)
    output = tmp_path / "model"
    generator.generate(output)
    assert len(json.loads((output / "golden.json").read_text())["steps"]) == 8


def test_reference_oracle_rejects_manifest_weight_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "model"
    run_generator(output)
    manifest = output / "manifest.textproto"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace(
            hashlib.sha256((output / "weights.i32le").read_bytes()).hexdigest(),
            "0" * 64,
        ),
        encoding="utf-8",
    )
    checked = subprocess.run(
        [sys.executable, str(REFERENCE_MODEL), "--check", str(output / "golden.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 1
    assert "manifest" in checked.stderr
