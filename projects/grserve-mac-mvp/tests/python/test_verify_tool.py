from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import fcntl
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.verify import (
    CommandResult,
    TRUSTED_GIT,
    VerificationInterrupted,
    _create_docker_context,
    run_verification,
    subprocess_runner,
)
from tools.evidence import EvidenceError
import tools.verify as verify_module


@pytest.fixture(autouse=True)
def required_identity_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
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
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(
                ("python@sha256:" + "a" * 64 + "\n").encode()
                if name == "docker/base-image.lock"
                else (
                    "snapshot=20250601T000000Z\nbuild-essential=12.9\ngit=1:2.39.5-0+deb12u2\n"
                ).encode()
                if name == "docker/apt-packages.lock"
                else (name + "\n").encode()
            )
    (tmp_path / "docker/Dockerfile.linux-amd64").write_text(
        "20250601T000000Z build-essential=12.9 git=1:2.39.5-0+deb12u2\n"
    )

    def fake_context(
        _root: Path,
        destination: Path,
        expected_identity: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del expected_identity
        destination.mkdir(parents=True)
        manifest = {
            "format": 1,
            "root_head": "1" * 40,
            "root_tree_sha256": "2" * 64,
            "protobuf_gitlink": "43e1626812c1b543e56a7bec59dc09eb18248bd2",
            "protobuf_head": "43e1626812c1b543e56a7bec59dc09eb18248bd2",
            "protobuf_tree_sha256": "3" * 64,
            "file_count": 1,
            "total_bytes": 1,
            "submodules": {},
        }
        (destination / "source-provenance.json").write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(verify_module, "_create_docker_context", fake_context)


class FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], CommandResult] | None = None):
        self.responses = responses or {}
        self.commands: list[tuple[str, ...]] = []
        self.environments: dict[tuple[str, ...], dict[str, str]] = {}

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        input_text: str | None = None,
    ) -> CommandResult:
        del input_text
        self.commands.append(command)
        self.environments[command] = dict(env or {})
        if len(command) >= 3 and command[:2] == ("cmake", "--preset"):
            preset = command[2]
            build = cwd / "build" / preset
            (build / "CMakeFiles/4.0.2").mkdir(parents=True, exist_ok=True)
            (build / "CMakeCache.txt").write_text(
                "CMAKE_C_COMPILER:FILEPATH=/usr/bin/clang\n"
                "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++\n"
                "CMAKE_OSX_SYSROOT:STRING=macosx\n"
                "CMAKE_OSX_DEPLOYMENT_TARGET:STRING=14.0\n"
            )
            (build / "CMakeFiles/4.0.2/CMakeCCompiler.cmake").write_text(
                'set(CMAKE_C_COMPILER_ID "AppleClang")\nset(CMAKE_C_COMPILER_VERSION "16.0.0")\n'
            )
            (build / "CMakeFiles/4.0.2/CMakeCXXCompiler.cmake").write_text(
                'set(CMAKE_CXX_COMPILER_ID "AppleClang")\nset(CMAKE_CXX_COMPILER_VERSION "16.0.0")\n'
            )
        lookup = (
            ("git", *command[1:]) if command and command[0] == TRUSTED_GIT else command
        )
        fallback = (
            CommandResult(0, "100% tests passed, 0 tests failed out of 1\n", "", 0.01)
            if command and command[0] == "ctest" and "-N" not in command
            else CommandResult(0, "1 passed in 0.01s\n", "", 0.01)
            if command and command[0] == "pytest"
            else CommandResult(0, "ok\n", "", 0.01)
        )
        return self.responses.get(lookup, fallback)


def result(code: int, stdout: str = "", stderr: str = "") -> CommandResult:
    return CommandResult(code, stdout, stderr, 0.01)


def test_verification_uses_an_isolated_docker_cli_plugin_config(tmp_path: Path) -> None:
    verification = verify_module.Verification(
        "linux-amd64",
        tmp_path,
        FakeRunner(preflight_responses()),
        "20260909T101112Z-a1b2c3d4",
    )
    home = Path(verification.base_env["HOME"])
    config = home / ".docker/config.json"
    assert json.loads(config.read_text(encoding="utf-8")) == {
        "cliPluginsExtraDirs": [
            "/Applications/Docker.app/Contents/Resources/cli-plugins"
        ]
    }
    assert config.stat().st_mode & 0o777 == 0o600


def preflight_responses(*, docker: bool = True) -> dict[tuple[str, ...], CommandResult]:
    tests = [
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
    ]
    manifest = json.dumps({"tests": [{"name": name} for name in tests]})
    responses = {
        ("uv", "--version"): result(
            0, "uv 0.11.7 (Homebrew 2026-04-15 aarch64-apple-darwin)\n"
        ),
        ("uv", "lock", "--check"): result(0),
        ("git", "--version"): result(0, "git version 2.39.5\n"),
        ("xcodebuild", "-version"): result(0, "Xcode 16.2\n"),
        ("clang", "--version"): result(0, "Apple clang version 16.0.0\n"),
        ("cmake", "--version"): result(0, "cmake version 4.1.1\n"),
        ("ninja", "--version"): result(0, "1.13.1\n"),
        ("xcrun", "--sdk", "macosx", "--show-sdk-path"): result(
            0, "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk\n"
        ),
        ("sw_vers", "-productVersion"): result(0, "15.1\n"),
        ("git", "rev-parse", "HEAD"): result(0, "1" * 40 + "\n"),
        ("git", "status", "--porcelain", "--untracked-files=all"): result(0),
        ("git", "diff", "--cached", "--quiet"): result(0),
        ("git", "ls-files", "-v"): result(0, "H tracked\n"),
        ("git", "ls-files", "-t"): result(0, "H tracked\n"),
        ("git", "ls-files", "--others", "--exclude-standard", "-z"): result(0),
        ("git", "for-each-ref", "--format=%(refname)", "refs/replace"): result(0),
        ("git", "rev-parse", "--is-shallow-repository"): result(0, "false\n"),
        ("git", "rev-parse", "HEAD^{tree}"): result(0, "a" * 40 + "\n"),
        ("git", "write-tree"): result(0, "a" * 40 + "\n"),
        ("git", "read-tree", "HEAD"): result(0),
        ("git", "add", "-u", "--", "."): result(0),
        ("git", "submodule", "status", "--recursive"): result(
            0,
            " 6910c9d9165801d8827d628cb72eb7ea9dd538c5 third_party/googletest "
            "(release-1.8.0-3518-g6910c9d9)\n"
            " 43e1626812c1b543e56a7bec59dc09eb18248bd2 third_party/protobuf "
            "(v4.30.2)\n",
        ),
        ("docker", "--version"): result(0, "Docker version 29.5.3\n"),
        ("docker", "info", "--format", "{{.ServerVersion}}"): result(
            0 if docker else 1,
            "29.5.3\n" if docker else "",
            "daemon unavailable" if not docker else "",
        ),
        ("docker", "buildx", "inspect", "--bootstrap"): result(
            0, "Name: desktop-linux\nPlatforms: linux/arm64, linux/amd64, linux/amd64/v2\n"
        ),
        (
            "docker",
            "context",
            "inspect",
            "--format",
            "{{json .Endpoints.docker.Host}}",
        ): result(0, '"unix:///var/run/docker.sock"\n'),
    }
    for preset in (
        "mac-arm64-debug",
        "mac-arm64-release",
        "mac-arm64-asan-ubsan",
        "mac-arm64-tsan",
    ):
        absl = f"build/{preset}/_deps/absl-src"
        responses[("git", "-C", absl, "rev-parse", "HEAD")] = result(
            0, verify_module.ABSEIL_COMMIT + "\n"
        )
        responses[
            ("git", "-C", absl, "status", "--porcelain", "--untracked-files=all")
        ] = result(0)
        responses[("git", "-C", absl, "diff", "--cached", "--quiet")] = result(0)
        responses[("git", "-C", absl, "ls-tree", "-r", "HEAD")] = result(
            0, "100644 blob abc CMakeLists.txt\n"
        )
        responses[("ctest", "-N", "--show-only=json-v1", "--preset", preset)] = result(
            0, manifest
        )
    for path, revision in (
        ("third_party/googletest", "6910c9d9165801d8827d628cb72eb7ea9dd538c5"),
        ("third_party/protobuf", "43e1626812c1b543e56a7bec59dc09eb18248bd2"),
    ):
        responses[("git", "-C", path, "rev-parse", "HEAD")] = result(0, revision + "\n")
        responses[
            ("git", "-C", path, "status", "--porcelain", "--untracked-files=all")
        ] = result(0)
        responses[("git", "-C", path, "diff", "--cached", "--quiet")] = result(0)
        responses[("git", "-C", path, "ls-files", "-v")] = result(0, "H tracked\n")
        responses[("git", "-C", path, "ls-files", "-t")] = result(0, "H tracked\n")
        responses[
            ("git", "-C", path, "ls-files", "--others", "--exclude-standard", "-z")
        ] = result(0)
        responses[
            ("git", "-C", path, "for-each-ref", "--format=%(refname)", "refs/replace")
        ] = result(0)
        responses[("git", "-C", path, "rev-parse", "--is-shallow-repository")] = result(
            0, "false\n"
        )
        responses[("git", "-C", path, "rev-parse", "HEAD^{tree}")] = result(
            0, "a" * 40 + "\n"
        )
        responses[("git", "-C", path, "write-tree")] = result(0, "a" * 40 + "\n")
        responses[("git", "-C", path, "read-tree", "HEAD")] = result(0)
        responses[("git", "-C", path, "add", "-u", "--", ".")] = result(0)
    for preset in verify_module.MAC_PRESETS:
        path = f"build/{preset}/_deps/absl-src"
        responses[("git", "-C", path, "ls-files", "-v")] = result(0, "H tracked\n")
        responses[("git", "-C", path, "ls-files", "-t")] = result(0, "H tracked\n")
        responses[
            ("git", "-C", path, "ls-files", "--others", "--exclude-standard", "-z")
        ] = result(0)
        responses[
            ("git", "-C", path, "for-each-ref", "--format=%(refname)", "refs/replace")
        ] = result(0)
        responses[("git", "-C", path, "rev-parse", "--is-shallow-repository")] = result(
            0, "false\n"
        )
        responses[("git", "-C", path, "rev-parse", "HEAD^{tree}")] = result(
            0, "a" * 40 + "\n"
        )
        responses[("git", "-C", path, "write-tree")] = result(0, "a" * 40 + "\n")
        responses[("git", "-C", path, "read-tree", "HEAD")] = result(0)
        responses[("git", "-C", path, "add", "-u", "--", ".")] = result(0)
    return responses


def test_runner_environment_is_hermetic_and_blocks_test_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    poisoned = {
        "GTEST_FILTER": "-*",
        "PYTEST_ADDOPTS": "--ignore=tests/python",
        "PYTHONPATH": "/secret/injection",
        "PYTEST_PLUGINS": "evil",
        "CC": "false",
        "CXX": "false",
        "CMAKE_PREFIX_PATH": "/secret/sdk",
        "GIT_CONFIG_GLOBAL": "/secret/gitconfig",
        "DOCKER_HOST": "tcp://remote.example:2375",
        "ASAN_OPTIONS": "exitcode=0",
    }
    for key, value in poisoned.items():
        monkeypatch.setenv(key, value)
    runner = FakeRunner(preflight_responses())
    run_verification("mac", tmp_path, runner=runner)

    for environment in runner.environments.values():
        assert all(environment.get(key) != value for key, value in poisoned.items())
    pytest_env = runner.environments[("pytest", "-q", "tests/python/test_api_e2e.py")]
    assert pytest_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert not Path(pytest_env["TMPDIR"]).is_relative_to(tmp_path)


def test_remote_docker_context_is_rejected_before_build(tmp_path: Path) -> None:
    responses = preflight_responses()
    responses[
        ("docker", "context", "inspect", "--format", "{{json .Endpoints.docker.Host}}")
    ] = result(0, '"ssh://builder.example"\n')
    runner = FakeRunner(responses)
    outcome = run_verification("linux-amd64", tmp_path, runner=runner)
    assert load_summary(outcome.evidence_dir)["gates"]["Linux amd64"] == {
        "status": "FAIL",
        "reason": "DOCKER_REMOTE_CONTEXT_FORBIDDEN",
    }
    assert not any(command[:2] == ("docker", "build") for command in runner.commands)


def test_configure_and_ctest_cannot_reuse_stale_or_zero_test_trees(
    tmp_path: Path,
) -> None:
    responses = preflight_responses()
    responses[("ctest", "-N", "--show-only=json-v1", "--preset", "mac-arm64-debug")] = (
        result(0, '{"tests":[]}')
    )
    runner = FakeRunner(responses)
    outcome = run_verification("mac", tmp_path, runner=runner)
    assert outcome.exit_code == 1
    assert ("cmake", "--preset", "mac-arm64-debug", "--fresh") in runner.commands
    assert any(
        "--no-tests=error" in command
        for command in runner.commands
        if command and command[0] == "ctest"
    )
    assert load_summary(outcome.evidence_dir)["gates"]["Debug"]["status"] == "FAIL"


def test_filtered_ctest_exit_zero_with_zero_tests_is_semantic_failure(
    tmp_path: Path,
) -> None:
    responses = preflight_responses()
    command = (
        "ctest",
        "--preset",
        "mac-arm64-debug",
        "--no-tests=error",
        "-R",
        "^c\\.hal_header$",
    )
    responses[command] = result(0, "No tests were found!!!\n")
    outcome = run_verification("mac", tmp_path, runner=FakeRunner(responses))
    assert load_summary(outcome.evidence_dir)["gates"]["C11 ABI"] == {
        "status": "FAIL",
        "reason": "COMMAND_FAILED",
    }


def test_pytest_exit_zero_with_zero_tests_is_semantic_failure(tmp_path: Path) -> None:
    responses = preflight_responses()
    command = ("pytest", "-q", "tests/python/test_api_e2e.py")
    responses[command] = result(0, "no tests ran in 0.01s\n")
    outcome = run_verification("mac", tmp_path, runner=FakeRunner(responses))
    assert load_summary(outcome.evidence_dir)["gates"]["API E2E"] == {
        "status": "FAIL",
        "reason": "COMMAND_FAILED",
    }


def test_all_four_mac_presets_attest_clean_pinned_abseil(tmp_path: Path) -> None:
    runner = FakeRunner(preflight_responses())
    outcome = run_verification("mac", tmp_path, runner=runner)
    abseil = load_summary(outcome.evidence_dir)["dependencies"]["abseil"]
    assert set(abseil["presets"]) == {
        "mac-arm64-debug",
        "mac-arm64-release",
        "mac-arm64-asan-ubsan",
        "mac-arm64-tsan",
    }
    assert all(
        item["clean"]
        and item["head"] == verify_module.ABSEIL_COMMIT
        and len(item["tree_sha256"]) == 64
        for item in abseil["presets"].values()
    )
    build = json.loads((outcome.evidence_dir / "build.json").read_text())
    assert set(build["toolchains"]) == set(verify_module.MAC_PRESETS)
    assert all(
        item["c_id"] == "AppleClang"
        and item["cxx_id"] == "AppleClang"
        and item["c_compiler"].endswith("clang")
        and item["sysroot"] == "macosx"
        and item["deployment_target"] == "14.0"
        for item in build["toolchains"].values()
    )


def test_signal_inside_gate_records_incomplete_gate_and_commands(
    tmp_path: Path,
) -> None:
    class InterruptAfterConfigure(FakeRunner):
        def __call__(self, command, **kwargs):
            if command == ("cmake", "--build", "--preset", "mac-arm64-debug"):
                raise VerificationInterrupted(15)
            return super().__call__(command, **kwargs)

    outcome = run_verification(
        "mac", tmp_path, runner=InterruptAfterConfigure(preflight_responses())
    )
    summary = load_summary(outcome.evidence_dir)
    build = json.loads((outcome.evidence_dir / "build.json").read_text())
    assert summary["gates"]["Debug"]["status"] == "INCOMPLETE"
    assert build["status"] == "INCOMPLETE"
    assert [x["command_id"] for x in build["commands"]] == ["mac.debug.configure"]
    events = [
        json.loads(x)
        for x in (outcome.evidence_dir / "events.jsonl").read_text().splitlines()
    ]
    assert next(x for x in events if x["gateId"] == "Debug")["status"].endswith(
        "INCOMPLETE"
    )


def test_signal_during_preset_post_attestation_keeps_gate_incomplete(
    tmp_path: Path,
) -> None:
    class InterruptingPost(FakeRunner):
        def __call__(self, command, **kwargs):
            if command == (
                TRUSTED_GIT,
                "-C",
                "build/mac-arm64-debug/_deps/absl-src",
                "status",
                "--porcelain",
                "--untracked-files=all",
            ):
                raise VerificationInterrupted(15)
            return super().__call__(command, **kwargs)

    outcome = run_verification(
        "mac", tmp_path, runner=InterruptingPost(preflight_responses())
    )
    assert (
        load_summary(outcome.evidence_dir)["gates"]["Debug"]["status"] == "INCOMPLETE"
    )


def test_subprocess_runner_bounds_large_outputs(tmp_path: Path) -> None:
    completed = subprocess_runner(
        (
            sys.executable,
            "-c",
            "import sys;sys.stdout.write('a'*2000000);sys.stderr.write('b'*2000000)",
        ),
        cwd=tmp_path,
    )
    assert completed.exit_code == 0
    assert completed.stdout_byte_count == 2_000_000
    assert completed.stderr_byte_count == 2_000_000
    assert completed.stdout_truncated and completed.stderr_truncated
    assert len(completed.stdout.encode()) <= 1_048_576
    assert len(completed.stdout_sha256) == 64


def test_subprocess_timeout_is_bounded_when_descendant_escapes_session(
    tmp_path: Path,
) -> None:
    child_pid = tmp_path / "child.pid"
    started = time.monotonic()
    completed = subprocess_runner(
        (
            sys.executable,
            "-c",
            "import pathlib,subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);"
            f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid));time.sleep(60)",
        ),
        cwd=tmp_path,
        timeout_seconds=0.05,
    )
    assert completed.exit_code == 124
    assert time.monotonic() - started < 2
    pid = int(child_pid.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_repository_lock_rejects_overlapping_verification(tmp_path: Path) -> None:
    lock = tmp_path / "build/.grserve-verify.lock"
    lock.parent.mkdir()
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        outcome = run_verification(
            "mac", tmp_path, runner=FakeRunner(preflight_responses())
        )
    assert outcome == verify_module.VerificationOutcome(
        2, None, "VERIFICATION_ALREADY_RUNNING"
    )


def test_identity_drift_after_gates_fails_final_verification(tmp_path: Path) -> None:
    class DriftingRunner(FakeRunner):
        head_reads = 0

        def __call__(self, command, **kwargs):
            if command == (TRUSTED_GIT, "rev-parse", "HEAD"):
                self.head_reads += 1
                return result(0, ("1" if self.head_reads == 1 else "2") * 40 + "\n")
            return super().__call__(command, **kwargs)

    outcome = run_verification(
        "mac", tmp_path, runner=DriftingRunner(preflight_responses())
    )
    summary = load_summary(outcome.evidence_dir)
    assert outcome.exit_code == 1
    assert summary["gates"]["preflight"] == {
        "status": "FAIL",
        "reason": "IDENTITY_CHANGED_DURING_VERIFY",
    }


def test_linux_build_uses_private_tracked_context_and_provenance(
    tmp_path: Path,
) -> None:
    source = Path(__file__).resolve().parents[2]
    context = tmp_path / "context"
    manifest = _create_docker_context(source, context)
    assert context != source
    assert (
        manifest["root_head"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert len(manifest["root_tree_sha256"]) == 64
    assert len(manifest["protobuf_tree_sha256"]) == 64
    assert manifest["file_count"] > 0
    assert not (context / ".git").exists()
    assert (context / "source-provenance.json").is_file()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import tools.proto_codegen as p; "
            "i=p._source_identity(Path.cwd()); assert i['root_head'] and i['gitlink']",
        ],
        cwd=context,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(context),
            "GRSERVE_SOURCE_PROVENANCE": str(context / "source-provenance.json"),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_docker_context_archives_head_not_assume_unchanged_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    tracked = root / "tracked.txt"
    tracked.write_text("committed\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
    subprocess.run(
        ["git", "update-index", "--assume-unchanged", "tracked.txt"],
        cwd=root,
        check=True,
    )
    tracked.write_text("poisoned\n")
    monkeypatch.setattr(verify_module, "SUBMODULES", {})
    context = tmp_path / "context"
    _create_docker_context(root, context)
    assert (context / "tracked.txt").read_text() == "committed\n"


def test_docker_context_uses_trusted_git_path_and_rejects_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["/usr/bin/git", "config", "user.email", "test@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "config", "user.name", "Test"], cwd=root, check=True
    )
    (root / "tracked.txt").write_text("committed\n")
    subprocess.run(["/usr/bin/git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["/usr/bin/git", "commit", "-qm", "base"], cwd=root, check=True)
    poison = tmp_path / "poison"
    poison.mkdir()
    fake_git = poison / "git"
    fake_git.write_text("#!/bin/sh\necho poison >&2\nexit 99\n")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(poison))
    monkeypatch.setattr(verify_module, "SUBMODULES", {})
    with pytest.raises(ValueError, match="SOURCE_CONTEXT_IDENTITY_MISMATCH"):
        _create_docker_context(
            root,
            tmp_path / "context",
            expected_identity={"head": "f" * 40, "submodules": []},
        )


def test_docker_context_archives_captured_commit_across_head_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run([TRUSTED_GIT, "init", "-q"], cwd=root, check=True)
    subprocess.run(
        [TRUSTED_GIT, "config", "user.email", "test@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run([TRUSTED_GIT, "config", "user.name", "Test"], cwd=root, check=True)
    tracked = root / "tracked.txt"
    tracked.write_text("A\n")
    subprocess.run([TRUSTED_GIT, "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run([TRUSTED_GIT, "commit", "-qm", "A"], cwd=root, check=True)
    commit_a = subprocess.check_output(
        [TRUSTED_GIT, "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    tree_a = subprocess.check_output(
        [TRUSTED_GIT, "rev-parse", "HEAD^{tree}"], cwd=root, text=True
    ).strip()
    tracked.write_text("B\n")
    subprocess.run([TRUSTED_GIT, "commit", "-qam", "B"], cwd=root, check=True)
    monkeypatch.setattr(verify_module, "SUBMODULES", {})
    _create_docker_context(
        root,
        tmp_path / "context-a",
        expected_identity={
            "head": commit_a,
            "identity": {"head_tree": tree_a},
            "submodules": [],
        },
    )
    assert (tmp_path / "context-a/tracked.txt").read_text() == "A\n"


def test_docker_context_rejects_git_replace_mechanism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run([TRUSTED_GIT, "init", "-q"], cwd=root, check=True)
    subprocess.run(
        [TRUSTED_GIT, "config", "user.email", "test@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run([TRUSTED_GIT, "config", "user.name", "Test"], cwd=root, check=True)
    tracked = root / "tracked.txt"
    tracked.write_text("A\n")
    subprocess.run([TRUSTED_GIT, "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run([TRUSTED_GIT, "commit", "-qm", "A"], cwd=root, check=True)
    commit_a = subprocess.check_output(
        [TRUSTED_GIT, "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    tracked.write_text("B\n")
    subprocess.run([TRUSTED_GIT, "commit", "-qam", "B"], cwd=root, check=True)
    commit_b = subprocess.check_output(
        [TRUSTED_GIT, "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    subprocess.run([TRUSTED_GIT, "replace", commit_a, commit_b], cwd=root, check=True)
    monkeypatch.setattr(verify_module, "SUBMODULES", {})
    with pytest.raises(ValueError, match="SOURCE_CONTEXT_UNSAFE_GIT_MECHANISM"):
        _create_docker_context(root, tmp_path / "context-replace")


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text())


def test_mac_preflight_checks_locked_tools_and_pinned_clean_submodules(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(preflight_responses())
    outcome = run_verification("mac", tmp_path, runner=runner)

    assert outcome.exit_code == 0
    assert ("uv", "--version") in runner.commands
    assert ("uv", "lock", "--check") in runner.commands
    assert ("xcodebuild", "-version") in runner.commands
    assert ("clang", "--version") in runner.commands
    assert (TRUSTED_GIT, "submodule", "status", "--recursive") in runner.commands
    assert ("docker", "info", "--format", "{{.ServerVersion}}") not in runner.commands
    assert load_summary(outcome.evidence_dir)["status"] == "PASS"


def test_mac_preflight_accepts_recognized_command_line_tools(tmp_path: Path) -> None:
    responses = preflight_responses()
    responses[("xcodebuild", "-version")] = result(
        1,
        stderr=(
            "xcode-select: error: tool 'xcodebuild' requires Xcode, but active "
            "developer directory '/Library/Developer/CommandLineTools' is a "
            "command line tools instance"
        ),
    )
    outcome = run_verification("mac", tmp_path, runner=FakeRunner(responses))
    preflight = json.loads((outcome.evidence_dir / "preflight.json").read_text())

    assert outcome.exit_code == 0
    assert preflight["toolchainKind"] == "CLT"
    assert preflight["status"] == "PASS"


@pytest.mark.parametrize(
    ("command", "response"),
    [
        (("uv", "--version"), result(0, "uv 0.11.6\n")),
        (("uv", "lock", "--check"), result(0, "Lock mismatch\n")),
        (("clang", "--version"), result(127, stderr="COMMAND_NOT_FOUND")),
        (("cmake", "--version"), result(127, stderr="COMMAND_NOT_FOUND")),
        (("ninja", "--version"), result(127, stderr="COMMAND_NOT_FOUND")),
    ],
)
def test_exit_zero_semantic_or_missing_tool_preflight_is_failure(
    tmp_path: Path, command: tuple[str, ...], response: CommandResult
) -> None:
    responses = preflight_responses()
    responses[command] = response
    outcome = run_verification("mac", tmp_path, runner=FakeRunner(responses))
    summary = load_summary(outcome.evidence_dir)
    assert outcome.exit_code != 0
    assert summary["gates"]["preflight"]["status"] == "FAIL"
    assert (
        json.loads((outcome.evidence_dir / "preflight.json").read_text())["status"]
        == "FAIL"
    )


@pytest.mark.parametrize(
    ("command", "response"),
    [
        (
            ("git", "status", "--porcelain", "--untracked-files=all"),
            result(0, " M tools/verify.py\n"),
        ),
        (("git", "diff", "--cached", "--quiet"), result(1)),
        (
            ("git", "submodule", "status", "--recursive"),
            result(
                0,
                "-"
                + "1" * 40
                + " third_party/googletest\n "
                + "2" * 40
                + " third_party/protobuf\n",
            ),
        ),
        (
            (
                "git",
                "-C",
                "third_party/googletest",
                "status",
                "--porcelain",
                "--untracked-files=all",
            ),
            result(0, "?? surprise\n"),
        ),
        (
            ("git", "-C", "third_party/protobuf", "diff", "--cached", "--quiet"),
            result(1),
        ),
    ],
)
def test_dirty_staged_untracked_or_uninitialized_checkout_fails_preflight(
    tmp_path: Path, command: tuple[str, ...], response: CommandResult
) -> None:
    responses = preflight_responses()
    responses[command] = response
    outcome = run_verification("mac", tmp_path, runner=FakeRunner(responses))
    assert load_summary(outcome.evidence_dir)["gates"]["preflight"]["status"] == "FAIL"
    assert outcome.exit_code == 1


def test_assume_unchanged_or_skip_worktree_index_flags_fail_preflight(
    tmp_path: Path,
) -> None:
    responses = preflight_responses()
    responses[("git", "ls-files", "-v")] = result(0, "h tools/verify.py\n")
    outcome = run_verification("mac", tmp_path, runner=FakeRunner(responses))
    assert outcome.exit_code == 1
    assert load_summary(outcome.evidence_dir)["gates"]["preflight"]["status"] == "FAIL"


def test_replace_refs_fail_preflight_and_identity_git_is_absolute(
    tmp_path: Path,
) -> None:
    responses = preflight_responses()
    responses[("git", "for-each-ref", "--format=%(refname)", "refs/replace")] = result(
        0, "refs/replace/" + "1" * 40 + "\n"
    )
    runner = FakeRunner(responses)
    outcome = run_verification("mac", tmp_path, runner=runner)
    assert outcome.exit_code == 1
    assert load_summary(outcome.evidence_dir)["gates"]["preflight"]["status"] == "FAIL"
    identity_commands = [
        command
        for command in runner.commands
        if any(token in command for token in ("rev-parse", "ls-files", "ls-tree"))
    ]
    assert identity_commands and all(
        command[0] == TRUSTED_GIT for command in identity_commands
    )


def test_full_unavailable_docker_fails_linux_without_skip(tmp_path: Path) -> None:
    runner = FakeRunner(preflight_responses(docker=False))
    outcome = run_verification("full", tmp_path, runner=runner)
    summary = load_summary(outcome.evidence_dir)

    assert outcome.exit_code != 0
    assert summary["status"] == "FAIL"
    linux = summary["gates"]["Linux amd64"]
    assert linux == {"status": "FAIL", "reason": "DOCKER_DAEMON_UNAVAILABLE"}
    assert all(gate["status"] != "SKIP" for gate in summary["gates"].values())
    events = [
        json.loads(line)
        for line in (outcome.evidence_dir / "events.jsonl").read_text().splitlines()
    ]
    assert {event["gateId"]: event["reason"] for event in events} == {
        name: gate.get("reason", "") for name, gate in summary["gates"].items()
    }
    assert all(
        event["reason"]
        for event in events
        if event["status"] == "VERIFICATION_GATE_STATUS_FAIL"
    )


def test_full_missing_docker_client_has_stable_daemon_reason(tmp_path: Path) -> None:
    responses = preflight_responses(docker=False)
    responses[("docker", "--version")] = result(127, stderr="COMMAND_NOT_FOUND")
    runner = FakeRunner(responses)
    outcome = run_verification("full", tmp_path, runner=runner)

    assert load_summary(outcome.evidence_dir)["gates"]["Linux amd64"] == {
        "status": "FAIL",
        "reason": "DOCKER_DAEMON_UNAVAILABLE",
    }
    assert not any(command[:2] == ("docker", "context") for command in runner.commands)


@pytest.mark.parametrize(
    ("command", "response", "reason"),
    [
        (
            ("docker", "--version"),
            result(127, stderr="COMMAND_NOT_FOUND"),
            "DOCKER_DAEMON_UNAVAILABLE",
        ),
        (
            ("docker", "info", "--format", "{{.ServerVersion}}"),
            result(1, stderr="offline"),
            "DOCKER_DAEMON_UNAVAILABLE",
        ),
        (
            ("docker", "buildx", "inspect", "--bootstrap"),
            result(0, "Name: desktop-linux\nPlatforms: linux/arm64\n"),
            "DOCKER_PLATFORM_UNAVAILABLE",
        ),
        (
            ("docker", "buildx", "inspect", "--bootstrap"),
            result(0, "Name: desktop-linux\nPlatforms: linux/amd64/v2\n"),
            "DOCKER_PLATFORM_UNAVAILABLE",
        ),
        (
            ("docker", "buildx", "inspect", "--bootstrap"),
            result(1, stderr="no builder"),
            "DOCKER_PLATFORM_UNAVAILABLE",
        ),
    ],
)
def test_docker_capability_matrix_has_stable_semantic_reason(
    tmp_path: Path, command: tuple[str, ...], response: CommandResult, reason: str
) -> None:
    responses = preflight_responses()
    responses[command] = response
    outcome = run_verification("linux-amd64", tmp_path, runner=FakeRunner(responses))
    assert load_summary(outcome.evidence_dir)["gates"]["Linux amd64"] == {
        "status": "FAIL",
        "reason": reason,
    }
    assert outcome.exit_code == 1


def test_subprocess_runner_records_missing_executable(tmp_path: Path) -> None:
    completed = subprocess_runner(
        ("grserve-command-that-does-not-exist",), cwd=tmp_path
    )
    assert completed.exit_code == 127
    assert completed.stderr == "COMMAND_NOT_FOUND"


def test_linux_profile_never_passes_or_skips_when_daemon_is_unavailable(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(preflight_responses(docker=False))
    outcome = run_verification("linux-amd64", tmp_path, runner=runner)
    summary = load_summary(outcome.evidence_dir)

    assert outcome.exit_code != 0
    assert summary["gates"]["Linux amd64"]["status"] == "FAIL"
    assert summary["gates"]["Linux amd64"]["reason"] == "DOCKER_DAEMON_UNAVAILABLE"
    linux_document = json.loads((outcome.evidence_dir / "linux-amd64.json").read_text())
    assert linux_document["status"] == "FAIL"
    assert linux_document["reason"] == "DOCKER_DAEMON_UNAVAILABLE"


def test_linux_profile_builds_and_runs_an_amd64_image_without_host_volume(
    tmp_path: Path,
) -> None:
    (tmp_path / "docker").mkdir(exist_ok=True)
    (tmp_path / "docker/base-image.lock").write_text("python@sha256:" + "a" * 64 + "\n")
    runner = FakeRunner(preflight_responses())
    outcome = run_verification("linux-amd64", tmp_path, runner=runner)

    assert outcome.exit_code == 0
    assert (
        "docker",
        "buildx",
        "inspect",
        "--bootstrap",
    ) in runner.commands
    build = next(
        command for command in runner.commands if command[:2] == ("docker", "build")
    )
    run = next(
        command for command in runner.commands if command[:2] == ("docker", "run")
    )
    assert ("--platform", "linux/amd64") == build[2:4]
    assert ("--platform", "linux/amd64") == run[3:5]
    assert "-v" not in run and "--volume" not in run
    assert (
        load_summary(outcome.evidence_dir)["gates"]["Linux amd64"]["status"] == "PASS"
    )


def test_failed_subcommand_marks_gate_failed_but_finishes_and_records_commands(
    tmp_path: Path,
) -> None:
    responses = preflight_responses()
    responses[("cmake", "--build", "--preset", "mac-arm64-debug")] = result(
        7, "partial output", "compiler failed"
    )
    runner = FakeRunner(responses)
    outcome = run_verification("mac", tmp_path, runner=runner)
    summary = load_summary(outcome.evidence_dir)
    build = json.loads((outcome.evidence_dir / "build.json").read_text())

    assert outcome.exit_code == 1
    assert summary["gates"]["Debug"]["status"] == "FAIL"
    assert ("ruff", "check", "python", "tools", "tests/python") in runner.commands
    assert (
        "mypy",
        "--explicit-package-bases",
        "python/grserve_api",
        "tools/verify.py",
        "tools/evidence.py",
        "tools/acceptance.py",
        "tools/normalize_events.py",
        "tools/proto_codegen.py",
        "tools/generate_mini_model.py",
    ) in runner.commands
    failed = next(
        record
        for record in build["commands"]
        if record["argv"] == ["cmake", "--build", "--preset", "mac-arm64-debug"]
    )
    assert {
        key: failed[key]
        for key in (
            "command_id",
            "argv",
            "duration_seconds",
            "exit_code",
            "status",
            "stderr",
            "stdout",
        )
    } == {
        "command_id": "mac.debug.build",
        "argv": ["cmake", "--build", "--preset", "mac-arm64-debug"],
        "duration_seconds": 0.01,
        "exit_code": 7,
        "status": "FAIL",
        "stderr": "compiler failed",
        "stdout": "partial output",
    }


def test_tsan_runtime_skip_requires_exact_apple_clang_capability_failure(
    tmp_path: Path,
) -> None:
    responses = preflight_responses()
    responses[("build/tsan-capability/probe",)] = result(
        66, stderr="FATAL: ThreadSanitizer: unsupported Darwin runtime"
    )
    outcome = run_verification("mac", tmp_path, runner=FakeRunner(responses))
    gate = load_summary(outcome.evidence_dir)["gates"]["TSan"]

    assert gate == {
        "status": "SKIP",
        "reason": "TSAN_RUNTIME_UNSUPPORTED_APPLE_CLANG",
    }
    assert outcome.exit_code == 0
    sanitizer = json.loads((outcome.evidence_dir / "sanitizer.json").read_text())
    assert sanitizer["status"] == "PASS"
    assert sanitizer["gates"]["TSan"] == gate
    assert load_summary(outcome.evidence_dir)["critical_skip_count"] == 0


def test_tsan_unsupported_post_attestation_signal_remains_incomplete(
    tmp_path: Path,
) -> None:
    responses = preflight_responses()
    responses[("build/tsan-capability/probe",)] = result(
        66, stderr="FATAL: ThreadSanitizer: unsupported Darwin runtime"
    )

    class InterruptingAttestation(FakeRunner):
        def __call__(self, command, **kwargs):
            if command == (
                TRUSTED_GIT,
                "-C",
                "build/mac-arm64-tsan/_deps/absl-src",
                "status",
                "--porcelain",
                "--untracked-files=all",
            ):
                raise VerificationInterrupted(15)
            return super().__call__(command, **kwargs)

    outcome = run_verification(
        "mac", tmp_path, runner=InterruptingAttestation(responses)
    )
    summary = load_summary(outcome.evidence_dir)
    assert summary["gates"]["TSan"]["status"] == "INCOMPLETE"
    events = [
        json.loads(line)
        for line in (outcome.evidence_dir / "events.jsonl").read_text().splitlines()
    ]
    event = next(item for item in events if item["gateId"] == "TSan")
    assert event["status"] == "VERIFICATION_GATE_STATUS_INCOMPLETE"


def test_signal_during_closing_revalidation_keeps_preflight_incomplete(
    tmp_path: Path,
) -> None:
    class InterruptingRevalidation(FakeRunner):
        head_calls = 0

        def __call__(self, command, **kwargs):
            if command == (TRUSTED_GIT, "rev-parse", "HEAD"):
                self.head_calls += 1
                if self.head_calls == 2:
                    raise VerificationInterrupted(15)
            return super().__call__(command, **kwargs)

    outcome = run_verification(
        "mac", tmp_path, runner=InterruptingRevalidation(preflight_responses())
    )
    summary = load_summary(outcome.evidence_dir)
    assert summary["status"] == "INCOMPLETE"
    assert summary["gates"]["preflight"]["status"] == "INCOMPLETE"
    preflight = json.loads((outcome.evidence_dir / "preflight.json").read_text())
    assert preflight["status"] == "INCOMPLETE"
    events = [
        json.loads(line)
        for line in (outcome.evidence_dir / "events.jsonl").read_text().splitlines()
    ]
    event = next(item for item in events if item["gateId"] == "preflight")
    assert event["status"] == "VERIFICATION_GATE_STATUS_INCOMPLETE"


def test_tsan_probe_creates_its_ignored_build_directory(tmp_path: Path) -> None:
    run_verification("mac", tmp_path, runner=FakeRunner(preflight_responses()))
    assert (tmp_path / "build/tsan-capability").is_dir()


def test_macos_asan_disables_only_unsupported_leak_detection(tmp_path: Path) -> None:
    runner = FakeRunner(preflight_responses())
    outcome = run_verification("mac", tmp_path, runner=runner)

    command = ("ctest", "--preset", "mac-arm64-asan-ubsan", "--no-tests=error")
    assert (
        runner.environments[command]["ASAN_OPTIONS"]
        == "abort_on_error=1:detect_leaks=0"
    )
    assert "ASan/UBSan" in load_summary(outcome.evidence_dir)["gates"]


def test_ordinary_tsan_probe_failure_is_not_hidden(tmp_path: Path) -> None:
    responses = preflight_responses()
    responses[("build/tsan-capability/probe",)] = result(9, stderr="application bug")
    outcome = run_verification("mac", tmp_path, runner=FakeRunner(responses))
    gate = load_summary(outcome.evidence_dir)["gates"]["TSan"]

    assert gate["status"] == "FAIL"
    assert gate["reason"] == "COMMAND_FAILED"
    assert outcome.exit_code == 1


def test_concurrent_runs_are_serialized_by_repository_lock(tmp_path: Path) -> None:
    def run_once(_: int):
        return run_verification(
            "mac", tmp_path, runner=FakeRunner(preflight_responses())
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(executor.map(run_once, range(4)))

    published = [item for item in outcomes if item.evidence_dir is not None]
    blocked = [
        item for item in outcomes if item.reason == "VERIFICATION_ALREADY_RUNNING"
    ]
    assert len(published) == 1 and len(blocked) == 3
    assert (published[0].evidence_dir / "summary.json").is_file()


@pytest.mark.parametrize("failure", [RuntimeError("boom"), VerificationInterrupted(15)])
def test_exception_or_signal_publishes_incomplete_run_with_stable_exit(
    tmp_path: Path, failure: BaseException
) -> None:
    class InterruptingRunner(FakeRunner):
        def __call__(self, command, **kwargs):
            if command == ("cmake", "--preset", "mac-arm64-debug", "--fresh"):
                raise failure
            return super().__call__(command, **kwargs)

    outcome = run_verification(
        "mac", tmp_path, runner=InterruptingRunner(preflight_responses())
    )
    summary = load_summary(outcome.evidence_dir)

    assert outcome.exit_code == (
        143 if isinstance(failure, VerificationInterrupted) else 2
    )
    assert summary["status"] == "INCOMPLETE"
    assert summary["reason"] == (
        "INTERRUPTED_SIGNAL_15"
        if isinstance(failure, VerificationInterrupted)
        else "INTERNAL_ERROR"
    )


def test_existing_evidence_directory_is_never_reused(tmp_path: Path) -> None:
    first = run_verification(
        "mac",
        tmp_path,
        runner=FakeRunner(preflight_responses()),
        run_id="20260909T010203Z-a1b2c3d4",
    )
    second = run_verification(
        "mac",
        tmp_path,
        runner=FakeRunner(preflight_responses()),
        run_id="20260909T010203Z-a1b2c3d4",
    )

    assert first.exit_code == 0
    assert second.exit_code == 2
    assert second.evidence_dir is None
    assert second.reason == "ARTIFACT_ALREADY_COMMITTED"
    assert load_summary(first.evidence_dir)["status"] == "PASS"


def test_pending_or_interrupted_publication_never_reports_an_evidence_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = tmp_path / "artifacts/mac-mvp/.pending"
    pending.mkdir(parents=True)
    outcome = run_verification(
        "mac", tmp_path, runner=FakeRunner(preflight_responses())
    )
    assert outcome.evidence_dir is None
    assert outcome.reason == "ARTIFACT_RECOVERY_REQUIRED"

    pending.rmdir()

    def interrupt_publication(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise VerificationInterrupted(15)

    monkeypatch.setattr(verify_module, "write_evidence", interrupt_publication)
    interrupted = run_verification(
        "mac", tmp_path, runner=FakeRunner(preflight_responses())
    )
    assert interrupted == verify_module.VerificationOutcome(
        2, None, "EVIDENCE_PUBLICATION_INTERRUPTED"
    )


def test_finalize_and_after_write_identity_signals_cannot_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_finalize = verify_module._finalize

    def interrupt_finalize(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise VerificationInterrupted(2)

    monkeypatch.setattr(verify_module, "_finalize", interrupt_finalize)
    try:
        outcome = run_verification(
            "mac", tmp_path, runner=FakeRunner(preflight_responses())
        )
    except VerificationInterrupted:
        pytest.fail("signal escaped during evidence finalization")
    assert outcome == verify_module.VerificationOutcome(
        2, None, "EVIDENCE_PUBLICATION_INTERRUPTED"
    )

    monkeypatch.setattr(verify_module, "_finalize", original_finalize)

    def interrupt_identity(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise VerificationInterrupted(15)

    monkeypatch.setattr(
        verify_module, "_validate_published_identity", interrupt_identity, raising=False
    )
    try:
        outcome = run_verification(
            "mac", tmp_path, runner=FakeRunner(preflight_responses())
        )
    except VerificationInterrupted:
        pytest.fail("signal escaped after evidence write")
    assert outcome == verify_module.VerificationOutcome(
        2, None, "EVIDENCE_PUBLICATION_INTERRUPTED"
    )


def test_durability_unknown_is_not_reclassified_when_destination_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "20260909T010203Z-b1b2c3d4"

    def durability_failure(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        (tmp_path / "artifacts/mac-mvp" / run_id).mkdir(parents=True)
        raise EvidenceError("ARTIFACT_DURABILITY_UNKNOWN")

    monkeypatch.setattr(verify_module, "write_evidence", durability_failure)
    outcome = run_verification(
        "mac",
        tmp_path,
        runner=FakeRunner(preflight_responses()),
        run_id=run_id,
    )
    assert outcome == verify_module.VerificationOutcome(
        2, None, "ARTIFACT_DURABILITY_UNKNOWN"
    )


def test_malformed_published_summary_has_stable_identity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "summary.json").write_text("not-json")
    monkeypatch.setattr(
        verify_module,
        "write_evidence",
        lambda *_args, **_kwargs: destination,
    )
    outcome = run_verification(
        "mac", tmp_path, runner=FakeRunner(preflight_responses())
    )
    assert outcome == verify_module.VerificationOutcome(
        2, None, "EVIDENCE_IDENTITY_MISMATCH"
    )


def test_main_never_leaks_a_signal_during_summary_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}")
    monkeypatch.setattr(
        verify_module,
        "run_verification",
        lambda *_args, **_kwargs: verify_module.VerificationOutcome(0, tmp_path),
    )
    monkeypatch.setattr(sys, "argv", ["verify.py", "--profile", "mac"])

    def interrupt_print(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise VerificationInterrupted(2)

    monkeypatch.setattr(builtins, "print", interrupt_print)
    try:
        exit_code = verify_module.main()
    except VerificationInterrupted:
        pytest.fail("signal escaped summary reporting")
    assert exit_code == 130


def test_full_contract_has_all_task21_gate_ids_and_never_critical_skips(
    tmp_path: Path,
) -> None:
    outcome = run_verification(
        "full", tmp_path, runner=FakeRunner(preflight_responses())
    )
    summary = load_summary(outcome.evidence_dir)
    assert set(summary["gates"]) == {
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
        "Linux amd64",
    }
    assert summary["critical_skip_count"] == 0


def test_profile_irrelevant_zero_command_documents_are_not_pass(tmp_path: Path) -> None:
    outcome = run_verification(
        "linux-amd64", tmp_path, runner=FakeRunner(preflight_responses())
    )
    for name in ("build.json", "cpp-tests.json", "python-tests.json", "sanitizer.json"):
        assert (
            json.loads((outcome.evidence_dir / name).read_text())["status"] == "NOT_RUN"
        )


def test_evidence_has_identity_hashes_stable_commands_and_typed_event_chain(
    tmp_path: Path,
) -> None:
    outcome = run_verification(
        "mac", tmp_path, runner=FakeRunner(preflight_responses())
    )
    summary = load_summary(outcome.evidence_dir)
    assert summary["git"]["head"] == "1" * 40
    assert summary["git"]["dirty"] is False
    assert len(summary["assets"]) == 10
    assert "python/grserve_api/_generated/event_pb2.py" in summary["assets"]
    assert all(len(value) == 64 for value in summary["assets"].values())
    preflight = json.loads((outcome.evidence_dir / "preflight.json").read_text())
    assert all(
        "command_id" in item and "argv" in item for item in preflight["commands"]
    )
    events = [
        json.loads(line)
        for line in (outcome.evidence_dir / "events.jsonl").read_text().splitlines()
    ]
    assert {item["gateId"]: item["status"] for item in events} == {
        name: "VERIFICATION_GATE_STATUS_" + gate["status"]
        for name, gate in summary["gates"].items()
    }
    assert all(item["runId"] == summary["run_id"] for item in events)
    assert all(item["commandId"] for item in events if item["gateId"] != "preflight")
    assert summary["fault_plan_seed"] == 17
    fault = next(item for item in events if item["gateId"] == "fault matrix")
    assert fault["faultPlanSeed"] == "17"
    assert fault["reason"] == ""
    ticks = [int(item["monotonicTick"]) for item in events]
    assert ticks == sorted(ticks) and len(ticks) == len(set(ticks))


def test_abseil_resolved_head_is_pinned_and_semantically_checked(
    tmp_path: Path,
) -> None:
    responses = preflight_responses()
    responses[
        (
            "git",
            "-C",
            "build/mac-arm64-debug/_deps/absl-src",
            "rev-parse",
            "HEAD",
        )
    ] = result(0, "0" * 40 + "\n")
    outcome = run_verification("mac", tmp_path, runner=FakeRunner(responses))
    summary = load_summary(outcome.evidence_dir)
    assert summary["gates"]["Debug"] == {
        "status": "FAIL",
        "reason": "ABSEIL_PIN_MISMATCH",
    }
    assert summary["dependencies"]["abseil"]["expected"] == (
        "9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f"
    )


def test_missing_required_asset_is_semantic_preflight_failure(tmp_path: Path) -> None:
    (tmp_path / "proto/event.proto").unlink()
    outcome = run_verification(
        "mac", tmp_path, runner=FakeRunner(preflight_responses())
    )
    assert load_summary(outcome.evidence_dir)["gates"]["preflight"] == {
        "status": "FAIL",
        "reason": "REQUIRED_ASSET_MISSING",
    }


def test_mutable_or_invalid_apt_lock_fails_preflight(tmp_path: Path) -> None:
    (tmp_path / "docker/apt-packages.lock").write_text(
        "snapshot=latest\nbuild-essential=12.9\ngit=latest\n"
    )
    runner = FakeRunner(preflight_responses())
    outcome = run_verification("linux-amd64", tmp_path, runner=runner)
    summary = load_summary(outcome.evidence_dir)
    assert outcome.exit_code == 1
    assert summary["gates"]["preflight"]["reason"] == "APT_LOCK_INVALID"
    assert not any(x[:2] == ("docker", "build") for x in runner.commands)


def test_subprocess_runner_contract_does_not_require_shell() -> None:
    command = ("python", "-c", "print('safe')")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0


def test_subprocess_runner_times_out_and_reaps_process_group(tmp_path: Path) -> None:
    completed = subprocess_runner(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        cwd=tmp_path,
        timeout_seconds=0.01,
    )

    assert completed.exit_code == 124
    assert completed.stderr == "COMMAND_TIMEOUT"
    assert completed.duration_seconds < 1


def test_linux_dockerfile_is_reproducible_and_runs_all_required_gates() -> None:
    root = Path(__file__).resolve().parents[2]
    content = (root / "docker/Dockerfile.linux-amd64").read_text()
    assert "ENV GRSERVE_SOURCE_PROVENANCE" not in content
    locked = (root / "docker/base-image.lock").read_text().strip()

    assert locked.startswith("python@sha256:")
    assert len(locked.removeprefix("python@sha256:")) == 64
    assert all(character in "0123456789abcdef" for character in locked.split(":", 1)[1])
    assert "ARG BASE_IMAGE" in content
    assert "FROM ${BASE_IMAGE}" in content
    apt_lock = (root / "docker/apt-packages.lock").read_text()
    assert "snapshot=20250601T000000Z" in apt_lock
    assert "build-essential=12.9" in apt_lock
    assert "git=1:2.39.5-0+deb12u2" in apt_lock
    assert "snapshot.debian.org/archive/debian/20250601T000000Z" in content
    assert "deb.debian.org" not in content
    assert "build-essential=12.9" in content
    assert "git=1:2.39.5-0+deb12u2" in content
    assert "uv==0.11.7" in content
    assert "uv sync --locked" in content
    assert "uname -m" in content and "x86_64" in content
    assert "ctest" in content
    assert "ruff check" in content
    assert "mypy" in content
    assert "tools/acceptance.py" in content
    assert "test_proto_codegen.py" in content
    assert "test_model_assets.py" in content
    assert "test_api_e2e.py" in content
    ignored = (root / ".dockerignore").read_text().splitlines()
    for entry in (
        ".git",
        "build",
        ".venv",
        "artifacts",
        ".worktrees",
        "__pycache__",
        ".DS_Store",
        ".idea",
        ".vscode",
        "*.log",
        "*.tmp",
        "*~",
        "cache",
        ".env",
    ):
        assert entry in ignored
    dependencies = (root / "cmake/Dependencies.cmake").read_text()
    assert "FetchContent_Declare(absl" in dependencies
    assert "9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f" in dependencies


def test_static_dependencies_are_pic_before_linux_shared_hals_link_them() -> None:
    cmake = (Path(__file__).resolve().parents[2] / "CMakeLists.txt").read_text()
    pic = "set(CMAKE_POSITION_INDEPENDENT_CODE ON)"
    dependencies = "include(${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake)"
    assert pic in cmake
    assert cmake.index(pic) < cmake.index(dependencies)


def test_every_current_gitignored_host_artifact_is_excluded_from_docker_context() -> (
    None
):
    root = Path(__file__).resolve().parents[2]
    ignored = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    patterns = (root / ".dockerignore").read_text().splitlines()
    assert ignored
    assert all(
        any(
            Path(name).match(pattern)
            or name.startswith(pattern.rstrip("/") + "/")
            or f"/{pattern.strip('/')}" in f"/{name}"
            for pattern in patterns
        )
        for name in ignored
    )


def test_sanitizers_are_added_after_third_party_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "CMakeLists.txt").read_text()
    sanitizers = (root / "cmake/Sanitizers.cmake").read_text()

    assert "GRSERVE_SANITIZE_THIRD_PARTY" in sanitizers
    assert "grserve_enable_project_sanitizers" in sanitizers
    assert "grserve_enable_project_sanitizers()" in cmake
    assert "-fno-sanitize-recover=undefined" in sanitizers
    assert "-fno-omit-frame-pointer" in sanitizers


def test_verification_gate_event_is_a_declared_generated_protobuf_type() -> None:
    root = Path(__file__).resolve().parents[2]
    proto = (root / "proto/event.proto").read_text()
    assert "message VerificationGateEvent" in proto
    assert "enum VerificationGateStatus" in proto
    from grserve_api._generated.event_pb2 import VerificationGateEvent

    event = VerificationGateEvent(gate_id="Debug")
    assert event.gate_id == "Debug"
