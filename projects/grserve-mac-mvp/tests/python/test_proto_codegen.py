from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CODEGEN = REPO_ROOT / "tools" / "proto_codegen.py"
GENERATED = REPO_ROOT / "python" / "grserve_api" / "_generated"


@dataclass(frozen=True)
class Toolchain:
    protoc: Path
    attestation: Path
    configuration: str


@dataclass(frozen=True)
class Sandbox:
    root: Path
    toolchain: Toolchain


def _git(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _pinned_protoc() -> Path:
    configured = os.environ.get("GRSERVE_PROTOC")
    candidates = (
        Path(configured) if configured else None,
        REPO_ROOT
        / "build"
        / "mac-arm64-debug"
        / "third_party"
        / "protobuf"
        / "protoc-30.2.0",
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_absolute() and candidate.is_file():
            return candidate
    pytest.fail(
        "build pinned protoc first: cmake --build --preset mac-arm64-debug --target protoc"
    )


def _configured_toolchain() -> Toolchain:
    protoc = _pinned_protoc()
    configured = os.environ.get("GRSERVE_PROTOC_ATTESTATION")
    attestation = (
        Path(configured)
        if configured
        else REPO_ROOT
        / "build"
        / "mac-arm64-debug"
        / "proto"
        / "Debug"
        / "protoc.attestation"
    )
    if not attestation.is_file():
        pytest.fail("configure mac-arm64-debug to create the protoc attestation")
    return Toolchain(protoc=protoc, attestation=attestation, configuration="Debug")


def _run(
    codegen: Path,
    toolchain: Toolchain,
    *arguments: str,
    cwd: Path | None = None,
    protoc: Path | None = None,
    attestation: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(codegen),
            *arguments,
            "--protoc",
            str(protoc if protoc is not None else toolchain.protoc),
            "--attestation",
            str(attestation if attestation is not None else toolchain.attestation),
            "--config",
            toolchain.configuration,
        ],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _digest_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def _package_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "python")
    return environment


def _replace_cache_value(sandbox: Sandbox, key: str, replacement: str) -> None:
    cache = sandbox.root / "build" / "test-debug" / "CMakeCache.txt"
    lines = cache.read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith(f"{key}:")]
    assert len(matches) == 1
    index = matches[0]
    declaration = lines[index].partition("=")[0]
    lines[index] = f"{declaration}={replacement}"
    cache.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_codegen(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("proto_codegen_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    root = tmp_path / "repo copy ; touch SHOULD_NOT_EXIST"
    (root / "tools").mkdir(parents=True)
    (root / "proto").mkdir()
    (root / "python" / "grserve_api" / "_generated").mkdir(parents=True)
    shutil.copy2(CODEGEN, root / "tools" / "proto_codegen.py")
    shutil.copy2(REPO_ROOT / "proto" / "engine.proto", root / "proto" / "engine.proto")
    shutil.copy2(REPO_ROOT / "proto" / "event.proto", root / "proto" / "event.proto")
    shutil.copy2(
        GENERATED / "__init__.py",
        root / "python" / "grserve_api" / "_generated" / "__init__.py",
    )
    shutil.copy2(
        GENERATED / "engine_pb2.py",
        root / "python" / "grserve_api" / "_generated" / "engine_pb2.py",
    )
    shutil.copy2(
        GENERATED / "event_pb2.py",
        root / "python" / "grserve_api" / "_generated" / "event_pb2.py",
    )
    (root / ".gitignore").write_text("build/\n", encoding="utf-8")
    (root / "CMakePresets.json").write_text(
        json.dumps(
            {
                "version": 6,
                "configurePresets": [
                    {
                        "name": "base",
                        "hidden": True,
                        "generator": "Ninja",
                        "binaryDir": "${sourceDir}/build/${presetName}",
                        "cacheVariables": {"CMAKE_OSX_ARCHITECTURES": "arm64"},
                    },
                    {
                        "name": "test-debug",
                        "inherits": "base",
                        "cacheVariables": {"CMAKE_BUILD_TYPE": "Debug"},
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.28)\n"
        "project(grserve_proto_fixture LANGUAGES NONE)\n"
        'set(GRSERVE_SANITIZERS "" CACHE STRING "")\n'
        'set(protobuf_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/protobuf" '
        'CACHE PATH "")\n'
        'set(protobuf_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/protobuf" '
        'CACHE PATH "")\n',
        encoding="utf-8",
    )

    protobuf_source = root / "third_party" / "protobuf"
    protobuf_source.mkdir(parents=True)
    _git("init", "-q", cwd=protobuf_source)
    _git("config", "user.email", "tests@grserve.invalid", cwd=protobuf_source)
    _git("config", "user.name", "GRServe Tests", cwd=protobuf_source)
    (protobuf_source / "version.json").write_text(
        '{"30.x":{"protoc_version":"30.2","languages":{"python":"6.30.2"}}}\n',
        encoding="utf-8",
    )
    _git("add", "version.json", cwd=protobuf_source)
    _git("commit", "-qm", "pin protobuf", cwd=protobuf_source)
    gitlink = _git("rev-parse", "HEAD", cwd=protobuf_source)

    _git("init", "-q", cwd=root)
    _git("config", "user.email", "tests@grserve.invalid", cwd=root)
    _git("config", "user.name", "GRServe Tests", cwd=root)
    _git(
        "add",
        ".gitignore",
        "CMakeLists.txt",
        "CMakePresets.json",
        "tools",
        "proto",
        "python",
        cwd=root,
    )
    _git(
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{gitlink},third_party/protobuf",
        cwd=root,
    )
    _git("commit", "-qm", "fixture", cwd=root)

    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.fail("cmake is required for proto codegen tests")
    configured = subprocess.run(
        [cmake, "--preset", "test-debug"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert configured.returncode == 0, configured.stderr

    build = root / "build" / "test-debug"
    protoc = build / "third_party" / "protobuf" / "protoc-30.2.0"
    protoc.parent.mkdir(parents=True)
    try:
        os.link(_pinned_protoc(), protoc)
    except OSError:
        shutil.copy2(_pinned_protoc(), protoc)
    attestation = build / "proto" / "Debug" / "protoc.attestation"
    attestation.parent.mkdir(parents=True)
    attested = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "proto_codegen.py"),
            "--write-attestation",
            str(attestation),
            "--protoc",
            str(protoc),
            "--config",
            "Debug",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert attested.returncode == 0, attested.stderr
    return Sandbox(
        root=root,
        toolchain=Toolchain(protoc, attestation, configuration="Debug"),
    )


def test_check_accepts_committed_output_without_modifying_it(tmp_path: Path) -> None:
    before = _digest_tree(GENERATED)
    result = _run(CODEGEN, _configured_toolchain(), "--check", cwd=tmp_path)
    after = _digest_tree(GENERATED)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "protobuf Python output is up to date\n"
    assert result.stderr == ""
    assert after == before


def test_package_import_does_not_import_generated_protocol() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import grserve_api; "
            "assert 'grserve_api._generated.engine_pb2' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_package_environment(),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_generated_protocol_imports_with_pinned_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import google.protobuf; "
            "from grserve_api._generated import engine_pb2; "
            "assert google.protobuf.__version__ == '6.30.2'; "
            "assert engine_pb2.DESCRIPTOR.package == 'grserve.rpc.v1'",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_package_environment(),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_check_reports_proto_change_as_stale(sandbox: Sandbox) -> None:
    proto = sandbox.root / "proto" / "engine.proto"
    proto.write_text(
        proto.read_text(encoding="utf-8").replace(
            "uint32 major = 1;", "uint64 major = 1;"
        ),
        encoding="utf-8",
    )
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "stale generated protobuf file: engine_pb2.py\n"


def test_check_reports_generated_change_as_stale(sandbox: Sandbox) -> None:
    generated = sandbox.root / "python" / "grserve_api" / "_generated" / "engine_pb2.py"
    generated.write_text(generated.read_text() + "# stale\n", encoding="utf-8")
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 1
    assert result.stderr == "stale generated protobuf file: engine_pb2.py\n"


def test_check_reports_missing_generated_file(sandbox: Sandbox) -> None:
    (sandbox.root / "python" / "grserve_api" / "_generated" / "engine_pb2.py").unlink()
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 1
    assert result.stderr == "missing generated protobuf file: engine_pb2.py\n"


def test_check_reports_extra_generated_file(sandbox: Sandbox) -> None:
    extra = sandbox.root / "python" / "grserve_api" / "_generated" / "obsolete_pb2.py"
    extra.write_text("# obsolete\n", encoding="utf-8")
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 1
    assert result.stderr == "extra generated protobuf file: obsolete_pb2.py\n"


def test_missing_protoc_has_stable_error(sandbox: Sandbox) -> None:
    missing = (
        sandbox.root / "build" / "test-debug" / "third_party" / "protobuf" / "missing"
    )
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py",
        sandbox.toolchain,
        "--check",
        protoc=missing,
    )
    assert result.returncode == 2
    assert result.stderr == f"protoc not found or not executable: {missing}\n"


def test_relative_and_path_protoc_are_rejected(sandbox: Sandbox) -> None:
    for untrusted in (Path("protoc"), Path("third_party/protobuf/protoc")):
        result = _run(
            sandbox.root / "tools" / "proto_codegen.py",
            sandbox.toolchain,
            "--check",
            protoc=untrusted,
        )
        assert result.returncode == 2
        assert result.stderr == f"protoc path must be absolute: {untrusted}\n"


def test_symlink_protoc_is_rejected(sandbox: Sandbox) -> None:
    linked = sandbox.toolchain.protoc.with_name("linked-protoc")
    linked.symlink_to(sandbox.toolchain.protoc)
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py",
        sandbox.toolchain,
        "--check",
        protoc=linked,
    )
    assert result.returncode == 2
    assert result.stderr == f"protoc path must not contain symlinks: {linked}\n"


def test_absolute_protoc_outside_repository_build_is_rejected(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    outside = tmp_path / "system-protoc-30.2.0"
    try:
        os.link(_pinned_protoc(), outside)
    except OSError:
        shutil.copy2(_pinned_protoc(), outside)
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py",
        sandbox.toolchain,
        "--check",
        protoc=outside,
    )
    assert result.returncode == 2
    assert result.stderr == "protoc is not a repository build artifact\n"


def test_fake_same_version_is_rejected_by_attested_digest(sandbox: Sandbox) -> None:
    fake = sandbox.toolchain.protoc
    fake.unlink()
    fake.write_text("#!/bin/sh\nprintf 'libprotoc 30.2\\n'\n", encoding="utf-8")
    fake.chmod(0o755)
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 2
    assert result.stderr == "protoc sha256 does not match CMake attestation\n"


def test_handwritten_attestation_without_declared_preset_is_rejected(
    sandbox: Sandbox,
) -> None:
    build = sandbox.root / "build" / "handwritten"
    protoc = build / "third_party" / "protobuf" / "protoc-30.2.0"
    protoc.parent.mkdir(parents=True)
    try:
        os.link(sandbox.toolchain.protoc, protoc)
    except OSError:
        shutil.copy2(sandbox.toolchain.protoc, protoc)
    attestation = build / "proto" / "Debug" / "protoc.attestation"
    attestation.parent.mkdir(parents=True)
    attestation.write_text(
        sandbox.toolchain.attestation.read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py",
        sandbox.toolchain,
        "--check",
        protoc=protoc,
        attestation=attestation,
    )
    assert result.returncode == 2
    assert result.stderr == (f"no declared configure preset uses build tree: {build}\n")


def test_missing_cmake_cache_is_rejected(sandbox: Sandbox) -> None:
    cache = sandbox.root / "build" / "test-debug" / "CMakeCache.txt"
    cache.unlink()
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 2
    assert result.stderr == f"CMake cache not found: {cache}\n"


def test_write_attestation_without_cmake_cache_is_rejected(sandbox: Sandbox) -> None:
    cache = sandbox.root / "build" / "test-debug" / "CMakeCache.txt"
    cache.unlink()
    result = subprocess.run(
        [
            sys.executable,
            str(sandbox.root / "tools" / "proto_codegen.py"),
            "--write-attestation",
            str(sandbox.toolchain.attestation),
            "--protoc",
            str(sandbox.toolchain.protoc),
            "--config",
            "Debug",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr == f"CMake cache not found: {cache}\n"


def test_attested_unknown_preset_is_rejected(sandbox: Sandbox) -> None:
    text = sandbox.toolchain.attestation.read_text(encoding="utf-8")
    sandbox.toolchain.attestation.write_text(
        text.replace("preset=test-debug", "preset=unknown"), encoding="utf-8"
    )
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 2
    assert result.stderr == "attested preset does not match configured preset\n"


def test_preset_binary_directory_drift_is_rejected(sandbox: Sandbox) -> None:
    presets = sandbox.root / "CMakePresets.json"
    presets.write_text(
        presets.read_text(encoding="utf-8").replace(
            '"binaryDir": "${sourceDir}/build/${presetName}"',
            '"binaryDir": "${sourceDir}/build/drift-${presetName}"',
        ),
        encoding="utf-8",
    )
    build = sandbox.root / "build" / "test-debug"
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 2
    assert result.stderr == f"no declared configure preset uses build tree: {build}\n"


@pytest.mark.parametrize(
    ("key", "replacement", "expected"),
    [
        (
            "CMAKE_HOME_DIRECTORY",
            "/tmp/not-the-repository",
            "CMake cache CMAKE_HOME_DIRECTORY does not match preset: ",
        ),
        (
            "CMAKE_GENERATOR",
            "Unix Makefiles",
            "CMake cache CMAKE_GENERATOR does not match preset: ",
        ),
        (
            "CMAKE_BUILD_TYPE",
            "Release",
            "CMake cache CMAKE_BUILD_TYPE does not match preset: ",
        ),
        (
            "GRSERVE_SANITIZERS",
            "thread",
            "CMake cache GRSERVE_SANITIZERS does not match preset: ",
        ),
        (
            "protobuf_SOURCE_DIR",
            "/tmp/not-pinned-protobuf",
            "CMake cache protobuf_SOURCE_DIR does not match preset: ",
        ),
        (
            "protobuf_BINARY_DIR",
            "/tmp/not-configured-protobuf",
            "CMake cache protobuf_BINARY_DIR does not match preset: ",
        ),
    ],
)
def test_cmake_cache_context_drift_is_rejected(
    sandbox: Sandbox, key: str, replacement: str, expected: str
) -> None:
    _replace_cache_value(sandbox, key, replacement)
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 2
    assert result.stderr.startswith(expected)


def test_configuration_identity_must_match_cache(sandbox: Sandbox) -> None:
    toolchain = Toolchain(
        sandbox.toolchain.protoc,
        sandbox.toolchain.attestation,
        configuration="Release",
    )
    result = _run(sandbox.root / "tools" / "proto_codegen.py", toolchain, "--check")
    assert result.returncode == 2
    assert result.stderr == (
        "configuration does not match CMake cache: expected Debug, got Release\n"
    )


def test_ninja_multi_config_uses_configuration_specific_protoc(
    sandbox: Sandbox,
) -> None:
    presets_path = sandbox.root / "CMakePresets.json"
    presets = json.loads(presets_path.read_text(encoding="utf-8"))
    presets["configurePresets"][0]["generator"] = "Ninja Multi-Config"
    del presets["configurePresets"][1]["cacheVariables"]["CMAKE_BUILD_TYPE"]
    presets_path.write_text(json.dumps(presets, indent=2) + "\n", encoding="utf-8")
    build = sandbox.root / "build" / "test-debug"
    shutil.rmtree(build)
    cmake = shutil.which("cmake")
    assert cmake is not None
    configured = subprocess.run(
        [cmake, "--preset", "test-debug"],
        cwd=sandbox.root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert configured.returncode == 0, configured.stderr
    protoc = build / "third_party" / "protobuf" / "Debug" / "protoc-30.2.0"
    protoc.parent.mkdir(parents=True)
    try:
        os.link(_pinned_protoc(), protoc)
    except OSError:
        shutil.copy2(_pinned_protoc(), protoc)
    attestation = build / "proto" / "Debug" / "protoc.attestation"
    attestation.parent.mkdir(parents=True)
    attested = subprocess.run(
        [
            sys.executable,
            str(sandbox.root / "tools" / "proto_codegen.py"),
            "--write-attestation",
            str(attestation),
            "--protoc",
            str(protoc),
            "--config",
            "Debug",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert attested.returncode == 0, attested.stderr
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py",
        Toolchain(protoc, attestation, configuration="Debug"),
        "--check",
    )
    assert result.returncode == 0, result.stderr


def test_wrong_protoc_version_is_rejected(sandbox: Sandbox) -> None:
    fake = sandbox.toolchain.protoc
    fake.unlink()
    fake.write_text("#!/bin/sh\nprintf 'libprotoc 29.0\\n'\n", encoding="utf-8")
    fake.chmod(0o755)
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 2
    assert result.stderr == "expected libprotoc 30.2, got libprotoc 29.0\n"


def test_submodule_head_drift_is_rejected(sandbox: Sandbox) -> None:
    source = sandbox.root / "third_party" / "protobuf"
    (source / "drift").write_text("drift\n", encoding="utf-8")
    _git("add", "drift", cwd=source)
    _git("commit", "-qm", "drift", cwd=source)
    actual = _git("rev-parse", "HEAD", cwd=source)
    expected = _git("rev-parse", "HEAD:third_party/protobuf", cwd=sandbox.root)
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 2
    assert result.stderr == (
        "protobuf submodule HEAD does not match repository gitlink: "
        f"expected {expected}, got {actual}\n"
    )


def test_attested_gitlink_mismatch_is_rejected(sandbox: Sandbox) -> None:
    text = sandbox.toolchain.attestation.read_text(encoding="utf-8")
    gitlink = _git("rev-parse", "HEAD:third_party/protobuf", cwd=sandbox.root)
    sandbox.toolchain.attestation.write_text(
        text.replace(f"gitlink={gitlink}", f"gitlink={'0' * 40}"), encoding="utf-8"
    )
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 2
    assert (
        result.stderr == "attested protobuf gitlink does not match repository gitlink\n"
    )


def test_sealed_source_provenance_validates_without_git_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sealed"
    (root / "tools").mkdir(parents=True)
    (root / "third_party/protobuf").mkdir(parents=True)
    shutil.copy2(CODEGEN, root / "tools/proto_codegen.py")
    (root / "tracked.txt").write_text("root\n")
    (root / "third_party/protobuf/version.json").write_text("protobuf\n")

    def digest(name: str, content: bytes) -> str:
        value = hashlib.sha256(content).hexdigest()
        aggregate = hashlib.sha256()
        encoded = name.encode()
        aggregate.update(len(encoded).to_bytes(4, "big"))
        aggregate.update(encoded)
        aggregate.update(bytes.fromhex(value))
        return aggregate.hexdigest()

    manifest = {
        "format": 1,
        "root_head": "1" * 40,
        "root_tree_sha256": digest("tracked.txt", b"root\n"),
        "protobuf_gitlink": "2" * 40,
        "protobuf_head": "2" * 40,
        "protobuf_tree_sha256": digest("version.json", b"protobuf\n"),
        "files": {
            "root": {"tracked.txt": hashlib.sha256(b"root\n").hexdigest()},
            "protobuf": {"version.json": hashlib.sha256(b"protobuf\n").hexdigest()},
        },
    }
    provenance = root / "source-provenance.json"
    provenance.write_text(json.dumps(manifest))
    module = _load_codegen(root / "tools/proto_codegen.py")
    old = os.environ.get("GRSERVE_SOURCE_PROVENANCE")
    os.environ["GRSERVE_SOURCE_PROVENANCE"] = str(provenance)
    try:
        assert module._source_identity(root) == {
            "root_head": "1" * 40,
            "gitlink": "2" * 40,
            "root_tree_sha256": manifest["root_tree_sha256"],
            "protobuf_tree_sha256": manifest["protobuf_tree_sha256"],
        }
        (root / "tracked.txt").write_text("tampered\n")
        with pytest.raises(
            module.CodegenError, match="sealed source provenance mismatch"
        ):
            module._source_identity(root)
    finally:
        if old is None:
            os.environ.pop("GRSERVE_SOURCE_PROVENANCE", None)
        else:
            os.environ["GRSERVE_SOURCE_PROVENANCE"] = old


def test_git_checkout_ignores_inherited_sealed_provenance_path(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GRSERVE_SOURCE_PROVENANCE", "/workspace/source-provenance.json")
    module = _load_codegen(sandbox.root / "tools/proto_codegen.py")
    identity = module._source_identity(sandbox.root)
    assert identity["gitlink"] == _git(
        "rev-parse", "HEAD:third_party/protobuf", cwd=sandbox.root
    )


def test_git_failure_is_fail_closed(sandbox: Sandbox) -> None:
    metadata = sandbox.root / ".git"
    hidden = sandbox.root / ".git-hidden"
    metadata.rename(hidden)
    try:
        result = _run(
            sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
        )
    finally:
        hidden.rename(metadata)
    assert result.returncode == 2
    assert result.stderr == "failed to read repository protobuf gitlink\n"


def test_paths_with_spaces_and_shell_metacharacters_are_literal(
    sandbox: Sandbox,
) -> None:
    marker = sandbox.root.parent / "SHOULD_NOT_EXIST"
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py",
        sandbox.toolchain,
        "--check",
        cwd=sandbox.root.parent,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("component", ["python", "grserve_api", "_generated"])
def test_generated_parent_symlink_never_writes_outside(
    sandbox: Sandbox, tmp_path: Path, component: str
) -> None:
    generated = sandbox.root / "python" / "grserve_api" / "_generated"
    if component == "python":
        original = sandbox.root / "python"
        outside = tmp_path / "outside-python"
        (outside / "grserve_api" / "_generated").mkdir(parents=True)
        outside_generated = outside / "grserve_api" / "_generated"
    elif component == "grserve_api":
        original = sandbox.root / "python" / "grserve_api"
        outside = tmp_path / "outside-api"
        (outside / "_generated").mkdir(parents=True)
        outside_generated = outside / "_generated"
    else:
        original = generated
        outside = tmp_path / "outside-generated"
        outside.mkdir()
        outside_generated = outside
    sentinel = outside_generated / "engine_pb2.py"
    sentinel.write_text("outside sentinel\n", encoding="utf-8")
    held = original.with_name(f"{original.name}-held")
    original.rename(held)
    original.symlink_to(outside, target_is_directory=True)
    result = _run(sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain)
    assert result.returncode == 2
    assert result.stderr == f"unsafe generated protobuf path component: {component}\n"
    assert sentinel.read_text(encoding="utf-8") == "outside sentinel\n"
    assert not list(outside_generated.glob("*.tmp"))


def test_existing_target_symlink_is_rejected_without_touching_target(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    target = sandbox.root / "python" / "grserve_api" / "_generated" / "engine_pb2.py"
    victim = tmp_path / "outside.py"
    victim.write_text("outside sentinel\n", encoding="utf-8")
    target.unlink()
    target.symlink_to(victim)
    result = _run(sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain)
    assert result.returncode == 2
    assert result.stderr == (
        "unsafe generated protobuf file: engine_pb2.py is not a regular file\n"
    )
    assert victim.read_text(encoding="utf-8") == "outside sentinel\n"
    assert not list(target.parent.glob("*.tmp"))


def test_check_refuses_target_symlink(sandbox: Sandbox, tmp_path: Path) -> None:
    target = sandbox.root / "python" / "grserve_api" / "_generated" / "engine_pb2.py"
    victim = tmp_path / "outside.py"
    victim.write_text("outside sentinel\n", encoding="utf-8")
    target.unlink()
    target.symlink_to(victim)
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
    )
    assert result.returncode == 2
    assert result.stderr == (
        "unsafe generated protobuf file: engine_pb2.py is not a regular file\n"
    )
    assert victim.read_text(encoding="utf-8") == "outside sentinel\n"


def test_open_directory_fd_prevents_parent_exchange_write(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    module = _load_codegen(sandbox.root / "tools" / "proto_codegen.py")
    generated = sandbox.root / "python" / "grserve_api" / "_generated"
    held = generated.with_name("_generated-held")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "engine_pb2.py"
    outside_target.write_text("outside sentinel\n", encoding="utf-8")
    with module._open_generated_directory(sandbox.root) as directory_fd:
        generated.rename(held)
        generated.symlink_to(outside, target_is_directory=True)
        module._replace_generated_file(directory_fd, "engine_pb2.py", b"replacement\n")
    assert outside_target.read_text(encoding="utf-8") == "outside sentinel\n"
    assert (held / "engine_pb2.py").read_bytes() == b"replacement\n"
    assert not list(held.glob("*.tmp"))
    assert not list(outside.glob("*.tmp"))


def test_parallel_checks_are_read_only_and_race_free(sandbox: Sandbox) -> None:
    generated = sandbox.root / "python" / "grserve_api" / "_generated"
    before = _digest_tree(generated)

    def check_once(_: int) -> subprocess.CompletedProcess[str]:
        return _run(
            sandbox.root / "tools" / "proto_codegen.py", sandbox.toolchain, "--check"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(check_once, range(16)))
    assert all(result.returncode == 0 for result in results), [
        result.stderr for result in results
    ]
    assert _digest_tree(generated) == before
    assert not list(generated.glob("*.tmp"))


def test_generate_atomically_replaces_only_managed_output(sandbox: Sandbox) -> None:
    generated = sandbox.root / "python" / "grserve_api" / "_generated"
    target = generated / "engine_pb2.py"
    init_before = (generated / "__init__.py").read_bytes()
    target.write_text("stale\n", encoding="utf-8")
    result = _run(
        sandbox.root / "tools" / "proto_codegen.py",
        sandbox.toolchain,
        cwd=sandbox.root.parent,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "generated committed Python protobuf modules\n"
    assert (generated / "__init__.py").read_bytes() == init_before
    assert "stale" not in target.read_text(encoding="utf-8")
    assert not list(generated.glob("*.tmp"))
    assert (
        _run(
            sandbox.root / "tools" / "proto_codegen.py",
            sandbox.toolchain,
            "--check",
        ).returncode
        == 0
    )
