#!/usr/bin/env python3
"""Generate or verify the committed Engine RPC Python protobuf module."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import tempfile


EXPECTED_PROTOC_VERSION = "libprotoc 30.2"
GENERATED_FILES = ("engine_pb2.py", "event_pb2.py")
ATTESTATION_FIELDS = (
    "format",
    "repo",
    "build",
    "preset",
    "generator",
    "configuration",
    "cmake_cache",
    "cache_sha256",
    "protobuf_source",
    "protobuf_build",
    "gitlink",
    "root_head",
    "root_tree_sha256",
    "protobuf_tree_sha256",
    "protoc",
    "sha256",
)
DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class CodegenError(RuntimeError):
    """A stable, user-facing code generation failure."""


@dataclass(frozen=True)
class ProtocLocation:
    executable: Path
    build_tree: Path


@dataclass(frozen=True)
class ConfigurePreset:
    name: str
    generator: str
    binary_dir: Path
    cache_variables: dict[str, str]


@dataclass(frozen=True)
class CMakeContext:
    preset: ConfigurePreset
    configuration: str
    cache_path: Path
    cache_values: dict[str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="verify without changing source files"
    )
    mode.add_argument(
        "--write-attestation",
        type=Path,
        metavar="PATH",
        help="write the build-local CMake protoc attestation",
    )
    parser.add_argument(
        "--protoc",
        required=True,
        type=Path,
        help="absolute path to the CMake-built protoc 30.2 binary",
    )
    parser.add_argument(
        "--attestation",
        type=Path,
        help="build-local attestation emitted for protobuf::protoc",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="CMake build configuration (for example Debug or Release)",
    )
    arguments = parser.parse_args()
    if arguments.write_attestation is None and arguments.attestation is None:
        parser.error("--attestation is required unless --write-attestation is used")
    if arguments.write_attestation is not None and arguments.attestation is not None:
        parser.error("--attestation cannot be combined with --write-attestation")
    return arguments


def _error(message: str, exit_code: int) -> int:
    print(message, file=sys.stderr)
    return exit_code


def _run_git(root: Path, *arguments: str, failure: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise CodegenError(failure) from error
    value = result.stdout.strip()
    if (
        result.returncode != 0
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CodegenError(failure)
    return value


def _repository_gitlink(root: Path) -> str:
    return _run_git(
        root,
        "rev-parse",
        "HEAD:third_party/protobuf",
        failure="failed to read repository protobuf gitlink",
    )


def _submodule_head(root: Path) -> str:
    return _run_git(
        root / "third_party" / "protobuf",
        "rev-parse",
        "HEAD",
        failure="failed to read protobuf submodule HEAD",
    )


def _content_tree_digest(root: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        value = _sha256(path)
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _source_identity(root: Path) -> dict[str, str]:
    sealed = root / "source-provenance.json"
    configured = (
        str(sealed) if not (root / ".git").exists() and sealed.exists() else None
    )
    if configured:
        path = Path(configured)
        try:
            if (
                path != root / "source-provenance.json"
                or path.resolve(strict=True) != path
            ):
                raise CodegenError("invalid sealed source provenance path")
            document = json.loads(path.read_text(encoding="utf-8"))
            files = document["files"]
            root_files = files["root"]
            protobuf_files = files["protobuf"]
            if not isinstance(root_files, dict) or not isinstance(protobuf_files, dict):
                raise ValueError
            for name, expected in root_files.items():
                if _sha256(root / name) != expected:
                    raise CodegenError("sealed source provenance mismatch")
            for name, expected in protobuf_files.items():
                if _sha256(root / "third_party/protobuf" / name) != expected:
                    raise CodegenError("sealed source provenance mismatch")
            identity = {
                "root_head": document["root_head"],
                "gitlink": document["protobuf_gitlink"],
                "root_tree_sha256": _content_tree_digest(root, list(root_files)),
                "protobuf_tree_sha256": _content_tree_digest(
                    root / "third_party/protobuf", list(protobuf_files)
                ),
            }
            if (
                document.get("protobuf_head") != identity["gitlink"]
                or document.get("root_tree_sha256") != identity["root_tree_sha256"]
                or document.get("protobuf_tree_sha256")
                != identity["protobuf_tree_sha256"]
            ):
                raise CodegenError("sealed source provenance mismatch")
            return identity
        except CodegenError:
            raise
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise CodegenError("invalid sealed source provenance") from error
    gitlink = _repository_gitlink(root)
    submodule = _submodule_head(root)
    if submodule != gitlink:
        raise CodegenError(
            "protobuf submodule HEAD does not match repository gitlink: "
            f"expected {gitlink}, got {submodule}"
        )

    def committed_tree(directory: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(directory), "ls-tree", "-r", "HEAD"],
                check=False,
                capture_output=True,
            )
        except OSError as error:
            raise CodegenError("failed to read tracked source provenance") from error
        if result.returncode != 0:
            raise CodegenError("failed to read tracked source provenance")
        return hashlib.sha256(result.stdout).hexdigest()

    return {
        "root_head": _run_git(
            root, "rev-parse", "HEAD", failure="failed to read repository HEAD"
        ),
        "gitlink": gitlink,
        "root_tree_sha256": committed_tree(root),
        "protobuf_tree_sha256": committed_tree(root / "third_party/protobuf"),
    }


def _validate_source_pin(root: Path) -> str:
    return _source_identity(root)["gitlink"]


def _preset_error(message: str) -> CodegenError:
    return CodegenError(f"invalid CMakePresets.json: {message}")


def _merge_preset(
    parent: dict[str, object], child: dict[str, object]
) -> dict[str, object]:
    merged = dict(parent)
    parent_cache = parent.get("cacheVariables", {})
    child_cache = child.get("cacheVariables", {})
    if not isinstance(parent_cache, dict) or not isinstance(child_cache, dict):
        raise _preset_error("cacheVariables must be an object")
    merged_cache = dict(parent_cache)
    for key, value in child_cache.items():
        if value is None:
            merged_cache.pop(key, None)
        else:
            merged_cache[key] = value
    merged.update(
        {
            key: value
            for key, value in child.items()
            if key not in {"name", "inherits", "hidden", "cacheVariables"}
        }
    )
    merged["cacheVariables"] = merged_cache
    return merged


def _expand_preset_value(value: str, root: Path, preset_name: str) -> str:
    replacements = {
        "sourceDir": str(root),
        "sourceParentDir": str(root.parent),
        "sourceDirName": root.name,
        "presetName": preset_name,
    }

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        try:
            return replacements[name]
        except KeyError as error:
            raise _preset_error(f"unsupported macro ${{{name}}}") from error

    expanded = re.sub(r"\$\{([^}]+)\}", replace, value)
    if "$" in expanded:
        raise _preset_error("unsupported macro syntax")
    return expanded


def _cache_value(value: object, root: Path, preset_name: str) -> str:
    if isinstance(value, dict):
        if set(value) - {"type", "value"} or "value" not in value:
            raise _preset_error("invalid typed cache variable")
        value = value["value"]
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if not isinstance(value, (str, int, float)):
        raise _preset_error("cache variable value must be scalar")
    return _expand_preset_value(str(value), root, preset_name)


def _load_configure_presets(root: Path) -> tuple[ConfigurePreset, ...]:
    path = root / "CMakePresets.json"
    try:
        if path.resolve(strict=True) != path or not path.is_file():
            raise _preset_error("file must be a regular repository file")
        document = json.loads(path.read_text(encoding="utf-8"))
    except CodegenError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _preset_error("cannot read file") from error
    if not isinstance(document, dict) or not isinstance(
        document.get("configurePresets"), list
    ):
        raise _preset_error("configurePresets must be an array")
    if document.get("include"):
        raise _preset_error("include files are not supported")

    raw_by_name: dict[str, dict[str, object]] = {}
    for item in document["configurePresets"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise _preset_error("every configure preset must have a string name")
        name = item["name"]
        if name in raw_by_name:
            raise _preset_error(f"duplicate configure preset: {name}")
        raw_by_name[name] = item

    resolved: dict[str, dict[str, object]] = {}
    resolving: set[str] = set()

    def resolve(name: str) -> dict[str, object]:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise _preset_error(f"inheritance cycle at preset: {name}")
        try:
            raw = raw_by_name[name]
        except KeyError as error:
            raise _preset_error(f"unknown inherited preset: {name}") from error
        resolving.add(name)
        inherited = raw.get("inherits", [])
        if isinstance(inherited, str):
            parents = [inherited]
        elif isinstance(inherited, list) and all(
            isinstance(parent, str) for parent in inherited
        ):
            parents = inherited
        else:
            raise _preset_error(f"invalid inherits for preset: {name}")
        merged: dict[str, object] = {}
        # CMake gives an earlier preset priority when multiple parents conflict.
        for parent in reversed(parents):
            merged = _merge_preset(merged, resolve(parent))
        merged = _merge_preset(merged, raw)
        resolving.remove(name)
        resolved[name] = merged
        return merged

    presets: list[ConfigurePreset] = []
    for name, raw in raw_by_name.items():
        if raw.get("hidden", False):
            continue
        merged = resolve(name)
        generator = merged.get("generator")
        binary_dir = merged.get("binaryDir")
        cache_variables = merged.get("cacheVariables", {})
        if not isinstance(generator, str) or not isinstance(binary_dir, str):
            raise _preset_error(f"preset lacks generator or binaryDir: {name}")
        if not isinstance(cache_variables, dict):
            raise _preset_error(f"invalid cacheVariables for preset: {name}")
        expanded_binary = Path(_expand_preset_value(binary_dir, root, name))
        if not expanded_binary.is_absolute():
            raise _preset_error(f"binaryDir must be absolute after expansion: {name}")
        normalized_binary = Path(os.path.normpath(expanded_binary))
        try:
            normalized_binary.relative_to(root / "build")
        except ValueError as error:
            raise _preset_error(
                f"binaryDir is outside repository build tree: {name}"
            ) from error
        presets.append(
            ConfigurePreset(
                name=name,
                generator=generator,
                binary_dir=normalized_binary,
                cache_variables={
                    key: _cache_value(value, root, name)
                    for key, value in cache_variables.items()
                    if value is not None
                },
            )
        )
    return tuple(presets)


def _parse_cmake_cache(path: Path) -> dict[str, str]:
    try:
        if path.resolve(strict=True) != path or not path.is_file():
            raise CodegenError(f"CMake cache not found: {path}")
        content = path.read_text(encoding="utf-8")
    except CodegenError:
        raise
    except FileNotFoundError as error:
        raise CodegenError(f"CMake cache not found: {path}") from error
    except (OSError, UnicodeError) as error:
        raise CodegenError(f"invalid CMake cache: {path}") from error
    values: dict[str, str] = {}
    for line in content.splitlines():
        if not line or line.startswith(("#", "//")) or ":" not in line:
            continue
        declaration, separator, value = line.partition("=")
        key, type_separator, _cache_type = declaration.partition(":")
        if not separator or not type_separator or not key or key in values:
            raise CodegenError(f"invalid CMake cache: {path}")
        values[key] = value
    return values


def _is_multi_config_generator(generator: str) -> bool:
    return (
        generator == "Ninja Multi-Config"
        or generator == "Xcode"
        or generator.startswith("Visual Studio ")
    )


def _require_cache_value(values: dict[str, str], key: str, expected: str) -> None:
    actual = values.get(key)
    if actual != expected:
        displayed = "<missing>" if actual is None else actual
        raise CodegenError(
            f"CMake cache {key} does not match preset: expected {expected}, got {displayed}"
        )


def _validate_cmake_context(
    root: Path, location: ProtocLocation, configuration: str
) -> CMakeContext:
    matches = [
        preset
        for preset in _load_configure_presets(root)
        if preset.binary_dir == location.build_tree
    ]
    if not matches:
        raise CodegenError(
            f"no declared configure preset uses build tree: {location.build_tree}"
        )
    if len(matches) != 1:
        raise CodegenError(
            f"multiple configure presets use build tree: {location.build_tree}"
        )
    preset = matches[0]
    cache_path = location.build_tree / "CMakeCache.txt"
    values = _parse_cmake_cache(cache_path)
    _require_cache_value(values, "CMAKE_HOME_DIRECTORY", str(root))
    _require_cache_value(values, "CMAKE_GENERATOR", preset.generator)
    for key, expected in preset.cache_variables.items():
        if key == "CMAKE_BUILD_TYPE" and _is_multi_config_generator(preset.generator):
            continue
        _require_cache_value(values, key, expected)
    _require_cache_value(
        values,
        "GRSERVE_SANITIZERS",
        preset.cache_variables.get("GRSERVE_SANITIZERS", ""),
    )
    if _is_multi_config_generator(preset.generator):
        configurations = values.get("CMAKE_CONFIGURATION_TYPES", "").split(";")
        if not configuration or configuration not in configurations:
            raise CodegenError(
                f"configuration is not available in CMake cache: {configuration}"
            )
    else:
        build_type = values.get("CMAKE_BUILD_TYPE", "")
        if not configuration or configuration != build_type:
            raise CodegenError(
                "configuration does not match CMake cache: "
                f"expected {build_type}, got {configuration}"
            )
    protobuf_source = root / "third_party" / "protobuf"
    protobuf_build = location.build_tree / "third_party" / "protobuf"
    _require_cache_value(values, "protobuf_SOURCE_DIR", str(protobuf_source))
    _require_cache_value(values, "protobuf_BINARY_DIR", str(protobuf_build))
    expected_protoc = protobuf_build / "protoc-30.2.0"
    if _is_multi_config_generator(preset.generator):
        expected_protoc = protobuf_build / configuration / "protoc-30.2.0"
    if location.executable != expected_protoc:
        raise CodegenError("protoc does not match configured protobuf build tree")
    return CMakeContext(
        preset=preset,
        configuration=configuration,
        cache_path=cache_path,
        cache_values=values,
    )


def _validate_protoc_path(protoc: Path, root: Path) -> ProtocLocation:
    if not protoc.is_absolute():
        raise CodegenError(f"protoc path must be absolute: {protoc}")
    if not protoc.is_file() or not os.access(protoc, os.X_OK):
        raise CodegenError(f"protoc not found or not executable: {protoc}")
    try:
        resolved = protoc.resolve(strict=True)
    except OSError as error:
        raise CodegenError(f"protoc not found or not executable: {protoc}") from error
    if resolved != protoc:
        raise CodegenError(f"protoc path must not contain symlinks: {protoc}")
    try:
        relative = protoc.relative_to(root / "build")
    except ValueError as error:
        raise CodegenError("protoc is not a repository build artifact") from error
    single_config_path = (
        len(relative.parts) == 4
        and relative.parts[1:3] == ("third_party", "protobuf")
        and relative.parts[3] == "protoc-30.2.0"
    )
    multi_config_path = (
        len(relative.parts) == 5
        and relative.parts[1:3] == ("third_party", "protobuf")
        and relative.parts[4] == "protoc-30.2.0"
    )
    if not single_config_path and not multi_config_path:
        raise CodegenError("protoc is not a repository build artifact")
    return ProtocLocation(
        executable=protoc, build_tree=root / "build" / relative.parts[0]
    )


def _protoc_version(protoc: Path) -> None:
    try:
        result = subprocess.run(
            [str(protoc), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise CodegenError(f"protoc not found or not executable: {protoc}") from error
    actual = result.stdout.strip()
    if result.returncode != 0 or actual != EXPECTED_PROTOC_VERSION:
        displayed = (
            actual or result.stderr.strip() or f"exit status {result.returncode}"
        )
        raise CodegenError(f"expected {EXPECTED_PROTOC_VERSION}, got {displayed}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_attestation(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in content.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise CodegenError("invalid protoc attestation")
        values[key] = value
    if tuple(values) != ATTESTATION_FIELDS or values["format"] != "3":
        raise CodegenError("invalid protoc attestation")
    return values


def _read_attestation(
    path: Path, root: Path, location: ProtocLocation, context: CMakeContext
) -> dict[str, str]:
    expected = (
        location.build_tree / "proto" / context.configuration / "protoc.attestation"
    )
    if not path.is_absolute() or path != expected:
        raise CodegenError("protoc attestation is not from the selected build tree")
    try:
        if path.resolve(strict=True) != path:
            raise CodegenError("protoc attestation must not contain symlinks")
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise CodegenError(f"protoc attestation not found: {path}") from error
    except (OSError, UnicodeError) as error:
        raise CodegenError("invalid protoc attestation") from error
    values = _parse_attestation(content)
    if values["repo"] != str(root):
        raise CodegenError("attested repository does not match current repository")
    if values["build"] != str(location.build_tree):
        raise CodegenError("attested build tree does not match selected build tree")
    if values["preset"] != context.preset.name:
        raise CodegenError("attested preset does not match configured preset")
    if values["generator"] != context.preset.generator:
        raise CodegenError("attested generator does not match configured preset")
    if values["configuration"] != context.configuration:
        raise CodegenError("attested configuration does not match CMake configuration")
    if values["cmake_cache"] != str(context.cache_path):
        raise CodegenError("attested CMake cache does not match configured build tree")
    if values["protobuf_source"] != str(root / "third_party" / "protobuf"):
        raise CodegenError(
            "attested protobuf source does not match repository submodule"
        )
    if values["protobuf_build"] != str(
        location.build_tree / "third_party" / "protobuf"
    ):
        raise CodegenError(
            "attested protobuf build does not match configured build tree"
        )
    return values


def _validate_toolchain(
    protoc: Path, attestation: Path, root: Path, configuration: str
) -> ProtocLocation:
    location = _validate_protoc_path(protoc, root)
    source = _source_identity(root)
    context = _validate_cmake_context(root, location, configuration)
    values = _read_attestation(attestation, root, location, context)
    if values["gitlink"] != source["gitlink"]:
        raise CodegenError(
            "attested protobuf gitlink does not match repository gitlink"
        )
    for key in ("root_head", "root_tree_sha256", "protobuf_tree_sha256"):
        if values[key] != source[key]:
            raise CodegenError("attested sealed source provenance mismatch")
    if values["protoc"] != str(protoc):
        raise CodegenError("protoc does not match CMake attestation")
    _protoc_version(protoc)
    if values["sha256"] != _sha256(protoc):
        raise CodegenError("protoc sha256 does not match CMake attestation")
    if values["cache_sha256"] != _sha256(context.cache_path):
        raise CodegenError("CMake cache sha256 does not match CMake attestation")
    return location


def _attestation_content(
    root: Path,
    location: ProtocLocation,
    context: CMakeContext,
    source: dict[str, str],
    digest: str,
) -> bytes:
    return (
        "format=3\n"
        f"repo={root}\n"
        f"build={location.build_tree}\n"
        f"preset={context.preset.name}\n"
        f"generator={context.preset.generator}\n"
        f"configuration={context.configuration}\n"
        f"cmake_cache={context.cache_path}\n"
        f"cache_sha256={_sha256(context.cache_path)}\n"
        f"protobuf_source={root / 'third_party' / 'protobuf'}\n"
        f"protobuf_build={location.build_tree / 'third_party' / 'protobuf'}\n"
        f"gitlink={source['gitlink']}\n"
        f"root_head={source['root_head']}\n"
        f"root_tree_sha256={source['root_tree_sha256']}\n"
        f"protobuf_tree_sha256={source['protobuf_tree_sha256']}\n"
        f"protoc={location.executable}\n"
        f"sha256={digest}\n"
    ).encode()


def _write_attestation(
    destination: Path, protoc: Path, root: Path, configuration: str
) -> None:
    location = _validate_protoc_path(protoc, root)
    context = _validate_cmake_context(root, location, configuration)
    expected = location.build_tree / "proto" / configuration / "protoc.attestation"
    if not destination.is_absolute() or destination != expected:
        raise CodegenError("attestation output is not in the selected build tree")
    source = _source_identity(root)
    _protoc_version(protoc)
    content = _attestation_content(root, location, context, source, _sha256(protoc))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".protoc.attestation.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_generated(content: bytes, root: Path) -> bytes:
    normalized = content.replace(b"\r\n", b"\n")
    coding_header = b"# -*- coding: utf-8 -*-\n"
    mypy_header = b"# mypy: ignore-errors\n"
    if normalized.startswith(coding_header) and not normalized.startswith(
        coding_header + mypy_header
    ):
        normalized = coding_header + mypy_header + normalized[len(coding_header) :]
    root_bytes = os.fsencode(str(root))
    if root_bytes in normalized:
        raise CodegenError(
            "generated protobuf output contains an absolute repository path"
        )
    if not normalized.endswith(b"\n"):
        normalized += b"\n"
    return normalized


def _generate(protoc: Path, destination: Path, root: Path) -> None:
    proto_dir = root / "proto"
    command = [
        str(protoc),
        f"--proto_path={proto_dir}",
        f"--python_out={destination}",
        str(proto_dir / "engine.proto"),
        str(proto_dir / "event.proto"),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit status {result.returncode}"
        )
        raise CodegenError(f"protoc failed: {detail}")
    for name in GENERATED_FILES:
        generated = destination / name
        generated.write_bytes(_normalize_generated(generated.read_bytes(), root))


@contextmanager
def _open_generated_directory(root: Path) -> Iterator[int]:
    descriptors: list[int] = []
    try:
        try:
            descriptors.append(os.open(root, DIRECTORY_FLAGS))
        except OSError as error:
            raise CodegenError("unsafe generated protobuf repository root") from error
        for component in ("python", "grserve_api", "_generated"):
            try:
                descriptors.append(
                    os.open(component, DIRECTORY_FLAGS, dir_fd=descriptors[-1])
                )
            except OSError as error:
                raise CodegenError(
                    f"unsafe generated protobuf path component: {component}"
                ) from error
        yield descriptors[-1]
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_generated_file(directory_fd: int, name: str) -> bytes | None:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        raise CodegenError(
            f"unsafe generated protobuf file: {name} is not a regular file"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CodegenError(
                f"unsafe generated protobuf file: {name} is not a regular file"
            )
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def _managed_generated_files(directory_fd: int) -> set[str]:
    managed: set[str] = set()
    for name in os.listdir(directory_fd):
        if not name.endswith("_pb2.py"):
            continue
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise CodegenError(
                f"unsafe generated protobuf file: {name} is not a regular file"
            )
        managed.add(name)
    return managed


def _check(protoc: Path, root: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="grserve-proto-check-") as temporary:
        temporary_dir = Path(temporary)
        _generate(protoc, temporary_dir, root)
        errors: list[str] = []
        with _open_generated_directory(root) as directory_fd:
            for name in GENERATED_FILES:
                committed = _read_generated_file(directory_fd, name)
                if committed is None:
                    errors.append(f"missing generated protobuf file: {name}")
                elif committed != (temporary_dir / name).read_bytes():
                    errors.append(f"stale generated protobuf file: {name}")
            for name in sorted(
                _managed_generated_files(directory_fd).difference(GENERATED_FILES)
            ):
                errors.append(f"extra generated protobuf file: {name}")
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
    print("protobuf Python output is up to date")
    return 0


def _validate_existing_target(directory_fd: int, name: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise CodegenError(
            f"unsafe generated protobuf file: {name} is not a regular file"
        )


def _replace_generated_file(directory_fd: int, name: str, content: bytes) -> None:
    _validate_existing_target(directory_fd, name)
    temporary_name = f".{name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise CodegenError("failed to write generated protobuf file")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _validate_existing_target(directory_fd, name)
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def main() -> int:
    arguments = _parse_arguments()
    root = _repo_root()
    try:
        if arguments.write_attestation is not None:
            _write_attestation(
                arguments.write_attestation,
                arguments.protoc,
                root,
                arguments.config,
            )
            return 0
        assert arguments.attestation is not None
        _validate_toolchain(
            arguments.protoc, arguments.attestation, root, arguments.config
        )
        if arguments.check:
            return _check(arguments.protoc, root)
        with tempfile.TemporaryDirectory(prefix="grserve-proto-generate-") as temporary:
            temporary_dir = Path(temporary)
            _generate(arguments.protoc, temporary_dir, root)
            with _open_generated_directory(root) as directory_fd:
                for name in GENERATED_FILES:
                    content = (temporary_dir / name).read_bytes()
                    _replace_generated_file(directory_fd, name, content)
        print("generated committed Python protobuf modules")
        return 0
    except (CodegenError, OSError) as error:
        return _error(str(error), 2)


if __name__ == "__main__":
    raise SystemExit(main())
