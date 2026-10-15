import json
from pathlib import Path

import pytest

from scripts import fant_sdpa_sdk_probe


EXPECTED_PATHS = (
    "/usr/local/fant/bin/hipcc",
    "/usr/local/fant/include/hip/hip_bfloat16.h",
    "/usr/local/fant/include/hip/devicelib/bfloat16/bfloat162_arithemtic.h",
    "/usr/local/fant/include/fhblas/fhblas.h",
    "/usr/local/fant/lib/libfhblas.so.4.0",
)


def test_sdk_path_status_records_every_required_exact_path():
    present = {EXPECTED_PATHS[0], EXPECTED_PATHS[-1]}

    status = fant_sdpa_sdk_probe.sdk_path_status(present.__contains__)

    assert tuple(status) == EXPECTED_PATHS
    assert status == {path: path in present for path in EXPECTED_PATHS}


def test_format_record_is_one_prefixed_json_object():
    record = {
        "paths": {path: True for path in EXPECTED_PATHS},
        "torch_version": "2.7.1",
        "torch_fant_file": "/opt/torch_fant/__init__.py",
        "compiler_version": "FANT clang version 17",
        "copy_smoke_available": True,
    }

    line = fant_sdpa_sdk_probe.format_record(record)

    prefix, payload = line.split(" ", 1)
    assert prefix == "FANT_SDPA_SDK"
    assert json.loads(payload) == record
    assert "\n" not in line


def test_collect_record_loads_library_before_checking_operator(tmp_path):
    events = []

    class FakeOps:
        class fant_sdpa:
            copy_smoke = object()

        @staticmethod
        def load_library(path):
            events.append(path)

    class FakeTorch:
        __version__ = "2.7.1"
        ops = FakeOps()

    class FakeTorchFant:
        __file__ = "/opt/torch_fant/__init__.py"

    library = tmp_path / "libfant_sdpa.so"
    record = fant_sdpa_sdk_probe.collect_record(
        torch_module=FakeTorch(),
        torch_fant_module=FakeTorchFant(),
        library_path=library,
        path_exists=lambda _path: True,
        compiler_version=lambda _path: "hipcc 17.0.0",
    )

    assert events == [str(library)]
    assert record["copy_smoke_available"] is True
    assert record["compiler_version"] == "hipcc 17.0.0"


def test_main_reports_missing_torch_without_import_time_failure(monkeypatch, capsys):
    real_import = __import__

    def missing_torch(name, *args, **kwargs):
        if name in {"torch", "torch_fant"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_torch)

    with pytest.raises(RuntimeError, match="requires torch and torch_fant"):
        fant_sdpa_sdk_probe.main([])
    assert capsys.readouterr().out == ""


def test_native_sources_declare_required_schemas_and_privateuse1_bindings():
    root = Path(__file__).parents[1]
    source = (root / "native/fant_sdpa/fant_sdpa.cpp").read_text()
    kernel = (root / "native/fant_sdpa/fant_sdpa_kernel.hip").read_text()

    assert "TORCH_LIBRARY(fant_sdpa, m)" in source
    assert 'm.def("copy_smoke(Tensor input) -> Tensor")' in source
    assert 'm.def("forward(Tensor query, Tensor key, Tensor value) -> Tensor")' in source
    assert "TORCH_LIBRARY_IMPL(fant_sdpa, PrivateUse1, m)" in source
    assert 'm.impl("copy_smoke", copy_smoke)' in source
    assert 'm.impl("forward", sdpa_forward)' in source
    assert "c10::DeviceType::PrivateUse1" in source
    assert "at::kBFloat16" in source
    assert "is_contiguous()" in source
    assert "empty_like" in source
    assert "hipLaunchKernelGGL" in kernel


def test_cmake_uses_fant_hipcc_and_torch_abi_inputs():
    root = Path(__file__).parents[1]
    cmake = (root / "native/fant_sdpa/CMakeLists.txt").read_text()

    assert "/usr/local/fant/bin/hipcc" in cmake
    assert "CXX_STANDARD 17" in cmake
    assert "POSITION_INDEPENDENT_CODE ON" in cmake
    assert "TORCH_CXX_FLAGS" in cmake
    assert "TORCH_LIBRARIES" in cmake
    assert "fhblas" in cmake
