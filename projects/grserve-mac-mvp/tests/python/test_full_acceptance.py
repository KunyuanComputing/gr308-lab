from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import stat

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.acceptance import AcceptanceError, load_full_summary, validate_full_summary
from tools.evidence import render_report, validate_report


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_GATES = (
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
    "Linux amd64",
    "API E2E",
)


def full_summary(*, tsan: dict[str, str] | None = None) -> dict[str, object]:
    gates = {name: {"status": "PASS"} for name in REQUIRED_GATES}
    gates["TSan"] = tsan or {"status": "PASS"}
    return {
        "schema_version": 1,
        "profile": "full",
        "status": "PASS",
        "run_id": "20260909T101112Z-a1b2c3d4",
        "critical_skip_count": 0,
        "gates": gates,
        "git": {
            "head": "1" * 40,
            "dirty": False,
            "submodules": [
                {
                    "name": "googletest",
                    "gitlink": "2" * 40,
                    "head": "2" * 40,
                    "clean": True,
                },
                {
                    "name": "protobuf",
                    "gitlink": "3" * 40,
                    "head": "3" * 40,
                    "clean": True,
                },
            ],
        },
        "dependencies": {
            "abseil": {"expected": "4" * 40, "resolved": "4" * 40}
        },
        "assets": {
            "models/mini-qwen38-hybrid/manifest.textproto": "5" * 64,
            "models/mini-qwen38-hybrid/golden.json": "6" * 64,
        },
    }


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")


def test_accepts_exact_full_gate_set_when_all_gates_pass() -> None:
    assert validate_full_summary(full_summary()) is True


def test_accepts_only_the_explicit_tsan_allowlist_reason() -> None:
    summary = full_summary(
        tsan={
            "status": "SKIP",
            "reason": "TSAN_RUNTIME_UNSUPPORTED_APPLE_CLANG",
        }
    )
    assert validate_full_summary(summary) is True


@pytest.mark.parametrize("mutation", ["missing", "unknown", "failed"])
def test_rejects_missing_unknown_or_nonpassing_required_gate(mutation: str) -> None:
    summary = full_summary()
    gates = summary["gates"]
    assert isinstance(gates, dict)
    if mutation == "missing":
        del gates["golden"]
    elif mutation == "unknown":
        gates["bonus"] = {"status": "PASS"}
    else:
        gates["API E2E"] = {"status": "FAIL"}
    with pytest.raises(AcceptanceError):
        validate_full_summary(summary)


@pytest.mark.parametrize(
    "tsan",
    [
        {"status": "FAIL"},
        {"status": "SKIP"},
        {"status": "SKIP", "reason": "COMMAND_FAILED"},
        {
            "status": "SKIP",
            "reason": "TSAN_RUNTIME_UNSUPPORTED_APPLE_CLANG",
            "extra": "not allowed",
        },
    ],
)
def test_rejects_nonpassing_or_ambiguous_tsan(tsan: dict[str, str]) -> None:
    with pytest.raises(AcceptanceError):
        validate_full_summary(full_summary(tsan=tsan))


@pytest.mark.parametrize("value", [True, False, 0.0, "0", None, -1, 1])
def test_critical_skip_count_is_exact_integer_zero(value: object) -> None:
    summary = full_summary()
    summary["critical_skip_count"] = value
    with pytest.raises(AcceptanceError):
        validate_full_summary(summary)


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", 1),
        ("profile", "mac"),
        ("status", True),
        ("status", "FAIL"),
        ("gates", []),
        ("extra", float("nan")),
        ("extra", float("inf")),
    ],
)
def test_rejects_wrong_types_profiles_statuses_and_nonfinite_values(
    field: str, value: object
) -> None:
    summary = full_summary()
    summary[field] = value
    with pytest.raises(AcceptanceError):
        validate_full_summary(summary)


@pytest.mark.parametrize(
    "content",
    [
        '{"profile":"full","profile":"full"}',
        '{"value":NaN}',
        '{"gates":{"preflight":{"status":"PASS","status":"PASS"}}}',
    ],
)
def test_json_loader_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path, content: str
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(content, encoding="utf-8")
    with pytest.raises(AcceptanceError):
        load_full_summary(summary_path)


def test_acceptance_cli_returns_nonzero_for_invalid_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    write_summary(summary_path, {**full_summary(), "critical_skip_count": True})
    completed = subprocess.run(
        [sys.executable, "tools/acceptance.py", "--summary", str(summary_path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "NOT_ACCEPTED" in completed.stderr


def test_rendered_accepted_report_is_bound_to_exact_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "run-a" / "summary.json"
    summary_path.parent.mkdir()
    report_path = tmp_path / "acceptance.md"
    write_summary(summary_path, full_summary())

    render_report(summary_path, report_path)

    report = report_path.read_text(encoding="utf-8")
    assert "ACCEPTED" in report
    assert "1" * 40 in report
    assert "protobuf" in report and "3" * 40 in report
    assert "manifest.textproto" in report and "5" * 64 in report
    assert "golden.json" in report and "6" * 64 in report
    assert all(name in report for name in (*REQUIRED_GATES, "TSan"))
    assert str(summary_path.resolve()) in report
    assert "Fant HAL/P0" in report
    assert validate_report(summary_path, report_path) is True


def test_failed_full_summary_renders_blocked_not_pseudo_acceptance(tmp_path: Path) -> None:
    summary = full_summary()
    summary["status"] = "FAIL"
    gates = summary["gates"]
    assert isinstance(gates, dict)
    gates["Linux amd64"] = {
        "status": "FAIL",
        "reason": "DOCKER_DAEMON_UNAVAILABLE",
    }
    summary_path = tmp_path / "run-blocked" / "summary.json"
    summary_path.parent.mkdir()
    report_path = tmp_path / "acceptance.md"
    write_summary(summary_path, summary)

    render_report(summary_path, report_path)

    report = report_path.read_text(encoding="utf-8")
    assert "NOT ACCEPTED / BLOCKED" in report
    assert "DOCKER_DAEMON_UNAVAILABLE" in report
    assert "本报告不构成 Mac MVP 通过认证" in report
    assert validate_report(summary_path, report_path) is True


def test_validate_report_rejects_tampering_or_different_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "run-a" / "summary.json"
    summary_path.parent.mkdir()
    report_path = tmp_path / "acceptance.md"
    write_summary(summary_path, full_summary())
    render_report(summary_path, report_path)
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace("Linux amd64", "Linux ARM"),
        encoding="utf-8",
    )
    with pytest.raises(AcceptanceError):
        validate_report(summary_path, report_path)


def test_evidence_report_cli_requires_explicit_summary_path(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "tools/evidence.py", "render-report", "--output", str(tmp_path / "x.md")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0


def test_verify_mac_mvp_is_the_exact_strict_full_profile_entrypoint() -> None:
    entrypoint = REPO_ROOT / "tools/verify_mac_mvp.sh"
    assert entrypoint.read_bytes() == (
        b"#!/usr/bin/env bash\n"
        b"set -Eeuo pipefail\n"
        b"exec uv run --locked python tools/verify.py --profile full\n"
    )
    assert entrypoint.stat().st_mode & stat.S_IXUSR


def test_readme_states_local_scope_usage_evidence_and_nonclaims() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "./tools/verify_mac_mvp.sh",
        "uv run --locked grserve-api",
        "GET /health",
        "GET /v1/models",
        "POST /v1/completions",
        "prompt_token_ids",
        "GRSERVE_SUMMARY=",
        "不证明 GR308 性能",
        "不证明 GR308 硬件正确性",
        "不代表服务器上线资格",
    ):
        assert required in readme
