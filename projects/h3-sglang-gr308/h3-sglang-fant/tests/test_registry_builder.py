from pathlib import Path

import pytest

from h3_sglang_fant.registry_builder import build_registry_overlay


SOURCE = '''forced_map = {
            "cpu": "sglang.multimodal_gen.runtime.platforms.cpu.CpuPlatform",
            "musa": "sglang.multimodal_gen.runtime.platforms.musa.MusaPlatform",
        }
'''


def test_adds_fant_override_once(tmp_path: Path):
    source = tmp_path / "__init__.py"
    output = tmp_path / "overlay.py"
    source.write_text(SOURCE)

    digest = build_registry_overlay(source, output)

    text = output.read_text()
    assert (
        text.count(
            '"fant": "h3_sglang_fant.platform.FantasyDiffusionPlatform"'
        )
        == 1
    )
    assert len(digest) == 64


def test_rejects_upstream_drift(tmp_path: Path):
    source = tmp_path / "__init__.py"
    output = tmp_path / "overlay.py"
    source.write_text("forced_map = {}\n")

    with pytest.raises(RuntimeError, match="registry anchor"):
        build_registry_overlay(source, output)
