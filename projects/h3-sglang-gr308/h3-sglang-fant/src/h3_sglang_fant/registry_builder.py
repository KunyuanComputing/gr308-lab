"""Build a narrow, drift-checked SGLang diffusion platform overlay."""

from hashlib import sha256
from pathlib import Path

ANCHOR = (
    '            "musa": '
    '"sglang.multimodal_gen.runtime.platforms.musa.MusaPlatform",\n'
)
ENTRY = (
    '            "fant": '
    '"h3_sglang_fant.platform.FantasyDiffusionPlatform",\n'
)


def build_registry_overlay(source: Path, output: Path) -> str:
    """Copy the upstream registry and add exactly one explicit Fantasy entry."""

    source_text = source.read_text()
    if ENTRY.strip() in source_text or source_text.count(ANCHOR) != 1:
        raise RuntimeError(
            "SGLang diffusion registry anchor is missing, duplicated, or already patched"
        )

    rendered = source_text.replace(ANCHOR, ANCHOR + ENTRY, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    return sha256(rendered.encode()).hexdigest()

