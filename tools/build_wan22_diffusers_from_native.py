#!/usr/bin/env python3
"""Build a Wan2.2 TI2V Diffusers checkpoint from the verified native checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--official-scaffold", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source))
    from wan_e2e_models import load_dit, load_text, load_vae

    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    for name in ("model_index.json", "scheduler"):
        source = args.official_scaffold / name
        target = args.output / name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)

    print("LOAD transformer", flush=True)
    transformer, transformer_info = load_dit(args.native)
    transformer.save_pretrained(
        args.output / "transformer",
        safe_serialization=True,
        max_shard_size="5GB",
    )
    del transformer
    gc.collect()
    print("SAVED transformer", flush=True)

    print("LOAD text_encoder", flush=True)
    text_encoder, tokenizer, text_info = load_text(args.native, args.source)
    text_encoder.save_pretrained(
        args.output / "text_encoder",
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(args.output / "tokenizer")
    del text_encoder, tokenizer
    gc.collect()
    print("SAVED text_encoder", flush=True)

    print("LOAD vae", flush=True)
    vae, vae_info = load_vae(args.native, args.source)
    vae.save_pretrained(
        args.output / "vae",
        safe_serialization=True,
        max_shard_size="5GB",
    )
    del vae
    gc.collect()
    print("SAVED vae", flush=True)

    files = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(args.output)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    manifest = {
        "native": str(args.native),
        "official_scaffold": str(args.official_scaffold),
        "transformer": transformer_info,
        "text_encoder": text_info,
        "vae": vae_info,
        "files": files,
    }
    (args.output / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"status": "complete", "files": len(files)}), flush=True)


if __name__ == "__main__":
    main()
