#!/usr/bin/env python3
"""Resumable segmented downloader for a public ModelScope repository."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_range(url: str, part: Path, start: int, end: int) -> None:
    part.parent.mkdir(parents=True, exist_ok=True)
    expected = end - start + 1
    existing = part.stat().st_size if part.exists() else 0
    if existing == expected:
        return
    if existing > expected:
        part.unlink()
        existing = 0

    for attempt in range(8):
        offset = start + existing
        request = Request(url, headers={"Range": f"bytes={offset}-{end}"})
        try:
            with urlopen(request, timeout=120) as response, part.open("ab") as output:
                if response.status != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status}")
                while True:
                    chunk = response.read(4 << 20)
                    if not chunk:
                        break
                    output.write(chunk)
                    existing += len(chunk)
            if existing == expected:
                return
            raise RuntimeError(f"short range: {existing}/{expected} bytes")
        except Exception as exc:
            if attempt == 7:
                raise RuntimeError(f"failed {part}: {exc}") from exc
            time.sleep(min(2**attempt, 30))
            existing = part.stat().st_size if part.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--include-prefix", action="append", default=[])
    args = parser.parse_args()

    namespace, name = args.repo.split("/", 1)
    api = (
        f"https://www.modelscope.cn/api/v1/models/{namespace}/{name}/repo/files"
        f"?Revision={quote(args.revision)}&Recursive=true"
    )
    payload = fetch_json(api)
    if payload.get("Code") != 200:
        raise RuntimeError(payload)
    files = [item for item in payload["Data"]["Files"] if item["Type"] == "blob"]
    if args.include_prefix:
        files = [
            item
            for item in files
            if any(item["Path"].startswith(prefix) for prefix in args.include_prefix)
        ]
    args.dest.mkdir(parents=True, exist_ok=True)
    parts_root = args.dest / ".segdl"
    tasks: list[tuple[str, Path, int, int]] = []
    manifests: list[tuple[Path, int, str, list[Path]]] = []

    for item in files:
        relative = item["Path"]
        size = int(item["Size"])
        expected_hash = item.get("Sha256") or ""
        target = args.dest / relative
        if target.exists() and target.stat().st_size == size:
            if not expected_hash or sha256(target) == expected_hash:
                print(f"SKIP {relative}", flush=True)
                continue
        segments = args.segments if size >= 100 << 20 else 1
        chunk_size = (size + segments - 1) // segments
        url = (
            f"https://www.modelscope.cn/models/{namespace}/{name}/resolve/"
            f"{quote(args.revision)}/{quote(relative)}"
        )
        parts: list[Path] = []
        for index in range(segments):
            start = index * chunk_size
            if start >= size:
                break
            end = min(size - 1, start + chunk_size - 1)
            part = parts_root / f"{relative}.part{index:02d}"
            parts.append(part)
            tasks.append((url, part, start, end))
        manifests.append((target, size, expected_hash, parts))

    print(f"DOWNLOAD tasks={len(tasks)} workers={args.workers}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_range, *task) for task in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(f"PART {index}/{len(futures)}", flush=True)

    for target, size, expected_hash, parts in manifests:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".assembling")
        with temporary.open("wb") as output:
            for part in parts:
                with part.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 << 20), b""):
                        output.write(chunk)
        if temporary.stat().st_size != size:
            raise RuntimeError(f"size mismatch for {target}")
        if expected_hash and sha256(temporary) != expected_hash:
            raise RuntimeError(f"sha256 mismatch for {target}")
        os.replace(temporary, target)
        for part in parts:
            part.unlink()
        print(f"OK {target.relative_to(args.dest)}", flush=True)

    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
