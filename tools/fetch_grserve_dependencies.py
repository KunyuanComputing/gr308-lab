#!/usr/bin/env python3
"""Restore pinned upstream sources only when explicitly invoked."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = json.loads((ROOT / 'upstream-dependencies.json').read_text())['grserve']
    base = ROOT / 'projects/grserve-mac-mvp'
    for item in entries:
        target = base / item['path']
        if target.exists():
            actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != item['commit']:
                raise SystemExit(f'Existing dependency has a different commit: {target}')
            print(f'Already at pinned commit: {item["path"]}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', item['url']], check=True)
        subprocess.run(['git', '-C', str(target), 'fetch', '--depth=1', 'origin', item['commit']], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        print(f'Restored: {item["path"]} @ {item["commit"]}')


if __name__ == '__main__':
    main()
