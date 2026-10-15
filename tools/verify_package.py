#!/usr/bin/env python3
"""Read-only syntax, provenance and optional CPU regression checks."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '__pycache__', '.venv', '.pytest_cache', 'third_party', 'vendor', 'build'}


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in IGNORED for x in p.relative_to(ROOT).parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-tests', action='store_true')
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    errors = []
    counts = {'python': 0, 'shell': 0, 'files': 0}
    for path in files():
        rel = str(path.relative_to(ROOT))
        counts['files'] += 1
        if path.stat().st_size >= 100 * 1024**2:
            errors.append({'file': rel, 'error': 'file exceeds package 100 MiB limit'})
        if path.suffix == '.py':
            counts['python'] += 1
            try:
                ast.parse(path.read_text(), filename=rel)
            except SyntaxError as exc:
                errors.append({'file': rel, 'error': f'Python syntax line {exc.lineno}: {exc.msg}'})
        if path.suffix == '.sh':
            counts['shell'] += 1
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append({'file': rel, 'error': result.stderr.replace(str(ROOT), '.')})
    checksums = ROOT / 'CHECKSUMS.sha256'
    checked_hashes = 0
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            digest, name = line.split('  ', 1)
            path = ROOT / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append({'file': name, 'error': 'checksum mismatch'})
            checked_hashes += 1
    for name in ['README.md', 'CODE_MAP.md', 'REPORTS.md', 'REPRODUCIBILITY.md', 'THIRD_PARTY_NOTICES.md']:
        path = ROOT / name
        if not path.exists():
            errors.append({'file': name, 'error': 'missing navigation document'})
            continue
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in link or link.startswith('#'):
                continue
            if not (path.parent / link.split('#')[0]).exists():
                errors.append({'file': name, 'error': f'broken local link: {link}'})
    suites = []
    if args.cpu_tests:
        groups = [
            ('baseline', '.', ['test_aggregate_graph_baseline.py', 'test_sglang_aggregate.py', 'test_fant_graph_baseline_script.py']),
            ('stability-protocol', '.', ['test_sglang_stability_protocol.py', 'run_sglang_post.sh']),
            ('wan-mixed-precision', '.', ['tools/test_fantasy_wanvae_mixed.py']),
            ('qwen-text-matrix', 'projects/gr308-three-card-mtp1-graph32', ['tests']),
            ('qwen-image', 'projects/qwen-image21-omni-fant', ['tests']),
            ('h3-pure-contracts', 'projects/h3-sglang-gr308/h3-sglang-fant', ['tests/test_registry_builder.py', 'tests/test_w4a8_gate.py']),
            ('h3-recovered', 'archive/recovered/h3/history', ['test_platform_capability_probe.py']),
        ]
        for label, directory, targets in groups:
            cwd = ROOT / directory
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            env['PYTHONPATH'] = os.pathsep.join([str(cwd), str(cwd/'src'), str(cwd/'scripts')])
            if label == 'stability-protocol':
                cmd = [sys.executable, '-B', *targets]
            else:
                cmd = [sys.executable, '-B', '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider', *targets]
            result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
            output = (result.stdout + result.stderr).replace(str(ROOT), '.')
            suites.append({'suite': label, 'returncode': result.returncode, 'output': output})
            print(f'[{label}] {output.strip()}', flush=True)
            if result.returncode:
                errors.append({'suite': label, 'error': 'CPU test suite failed'})
    summary = {'counts': counts, 'checked_hashes': checked_hashes, 'cpu_suites': suites, 'errors': errors, 'passed': not errors,
               'not_run': ['GPU inference or benchmarks', 'GRServe full C++ build', 'FANT runtime platform tests', 'native HIP compilation']}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'cpu_suites'}, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
