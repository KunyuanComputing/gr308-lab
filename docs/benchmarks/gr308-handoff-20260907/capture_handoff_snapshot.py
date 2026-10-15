#!/usr/bin/env python3
"""Read-only GR308 handoff inventory. No inference, runtime loading or service writes."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SUITE = Path('/data/infra/benchmarks/gr308-ftuca-20260906/production-suite')
sys.path.insert(0, str(SUITE))
import qwen_stream_only as f
import operator_worker_v2 as w


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refuse to overwrite evidence')
    a = f.load_core()
    baseline = a.strict_json((SUITE/a.BASELINE).read_text())
    stack = a.strict_json((SUITE/w.STACK_NAME).read_text())
    f.require(f.sha(SUITE/a.BASELINE) == a.BASELINE_SHA, 'Baseline changed')
    f.require(f.sha(SUITE/w.STACK_NAME) == w.STACK_SHA, 'Stack evidence changed')
    commands = []
    def read_command(argv):
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        row = dict(argv=argv, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        commands.append(row)
        return row
    with a.LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        w.verify_files(stack)
        before = a.verify_identity(baseline)
        idle = a.wait_idle(timeout=10)
        markers = a.verify_history()
        metrics = a.s.live_metrics()
        read_command(['fantgpu-smi'])
        read_command(['systemctl', 'show', 'monitor_server.service',
                      '--property=ActiveState,SubState,MainPID,NRestarts,UnitFileState'])
        # Names/status only: do not collect container environments or sensitive service logs.
        read_command(['docker', 'ps', '-a', '--format', '{{.Names}}\t{{.Status}}'])
        owned_names = {'peer_copy_probe', 'peer_copy_guard.py', 'route_ab_followup.py',
                       'route_ab_controller.py', 'soak_controller.py', 'attention_worker.py'}
        matching = []
        for directory in Path('/proc').iterdir():
            if not directory.name.isdigit():
                continue
            try:
                argv = (directory/'cmdline').read_bytes().split(b'\0')
                names = [Path(x.decode('utf-8', errors='replace')).name for x in argv if x]
                selected = sorted(set(names) & owned_names)
                if selected:
                    matching.append(dict(pid=int(directory.name), matched_executables=selected))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        download_records = {}
        for stage in ('gr308-h3-20260905', 'gr308-wan22-20260905', 'gr308-dsv4-20260905',
                      'gr308-qwen35-4b-20260905'):
            path = SUITE.parent.parent/stage/'download/download-final.json'
            entry = dict(path=str(path), exists=path.is_file(), current_weights_rehashed=False)
            if path.is_file():
                entry.update(sha256=f.sha(path), record=json.loads(path.read_text()))
            download_records[stage] = entry
        after = a.verify_identity(baseline)
        row = dict(schema_version=1, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            read_only=True, inference_requests_sent=0, native_gpu_test_calls=0,
            account_usage_queried=False, production_modified=False,
            source_sha256=f.sha(__file__), baseline_sha256=a.BASELINE_SHA,
            stack_sha256=w.STACK_SHA, identity_before=before, identity_after=after,
            idle_samples=idle, metrics=metrics, preserved_historical_markers=markers,
            verified_library_files=stack['files'], commands=commands,
            matching_owned_test_processes=matching,
            peer_copy_implementation_exists=(SUITE/'peer-copy-implementation.json').exists(),
            peer_copy_run_files=sorted(str(p.relative_to(SUITE)) for p in (SUITE/'peer-copy-wall').rglob('*') if p.is_file()),
            prior_route_controller_pids_exist={str(p):Path('/proc', str(p)).exists() for p in (2784064,2824865)},
            disk=dict(zip(('total','used','free'),shutil.disk_usage('/data'))),
            download_final_records=download_records)
        a.s.write_new(args.output, row)
        print(json.dumps(dict(output=str(args.output), sha256=f.sha(args.output),
            timestamp_utc=row['timestamp_utc'], matching_owned_test_processes=matching,
            peer_copy_implementation_exists=row['peer_copy_implementation_exists'],
            health=row['identity_after']['health'], disk=row['disk']), ensure_ascii=False))


if __name__ == '__main__':
    main()
