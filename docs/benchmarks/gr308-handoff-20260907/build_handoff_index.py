#!/usr/bin/env python3
"""Print a bounded local evidence index and validate the handoff; never run workloads."""
import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'benchmarks/GR308_TEST_HANDOFF_20260907_ZH.md'
SUITE = ROOT/'benchmarks/gr308-ftuca-20260906/production-suite'
SUPPORT = Path(__file__).resolve().parent
LOCAL_ONLY = {'peer_copy_evidence.py','peer_copy_guard.py','audit_peer_copy.py','test_peer_copy.py'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    content = DOC.read_text()
    targets = sorted(set(re.findall(r'\]\((/[^)]+)\)', content)))
    pending_generated = {str(SUPPORT/'EVIDENCE_INDEX.json')}
    missing = [p for p in targets if not Path(p).exists() and p not in pending_generated]
    if missing:
        raise RuntimeError('Missing handoff links: '+repr(missing))
    rows = re.findall(r'^\|([A-Z0-9]+-\d+)\|([^\n]+)\|$', content, re.M)
    if len(rows) != 31 or len({name for name,_ in rows}) != 31:
        raise RuntimeError('The 31-item handoff table is incomplete or duplicated')
    statuses = [rest.split('|')[1] for _,rest in rows]
    counts = {name:statuses.count(name) for name in ('完成','部分/问题','待完成')}
    if counts != {'完成':1,'部分/问题':22,'待完成':8}:
        raise RuntimeError('Task count mismatch: '+repr(counts))
    chosen = {Path(p) for p in targets if Path(p).is_file() and p not in pending_generated}
    chosen.add(DOC)
    # Top-level suite sources and audits. Recursive large tensor/log/run payloads are intentionally excluded.
    chosen.update(p for p in SUITE.iterdir() if p.is_file() and not p.is_symlink()
                  and p.suffix in ('.json','.md','.py','.cpp','.h'))
    for dirname in ('peer-copy-build-20260907T022721Z', 'pageable-build-20260906T220752Z', 'pinned-build-20260906T223514Z'):
        chosen.update(p for p in (SUITE/dirname).iterdir() if p.is_file() and not p.is_symlink())
    chosen.update(p for p in SUPPORT.iterdir() if p.is_file()
                  and p.name not in ('EVIDENCE_INDEX.json','HANDOFF_SHA256SUMS','validation.json'))
    for dirname in ('gr308-dsv4-20260905','gr308-h3-20260905','gr308-qwen35-4b-20260905','gr308-wan22-20260905'):
        base = ROOT/'benchmarks'/dirname
        chosen.update(p for p in base.iterdir() if p.is_file() and p.suffix in ('.md','.json','.py'))
    for dirname in ('gr308-iteration-v2-20260905','gr308-optimization-e2e-20260905','gr308-optimization-review-20260905'):
        base = ROOT/'benchmarks'/dirname
        chosen.update(p for p in base.iterdir() if p.is_file() and p.suffix == '.md')
    frozen_checks = []
    for path in sorted(SUITE.glob('*implementation*.json')):
        data = json.loads(path.read_text())
        if not isinstance(data,dict) or not data or not all(isinstance(v,str) and re.fullmatch('[a-f0-9]{64}',v) for v in data.values()):
            continue
        for relative, expected in data.items():
            target = SUITE/relative
            if not target.is_file() or digest(target) != expected:
                raise RuntimeError('Frozen source mismatch: '+str(target))
        frozen_checks.append(dict(file=str(path.relative_to(ROOT)),entries=len(data),passed=True))
    files = []
    for path in sorted(chosen):
        relative = path.relative_to(ROOT)
        files.append(dict(relative_path=str(relative),local_path=str(path),
            mapped_remote_path='/data/infra/'+str(relative),bytes=path.stat().st_size,sha256=digest(path),
            remote_copy_status='not_synced_unfrozen_candidate' if path.parent==SUITE and path.name in LOCAL_ONLY else 'mapping_only_not_individually_rechecked_by_this_index'))
    result = dict(schema_version=1,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Local handoff links, suite top-level sources/audits, selected model/optimization reports and handoff support files. Not a full raw-data or checkpoint archive.',
        read_only=True,new_native_gpu_tests=0,new_inference_requests=0,
        local_root=str(ROOT),remote_root_mapping='/data/infra',
        validation=dict(local_links_checked=len(targets),missing_links=missing,requirements=len(rows),
            requirement_status_counts=counts,http_accounting=74+18+6+246+5+116+30+72,
            frozen_implementation_manifests=frozen_checks),
        file_count=len(files),indexed_bytes=sum(f['bytes'] for f in files),files=files,
        full_transfer_raw_local=False,
        complete_transfer_raw_remote_directories=[
            '/data/infra/benchmarks/gr308-ftuca-20260906/production-suite/'+name+'/runs'
            for name in ('pageable-wall','pinned-wall','pinned-wall-followup')],
        note='Server path mapping is not a claim of a fresh per-file server comparison. Existing stage manifests remain the authority for recursive evidence. This index deliberately excludes itself and large raw payload collections.')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
