"""One isolated Next MTP1 experiment; preserve the completed baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
import traceback

from goal_bench_contract import admit_capacity, bench_command, run_case
from goal_preflight import MODELS, capabilities, health_gate
from inventory import inspect_model
from run_legacy_multi import IMAGE, CLIENT, QUALITY, command, emit, load, quality, save, smi
from mtp_runtime_gate import admit_mtp_counters

ROOT = Path(__file__).resolve().parent


def check_port_available(port):
    # Match server reuse semantics: TIME_WAIT is not an active listener.
    # A real listener still makes bind fail; never kill a process to free a port.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))


def scoped_health_gate(raw, memories, running, quarantine, baseline=None):
    if not quarantine:
        return health_gate(raw, memories, running)
    # Quarantine the complete failed physical board, never just one function.
    if quarantine != [0, 1] or len(memories) != 8 or running:
        return False
    rows = re.findall(r'Device (\d+):\d+ Status:\[([^]]+)\]', raw)
    healthy = len(rows) == 8 and {int(i) for i, _ in rows} == set(range(8)) and all(s == 'OK' for _, s in rows)
    stable = baseline is None or all(memories[i] == baseline[i] for i in quarantine)
    return healthy and stable and all(memories[i] == 540 for i in range(2, 8))


def serve_command(model, tp, kv_gib, batch_tokens, mtp_tokens, block_size, port, max_seqs=64):
    args = ['vllm', 'serve', '/model', '--served-model-name', model,
            '--host', '127.0.0.1', '--port', str(port), '--tensor-parallel-size', str(tp),
            '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.9',
            '--max-model-len', '4096', '--max-num-seqs', str(max_seqs),
            '--max-num-batched-tokens', str(batch_tokens), '--kv-cache-memory-bytes', str(kv_gib*1024**3),
            '--no-enable-prefix-caching', '--no-async-scheduling', '--disable-custom-all-reduce',
            '--no-enable-log-requests', '--enforce-eager']
    if model.startswith(('Qwen3.5-', 'Qwen3.6-')):
        args.append('--language-model-only')
    if block_size:
        args += ['--block-size', str(block_size)]
    if mtp_tokens:
        args += ['--speculative-config', json.dumps(dict(method='mtp', num_speculative_tokens=mtp_tokens))]
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--profile', choices=['baseline', 'mtp3', 'tuned'], required=True)
    parser.add_argument('--devices', nargs='+', type=int, required=True)
    parser.add_argument('--kv-gib', type=int, required=True)
    parser.add_argument('--batch-tokens', type=int, default=2048)
    parser.add_argument('--mtp-tokens', type=int, default=3)
    parser.add_argument('--block-size', type=int)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--port', type=int, default=18182)
    parser.add_argument('--concurrencies', nargs='+', type=int, choices=[1, 16, 64], default=[1, 16])
    parser.add_argument('--quarantine-devices', nargs='+', type=int, default=[])
    args = parser.parse_args()
    assert (args.model == 'Qwen3-Next-80B-A3B-Instruct' and args.profile == 'tuned'
            and args.mtp_tokens == 1 and args.devices == [2, 3, 4, 5]
            and args.quarantine_devices == [0, 1] and args.kv_gib == 8
            and args.batch_tokens == 2048 and args.block_size == 1088
            and args.concurrencies == [1, 16] and args.attempt == 1 and args.port == 20911)
    assert len(args.devices) in (1, 2, 4, 8) and len(set(args.devices)) == len(args.devices)
    assert all(0 <= d < 8 for d in args.devices)
    assert args.concurrencies in ([1, 16], [1, 16, 64])
    assert args.quarantine_devices in ([], [0, 1])
    assert not set(args.devices) & set(args.quarantine_devices)
    assert 1 <= args.kv_gib <= 40 and args.batch_tokens >= 256
    mtp = 0 if args.profile == 'baseline' else args.mtp_tokens
    assert args.profile != 'mtp3' or mtp == 3
    assert 0 <= mtp <= 4
    slug = re.sub('[^a-z0-9]+', '-', args.model.lower()).strip('-')
    label = f'goal-{slug}-{args.profile}-eager-r{args.attempt}'
    name = f'gr308-{label}'
    out = ROOT / label
    out.mkdir(exist_ok=False)
    result = dict(status='failed', execution_mode='eager_no_graph', model=args.model, profile=args.profile, started=time.time(),
                  logical_functions=args.devices, physical_cards=sorted({d//2 for d in args.devices}),
                  container=name, cases=[])
    result.update(requested_concurrencies=args.concurrencies, quarantined_devices=args.quarantine_devices)
    cid = None
    stop = threading.Event()
    failures = []
    monitor = None
    follower = None
    start = time.monotonic()
    current_case = None
    try:
        initial = smi()
        health = command(['sudo', '-n', 'fantgpu-smi', '--query-core-health']).stdout
        assert scoped_health_gate(health, initial['memories_mib'], command(['docker', 'ps', '-q']).stdout.split(), args.quarantine_devices)
        save(out, 'hardware-before.json', dict(**initial, health=health))
        assert command(['docker', 'inspect', name], check=False).returncode != 0
        check_port_available(args.port)
        model_root = Path('/data/models') / args.model
        manifest = inspect_model(model_root)
        assert manifest['weights_header_qualified'], 'Weights not fully admitted'
        config = json.loads((model_root / 'config.json').read_text())
        weight_map = json.loads((model_root / 'model.safetensors.index.json').read_text())['weight_map']
        cap = capabilities(config, weight_map)
        assert not mtp or cap['native_mtp_weight_evidence'], 'No native MTP weights; do not substitute'
        save(out, 'model-manifest.json', manifest)
        save(out, 'model-capabilities.json', cap)
        sources = [(Path(__file__), 'runner.py'), (CLIENT, 'client-frozen.py'),
                   (QUALITY, 'quality-frozen.py'), (ROOT/'goal_bench_contract.py', 'bench-frozen.py'),
                   (ROOT/'goal_preflight.py', 'preflight-frozen.py'), (ROOT/'inventory.py', 'inventory-frozen.py'),
                   (ROOT/'run_legacy_multi.py', 'helper-frozen.py'),
                   (ROOT/'trace_native_bench.py', 'trace-native-bench-frozen.py'),
                   (ROOT/'mtp_runtime_gate.py', 'mtp-runtime-gate-frozen.py'),
                   (Path('/etc/fant/fantHipSettings.cfg'), 'fantHipSettings.cfg')]
        for source, destination in sources:
            shutil.copyfile(source, out/destination)
        serve = serve_command(args.model, len(args.devices), args.kv_gib, args.batch_tokens, mtp, args.block_size, args.port, max(args.concurrencies))
        launch = ['docker', 'run', '-d', '--name', name, '--restart', 'no', '--network', 'host',
                  '--ipc', 'private', '--shm-size', '8g', '--privileged', '--security-opt', 'label=disable',
                  '--memory', '256g', '--cpuset-cpus', '56-63', '--log-opt', 'max-size=40m', '--log-opt', 'max-file=2',
                  '-e', 'CUDA_VISIBLE_DEVICES='+','.join(map(str, args.devices)), '-e', 'OMP_NUM_THREADS=2',
                  '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
                  '-e', 'XDG_CACHE_HOME=/cache/xdg', '-e', 'TORCH_EXTENSIONS_DIR=/cache/torch_extensions']
        for path in ('/usr/lib/x86_64-linux-gnu/fantgpu-fh3', '/usr/local/fant/include', '/usr/local/fant/lib/llvm'):
            launch += ['-v', path+':'+path+':ro']
        launch += ['-v', str(model_root)+':/model:ro',
                   '-v', str(out/'trace-native-bench-frozen.py')+':/opt/trace_native_bench.py:ro',
                   '-v', str(out/'fantHipSettings.cfg')+':/etc/fant/fantHipSettings.cfg:ro',
                   '-v', '/data/infra/benchmarks/.grserve-qwen36-work/runtime-cache:/cache',
                   IMAGE, 'timeout', '--signal=TERM', '--kill-after=20', '34000', *serve]
        save(out, 'launch.json', dict(docker_args=launch, vllm_args=serve,
             logical_functions=args.devices, physical_cards=result['physical_cards'],
             profile=args.profile, execution_mode='eager_no_graph', mtp_tokens=mtp, requested_concurrencies=args.concurrencies,
             quarantined_devices=args.quarantine_devices,
             source_hashes={dst:hashlib.sha256((out/dst).read_bytes()).hexdigest() for _,dst in sources}))
        cid = command(launch).stdout.strip()
        result['container_id'] = cid
        follower = subprocess.Popen(['docker', 'logs', '--follow', '--timestamps', cid],
                                    stdout=(out/'server.log').open('w'), stderr=subprocess.STDOUT)
        emit(stage='server_started', model=args.model, profile=args.profile, container=name,
             logical_devices=args.devices, kv_gib_per_worker=args.kv_gib, cases=args.concurrencies,
             quarantined_devices=args.quarantine_devices)

        def watch():
            while not stop.is_set():
                try:
                    row = smi()
                    with (out/'telemetry.jsonl').open('a') as handle:
                        handle.write(json.dumps(row)+'\n')
                    assert max(row['temperatures_c']) < 80, 'Temperature limit'
                    assert all(v == initial['memories_mib'][i] for i,v in enumerate(row['memories_mib']) if i not in args.devices), 'Non-target memory changed'
                    assert time.monotonic()-start < 33300, 'Overall run deadline'
                    assert set(command(['docker','ps','-q','--no-trunc']).stdout.split()) <= {cid}, 'Another container is running'
                    raw = command(['sudo','-n','fantgpu-smi','--query-core-health']).stdout
                    assert len(re.findall(r'Device \d+:\d+ Status:\[OK\]', raw)) == 8, 'Device health failed'
                    logs = (out/'server.log').read_text(errors='replace')
                    assert not any(e in logs for e in ('HIP error: out of memory', 'Guilty of lockup', 'Compute Context has been reset', 'Segfault encountered')), 'Backend fatal error'
                    completed = None
                    if current_case and (out/f'c{current_case}'/'benchmark.log').exists():
                        progress = re.findall(r'(\d+)/64', (out/f'c{current_case}'/'benchmark.log').read_text(errors='replace'))
                        completed = int(progress[-1]) if progress else None
                    emit(stage='heartbeat', case=current_case, native_progress=completed,
                         target_memory_mib=[row['memories_mib'][i] for i in args.devices])
                except Exception as exc:
                    failures.append(repr(exc))
                    emit(stage='safety_stop', error=repr(exc))
                    command(['docker','stop','--timeout','20',cid], check=False)
                    return
                stop.wait(15)

        monitor = threading.Thread(target=watch, daemon=True)
        monitor.start()
        client = load(out/'client-frozen.py', 'goal_client')
        client.ROOT = out
        client.MODEL = args.model
        client.PORT = args.port
        helper = load(out/'quality-frozen.py', 'goal_quality')

        def safety_check():
            assert not failures, failures
            assert command(['docker','inspect','--format','{{.State.Running}}',cid]).stdout.strip() == 'true', 'Server exited'

        boot = time.monotonic()
        while time.monotonic()-boot < 900:
            safety_check()
            try:
                models = client.api('/v1/models', timeout=2)
            except (OSError, client.http.client.HTTPException):
                time.sleep(3)
                continue
            assert models['data'][0]['id'] == args.model, 'Wrong server model identity'
            save(out, 'models.json', models)
            break
        else:
            raise TimeoutError('Model startup deadline')
        result['startup_s'] = time.monotonic()-boot
        capacity = admit_capacity(serve, (out/'server.log').read_text(), max(args.concurrencies))
        save(out, 'capacity-admission.json', capacity)
        emit(stage='ready', model=args.model, **capacity)
        metrics_before_quality = client.api('/metrics')
        (out/'quality-metrics-before.prom').write_text(metrics_before_quality)
        correctness = quality(client, helper, out, dict(vision=False, reasoning=False))
        result['correctness'] = correctness
        assert all(r['passed'] for r in correctness), 'Correctness gate failed'
        metrics_after_quality = client.api('/metrics')
        (out/'quality-metrics-after.prom').write_text(metrics_after_quality)
        mtp_gate = admit_mtp_counters(metrics_before_quality, metrics_after_quality)
        save(out, 'mtp-runtime-admission.json', mtp_gate)
        emit(stage='mtp_runtime_admitted', **mtp_gate)
        helptext = command(['docker','exec',cid,'vllm','bench','serve','--help=all'], timeout=60).stdout
        (out/'bench-help.txt').write_text(helptext)
        assert all(flag in helptext for flag in ('--random-range-ratio','--tokenizer','--save-detailed','--seed'))
        for c in args.concurrencies:
            current_case = c
            (out/f'c{c}-metrics-before.prom').write_text(client.api('/metrics'))
            emit(stage='benchmark_started', model=args.model, concurrency=c, requests=64, input_tokens=3072, output_tokens=1024)
            case = run_case(container_id=cid, model=args.model, concurrency=c, port=args.port,
                            output_dir=out/f'c{c}', serve_args=serve,
                            server_log=(out/'server.log').read_text(), safety_check=safety_check)
            (out/f'c{c}-metrics-after.prom').write_text(client.api('/metrics'))
            result['cases'].append(case)
            save(out, 'progress.json', result)
            emit(stage='case_complete', model=args.model, **case)
        result['status'] = 'native_cases_passed_pending_independent_audit'
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
        emit(stage='failed', error=repr(exc))
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=45)
        if cid:
            command(['docker','stop','--timeout','20',cid], check=False)
            save(out, 'container-after-stop.json', json.loads(command(['docker','inspect',cid]).stdout))
        if follower:
            try:
                follower.wait(timeout=10)
            except subprocess.TimeoutExpired:
                follower.terminate()
        deadline = time.monotonic()+60
        while True:
            after = smi()
            if (all(after['memories_mib'][i] == 540 for i in range(8) if i not in args.quarantine_devices)
                    or time.monotonic() >= deadline):
                break
            time.sleep(2)
        health = command(['sudo','-n','fantgpu-smi','--query-core-health']).stdout
        save(out, 'hardware-after.json', dict(**after, health=health))
        scoped_ok = scoped_health_gate(health, after['memories_mib'],
                       command(['docker', 'ps', '-q']).stdout.split(), args.quarantine_devices,
                       initial['memories_mib'])
        result.update(memory_released=after['memories_mib']==[540]*8, health_ok=health_gate(health,after['memories_mib'],[]),
                      scoped_cleanup_ok=scoped_ok,
                      finished=time.time(), safety_failures=failures)
        save(out, 'result.json', result)
        emit(stage='complete', status=result['status'], memory_released=result['memory_released'], health_ok=result['health_ok'])


if __name__ == '__main__':
    main()
