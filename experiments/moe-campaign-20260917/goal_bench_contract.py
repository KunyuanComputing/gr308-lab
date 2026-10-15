"""Full native benchmark cases, intended for a health-gated goal server runner.

This module does NOT launch or stop a server and is not a performance result.
The caller must validate model identity/weights, semantic quality, health,
ownership and cleanup; it supplies a live safety callback throughout each run.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from goal_preflight import CONCURRENCIES, MODELS

REQUESTS = 64
INPUT_TOKENS = 3072
OUTPUT_TOKENS = 1024
SEED = 2026


def require(condition, message):
    if not condition:
        raise ValueError(message)


def option(args, key):
    require(args.count(key) == 1, f'Missing or ambiguous option: {key}')
    idx = args.index(key)
    require(idx + 1 < len(args), f'Missing value: {key}')
    return args[idx + 1]


def admit_capacity(serve_args, server_log, concurrency):
    require(concurrency in CONCURRENCIES, 'Concurrency outside the goal')
    maximum = int(option(serve_args, '--max-num-seqs'))
    context = int(option(serve_args, '--max-model-len'))
    require(maximum >= concurrency, 'Scheduler cannot hold the requested concurrency')
    require(context >= INPUT_TOKENS + OUTPUT_TOKENS, 'Context shorter than the workload')
    matches = re.findall(r'Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x', server_log)
    relevant = [float(capacity) for length, capacity in matches if int(length.replace(',', '')) == context]
    require(relevant, 'No matching runtime KV concurrency estimate in startup log')
    # Take the minimum across workers, not the most favorable reported capacity.
    require(min(relevant) >= concurrency, 'KV capacity would queue/preempt full-context requests')
    return dict(max_num_seqs=maximum, max_model_len=context,
                runtime_reported_capacity=min(relevant),
                capacity_basis='startup estimate at server maximum context; not a measurement of actual simultaneous requests')


def deadline_s(concurrency):
    require(concurrency in CONCURRENCIES, 'Concurrency outside the goal')
    # 64 full outputs at C1 can take over an hour. Do not reuse the old 780s C16 limit.
    return {1: 14400, 16: 7200, 64: 7200}[concurrency]


def bench_command(model, concurrency, port, result_path, tokenizer_path='/model'):
    require(model in MODELS, 'Model outside the exact goal')
    require(concurrency in CONCURRENCIES, 'Concurrency outside the goal')
    require(1024 <= port <= 65535, 'Invalid loopback port')
    path = Path(result_path)
    require(path.is_absolute(), 'Container result path must be absolute')
    require(Path(tokenizer_path).is_absolute(), 'Use the admitted local tokenizer, not a remote name')
    return ['vllm', 'bench', 'serve', '--backend', 'vllm',
            '--base-url', f'http://127.0.0.1:{port}', '--model', model, '--tokenizer', tokenizer_path,
            '--dataset-name', 'random', '--random-input-len', str(INPUT_TOKENS),
            '--random-output-len', str(OUTPUT_TOKENS), '--random-range-ratio', '0',
            '--num-prompts', str(REQUESTS), '--max-concurrency', str(concurrency),
            '--ignore-eos', '--request-rate', 'inf', '--temperature', '0', '--seed', str(SEED),
            '--percentile-metrics', 'ttft,tpot,itl', '--metric-percentiles', '25,50,75,90,99',
            '--save-result', '--save-detailed', '--result-dir', str(path.parent),
            '--result-filename', path.name]


def audit_case(summary, concurrency):
    require(summary.get('num_prompts') == REQUESTS, 'Wrong request count')
    require(summary.get('completed') == REQUESTS and summary.get('failed') == 0, 'Incomplete requests')
    require(summary.get('max_concurrency') == concurrency, 'Wrong concurrency')
    require(summary.get('request_rate') == 'inf', 'Wrong request rate')
    for key in ('input_lens', 'output_lens', 'ttfts', 'itls', 'start_times', 'generated_texts', 'errors'):
        require(isinstance(summary.get(key), list) and len(summary[key]) == REQUESTS, f'Missing detailed records: {key}')
    require(not any(summary['errors']), 'Native client reported errors')
    require(summary['input_lens'] == [INPUT_TOKENS] * REQUESTS, 'Input length mismatch')
    require(summary['output_lens'] == [OUTPUT_TOKENS] * REQUESTS, 'Truncated or mismatched outputs')
    require(summary.get('total_input_tokens') == REQUESTS * INPUT_TOKENS, 'Wrong input total')
    require(summary.get('total_output_tokens') == REQUESTS * OUTPUT_TOKENS, 'Wrong output total')
    duration = summary.get('duration', 0)
    require(isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0, 'Invalid duration')
    throughput = REQUESTS * OUTPUT_TOKENS / duration
    require(math.isclose(throughput, summary.get('output_throughput', -1), rel_tol=1e-8), 'Throughput arithmetic mismatch')
    percentiles = {}
    for metric in ('ttft', 'tpot', 'itl'):
        percentiles[metric] = {}
        for percentile in (25, 50, 75, 90, 99):
            value = summary.get(f'p{percentile}_{metric}_ms')
            require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, 'Missing/invalid percentiles')
            percentiles[metric][str(percentile)] = value
    return dict(native_case_complete=True, concurrency=concurrency, requests=REQUESTS,
                input_tokens_per_request=INPUT_TOKENS, output_tokens_per_request=OUTPUT_TOKENS,
                output_tokens_per_s=throughput, duration_s=duration, percentiles_ms=percentiles,
                full_goal_qualified=False,
                remaining_gates=['server/model identity', 'semantic quality', 'server usage/metrics cross-check',
                                 'MTP and Graph runtime counters', 'hardware health and cleanup',
                                 'matched baseline/tuned comparison'],
                timing_note='ITL is native streamed text-chunk latency; MTP may bundle multiple tokens.')


def run_case(*, container_id, model, concurrency, port, output_dir,
             serve_args, server_log, safety_check):
    """Run one case on an already admitted server; callback must raise on unsafe state.

    Any raised error must cause the caller to stop its owned server and preserve
    evidence. Short output / benchmark failure never returns a successful audit.
    """
    require(re.fullmatch(r'[0-9a-f]{64}', container_id) is not None, 'Use the inspected full container ID')
    require(callable(safety_check), 'A live safety monitor is required')
    capacity = admit_capacity(serve_args, server_log, concurrency)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    path = f'/tmp/goal-bench-c{concurrency}-{time.time_ns()}.json'
    bench = bench_command(model, concurrency, port, path)
    traced_bench = ['python3', '/opt/trace_native_bench.py', path + '.inputs.json', *bench]
    command = ['docker', 'exec', container_id, 'timeout', '--signal=TERM', '--kill-after=20',
               str(deadline_s(concurrency)), *traced_bench]
    manifest = dict(command=command, concurrency_capacity=capacity, timeout_s=deadline_s(concurrency),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    status='started', performance_result=False)
    (out / 'command.json').write_text(json.dumps(manifest, indent=2) + '\n')
    safety_check()
    with (out / 'benchmark.log').open('w') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while process.poll() is None:
                safety_check()
                require(time.monotonic() - started < deadline_s(concurrency) + 60, 'Benchmark supervisor deadline')
                time.sleep(2)
            require(process.returncode == 0, 'Native benchmark process failed')
            safety_check()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    # The remote timeout remains bounded even if docker exec's local observer dies.
    # Caller must stop the owned server on an exception; terminating exec alone
    # does not promise the in-container request process has terminated.
    subprocess.run(['docker', 'cp', f'{container_id}:{path}', str(out / 'benchmark.json')],
                   check=True, timeout=30)
    subprocess.run(['docker', 'cp', f'{container_id}:{path}.inputs.json', str(out / 'native-inputs.json')],
                   check=True, timeout=30)
    trace = json.loads((out / 'native-inputs.json').read_text())
    require(len(trace['requests']) == REQUESTS and trace['seed'] == SEED,
            'Native input trace missing or mismatched')
    summary = json.loads((out / 'benchmark.json').read_text())
    audited = audit_case(summary, concurrency)
    (out / 'case-audit.json').write_text(json.dumps(audited, indent=2) + '\n')
    return audited
