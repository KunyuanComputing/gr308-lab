"""CPU-only admission for the exact five-model goal; never launches inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from inventory import inspect_model

MODELS = (
    'Qwen3.6-35B-A3B', 'Qwen3.6-27B', 'Qwen3.5-9B',
    'Qwen3-30B-A3B', 'Qwen3-Next-80B-A3B-Instruct',
)
CONCURRENCIES = (1, 16, 64)
GIB = 1024 ** 3


def capabilities(config, weight_map):
    text = config.get('text_config', config)
    model_type = config['model_type']
    mtp_keys = sorted(k for k in weight_map if re.search(r'(^|\.)(mtp|nextn)(\.|$)', k))
    declared = text.get('mtp_num_hidden_layers', text.get('num_nextn_predict_layers'))
    recognized = model_type in ('qwen3_5', 'qwen3_5_moe', 'qwen3_next')
    # Next's official config omits layer count; its actual tensor index is decisive.
    native = recognized and bool(mtp_keys)
    return dict(model_type=model_type, declared_mtp_layers=declared,
                mtp_tensor_count=len(mtp_keys), mtp_key_examples=mtp_keys[:5],
                native_mtp_weight_evidence=native,
                native_mtp_runtime_status='unverified_for_this_matrix' if native else 'not_applicable_no_native_weights',
                method='mtp' if native else None,
                hybrid=model_type in ('qwen3_5', 'qwen3_5_moe', 'qwen3_next'))


def health_gate(raw, memories, running_containers):
    rows = re.findall(r'Device (\d+):\d+ Status:\[([^]]+)\]', raw)
    return (len(rows) == 8 and {int(i) for i, _ in rows} == set(range(8))
            and all(status == 'OK' for _, status in rows)
            and memories == [540] * 8 and not running_containers)


def workload(concurrency):
    if concurrency not in CONCURRENCIES:
        raise ValueError('Concurrency outside the requested matrix')
    return dict(concurrency=concurrency, requests=64, input_tokens=3072,
                output_tokens=1024, seed=2026, temperature=0, ignore_eos=True,
                server_min_max_num_seqs=concurrency,
                required_completed=64, required_total_output_tokens=65536,
                dataset='random', counts_as_result=False)


def inspect_goal_model(root):
    row = inspect_model(root)
    if 'error' in row:
        return row
    config = json.loads((root / 'config.json').read_text())
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    row['capabilities'] = capabilities(config, index['weight_map'])
    weight_bytes = index.get('metadata', {}).get('total_size')
    row['indexed_tensor_bytes'] = weight_bytes
    if weight_bytes:
        # This is an optimistic lower bound, NOT a safe placement recommendation.
        row['physical_cards_weights_only_lower_bound'] = (weight_bytes + 112 * GIB - 1) // (112 * GIB)
        row['placement_warning'] = 'Excludes KV, state, activations, MTP duplication, runtime and sharding overhead; runtime admission required.'
    row['target_workloads'] = [workload(c) for c in CONCURRENCIES]
    row['variants_required'] = ['baseline', 'native_mtp', 'combined_tuned']
    if not row['capabilities']['native_mtp_weight_evidence']:
        row['variant_decision_required'] = 'User decision on non-MTP alternative; do not silently substitute or mark goal complete.'
    row['performance_status'] = 'not_run_in_this_matrix'
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-root', type=Path, default=Path('/data/models'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    health = subprocess.run(['sudo', '-n', 'fantgpu-smi', '--query-core-health'],
                            capture_output=True, text=True, timeout=20)
    smi = subprocess.run(['sudo', '-n', 'fantgpu-smi'], capture_output=True, text=True, timeout=20)
    running = subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True, timeout=20)
    memories = [int(x) for x in re.findall(r'(\d+)MiB / 57344MiB', smi.stdout)]
    result = dict(timestamp=time.time(), kind='CPU_only_no_inference',
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  models=[inspect_goal_model(args.models_root / name) for name in MODELS],
                  health=dict(raw=health.stdout, memory_mib=memories,
                              running_containers=running.stdout.split(),
                              command_exit_codes=[health.returncode, smi.returncode, running.returncode]),
                  gpu_admission_passed=(all(p.returncode == 0 for p in (health, smi, running))
                                        and health_gate(health.stdout, memories, running.stdout.split())),
                  inference_launched=False, goal_complete=False)
    # Exclusive output creation prevents overwriting evidence from a previous pass.
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('timestamp', 'gpu_admission_passed', 'inference_launched')}))
    for row in result['models']:
        print(json.dumps({k: row.get(k) for k in ('name', 'valid_shards', 'expected_shards', 'capabilities', 'physical_cards_weights_only_lower_bound')}))


if __name__ == '__main__':
    main()
