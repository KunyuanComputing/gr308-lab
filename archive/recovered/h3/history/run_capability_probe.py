#!/usr/bin/env python3
"""Guarded single-card diagnostics; no production restart or full H3 pipeline."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from run_device_preflight import ROOT, IMAGE, http


def metrics_idle(raw):
    for metric in ('num_requests_running', 'num_requests_waiting'):
        values = [float(line.split()[-1]) for line in raw.splitlines()
                  if line.startswith('vllm:' + metric + '{')]
        if not values or not all(value == 0 for value in values):
            return False
    return True


def require_idle():
    for port in (8000, 8001):
        if not metrics_idle(http(port, '/metrics')):
            raise RuntimeError('Live traffic or missing metrics; defer GPU checks')


def validated_run_artifact(root, raw_path, required_name):
    """Only reuse an in-scope, frozen and integrity-checked probe artifact."""
    path=Path(raw_path).resolve()
    path.relative_to(Path(root).resolve())
    if path.name!=required_name:
        raise ValueError('Unexpected probe artifact name')
    manifest=json.loads((path.parent/'manifest.json').read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get(path.name)!=digest:
        raise ValueError('Probe artifact manifest hash mismatch')
    return path,digest


def main():
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--experimental-platform', action='store_true')
    mode_group.add_argument('--full-projections', action='store_true')
    mode_group.add_argument('--loader-offload', action='store_true')
    mode_group.add_argument('--full-dit', action='store_true')
    mode_group.add_argument('--dit-boundary', action='store_true')
    mode_group.add_argument('--dit-teacher', action='store_true')
    mode_group.add_argument('--full-encoder', action='store_true')
    mode_group.add_argument('--audio-vae', action='store_true')
    parser.add_argument('--dit-reference')
    parser.add_argument('--dit-prompt')
    parser.add_argument('--no-block-diagnostics',action='store_true')
    parser.add_argument('--fp32-sdpa',action='store_true')
    args = parser.parse_args()
    dit_component = args.full_dit or args.dit_boundary or args.dit_teacher
    assert not args.dit_teacher or args.dit_reference
    reference=None
    if args.dit_reference:
        assert args.full_dit or args.dit_teacher
        reference,reference_sha=validated_run_artifact(ROOT,args.dit_reference,'cpu-reference.safetensors')
    prompt=None
    if args.dit_prompt:
        assert args.full_dit
        prompt,prompt_sha=validated_run_artifact(ROOT,args.dit_prompt,'prompt-embeddings.safetensors')
    full_component = dit_component or args.full_encoder or args.audio_vae
    storage_compat = args.loader_offload or full_component
    lock = open('/data/infra/benchmarks/gr308-iteration-v2-20260905/maintenance.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(3):
        require_idle()
        time.sleep(1)
    for port in (8000, 8001, 8080, 8180, 8181):
        http(port, '/health')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    kind = ('audio-vae' if args.audio_vae else
            'full-encoder' if args.full_encoder else
            'dit-teacher' if args.dit_teacher else
            'dit-prompt' if args.dit_prompt else
            'dit-boundary' if args.dit_boundary else
            'full-dit' if args.full_dit else
            'loader-offload' if args.loader_offload else
            'full-projections' if args.full_projections else
            'adapter' if args.experimental_platform else 'capability')
    name = 'gr308-h3-' + kind + '-' + stamp.lower()
    out = ROOT / (kind + '-run-' + stamp)
    out.mkdir()
    assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
    argv = ['docker', 'run', '-d', '--name', name, '--restart', 'no',
            '--network', 'none', '--ipc', 'host', '--privileged',
            '--security-opt', 'label=disable', '--cpuset-cpus', '56,57',
            '--memory', '48g' if full_component else '8g' if args.full_projections or args.loader_offload else '4g',
            '--log-opt', 'max-size=10m', '--log-opt', 'max-file=2',
            '-e', 'CUDA_VISIBLE_DEVICES=2', '-e', 'OMP_NUM_THREADS=1',
            '-e', 'PYTHONPATH=/src/python:/kitchen:/deps',
            '-v', '/usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro',
            '-v', '/usr/local/fant/include:/usr/local/fant/include:ro',
            '-v', '/usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro',
            '-v', str(ROOT/'sglang-da76fa073f8e7df4bf30d7049dd4609d9ca3f23c')+':/src:ro',
            '-v', str(ROOT/'comfy-kitchen-b678fdf63378409676aa5596721445d33794d0ea')+':/kitchen:ro',
            '-v', str(ROOT/'runtime-deps')+':/deps:ro',
            '-v', str(ROOT)+':/work:ro', IMAGE, 'sleep', '1080' if full_component else '600']
    if args.experimental_platform or args.full_projections or storage_compat:
        argv[-3:-3] = ['-e', 'SGLANG_DIFFUSION_PLATFORM_OVERRIDE=fant',
            '-v', str(ROOT/'fantasy_platform_registry.py')+
            ':/src/python/sglang/multimodal_gen/runtime/platforms/__init__.py:ro']
    if args.full_projections or storage_compat:
        checkpoint = Path('/data/models/MiniMax-H3-INT4-W4A8/diffusion_models/'
                          'minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors')
        assert checkpoint.is_file() and checkpoint.stat().st_size == 12540857840
        argv[-3:-3] = ['-v', '/data/models/MiniMax-H3-INT4-W4A8:/model:ro']
    if storage_compat:
        argv[-3:-3] = ['-e', 'H3_FANT_W4A8_COMPAT=1',
            '-v', str(ROOT/'fantasy_layerwise_offload.py')+
            ':/src/python/sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload.py:ro']
    if full_component:
        argv[-3:-3] = ['-v', str(out)+':/out']
    (out/'launch.json').write_text(json.dumps(argv, indent=2)+'\n')
    for file in ('platform_capability_probe.py', 'run_capability_probe.py',
                 'fantasy_platform.py', 'fantasy_platform_smoke.py', 'fantasy_platform_registry.py'):
        (out/file).write_bytes((ROOT/file).read_bytes())
    if args.full_projections:
        (out/'operator_probe.py').write_bytes((ROOT/'operator_probe.py').read_bytes())
    if storage_compat:
        for file in ('fantasy_quant_compat.py', 'fantasy_loader_offload_probe.py',
                     'fantasy_parameter_bridge.py', 'build_fantasy_offload_overlay.py',
                     'fantasy_layerwise_offload.py'):
            (out/file).write_bytes((ROOT/file).read_bytes())
    if dit_component:
        for file in ('fantasy_full_dit_probe.py','fantasy_dit_diagnostics.py','fantasy_sdpa_precision.py'):
            (out/file).write_bytes((ROOT/file).read_bytes())
    if args.full_encoder:
        (out/'fantasy_full_encoder_probe.py').write_bytes((ROOT/'fantasy_full_encoder_probe.py').read_bytes())
        (out/'h3-text-encoder-config.json').write_bytes((ROOT/'base-config/FL2VA/text_encoder/config.json').read_bytes())
    if args.audio_vae:
        (out/'fantasy_audio_vae_probe.py').write_bytes((ROOT/'fantasy_audio_vae_probe.py').read_bytes())
        (out/'h3-audio-vae-config.json').write_bytes((ROOT/'base-config/FL2VA/audio_vae/config.json').read_bytes())
    if args.experimental_platform or dit_component:
        (out/'h3-transformer-config.json').write_bytes(
            (ROOT/'base-config/FL2VA/transformer/config.json').read_bytes())
    result = dict(started=time.time(), model_e2e=False, production_changed=False,
                  owned_container=name, run_dir=str(out))
    created = False
    active_exec = None
    try:
        subprocess.run(argv, capture_output=True, check=True, timeout=60)
        created = True
        modes = (('audio-vae',) if args.audio_vae else
                 ('encoder',) if args.full_encoder else
                 ('dit',) if dit_component else
                 ('loader',) if args.loader_offload else
                 ('w4a8',) if args.full_projections else
                 ('adapter',) if args.experimental_platform else ('api', 'ops'))
        for mode in modes:
            require_idle()
            with (out/(mode+'.log')).open('wb') as log:
                script_args = (['/work/fantasy_platform_smoke.py'] if mode == 'adapter' else
                    ['/work/platform_capability_probe.py', '--mode', mode, '--device', 'fant:0'])
                if mode == 'w4a8':
                    script_args = ['/work/operator_probe.py', '--device', 'fant:0',
                        '--full-projections', '--compat-only', '--checkpoint',
                        '/model/diffusion_models/minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors']
                if mode == 'loader':
                    script_args = ['/work/fantasy_loader_offload_probe.py']
                if mode == 'dit':
                    script_args = ['/work/fantasy_full_dit_probe.py']
                    if args.dit_boundary:
                        script_args.append('--boundary')
                    if args.dit_teacher:
                        script_args.append('--teacher-force')
                    if reference is not None:
                        script_args.extend(['--reference-file','/work/'+str(reference.relative_to(ROOT)),
                            '--reference-sha256',reference_sha])
                    if prompt is not None:
                        script_args.extend(['--prompt-file','/work/'+str(prompt.relative_to(ROOT)),
                            '--prompt-sha256',prompt_sha])
                    if args.no_block_diagnostics:
                        script_args.append('--no-block-diagnostics')
                    if args.fp32_sdpa:
                        script_args.append('--fp32-sdpa')
                if mode == 'encoder':
                    script_args = ['/work/fantasy_full_encoder_probe.py']
                if mode == 'audio-vae':
                    script_args = ['/work/fantasy_audio_vae_probe.py']
                active_exec = subprocess.Popen(['docker', 'exec', name, 'python', *script_args],
                    stdout=log, stderr=subprocess.STDOUT)
                timeout_s = 300 if args.dit_boundary or args.audio_vae else 900 if full_component else 300 if args.loader_offload else 180
                deadline = time.monotonic() + timeout_s
                while active_exec.poll() is None:
                    time.sleep(1)
                    require_idle()
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'Bounded device test exceeded {timeout_s} seconds')
                result[mode+'_exit'] = active_exec.returncode
                active_exec = None
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        if created:
            for command in (['docker','stop','--time','10',name], ['docker','rm',name]):
                try:
                    cleanup = subprocess.run(command, capture_output=True, timeout=30)
                    if cleanup.returncode:
                        result.setdefault('cleanup_errors', []).append(cleanup.stderr.decode())
                except Exception as exc:
                    result.setdefault('cleanup_errors', []).append(repr(exc))
        if active_exec is not None:
            try:
                active_exec.wait(timeout=10)
            except subprocess.TimeoutExpired:
                active_exec.kill()
                active_exec.wait(timeout=10)
        result['health_after'] = {}
        for port in (8000, 8001, 8080, 8180, 8181):
            try:
                http(port, '/health')
                result['health_after'][str(port)] = True
            except Exception as exc:
                result['health_after'][str(port)] = repr(exc)
        result['finished'] = time.time()
        (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in out.iterdir() if p.is_file()}
        (out/'manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    probes_passed = (result.get('audio-vae_exit') == 0 if args.audio_vae else
                     result.get('encoder_exit') == 0 if args.full_encoder else
                     result.get('dit_exit') == 0 if dit_component else
                     result.get('loader_exit') == 0 if args.loader_offload else
                     result.get('w4a8_exit') == 0 if args.full_projections else
                     result.get('adapter_exit') == 0 if args.experimental_platform else
                     result.get('api_exit') == 0 and result.get('ops_exit') == 0)
    return 0 if (probes_passed
                 and not result.get('cleanup_errors')
                 and all(v is True for v in result['health_after'].values())) else 2


if __name__ == '__main__':
    raise SystemExit(main())
