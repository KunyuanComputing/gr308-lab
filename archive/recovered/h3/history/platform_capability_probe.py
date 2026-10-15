#!/usr/bin/env python3
"""Bounded GR308 API/operator diagnostic, not an H3 deployment adapter.

The default API mode allocates no accelerator tensors in this script. Ops mode
must be run in an isolated test container after the host maintenance/idle guard.
No full checkpoint is loaded, platform is not overridden, and no files are edited.
"""
import argparse
import json
import traceback

API_NAMES = (
    'is_available', 'device_count', 'current_device', 'set_device',
    'get_device_properties', 'mem_get_info', 'memory_allocated',
    'Stream', 'stream', 'current_stream', 'Event', 'synchronize',
)


def finalize(report):
    report['platform_is_fant'] = report.get('platform_device_type') == 'fant'
    ops = report.get('operators', [])
    report['small_ops_passed'] = len(ops) == 6 and all(r.get('passed') for r in ops)
    report['model_e2e'] = False
    report['backend_adapter_installed'] = False
    report['native_kernel_acceleration_proven'] = False
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('api', 'ops'), default='api')
    ap.add_argument('--device', default='fant:0')
    args = ap.parse_args()
    if not args.device.startswith('fant:') or not args.device[5:].isdigit():
        ap.error('An explicit fant:N device is required; no CPU/CUDA substitution')
    report = dict(mode=args.mode, requested_device=args.device, operators=[])
    try:
        import torch
        import torch_fant  # noqa: F401 -- registers PrivateUse1
        import torch.nn.functional as F
        from sglang.multimodal_gen.runtime.platforms import current_platform

        report.update(torch=torch.__version__,
                      privateuse_backend=torch._C._get_privateuse1_backend_name(),
                      platform_class=type(current_platform).__name__,
                      platform_device_type=current_platform.device_type)
        module = torch.get_device_module('fant')
        report['api'] = {name: callable(getattr(module, name, None)) for name in API_NAMES}
        if args.mode == 'ops':
            # Never assume a CPU fallback platform provides the correct device API.
            assert report['privateuse_backend'] == 'fant'
            assert all(report['api'][name] for name in
                       ('is_available', 'device_count', 'set_device', 'mem_get_info', 'synchronize'))
            index = int(args.device[5:])
            assert module.is_available() and index < module.device_count()
            module.set_device(index)
            free, total = module.mem_get_info(index)
            report['memory_before'] = dict(free_bytes=free, total_bytes=total)
            assert free >= 512 * 1024**2, 'Less than 512 MiB free; skip even tiny operator checks'
            gen = torch.Generator(device='cpu').manual_seed(20260905)
            for dtype in (torch.float16, torch.bfloat16):
                inputs = {
                    'matmul': [torch.randn(s, generator=gen) * .1 for s in ((16,128),(128,128))],
                    'sdpa': [torch.randn((1,42,16,128), generator=gen) * .1 for _ in range(3)],
                    'conv3d': [torch.randn(s, generator=gen) * .1 for s in ((1,4,3,8,8),(8,4,3,3,3))],
                }
                functions = {
                    'matmul': lambda a,b: a @ b,
                    'sdpa': lambda q,k,v: F.scaled_dot_product_attention(q,k,v,dropout_p=0.0),
                    'conv3d': lambda a,w: F.conv3d(a,w,padding=1),
                }
                for name, values in inputs.items():
                    row = dict(operator=name, dtype=str(dtype), passed=False)
                    device_values = None
                    result = None
                    try:
                        rounded = [v.to(dtype) for v in values]
                        reference = functions[name](*[v.float() for v in rounded])
                        device_values = [v.to(args.device) for v in rounded]
                        result = functions[name](*device_values)
                        module.synchronize()
                        row['output_device'] = str(result.device)
                        actual = result.detach().cpu().float()
                        row['finite'] = bool(torch.isfinite(actual).all())
                        row['relative_l2'] = float((actual-reference).norm()/reference.norm().clamp_min(1e-12))
                        row['passed'] = (result.device.type == 'fant' and row['finite']
                                         and row['relative_l2'] < .03
                                         and torch.allclose(actual, reference, rtol=.05, atol=.01))
                    except Exception as exc:
                        row['error'] = repr(exc)
                    finally:
                        del result, device_values
                    report['operators'].append(row)
    except Exception as exc:
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
    print(json.dumps(finalize(report), indent=2))
    # A diagnostic may collect useful results while the runtime gate still fails.
    return 0 if (not report.get('error') and report['platform_is_fant']
                 and (args.mode == 'api' or report['small_ops_passed'])) else 2


if __name__ == '__main__':
    raise SystemExit(main())
