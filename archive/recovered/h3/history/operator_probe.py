#!/usr/bin/env python3
"""Bounded real-H3 weight-row probe, explicitly not a model/video E2E."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--rows', type=int, default=128)
    ap.add_argument('--full-projections', action='store_true')
    ap.add_argument('--compat-only', action='store_true')
    args = ap.parse_args()
    assert 1 <= args.rows <= 128
    import torch
    if args.device != 'cpu':
        import torch_fant
        if args.full_projections:
            from sglang.multimodal_gen.runtime.platforms import current_platform
            from fantasy_platform import require_active
            assert require_active() is current_platform
            torch_fant.set_per_process_memory_fraction(.10, 0)
    torch.set_num_threads(2)
    print(json.dumps(dict(stage='runtime', torch=torch.__version__, device=args.device,
                         int_mm_privateuse1=torch._C._dispatch_has_kernel_for_dispatch_key(
                             'aten::_int_mm', 'PrivateUse1'))), flush=True)
    if args.device != 'cpu':
        a = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        got = (a.to(args.device) @ a.to(args.device)).cpu()
        control_ok = torch.equal(got, a @ a)
        print(json.dumps(dict(stage='fp32_control', passed=control_ok,
                              device=args.device, values=got.tolist())), flush=True)
        if not control_ok:
            return 3
        try:
            a8 = torch.ones((32, 32), dtype=torch.int8).to(args.device)
            got8 = torch._int_mm(a8, a8).cpu()
            print(json.dumps(dict(stage='int8_primitive', passed=bool((got8 == 32).all()),
                                  output_dtype=str(got8.dtype))), flush=True)
        except Exception as exc:
            print(json.dumps(dict(stage='int8_primitive', passed=False,
                                  error=repr(exc))), flush=True)
    import comfy_kitchen as ck
    from comfy_kitchen.tensor.w4a8_int8 import w4a8_int8_linear
    path = Path(args.checkpoint)
    with path.open('rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        assert 0 < hlen < 64*1024*1024
        raw_header = f.read(hlen)
    head = json.loads(raw_header)
    dtypes = dict(I8=torch.int8, F32=torch.float32, F8_E4M3=torch.float8_e4m3fn)

    def read(name, limit=True):
        h = head[name]
        shape = list(h['shape'])
        if limit:
            if args.full_projections:
                assert 1 <= shape[0] <= 32768, 'projection row cap exceeded'
            else:
                shape[0] = min(shape[0], args.rows)
        dtype = dtypes[h['dtype']]
        count = math.prod(shape)
        nbytes = count * torch.tensor([], dtype=dtype).element_size()
        assert nbytes <= 512*1024*1024, 'single tensor read budget exceeded'
        offset = 8 + hlen + h['data_offsets'][0]
        assert offset+nbytes <= path.stat().st_size, 'requested tensor range not downloaded'
        with path.open('rb') as f:
            f.seek(offset)
            raw = bytearray(f.read(nbytes))
        assert len(raw) == nbytes
        return torch.frombuffer(raw, dtype=dtype).clone().reshape(shape)

    for stem in ('blocks.0.attn.out_proj', 'blocks.0.attn.qkv_proj',
                 'blocks.0.mlp.fc1', 'blocks.0.mlp.fc2'):
        q = read(stem+'.weight')
        scales = read(stem+'.weight_s_rel')
        channels = read(stem+'.weight_s_channel')
        cb = read(stem+'.weight_codebook', limit=False)
        torch.manual_seed(1101)
        for m in (1, 4, 32):
            x = torch.randn(m, q.shape[1]*2, dtype=torch.bfloat16)
            common = dict(group_size=16, convrot_groupsize=256, out_dtype=torch.bfloat16)
            with ck.use_backend('eager'):
                reference = w4a8_int8_linear(x, q, scales, channels, codebook=cb, **common)
            modes = (('serialized_fp8',) if args.device == 'cpu' else
                     ('cpu_decoded_fp32_scale',) if args.compat_only else
                     ('serialized_fp8', 'cpu_decoded_fp32_scale'))
            for scale_mode in modes:
                row = dict(stage='operator', stem=stem, m=m, n=q.shape[0], k=x.shape[1],
                           full_original_n=head[stem+'.weight']['shape'][0],
                           full_projection=args.full_projections,
                           device=args.device, scale_mode=scale_mode,
                           header_sha256=hashlib.sha256(raw_header).hexdigest())
                try:
                    sd = scales if scale_mode == 'serialized_fp8' else scales.float()
                    operands = [v.to(args.device) for v in (x, q, sd, channels, cb)]
                    t0 = time.perf_counter()
                    with ck.use_backend('eager'):
                        got = w4a8_int8_linear(*operands[:4], codebook=operands[4], **common)
                    got = got.cpu().float()
                    row['wall_s'] = time.perf_counter()-t0
                    row['finite'] = bool(torch.isfinite(got).all())
                    row['rel_l2'] = float(torch.linalg.vector_norm(got-reference.float()) /
                                          torch.linalg.vector_norm(reference.float()).clamp_min(1e-12))
                    row['passed'] = row['finite'] and row['rel_l2'] <= 0.02
                except Exception as exc:
                    row.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
                print(json.dumps(row), flush=True)
                if not row['passed'] and (args.device == 'cpu' or scale_mode == 'cpu_decoded_fp32_scale'):
                    if args.device != 'cpu':
                        # Capacity/compute diagnostic only: removing A8 quantization
                        # changes the inference contract and does not pass the W4A8 gate.
                        control = dict(stage='cpu_dequant_bf16_control', stem=stem,
                                       m=m, n=q.shape[0], k=x.shape[1],
                                       w4a8_gate_still_failed=True)
                        try:
                            with ck.use_backend('eager'):
                                dense = ck.dequantize_w4a8_int8_weight(
                                    q, scales, channels, codebook=cb, group_size=16,
                                    convrot_groupsize=256, output_dtype=torch.bfloat16)
                            dense_cpu = torch.nn.functional.linear(x, dense).float()
                            dense_gpu = torch.nn.functional.linear(
                                x.to(args.device), dense.to(args.device)).cpu().float()
                            def rel(a, b):
                                return float(torch.linalg.vector_norm(a-b) /
                                             torch.linalg.vector_norm(b).clamp_min(1e-12))
                            control['finite'] = bool(torch.isfinite(dense_gpu).all())
                            control['rel_l2_vs_dense_cpu'] = rel(dense_gpu, dense_cpu)
                            control['rel_l2_vs_w4a8_cpu'] = rel(dense_gpu, reference.float())
                            control['passed'] = control['finite'] and control['rel_l2_vs_dense_cpu'] <= .02
                        except Exception as exc:
                            control.update(passed=False, error=repr(exc))
                        print(json.dumps(control), flush=True)
                    return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
