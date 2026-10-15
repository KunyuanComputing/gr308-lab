import json, os, time, traceback
from pathlib import Path
import torch
import torch_fant  # registers the `fant` PrivateUse1 backend
from diffusers import QwenImage21Pipeline

base = Path('/work')
steps = int(os.environ.get('STEPS', '4'))
width = int(os.environ.get('WIDTH', '1024'))
height = int(os.environ.get('HEIGHT', '1024'))
dtype_name = os.environ.get('DTYPE', 'bfloat16')
dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_name]
device = torch.device('fant:0')
out = base / 'output' / f'qwen-image-2.1-{dtype_name}-{width}x{height}-{steps}steps.png'
result = {'steps': steps, 'width': width, 'height': height, 'dtype': dtype_name, 'device': str(device), 'output': str(out)}
start = time.time()
try:
    result['torch'] = torch.__version__
    result['device_count'] = torch.fant.device_count()
    result['device_name'] = torch.fant.get_device_name(0)
    pipe = QwenImage21Pipeline.from_pretrained('/work/model', torch_dtype=dtype, low_cpu_mem_usage=True)
    result['load_seconds'] = time.time() - start
    pipe.to(device)
    if hasattr(torch.fant, 'reset_peak_memory_stats'):
        torch.fant.reset_peak_memory_stats(0)
    gen = torch.Generator(device='cpu').manual_seed(42)
    infer_start = time.time()
    image = pipe(
        prompt='A clean product photograph of a red ceramic teapot on a light wooden table, soft studio lighting, with the Chinese text “风华三号” clearly printed on the teapot.',
        width=width, height=height, num_inference_steps=steps, guidance_scale=1.0, generator=gen,
    ).images[0]
    if hasattr(torch.fant, 'synchronize'):
        torch.fant.synchronize()
    result['inference_seconds'] = time.time() - infer_start
    result['peak_memory_bytes'] = torch.fant.max_memory_allocated(0) if hasattr(torch.fant, 'max_memory_allocated') else None
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    result['ok'] = True
except Exception as e:
    result['ok'] = False
    result['error'] = repr(e)
    result['traceback'] = traceback.format_exc()
finally:
    result['total_seconds'] = time.time() - start
    (base / 'logs' / f'probe-{dtype_name}-{steps}steps.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result.get('ok'):
        raise SystemExit(1)
