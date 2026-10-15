import json, time, traceback
from pathlib import Path
import torch
import torch_fant
from PIL import Image
from diffusers import AutoencoderKLQwenImage21
base=Path('/work')
latent_path=base/'output/qwen-image-2.1-bfloat16-1024x1024-4steps-latents.pt'
out=base/'output/qwen-image-2.1-bfloat16-1024x1024-4steps.png'
result={'device':'fant:0','dtype':'bfloat16','latent':str(latent_path),'output':str(out)}
start=time.time()
try:
    vae=AutoencoderKLQwenImage21.from_pretrained('/work/model/vae', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    vae.to('fant:0')
    packed=torch.load(latent_path,map_location='cpu').to('fant:0')
    batch,_,channels=packed.shape
    latent_h=latent_w=64
    latents=packed.transpose(1,2).reshape(batch,channels,1,latent_h,latent_w).to(vae.dtype)
    mean=torch.tensor(vae.config.latents_mean).view(1,vae.config.z_dim,1,1,1).to(latents.device,latents.dtype)
    std=torch.tensor(vae.config.latents_std).view(1,vae.config.z_dim,1,1,1).to(latents.device,latents.dtype)
    latents=latents*std+mean
    if hasattr(torch.fant,'reset_peak_memory_stats'): torch.fant.reset_peak_memory_stats(0)
    t=time.time()
    decoded=vae.decode(latents,return_dict=False)[0][:,:,0]
    if hasattr(torch.fant,'synchronize'): torch.fant.synchronize()
    result['decode_seconds']=time.time()-t
    result['peak_memory_bytes']=torch.fant.max_memory_allocated(0) if hasattr(torch.fant,'max_memory_allocated') else None
    arr=((decoded[0].float().cpu().permute(1,2,0)/2+0.5).clamp(0,1)*255).round().to(torch.uint8).numpy()
    Image.fromarray(arr).save(out)
    result['ok']=True
except Exception as e:
    result['ok']=False; result['error']=repr(e); result['traceback']=traceback.format_exc()
finally:
    result['total_seconds']=time.time()-start
    (base/'logs/decode-bfloat16-4steps.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if not result.get('ok'): raise SystemExit(1)
