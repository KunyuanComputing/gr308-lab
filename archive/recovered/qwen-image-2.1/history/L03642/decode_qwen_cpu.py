import json,time,traceback
from pathlib import Path
import torch
from PIL import Image
from diffusers import AutoencoderKLQwenImage21
base=Path('/work'); latent_path=base/'output/qwen-image-2.1-bfloat16-1024x1024-4steps-latents.pt'; out=base/'output/qwen-image-2.1-bfloat16-1024x1024-4steps-cpu-vae.png'
result={'device':'cpu','dtype':'float32','latent':str(latent_path),'output':str(out)}; start=time.time()
try:
 torch.set_num_threads(64); torch.set_num_interop_threads(4)
 vae=AutoencoderKLQwenImage21.from_pretrained('/work/model/vae',torch_dtype=torch.float32,low_cpu_mem_usage=True)
 packed=torch.load(latent_path,map_location='cpu').float(); b,_,c=packed.shape
 latents=packed.transpose(1,2).reshape(b,c,1,64,64)
 mean=torch.tensor(vae.config.latents_mean).view(1,vae.config.z_dim,1,1,1); std=torch.tensor(vae.config.latents_std).view(1,vae.config.z_dim,1,1,1)
 t=time.time(); decoded=vae.decode(latents*std+mean,return_dict=False)[0][:,:,0]; result['decode_seconds']=time.time()-t
 arr=((decoded[0].permute(1,2,0)/2+0.5).clamp(0,1)*255).round().to(torch.uint8).numpy(); Image.fromarray(arr).save(out); result['ok']=True
except Exception as e: result.update(ok=False,error=repr(e),traceback=traceback.format_exc())
finally:
 result['total_seconds']=time.time()-start; (base/'logs/decode-cpu-4steps.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)); print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
 if not result.get('ok'): raise SystemExit(1)
