from pathlib import Path
p=Path('/data/infra/qwen-image-2.1-probe-20260921/probe.py')
s=p.read_text()
old='''    image = pipe(
        prompt='A clean product photograph of a red ceramic teapot on a light wooden table, soft studio lighting, with the Chinese text “风华三号” clearly printed on the teapot.',
        width=width, height=height, num_inference_steps=steps, generator=gen,
    ).images[0]
'''
new='''    packed_latents = pipe(
        prompt='A clean product photograph of a red ceramic teapot on a light wooden table, soft studio lighting, with the Chinese text “风华三号” clearly printed on the teapot.',
        width=width, height=height, num_inference_steps=steps, generator=gen, output_type="latent",
    ).images
    latent_path = base / 'output' / f'qwen-image-2.1-{dtype_name}-{width}x{height}-{steps}steps-latents.pt'
    torch.save(packed_latents.cpu(), latent_path)
    result['latent_output'] = str(latent_path)
    latents = pipe._unpack_latents(packed_latents, height, width, pipe.vae_scale_factor).to(pipe.vae.dtype)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std + latents_mean
    decoded = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
'''
assert old in s
p.write_text(s.replace(old,new,1))
