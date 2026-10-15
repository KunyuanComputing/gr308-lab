from pathlib import Path
p=Path('/data/infra/qwen-image-2.1-probe-20260921/deps/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py')
s=p.read_text()
start=s.index('        spatial_channels = self.out_channels * self.factor_t', s.index('class QwenImage21DupUp3D'))
end=s.index('        if first_chunk:', start)
new='''        spatial_channels = self.out_channels * self.factor_t
        x = x.reshape(batch * frames, spatial_channels * self.factor_s**2, height, width)
        out_height, out_width = height * self.factor_s, width * self.factor_s
        spatial = torch.empty(
            batch * frames, spatial_channels, out_height, out_width,
            device=x.device, dtype=x.dtype,
        )
        for offset_h in range(self.factor_s):
            for offset_w in range(self.factor_s):
                phase = offset_h * self.factor_s + offset_w
                spatial[:, :, offset_h::self.factor_s, offset_w::self.factor_s].copy_(
                    x[:, phase::self.factor_s**2]
                )
        x = spatial.reshape(batch, frames, spatial_channels, out_height, out_width)

        temporal = torch.empty(
            batch, self.out_channels, frames * self.factor_t, out_height, out_width,
            device=x.device, dtype=x.dtype,
        )
        for offset_t in range(self.factor_t):
            temporal[:, :, offset_t::self.factor_t].copy_(
                x[:, :, offset_t::self.factor_t].permute(0, 2, 1, 3, 4)
            )
        x = temporal
'''
p.write_text(s[:start]+new+s[end:])
