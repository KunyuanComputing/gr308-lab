from pathlib import Path
p=Path('/data/infra/qwen-image-2.1-probe-20260921/deps/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py')
s=p.read_text()
old='''        x = x.reshape(batch * frames, self.out_channels * self.factor_t * self.factor_s**2, height, width)
        x = F.pixel_shuffle(x, self.factor_s)
        out_height, out_width = height * self.factor_s, width * self.factor_s
        x = x.reshape(batch, frames, self.out_channels * self.factor_t, out_height, out_width)
'''
new='''        spatial_channels = self.out_channels * self.factor_t
        x = x.reshape(batch * frames, spatial_channels * self.factor_s**2, height, width)
        out_height, out_width = height * self.factor_s, width * self.factor_s
        target = torch.arange(spatial_channels * out_height * out_width, device="cpu")
        out_w = target.remainder(out_width)
        target = target.div(out_width, rounding_mode="floor")
        out_h = target.remainder(out_height)
        out_c = target.div(out_height, rounding_mode="floor")
        source_c = out_c * self.factor_s**2 + out_h.remainder(self.factor_s) * self.factor_s + out_w.remainder(self.factor_s)
        spatial_idx = ((source_c * height + out_h.div(self.factor_s, rounding_mode="floor")) * width + out_w.div(self.factor_s, rounding_mode="floor")).to(x.device)
        x = x.reshape(batch * frames, -1).index_select(1, spatial_idx)
        x = x.reshape(batch, frames, spatial_channels, out_height, out_width)
'''
assert old in s
p.write_text(s.replace(old,new,1))
