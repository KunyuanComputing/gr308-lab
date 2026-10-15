from pathlib import Path
p = Path('/data/infra/qwen-image-2.1-probe-20260921/deps/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py')
s = p.read_text()
start = s.index('    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:\n', s.index('class QwenImage21DupUp3D'))
end = s.index('\n\n\nclass QwenImage21CausalConv3d', start)
new = '''    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        # FANT currently rejects the upstream 8D permute. Perform the same
        # spatial and temporal pixel shuffle using tensors of at most 5 dims.
        batch, _, frames, height, width = x.shape
        repeat_idx = torch.arange(x.size(1), device="cpu").repeat_interleave(self.repeats).to(x.device)
        x = x.index_select(1, repeat_idx)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(batch * frames, self.out_channels * self.factor_t * self.factor_s**2, height, width)
        x = F.pixel_shuffle(x, self.factor_s)
        out_height, out_width = height * self.factor_s, width * self.factor_s
        x = x.reshape(batch, frames, self.out_channels * self.factor_t, out_height, out_width)

        temporal_idx = (
            torch.arange(frames * self.out_channels * self.factor_t, device="cpu")
            .reshape(frames, self.out_channels, self.factor_t)
            .permute(0, 2, 1)
            .reshape(-1)
            .to(x.device)
        )
        x = x.reshape(batch, frames * self.out_channels * self.factor_t, out_height, out_width)
        x = x.index_select(1, temporal_idx)
        x = x.reshape(batch, frames * self.factor_t, self.out_channels, out_height, out_width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x'''
p.write_text(s[:start] + new + s[end:])
