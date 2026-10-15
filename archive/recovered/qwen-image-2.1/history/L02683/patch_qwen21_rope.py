from pathlib import Path
p=Path('/data/infra/qwen-image-2.1-probe-20260921/deps/diffusers/models/transformers/transformer_qwenimage21.py')
s=p.read_text()
old='''        self.freqs = [freq.to(device) for freq in self.freqs]\n\n        frame_index, height_index, width_index = [], [], []\n'''
new='''        # Fantasy/GR308 does not support complex64 indexing or arithmetic.\n        # Keep the immutable RoPE table and all indexing on CPU, then send the\n        # mathematically equivalent real cos/sin representation to the device.\n        self.freqs = [freq.cpu() for freq in self.freqs]\n\n        frame_index, height_index, width_index = [], [], []\n'''
if old not in s: raise SystemExit('first patch anchor missing')
s=s.replace(old,new,1)
old2='''        frame_index = torch.tensor(frame_index, dtype=torch.long, device=device)\n        height_index = frame_index.clone()\n        width_index = frame_index.clone()\n        height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long, device=device)\n        width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long, device=device)\n\n        return torch.cat([self.freqs[0][frame_index], self.freqs[1][height_index], self.freqs[2][width_index]], dim=-1)\n'''
new2='''        frame_index = torch.tensor(frame_index, dtype=torch.long, device="cpu")\n        height_index = frame_index.clone()\n        width_index = frame_index.clone()\n        image_pad_mask_cpu = image_pad_mask.to("cpu")\n        height_index[image_pad_mask_cpu] = torch.tensor(image_height_index, dtype=torch.long, device="cpu")\n        width_index[image_pad_mask_cpu] = torch.tensor(image_width_index, dtype=torch.long, device="cpu")\n\n        freqs_complex = torch.cat(\n            [self.freqs[0][frame_index], self.freqs[1][height_index], self.freqs[2][width_index]], dim=-1\n        )\n        cos = torch.repeat_interleave(freqs_complex.real, 2, dim=-1).to(device)\n        sin = torch.repeat_interleave(freqs_complex.imag, 2, dim=-1).to(device)\n        return cos, sin\n'''
if old2 not in s: raise SystemExit('second patch anchor missing')
s=s.replace(old2,new2,1)
count=s.count('rotary_emb, use_real=False')
if count != 2: raise SystemExit(f'expected 2 use_real anchors, got {count}')
s=s.replace('rotary_emb, use_real=False','rotary_emb, use_real=True')
backup=p.with_suffix('.py.upstream')
if not backup.exists(): backup.write_text(p.read_text())
p.write_text(s)
print('patched',p)
