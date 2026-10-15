import torch
import torch_fant
for name in (
    'aten::scaled_dot_product_attention',
    'aten::_scaled_dot_product_attention_math',
    'aten::linear',
    'aten::mm',
    'aten::bmm',
):
    print('OP', name)
    print(torch._C._dispatch_dump_table(name))
print('TORCH_FANT', torch_fant.__file__)
