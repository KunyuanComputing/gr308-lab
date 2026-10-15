import torch
import torch_fant
for name in (
    'aten::scaled_dot_product_attention',
    'aten::_scaled_dot_product_attention_math',
    'aten::linear',
    'aten::addmm',
    'aten::mm',
    'aten::bmm',
    'aten::_softmax',
):
    table = torch._C._dispatch_dump_table(name)
    selected = [line for line in table.splitlines() if line.startswith('PrivateUse1:')]
    print(name, *(selected or ['NO_PRIVATEUSE1']), sep='\n  ')
print('torch_fant', torch_fant.__file__)
