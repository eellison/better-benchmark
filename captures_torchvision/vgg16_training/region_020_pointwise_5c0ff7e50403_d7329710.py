"""
Standalone repro captured via capture_hook.
Label: vgg16_training
Pattern hash: 5c0ff7e50403
Shape hash: d7329710
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_16: "f32[4, 512, 14, 14]", _shape_param_0, getitem_7: "i8[4, 512, 14, 14]", _shape_param_1, _shape_param_2, le_5: "b8[4, 512, 28, 28]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        full_default: "f32[2048, 784]" = torch.ops.aten.full.default([2048, 784], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[2048, 196]" = torch.ops.aten.reshape.default(getitem_16, _shape_param_0);  getitem_16 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[4, 512, 14, 14]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_7, [2, 2], [28, 28], [2, 2], [0, 0], [1, 1]);  getitem_7 = None
        reshape_default_1: "i64[2048, 196]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[2048, 784]" = torch.ops.aten.scatter_add.default(full_default, 1, reshape_default_1, reshape_default);  full_default = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[4, 512, 28, 28]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        full_default_1 = full_default
        where_self: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_5, full_default_1, reshape_default_2);  le_5 = full_default_1 = reshape_default_2 = None
        return where_self



def make_inputs():
    return [
    torch.randn(401408, dtype=torch.float32, device='cuda').as_strided([4, 512, 14, 14], [100352, 1, 7168, 512]),  # getitem_16
    [2048, 196],  # _shape_param_0
    torch.randint(0, 2, (401408,), dtype=torch.int8, device='cuda').as_strided([4, 512, 14, 14], [100352, 1, 7168, 512]),  # getitem_7
    [2048, 196],  # _shape_param_1
    [4, 512, 28, 28],  # _shape_param_2
    torch.randn(1605632, dtype=torch.bool, device='cuda').as_strided([4, 512, 28, 28], [401408, 1, 14336, 512]),  # le_5
    torch.tensor(1),  # full_default_1 (unknown shape)
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
