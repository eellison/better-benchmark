"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch14_dinov2.lvd142m_train
Pattern hash: 3ec9e91bb89f
Shape hash: ffe71fd1
"""
_shapes_config = "(T([128, 12, 1370, 64], f32, stride=(1052160, 64, 768, 1)), T([128, 12, 1370, 64], f32, stride=(1052160, 64, 768, 1)), T([128, 12, 1370, 64], f32, stride=(1052160, 64, 768, 1)), T([2304, 768], f32), S([3, 128, 12, 1370, 64]), S([128, 1370, 2304]), S([175360, 2304]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_178: "f32[128, 12, 1370, 64]", getitem_179: "f32[128, 12, 1370, 64]", getitem_180: "f32[128, 12, 1370, 64]", primals_8: "f32[2304, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_default: "f32[384, 12, 1370, 64]" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
        reshape_default: "f32[3, 128, 12, 1370, 64]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default: "f32[128, 1370, 3, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [1, 3, 0, 2, 4]);  reshape_default = None
        clone_default: "f32[128, 1370, 3, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[128, 1370, 2304]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[175360, 2304]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_2: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_2, permute_default_2)



def make_inputs():
    return [
    torch.randn(134676480, dtype=torch.float32, device='cuda').as_strided([128, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_178
    torch.randn(134676480, dtype=torch.float32, device='cuda').as_strided([128, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_179
    torch.randn(134676480, dtype=torch.float32, device='cuda').as_strided([128, 12, 1370, 64], [1052160, 64, 768, 1]),  # getitem_180
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    [3, 128, 12, 1370, 64],  # _shape_param_0
    [128, 1370, 2304],  # _shape_param_1
    [175360, 2304],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
