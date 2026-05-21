"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 3ec9e91bb89f
Shape hash: a4ebf2b9
"""
_shapes_config = "(T([512, 4, 16, 60], f32, stride=(3840, 60, 240, 1)), T([512, 4, 16, 60], f32, stride=(3840, 60, 240, 1)), T([512, 4, 16, 60], f32, stride=(3840, 60, 240, 1)), T([720, 240], f32), S([3, 512, 4, 16, 60]), S([512, 16, 720]), S([8192, 720]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_186: "f32[512, 4, 16, 60]", getitem_187: "f32[512, 4, 16, 60]", getitem_188: "f32[512, 4, 16, 60]", primals_257: "f32[720, 240]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_default: "f32[1536, 4, 16, 60]" = torch.ops.aten.cat.default([getitem_186, getitem_187, getitem_188]);  getitem_186 = getitem_187 = getitem_188 = None
        reshape_default: "f32[3, 512, 4, 16, 60]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default: "f32[512, 16, 3, 4, 60]" = torch.ops.aten.permute.default(reshape_default, [1, 3, 0, 2, 4]);  reshape_default = None
        clone_default: "f32[512, 16, 3, 4, 60]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[512, 16, 720]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[8192, 720]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[240, 720]" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
        permute_default_2: "f32[720, 240]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_2, permute_default_2)



def make_inputs():
    return [
    torch.randn(1966080, dtype=torch.float32, device='cuda').as_strided([512, 4, 16, 60], [3840, 60, 240, 1]),  # getitem_186
    torch.randn(1966080, dtype=torch.float32, device='cuda').as_strided([512, 4, 16, 60], [3840, 60, 240, 1]),  # getitem_187
    torch.randn(1966080, dtype=torch.float32, device='cuda').as_strided([512, 4, 16, 60], [3840, 60, 240, 1]),  # getitem_188
    torch.randn([720, 240], dtype=torch.float32, device='cuda'),
    [3, 512, 4, 16, 60],  # _shape_param_0
    [512, 16, 720],  # _shape_param_1
    [8192, 720],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
