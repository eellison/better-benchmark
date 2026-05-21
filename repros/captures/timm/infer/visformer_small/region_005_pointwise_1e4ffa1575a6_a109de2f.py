"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_infer
Pattern hash: 1e4ffa1575a6
Shape hash: a109de2f
"""
_shapes_config = "(T([128, 2304, 7, 7], f32, stride=(112896, 1, 16128, 2304)), S([128, 3, 6, 128, 49]), S([128, 6, 49, 128]), S([768, 49, 128]), S([128, 6, 128, 49]), S([768, 128, 49]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_53: "f32[128, 2304, 7, 7]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        reshape_default: "f32[128, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_53, _shape_param_0);  convolution_53 = _shape_param_0 = None
        permute_default: "f32[3, 128, 6, 49, 128]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2, 4, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[128, 6, 49, 128]" = unbind_int[0]
        getitem_1: "f32[128, 6, 49, 128]" = unbind_int[1]
        getitem_2: "f32[128, 6, 49, 128]" = unbind_int[2];  unbind_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_default: "f32[128, 6, 49, 128]" = torch.ops.aten.expand.default(getitem, _shape_param_1);  getitem = _shape_param_1 = None
        clone_default: "f32[128, 6, 49, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_1: "f32[768, 49, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        permute_default_1: "f32[128, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand_default_1: "f32[128, 6, 128, 49]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        clone_default_1: "f32[128, 6, 128, 49]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_2: "f32[768, 128, 49]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        return (reshape_default_1, reshape_default_2, getitem_2)



def make_inputs():
    return [
    torch.randn(14450688, dtype=torch.float32, device='cuda').as_strided([128, 2304, 7, 7], [112896, 1, 16128, 2304]),  # convolution_53
    [128, 3, 6, 128, 49],  # _shape_param_0
    [128, 6, 49, 128],  # _shape_param_1
    [768, 49, 128],  # _shape_param_2
    [128, 6, 128, 49],  # _shape_param_3
    [768, 128, 49],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
