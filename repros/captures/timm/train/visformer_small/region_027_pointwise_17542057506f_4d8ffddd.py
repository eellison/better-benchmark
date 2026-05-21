"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 17542057506f
Shape hash: 4d8ffddd
"""
_shapes_config = "(T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), S([128, 6, 64, 196]), S([768, 196, 64]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_173: "f32[128, 384, 14, 14]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        reshape_default: "f32[128, 6, 64, 196]" = torch.ops.aten.reshape.default(getitem_173, _shape_param_0);  getitem_173 = _shape_param_0 = None
        permute_default: "f32[128, 6, 196, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 2]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        clone_default: "f32[128, 6, 196, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[768, 196, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        return reshape_default_1



def make_inputs():
    return [
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # getitem_173
    [128, 6, 64, 196],  # _shape_param_0
    [768, 196, 64],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
