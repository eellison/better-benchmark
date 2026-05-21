"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_train
Pattern hash: 66a849609a78
Shape hash: 52ee260d
"""
_shapes_config = "(T([32768, 2304], f32), S([128, 256, 2304]), S([128, 256, 3, 12, 64]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_44: "f32[32768, 2304]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[128, 256, 2304]" = torch.ops.aten.reshape.default(addmm_44, _shape_param_0);  addmm_44 = _shape_param_0 = None
        reshape_default_1: "f32[128, 256, 3, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[3, 128, 12, 256, 64]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[128, 12, 256, 64]" = unbind_int[0]
        getitem_1: "f32[128, 12, 256, 64]" = unbind_int[1]
        getitem_2: "f32[128, 12, 256, 64]" = unbind_int[2];  unbind_int = None
        return (getitem, getitem_1, getitem_2)



def make_inputs():
    return [
    torch.randn([32768, 2304], dtype=torch.float32, device='cuda'),
    [128, 256, 2304],  # _shape_param_0
    [128, 256, 3, 12, 64],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
