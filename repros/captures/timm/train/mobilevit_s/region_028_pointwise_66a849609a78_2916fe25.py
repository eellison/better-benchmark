"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 66a849609a78
Shape hash: 2916fe25
"""
_shapes_config = "(T([131072, 432], f32), S([512, 256, 432]), S([512, 256, 3, 4, 36]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_4: "f32[131072, 432]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[512, 256, 432]" = torch.ops.aten.reshape.default(addmm_4, _shape_param_0);  addmm_4 = _shape_param_0 = None
        reshape_default_1: "f32[512, 256, 3, 4, 36]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[3, 512, 4, 256, 36]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[512, 4, 256, 36]" = unbind_int[0]
        getitem_1: "f32[512, 4, 256, 36]" = unbind_int[1]
        getitem_2: "f32[512, 4, 256, 36]" = unbind_int[2];  unbind_int = None
        return (getitem, getitem_1, getitem_2)



def make_inputs():
    return [
    torch.randn([131072, 432], dtype=torch.float32, device='cuda'),
    [512, 256, 432],  # _shape_param_0
    [512, 256, 3, 4, 36],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
