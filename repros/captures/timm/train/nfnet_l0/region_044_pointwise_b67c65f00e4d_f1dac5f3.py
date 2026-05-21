"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: b67c65f00e4d
Shape hash: f1dac5f3
"""
_shapes_config = "(T([128, 128, 1, 1], f32), T([], f32), T([128, 128, 1, 1], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_1: "f32[128, 128, 1, 1]", full_default: "f32[]", getitem_303: "f32[128, 128, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_scalar: "b8[128, 128, 1, 1]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_self: "f32[128, 128, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_303);  le_scalar = full_default = getitem_303 = None
        return where_self



def make_inputs():
    return [
    torch.randn([128, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
