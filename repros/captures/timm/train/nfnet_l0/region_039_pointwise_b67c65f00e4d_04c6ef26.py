"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: b67c65f00e4d
Shape hash: 04c6ef26
"""
_shapes_config = "(T([128, 64, 1, 1], f32), T([], f32), T([128, 64, 1, 1], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu: "f32[128, 64, 1, 1]", full_default: "f32[]", getitem_324: "f32[128, 64, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_scalar: "b8[128, 64, 1, 1]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_self: "f32[128, 64, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_324);  le_scalar = full_default = getitem_324 = None
        return where_self



def make_inputs():
    return [
    torch.randn([128, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
