"""
Standalone repro captured via capture_hook.
Label: vgg16_training
Pattern hash: 1465bd877bdb
Shape hash: eb68e349
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_4: "f32[4, 256, 56, 56]", full_default: "f32[]", getitem_31: "f32[4, 256, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        le_scalar: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_self: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_31);  le_scalar = full_default = getitem_31 = None
        return where_self



def make_inputs():
    return [
    torch.randn(3211264, dtype=torch.float32, device='cuda').as_strided([4, 256, 56, 56], [802816, 1, 14336, 256]),  # relu_4
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(3211264, dtype=torch.float32, device='cuda').as_strided([4, 256, 56, 56], [802816, 1, 14336, 256]),  # getitem_31
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
