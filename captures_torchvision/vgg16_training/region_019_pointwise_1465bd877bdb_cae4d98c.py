"""
Standalone repro captured via capture_hook.
Label: vgg16_training
Pattern hash: 1465bd877bdb
Shape hash: cae4d98c
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_7: "f32[4, 512, 28, 28]", full_default: "f32[]", getitem_22: "f32[4, 512, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        le_scalar: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_self: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_22);  le_scalar = full_default = getitem_22 = None
        return where_self



def make_inputs():
    return [
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([4, 512, 28, 28], [401408, 1, 14336, 512]),  # relu_7
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([4, 512, 28, 28], [401408, 1, 14336, 512]),  # getitem_22
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
