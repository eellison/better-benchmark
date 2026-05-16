"""
Standalone repro captured via capture_hook.
Label: vgg16_training
Pattern hash: 1465bd877bdb
Shape hash: 3a464cb8
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_2: "f32[4, 128, 112, 112]", full_default: "f32[]", getitem_37: "f32[4, 128, 112, 112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        le_scalar: "b8[4, 128, 112, 112]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_self: "f32[4, 128, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_37);  le_scalar = full_default = getitem_37 = None
        return where_self



def make_inputs():
    return [
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([4, 128, 112, 112], [1605632, 1, 14336, 128]),  # relu_2
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(6422528, dtype=torch.float32, device='cuda').as_strided([4, 128, 112, 112], [1605632, 1, 14336, 128]),  # getitem_37
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
