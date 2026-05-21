"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: c9e1a8113328
Shape hash: 652992e8
"""
_shapes_config = "(T([128, 512, 28, 28], f32, stride=(401408, 1, 14336, 512)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_21: "f32[128, 512, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[128, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_21, [2, 3], True);  convolution_21 = None
        return mean_dim



def make_inputs():
    return [
    torch.randn(51380224, dtype=torch.float32, device='cuda').as_strided([128, 512, 28, 28], [401408, 1, 14336, 512]),  # convolution_21
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
