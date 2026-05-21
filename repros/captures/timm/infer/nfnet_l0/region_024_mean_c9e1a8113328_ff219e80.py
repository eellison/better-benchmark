"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer
Pattern hash: c9e1a8113328
Shape hash: ff219e80
"""
_shapes_config = "(T([128, 256, 56, 56], f32, stride=(802816, 1, 14336, 256)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_8: "f32[128, 256, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[128, 256, 1, 1]" = torch.ops.aten.mean.dim(convolution_8, [2, 3], True);  convolution_8 = None
        return mean_dim



def make_inputs():
    return [
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([128, 256, 56, 56], [802816, 1, 14336, 256]),  # convolution_8
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
