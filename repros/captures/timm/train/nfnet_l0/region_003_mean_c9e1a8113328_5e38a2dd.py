"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: c9e1a8113328
Shape hash: 5e38a2dd
"""
_shapes_config = "(T([128, 1536, 7, 7], f32, stride=(75264, 1, 10752, 1536)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_77: "f32[128, 1536, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_77, [2, 3], True);  convolution_77 = None
        return mean_dim



def make_inputs():
    return [
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 1536, 7, 7], [75264, 1, 10752, 1536]),  # convolution_77
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
