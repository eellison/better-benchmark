"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: 182f6f9450b9
Shape hash: 0a954b74
"""
_shapes_config = "(T([128, 144, 1, 1], f32), T([128, 144, 28, 28], f32, stride=(112896, 1, 4032, 144)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_18: "f32[128, 144, 1, 1]", div_10: "f32[128, 144, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
        mul_tensor: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(div_10, sigmoid_default);  div_10 = sigmoid_default = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([128, 144, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(14450688, dtype=torch.float32, device='cuda').as_strided([128, 144, 28, 28], [112896, 1, 4032, 144]),  # div_10
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
