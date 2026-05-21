"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: 182f6f9450b9
Shape hash: 71d37421
"""
_shapes_config = "(T([128, 96, 1, 1], f32), T([128, 96, 56, 56], f32, stride=(301056, 1, 5376, 96)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_8: "f32[128, 96, 1, 1]", div_4: "f32[128, 96, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_8);  convolution_8 = None
        mul_tensor: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(div_4, sigmoid_default);  div_4 = sigmoid_default = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([128, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # div_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
