"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: 182f6f9450b9
Shape hash: df7d9f07
"""
_shapes_config = "(T([128, 672, 1, 1], f32), T([128, 672, 7, 7], f32, stride=(32928, 1, 4704, 672)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_58: "f32[128, 672, 1, 1]", div_34: "f32[128, 672, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
        mul_tensor: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(div_34, sigmoid_default);  div_34 = sigmoid_default = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([128, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(4214784, dtype=torch.float32, device='cuda').as_strided([128, 672, 7, 7], [32928, 1, 4704, 672]),  # div_34
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
