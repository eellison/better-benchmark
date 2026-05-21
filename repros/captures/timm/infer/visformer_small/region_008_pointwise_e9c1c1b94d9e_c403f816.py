"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_infer
Pattern hash: e9c1c1b94d9e
Shape hash: c403f816
"""
_shapes_config = "(T([128, 1536, 14, 14], f32, stride=(301056, 1, 21504, 1536)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_38: "f32[128, 1536, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        mul_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, 0.5)
        mul_tensor_1: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476);  convolution_38 = None
        erf_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        return mul_tensor_2



def make_inputs():
    return [
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 1536, 14, 14], [301056, 1, 21504, 1536]),  # convolution_38
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
