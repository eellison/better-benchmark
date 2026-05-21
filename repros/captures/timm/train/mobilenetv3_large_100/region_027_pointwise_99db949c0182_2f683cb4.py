"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 99db949c0182
Shape hash: 2f683cb4
"""
_shapes_config = "(T([512, 120, 1, 1], f32), T([512, 120, 28, 28], f32, stride=(94080, 1, 3360, 120)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_22: "f32[512, 120, 1, 1]", relu_12: "f32[512, 120, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        add_tensor: "f32[512, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_22, 3);  convolution_22 = None
        clamp_min_default: "f32[512, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor: "f32[512, 120, 28, 28]" = torch.ops.aten.mul.Tensor(relu_12, div_tensor);  relu_12 = div_tensor = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([512, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(48168960, dtype=torch.float32, device='cuda').as_strided([512, 120, 28, 28], [94080, 1, 3360, 120]),  # relu_12
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
