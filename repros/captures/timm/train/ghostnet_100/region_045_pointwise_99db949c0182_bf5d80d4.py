"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 99db949c0182
Shape hash: bf5d80d4
"""
_shapes_config = "(T([512, 72, 1, 1], f32), T([512, 72, 28, 28], f32, stride=(56448, 1, 2016, 72)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_20: "f32[512, 72, 1, 1]", add_97: "f32[512, 72, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        add_tensor: "f32[512, 72, 1, 1]" = torch.ops.aten.add.Tensor(convolution_20, 3);  convolution_20 = None
        clamp_min_default: "f32[512, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(add_97, div_tensor);  add_97 = div_tensor = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([512, 72, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(28901376, dtype=torch.float32, device='cuda').as_strided([512, 72, 28, 28], [56448, 1, 2016, 72]),  # add_97
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
