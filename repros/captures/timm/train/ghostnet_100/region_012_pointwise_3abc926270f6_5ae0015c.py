"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 3abc926270f6
Shape hash: 5ae0015c
"""
_shapes_config = "(T([512, 168, 1, 1], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_67: "f32[512, 168, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_default: "f32[512, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_67);  convolution_67 = None
        return relu_default



def make_inputs():
    return [
    torch.randn([512, 168, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
