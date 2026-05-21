"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: 3abc926270f6
Shape hash: 8f95d7d1
"""
_shapes_config = "(T([128, 128, 1, 1], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_22: "f32[128, 128, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_default: "f32[128, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
        return relu_default



def make_inputs():
    return [
    torch.randn([128, 128, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
