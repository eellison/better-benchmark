"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: b76cd296cd55
Shape hash: 34f714a8
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_23: "f32[4, 240, 1, 1]", div_14: "f32[4, 240, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        sigmoid_default: "f32[4, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_tensor: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_default, div_14);  sigmoid_default = div_14 = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([4, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 240, 28, 28], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
