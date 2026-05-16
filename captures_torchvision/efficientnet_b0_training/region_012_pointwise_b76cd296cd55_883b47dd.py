"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: b76cd296cd55
Shape hash: 883b47dd
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_58: "f32[4, 672, 1, 1]", div_40: "f32[4, 672, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        sigmoid_default: "f32[4, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_tensor: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_default, div_40);  sigmoid_default = div_40 = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([4, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 672, 7, 7], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
