"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_inference
Pattern hash: b76cd296cd55
Shape hash: f5482f7d
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_43: "f32[1, 480, 1, 1]", div_25: "f32[1, 480, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        sigmoid_default: "f32[1, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_tensor: "f32[1, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_default, div_25);  sigmoid_default = div_25 = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([1, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 480, 14, 14], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
