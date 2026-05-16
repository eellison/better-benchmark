"""
Standalone repro captured via capture_hook.
Label: vgg16_inference
Pattern hash: fe73768527e6
Shape hash: a43b1211
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_11: "f32[1, 512, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        relu_default: "f32[1, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        return relu_default



def make_inputs():
    return [
    torch.randn([1, 512, 14, 14], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
