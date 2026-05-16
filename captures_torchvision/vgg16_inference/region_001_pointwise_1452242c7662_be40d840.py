"""
Standalone repro captured via capture_hook.
Label: vgg16_inference
Pattern hash: 1452242c7662
Shape hash: be40d840
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[1, 4096]", arg29_1: "f32[4096, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        relu_default: "f32[1, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        return (relu_default, permute_default)



def make_inputs():
    return [
    torch.randn([1, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
