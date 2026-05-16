"""
Standalone repro captured via capture_hook.
Label: resnet18_training
Pattern hash: 6bba1857ce49
Shape hash: 864a1592
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_16: "f32[4, 512, 7, 7]", convolution_17: "f32[4, 512, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True);  convolution_16 = None
        getitem: "f32[1, 512, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True);  convolution_17 = None
        getitem_2: "f32[1, 512, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 512, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        return (getitem, getitem_1, getitem_2, getitem_3)



def make_inputs():
    return [
    torch.randn([4, 512, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 7, 7], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
