"""
Standalone repro captured via capture_hook.
Label: resnext50_32x4d_training
Pattern hash: 6bba1857ce49
Shape hash: a62fc37d
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_45: "f32[4, 2048, 7, 7]", convolution_46: "f32[4, 2048, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True);  convolution_45 = None
        getitem: "f32[1, 2048, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 2048, 1, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True);  convolution_46 = None
        getitem_2: "f32[1, 2048, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 2048, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        return (getitem, getitem_1, getitem_2, getitem_3)



def make_inputs():
    return [
    torch.randn([4, 2048, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 2048, 7, 7], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
