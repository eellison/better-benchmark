"""
Standalone repro captured via capture_hook.
Label: resnet18_training
Pattern hash: 6bba1857ce49
Shape hash: 40390331
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_6: "f32[4, 128, 28, 28]", convolution_7: "f32[4, 128, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True);  convolution_6 = None
        getitem: "f32[1, 128, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 128, 1, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True);  convolution_7 = None
        getitem_2: "f32[1, 128, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 128, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        return (getitem, getitem_1, getitem_2, getitem_3)



def make_inputs():
    return [
    torch.randn([4, 128, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 28, 28], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
