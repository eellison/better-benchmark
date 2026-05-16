"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: a9742808a25b
Shape hash: a89ca877
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_2: "f32[4, 64, 56, 56]", convolution_2: "f32[4, 32, 56, 56]", convolution_4: "f32[4, 32, 56, 56]", convolution_6: "f32[4, 32, 56, 56]", convolution_8: "f32[4, 32, 56, 56]", convolution_10: "f32[4, 32, 56, 56]", convolution_12: "f32[4, 32, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        cat_default: "f32[4, 256, 56, 56]" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4, convolution_6, convolution_8, convolution_10, convolution_12], 1);  getitem_2 = convolution_2 = convolution_4 = convolution_6 = convolution_8 = convolution_10 = convolution_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [0, 2, 3], correction = 0, keepdim = True);  cat_default = None
        getitem: "f32[1, 256, 1, 1]" = var_mean_correction[0]
        getitem_3: "f32[1, 256, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_3)



def make_inputs():
    return [
    torch.randn([4, 64, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 56, 56], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
