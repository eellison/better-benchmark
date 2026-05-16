"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: a9742808a25b
Shape hash: 9f60dbdd
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, avg_pool2d: "f32[4, 128, 28, 28]", convolution_15: "f32[4, 32, 28, 28]", convolution_17: "f32[4, 32, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_default: "f32[4, 192, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17], 1);  avg_pool2d = convolution_15 = convolution_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [0, 2, 3], correction = 0, keepdim = True);  cat_default = None
        getitem: "f32[1, 192, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 192, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([4, 128, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 28, 28], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
