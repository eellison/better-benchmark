"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: a9742808a25b
Shape hash: d0c74152
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, avg_pool2d_1: "f32[4, 256, 14, 14]", convolution_40: "f32[4, 32, 14, 14]", convolution_42: "f32[4, 32, 14, 14]", convolution_44: "f32[4, 32, 14, 14]", convolution_46: "f32[4, 32, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_default: "f32[4, 384, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46], 1);  avg_pool2d_1 = convolution_40 = convolution_42 = convolution_44 = convolution_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [0, 2, 3], correction = 0, keepdim = True);  cat_default = None
        getitem: "f32[1, 384, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([4, 256, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 14, 14], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
