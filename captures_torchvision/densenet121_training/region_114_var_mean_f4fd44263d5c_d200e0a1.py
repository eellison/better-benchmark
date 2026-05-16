"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: f4fd44263d5c
Shape hash: d200e0a1
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_13: "f32[4, 128, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        avg_pool2d_default: "f32[4, 128, 28, 28]" = torch.ops.aten.avg_pool2d.default(convolution_13, [2, 2], [2, 2]);  convolution_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        var_mean_correction = torch.ops.aten.var_mean.correction(avg_pool2d_default, [0, 2, 3], correction = 0, keepdim = True);  avg_pool2d_default = None
        getitem: "f32[1, 128, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 128, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([4, 128, 56, 56], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
