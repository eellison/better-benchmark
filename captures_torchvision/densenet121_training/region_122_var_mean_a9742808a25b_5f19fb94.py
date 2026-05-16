"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: a9742808a25b
Shape hash: 5f19fb94
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_2: "f32[4, 64, 56, 56]", convolution_2: "f32[4, 32, 56, 56]", convolution_4: "f32[4, 32, 56, 56]", convolution_6: "f32[4, 32, 56, 56]", convolution_8: "f32[4, 32, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_default: "f32[4, 192, 56, 56]" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4, convolution_6, convolution_8], 1);  getitem_2 = convolution_2 = convolution_4 = convolution_6 = convolution_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [0, 2, 3], correction = 0, keepdim = True);  cat_default = None
        getitem: "f32[1, 192, 1, 1]" = var_mean_correction[0]
        getitem_3: "f32[1, 192, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_3)



def make_inputs():
    return [
    torch.randn([4, 64, 56, 56], dtype=torch.float32, device='cuda'),
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
