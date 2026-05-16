"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: f88be5a1875f
Shape hash: c2e21c9d
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, cat_51: "f32[4, 832, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_51, [0, 2, 3], correction = 0, keepdim = True);  cat_51 = None
        getitem: "f32[1, 832, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 832, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([4, 832, 7, 7], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
