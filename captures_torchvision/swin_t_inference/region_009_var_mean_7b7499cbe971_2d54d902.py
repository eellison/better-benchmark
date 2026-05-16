"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: 7b7499cbe971
Shape hash: 2d54d902
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[49, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:85 in forward, code: x = self.reduction(x)  # ... H/2 W/2 2*C
        reshape_default: "f32[1, 7, 7, 768]" = torch.ops.aten.reshape.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [3], correction = 0, keepdim = True);  reshape_default = None
        getitem: "f32[1, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([49, 768], dtype=torch.float32, device='cuda'),
    [1, 7, 7, 768],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
