"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: a7cbd072693b
Shape hash: 881ec44a
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_3: "f32[12544, 96]", _shape_param_0, add_5: "f32[4, 56, 56, 96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[4, 56, 56, 96]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_0);  addmm_3 = _shape_param_0 = None
        add_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.add.Tensor(add_5, reshape_default);  add_5 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [3], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[4, 56, 56, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 56, 56, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([12544, 96], dtype=torch.float32, device='cuda'),
    [4, 56, 56, 96],  # _shape_param_0
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
