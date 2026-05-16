"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: a7cbd072693b
Shape hash: eaf77e5b
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f32[49, 768]", _shape_param_0, add_124: "f32[1, 7, 7, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[1, 7, 7, 768]" = torch.ops.aten.reshape.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None
        add_tensor: "f32[1, 7, 7, 768]" = torch.ops.aten.add.Tensor(add_124, reshape_default);  add_124 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:609 in forward, code: x = self.norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [3], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[1, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([49, 768], dtype=torch.float32, device='cuda'),
    [1, 7, 7, 768],  # _shape_param_0
    torch.randn([1, 7, 7, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
