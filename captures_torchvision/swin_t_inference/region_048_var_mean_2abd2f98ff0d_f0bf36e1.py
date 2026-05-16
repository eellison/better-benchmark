"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: 2abd2f98ff0d
Shape hash: f0bf36e1
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_1: "f32[3136, 96]", _shape_param_0, _shape_param_1, _shape_param_2, add_1: "f32[1, 56, 56, 96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        reshape_default: "f32[64, 49, 96]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_0);  addmm_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:219 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_1: "f32[1, 8, 8, 7, 7, 96]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:220 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
        permute_default: "f32[1, 8, 7, 8, 7, 96]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None
        clone_default: "f32[1, 8, 7, 8, 7, 96]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[1, 56, 56, 96]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        add_tensor: "f32[1, 56, 56, 96]" = torch.ops.aten.add.Tensor(add_1, reshape_default_2);  add_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [3], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[1, 56, 56, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 56, 56, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([3136, 96], dtype=torch.float32, device='cuda'),
    [64, 49, 96],  # _shape_param_0
    [1, 8, 8, 7, 7, 96],  # _shape_param_1
    [1, 56, 56, 96],  # _shape_param_2
    torch.randn([1, 56, 56, 96], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
