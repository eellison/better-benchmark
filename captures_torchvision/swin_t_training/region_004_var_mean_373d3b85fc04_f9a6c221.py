"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 373d3b85fc04
Shape hash: f9a6c221
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_45: "f32[196, 768]", _shape_param_0, _shape_param_1, _shape_param_2, inductor_seeds_default: "i64[22]", add_120: "f32[4, 7, 7, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        reshape_default: "f32[4, 49, 768]" = torch.ops.aten.reshape.default(addmm_45, _shape_param_0);  addmm_45 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:219 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_1: "f32[4, 1, 1, 7, 7, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:220 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
        permute_default: "f32[4, 1, 7, 1, 7, 768]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None
        reshape_default_2: "f32[4, 7, 7, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 1, 1, 1]" = torch.ops.prims.inductor_random.default([4, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[4, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.8);  inductor_random_default = None
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.8);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(reshape_default_2, div_tensor);  reshape_default_2 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        add_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.add.Tensor(add_120, mul_tensor);  add_120 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [3], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[4, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([196, 768], dtype=torch.float32, device='cuda'),
    [4, 49, 768],  # _shape_param_0
    [4, 1, 1, 7, 7, 768],  # _shape_param_1
    [4, 7, 7, 768],  # _shape_param_2
    torch.randint(0, 2, [22], dtype=torch.int64, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
