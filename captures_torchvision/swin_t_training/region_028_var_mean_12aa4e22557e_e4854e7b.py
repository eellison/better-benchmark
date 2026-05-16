"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 12aa4e22557e
Shape hash: e4854e7b
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_15: "f32[3136, 192]", _shape_param_0, inductor_seeds_default: "i64[22]", add_41: "f32[4, 28, 28, 192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[4, 28, 28, 192]" = torch.ops.aten.reshape.default(addmm_15, _shape_param_0);  addmm_15 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 1, 1, 1]" = torch.ops.prims.inductor_random.default([4, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[4, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.9454545454545454);  inductor_random_default = None
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9454545454545454);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor: "f32[4, 28, 28, 192]" = torch.ops.aten.mul.Tensor(reshape_default, div_tensor);  reshape_default = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        add_tensor: "f32[4, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_41, mul_tensor);  add_41 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:38 in _patch_merging_pad, code: x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        slice_tensor: "f32[4, 14, 28, 192]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 0, 9223372036854775807, 2)
        slice_tensor_1: "f32[4, 14, 14, 192]" = torch.ops.aten.slice.Tensor(slice_tensor, 2, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:39 in _patch_merging_pad, code: x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        slice_tensor_2: "f32[4, 14, 28, 192]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 1, 9223372036854775807, 2);  add_tensor = None
        slice_tensor_3: "f32[4, 14, 14, 192]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 2, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:40 in _patch_merging_pad, code: x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        slice_tensor_4: "f32[4, 14, 14, 192]" = torch.ops.aten.slice.Tensor(slice_tensor, 2, 1, 9223372036854775807, 2);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:41 in _patch_merging_pad, code: x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        slice_tensor_5: "f32[4, 14, 14, 192]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 2, 1, 9223372036854775807, 2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:42 in _patch_merging_pad, code: x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        cat_default: "f32[4, 14, 14, 768]" = torch.ops.aten.cat.default([slice_tensor_1, slice_tensor_3, slice_tensor_4, slice_tensor_5], -1);  slice_tensor_1 = slice_tensor_3 = slice_tensor_4 = slice_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:84 in forward, code: x = self.norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [3], correction = 0, keepdim = True);  cat_default = None
        getitem: "f32[4, 14, 14, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 14, 14, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([3136, 192], dtype=torch.float32, device='cuda'),
    [4, 28, 28, 192],  # _shape_param_0
    torch.randint(0, 2, [22], dtype=torch.int64, device='cuda'),
    torch.randn([4, 28, 28, 192], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
