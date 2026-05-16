"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 04ce07fe6207
Shape hash: 0090b852
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_72: "f32[784, 768]", _shape_param_0, primals_65: "f32[768]", cat_1: "f32[4, 14, 14, 768]", getitem_21: "f32[4, 14, 14, 1]", rsqrt_10: "f32[4, 14, 14, 1]", lt_5: "b8[4, 1, 1, 1]", _shape_param_1, primals_63: "f32[192, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:85 in forward, code: x = self.reduction(x)  # ... H/2 W/2 2*C
        reshape_default: "f32[4, 14, 14, 768]" = torch.ops.aten.reshape.default(mm_72, _shape_param_0);  mm_72 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:84 in forward, code: x = self.norm(x)
        mul_tensor: "f32[4, 14, 14, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_65);  reshape_default = primals_65 = None
        mul_tensor_1: "f32[4, 14, 14, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        sub_tensor: "f32[4, 14, 14, 768]" = torch.ops.aten.sub.Tensor(cat_1, getitem_21);  cat_1 = getitem_21 = None
        mul_tensor_2: "f32[4, 14, 14, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_10);  sub_tensor = None
        mul_tensor_3: "f32[4, 14, 14, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[4, 14, 14, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[4, 14, 14, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[4, 14, 14, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        mul_tensor_5: "f32[4, 14, 14, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:42 in _patch_merging_pad, code: x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        slice_tensor: "f32[4, 14, 14, 192]" = torch.ops.aten.slice.Tensor(mul_tensor_5, 3, 0, 192)
        slice_tensor_1: "f32[4, 14, 14, 192]" = torch.ops.aten.slice.Tensor(mul_tensor_5, 3, 192, 384)
        slice_tensor_2: "f32[4, 14, 14, 192]" = torch.ops.aten.slice.Tensor(mul_tensor_5, 3, 384, 576)
        slice_tensor_3: "f32[4, 14, 14, 192]" = torch.ops.aten.slice.Tensor(mul_tensor_5, 3, 576, 768);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:41 in _patch_merging_pad, code: x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        full_default: "f32[4, 14, 28, 192]" = torch.ops.aten.full.default([4, 14, 28, 192], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[4, 14, 28, 192]" = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_3, 2, 1, 9223372036854775807, 2);  slice_tensor_3 = None
        full_default_1: "f32[4, 28, 28, 192]" = torch.ops.aten.full.default([4, 28, 28, 192], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_1: "f32[4, 28, 28, 192]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_scatter_default, 1, 1, 9223372036854775807, 2);  slice_scatter_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:40 in _patch_merging_pad, code: x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        slice_scatter_default_2: "f32[4, 14, 28, 192]" = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_2, 2, 1, 9223372036854775807, 2);  slice_tensor_2 = None
        slice_scatter_default_3: "f32[4, 28, 28, 192]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_scatter_default_2, 1, 0, 9223372036854775807, 2);  slice_scatter_default_2 = None
        add_tensor: "f32[4, 28, 28, 192]" = torch.ops.aten.add.Tensor(slice_scatter_default_1, slice_scatter_default_3);  slice_scatter_default_1 = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:39 in _patch_merging_pad, code: x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        slice_scatter_default_4: "f32[4, 14, 28, 192]" = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_1, 2, 0, 9223372036854775807, 2);  slice_tensor_1 = None
        slice_scatter_default_5: "f32[4, 28, 28, 192]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_scatter_default_4, 1, 1, 9223372036854775807, 2);  slice_scatter_default_4 = None
        add_tensor_1: "f32[4, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_tensor, slice_scatter_default_5);  add_tensor = slice_scatter_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:38 in _patch_merging_pad, code: x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        slice_scatter_default_6: "f32[4, 14, 28, 192]" = torch.ops.aten.slice_scatter.default(full_default, slice_tensor, 2, 0, 9223372036854775807, 2);  full_default = slice_tensor = None
        slice_scatter_default_7: "f32[4, 28, 28, 192]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_scatter_default_6, 1, 0, 9223372036854775807, 2);  full_default_1 = slice_scatter_default_6 = None
        add_tensor_2: "f32[4, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_scatter_default_7);  add_tensor_1 = slice_scatter_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_5, torch.float32);  lt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor_1: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9454545454545454);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor_6: "f32[4, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_tensor_2, div_tensor_1);  add_tensor_2 = div_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default_1: "f32[3136, 192]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_1);  mul_tensor_6 = _shape_param_1 = None
        permute_default: "f32[768, 192]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_default_1: "f32[192, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([784, 768], dtype=torch.float32, device='cuda'),
    [4, 14, 14, 768],  # _shape_param_0
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1, 1, 1], dtype=torch.bool, device='cuda'),
    [3136, 192],  # _shape_param_1
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
