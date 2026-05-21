"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: 26d1711c064d
Shape hash: 57f88806
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 2560, 7, 7], f32, stride=(125440, 1, 17920, 2560)), T([128, 2560, 1, 1], f32), T([128, 1, 1, 1], f32), T([128, 2560, 7, 7], f32, stride=(125440, 1, 17920, 2560)), T([2560], f32), S([2560]), S([2560]), S([128, 2560, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_44: "f32[128, 2560, 7, 7]", pow_28: "f32[128, 2560, 1, 1]", add_89: "f32[128, 1, 1, 1]", getitem_38: "f32[128, 2560, 7, 7]", primals_155: "f32[2560]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_tensor: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_44, 0.5)
        mul_tensor_1: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_44, 0.7071067811865476)
        erf_default: "f32[128, 2560, 7, 7]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 2560, 7, 7]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_tensor: "f32[128, 2560, 1, 1]" = torch.ops.aten.div.Tensor(pow_28, add_89)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor_3: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, div_tensor)
        mul_scalar: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Scalar(mul_tensor_3, 1);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_38, mul_scalar);  mul_scalar = None
        reshape_default: "f32[1, 2560, 1, 1]" = torch.ops.aten.reshape.default(primals_155, [1, -1, 1, 1]);  primals_155 = None
        mul_scalar_1: "f32[1, 2560, 1, 1]" = torch.ops.aten.mul.Scalar(reshape_default, 1);  reshape_default = None
        mul_tensor_5: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_38, mul_scalar_1);  mul_scalar_1 = None
        sum_dim_int_list: "f32[1, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(getitem_38, [0, 2, 3], True)
        sum_dim_int_list_1: "f32[1, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3], True);  mul_tensor_4 = None
        mul_tensor_6: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_tensor_2)
        mul_tensor_7: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_5, div_tensor);  mul_tensor_5 = None
        sum_dim_int_list_2: "f32[128, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2, 3], True);  mul_tensor_6 = None
        add_tensor_1: "f32[128, 2560, 7, 7]" = torch.ops.aten.add.Tensor(getitem_38, mul_tensor_7);  getitem_38 = mul_tensor_7 = None
        reshape_default_1: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_0);  sum_dim_int_list_1 = _shape_param_0 = None
        reshape_default_2: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_tensor_1: "f32[128, 2560, 1, 1]" = torch.ops.aten.div.Tensor(div_tensor, add_89);  div_tensor = None
        neg_default: "f32[128, 2560, 1, 1]" = torch.ops.aten.neg.default(sum_dim_int_list_2)
        mul_tensor_8: "f32[128, 2560, 1, 1]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[128, 2560, 1, 1]" = torch.ops.aten.div.Tensor(sum_dim_int_list_2, add_89);  sum_dim_int_list_2 = add_89 = None
        sum_dim_int_list_3: "f32[128, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [1], True);  mul_tensor_8 = None
        expand_default: "f32[128, 2560, 1, 1]" = torch.ops.aten.expand.default(sum_dim_int_list_3, _shape_param_2);  sum_dim_int_list_3 = _shape_param_2 = None
        div_scalar: "f32[128, 2560, 1, 1]" = torch.ops.aten.div.Scalar(expand_default, 2560);  expand_default = None
        add_tensor_2: "f32[128, 2560, 1, 1]" = torch.ops.aten.add.Tensor(div_tensor_2, div_scalar);  div_tensor_2 = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_tensor_3: "f32[128, 2560, 7, 7]" = torch.ops.aten.div.Tensor(mul_tensor_2, pow_28);  mul_tensor_2 = None
        eq_scalar: "b8[128, 2560, 1, 1]" = torch.ops.aten.eq.Scalar(pow_28, 0);  pow_28 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 2560, 7, 7]" = torch.ops.aten.where.self(eq_scalar, full_default, div_tensor_3);  eq_scalar = full_default = div_tensor_3 = None
        clone_default: "f32[128, 2560, 7, 7]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        mul_tensor_9: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor_2, clone_default);  add_tensor_2 = clone_default = None
        add_tensor_3: "f32[128, 2560, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_9);  add_tensor_1 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_tensor_10: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_11: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_44, convolution_44)
        mul_tensor_12: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_11, -0.5);  mul_tensor_11 = None
        exp_default: "f32[128, 2560, 7, 7]" = torch.ops.aten.exp.default(mul_tensor_12);  mul_tensor_12 = None
        mul_tensor_13: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_14: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_44, mul_tensor_13);  convolution_44 = mul_tensor_13 = None
        add_tensor_4: "f32[128, 2560, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_14);  mul_tensor_10 = mul_tensor_14 = None
        mul_tensor_15: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor_3, add_tensor_4);  add_tensor_3 = add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_dim_int_list_4: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 2, 3]);  mul_tensor_15 = None
        return (reshape_default_1, reshape_default_2, sum_dim_int_list_4)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
