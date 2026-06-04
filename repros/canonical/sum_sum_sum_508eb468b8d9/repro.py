"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train_001
Pattern hash: 508eb468b8d9
Shape hash: 7bb57998
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 80, 56, 56], f32), T([128, 80, 56, 56], f32), T([80], f32), T([128, 80, 56, 56], f32), T([128, 56, 56, 1], f32), T([128, 56, 56, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, add_77: "f32[128, 80, 56, 56]", getitem_132: "f32[128, 80, 56, 56]", arg2_1: "f32[80]", arg81_1: "f32[128, 80, 56, 56]", arg82_1: "f32[128, 56, 56, 1]", arg83_1: "f32[128, 56, 56, 1]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[128, 80, 56, 56]" = torch.ops.aten.add.Tensor(add_77, getitem_132);  add_77 = getitem_132 = None
        permute_default: "f32[128, 56, 56, 80]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 3, 1]);  add_tensor = None
        mul_tensor: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(permute_default, arg2_1);  arg2_1 = None
        mul_tensor_1: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(mul_tensor, 80)
        sum_dim_int_list: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        permute_default_1: "f32[128, 56, 56, 80]" = torch.ops.aten.permute.default(arg81_1, [0, 2, 3, 1]);  arg81_1 = None
        sub_tensor: "f32[128, 56, 56, 80]" = torch.ops.aten.sub.Tensor(permute_default_1, arg82_1);  permute_default_1 = arg82_1 = None
        mul_tensor_2: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(sub_tensor, arg83_1);  sub_tensor = None
        mul_tensor_3: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 56, 56, 80]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 56, 56, 80]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(arg83_1, 80);  arg83_1 = None
        mul_tensor_5: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(permute_default, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1, 2]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[80]" = torch.ops.aten.sum.dim_IntList(permute_default, [0, 1, 2]);  permute_default = None
        permute_default_2: "f32[128, 80, 56, 56]" = torch.ops.aten.permute.default(mul_tensor_5, [0, 3, 1, 2]);  mul_tensor_5 = None
        sum_dim_int_list_4: "f32[80]" = torch.ops.aten.sum.dim_IntList(permute_default_2, [0, 2, 3]);  permute_default_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, sum_dim_int_list_4)

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
