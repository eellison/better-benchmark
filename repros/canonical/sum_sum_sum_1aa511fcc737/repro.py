"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train_001
Pattern hash: 1aa511fcc737
Shape hash: 18e94aae
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 96, 56, 56], f32), T([128, 96, 56, 56], f32), T([128, 96, 56, 56], f32), T([128, 96, 56, 56], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32), T([], f32), T([128, 96, 56, 56], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32), T([128, 96, 56, 56], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_114: "f32[128, 96, 56, 56]", getitem_117: "f32[128, 96, 56, 56]", where_19: "f32[128, 96, 56, 56]", arg118_1: "f32[128, 96, 56, 56]", arg292_1: "f32[1, 96, 1, 1]", arg119_1: "f32[96]", arg9_1: "f32[96]", full_1: "f32[]", arg116_1: "f32[128, 96, 56, 56]", arg293_1: "f32[1, 96, 1, 1]", arg117_1: "f32[96]", arg8_1: "f32[96]", arg114_1: "f32[128, 96, 56, 56]", arg294_1: "f32[1, 96, 1, 1]", arg115_1: "f32[96]", arg6_1: "f32[96]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(getitem_114, getitem_117);  getitem_114 = getitem_117 = None
        sum_dim_int_list: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_tensor: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(arg118_1, arg292_1);  arg292_1 = None
        mul_tensor: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_19, sub_tensor)
        sum_dim_int_list_1: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06)
        mul_tensor_3: "f32[96]" = torch.ops.aten.mul.Tensor(arg119_1, arg119_1)
        mul_tensor_4: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[96]" = torch.ops.aten.mul.Tensor(arg119_1, arg9_1);  arg9_1 = None
        unsqueeze_default_6: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_19, mul_tensor_6);  where_19 = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg119_1);  sum_dim_int_list_1 = arg119_1 = None
        add_tensor_1: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_7);  add_tensor = mul_tensor_7 = None
        le_scalar: "b8[128, 96, 56, 56]" = torch.ops.aten.le.Scalar(arg118_1, 0);  arg118_1 = None
        where_self: "f32[128, 96, 56, 56]" = torch.ops.aten.where.self(le_scalar, full_1, add_tensor_1);  le_scalar = full_1 = add_tensor_1 = None
        sum_dim_int_list_2: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_3: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(arg116_1, arg293_1);  arg116_1 = arg293_1 = None
        mul_tensor_9: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_3)
        sum_dim_int_list_3: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 2, 3]);  mul_tensor_9 = None
        mul_tensor_10: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.4912308673469386e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_10: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_11: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.4912308673469386e-06)
        mul_tensor_12: "f32[96]" = torch.ops.aten.mul.Tensor(arg117_1, arg117_1)
        mul_tensor_13: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_12: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_13: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_14: "f32[96]" = torch.ops.aten.mul.Tensor(arg117_1, arg8_1);  arg8_1 = None
        unsqueeze_default_15: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_16: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_15: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_15);  mul_tensor_15 = None
        sub_tensor_5: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_16: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        mul_tensor_17: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, arg117_1);  sum_dim_int_list_3 = arg117_1 = None
        sum_dim_int_list_4: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_6: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(arg114_1, arg294_1);  arg114_1 = arg294_1 = None
        mul_tensor_18: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_6)
        sum_dim_int_list_5: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 2, 3]);  mul_tensor_18 = None
        mul_tensor_19: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 2.4912308673469386e-06);  sum_dim_int_list_4 = None
        unsqueeze_default_18: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_19, 0);  mul_tensor_19 = None
        unsqueeze_default_19: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 2);  unsqueeze_default_18 = None
        unsqueeze_default_20: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 3);  unsqueeze_default_19 = None
        mul_tensor_20: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, 2.4912308673469386e-06)
        mul_tensor_21: "f32[96]" = torch.ops.aten.mul.Tensor(arg115_1, arg115_1)
        mul_tensor_22: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        unsqueeze_default_21: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_22, 0);  mul_tensor_22 = None
        unsqueeze_default_22: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 2);  unsqueeze_default_21 = None
        unsqueeze_default_23: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 3);  unsqueeze_default_22 = None
        mul_tensor_23: "f32[96]" = torch.ops.aten.mul.Tensor(arg115_1, arg6_1);  arg6_1 = None
        unsqueeze_default_24: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_23, 0);  mul_tensor_23 = None
        unsqueeze_default_25: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 2);  unsqueeze_default_24 = None
        unsqueeze_default_26: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 3);  unsqueeze_default_25 = None
        mul_tensor_24: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_23);  sub_tensor_6 = unsqueeze_default_23 = None
        sub_tensor_7: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_24);  where_self = mul_tensor_24 = None
        sub_tensor_8: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_7, unsqueeze_default_20);  sub_tensor_7 = unsqueeze_default_20 = None
        mul_tensor_25: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_8, unsqueeze_default_26);  sub_tensor_8 = unsqueeze_default_26 = None
        mul_tensor_26: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, arg115_1);  sum_dim_int_list_5 = arg115_1 = None
        return (mul_tensor_8, mul_tensor_16, mul_tensor_17, mul_tensor_25, mul_tensor_26)

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
