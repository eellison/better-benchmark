"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g53
Pattern hash: 57c5e3ecd58f
Shape hash: 0e301bf0
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_108: "f16[128, 96, 56, 56]", getitem_111: "f16[128, 96, 56, 56]", arg85_1: "f16[128, 96, 56, 56]", full: "f16[]", arg83_1: "f16[128, 96, 56, 56]", arg290_1: "f32[1, 96, 1, 1]", arg84_1: "f32[96]", arg6_1: "f32[96]", arg80_1: "f16[128, 96, 56, 56]", arg291_1: "f32[1, 96, 1, 1]", arg81_1: "f32[96]", arg5_1: "f32[96]"):
        # No stacktrace found for following nodes
        add_tensor: "f16[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(getitem_108, getitem_111);  getitem_108 = getitem_111 = None
        le_scalar: "b8[128, 96, 56, 56]" = torch.ops.aten.le.Scalar(arg85_1, 0);  arg85_1 = None
        where_self: "f16[128, 96, 56, 56]" = torch.ops.aten.where.self(le_scalar, full, add_tensor);  le_scalar = full = add_tensor = None
        convert_element_type_default: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(where_self, torch.float32);  where_self = None
        sum_dim_int_list: "f32[96]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [0, 2, 3])
        convert_element_type_default_1: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.float32);  arg83_1 = None
        sub_tensor: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, arg290_1);  convert_element_type_default_1 = arg290_1 = None
        mul_tensor: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor)
        sum_dim_int_list_1: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[96]" = torch.ops.aten.mul.Tensor(arg84_1, arg84_1)
        mul_tensor_4: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[96]" = torch.ops.aten.mul.Tensor(arg84_1, arg6_1);  arg84_1 = arg6_1 = None
        unsqueeze_default_6: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_default, mul_tensor_6);  mul_tensor_6 = None
        sub_tensor_2: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        convert_element_type_default_2: "f16[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.float16);  mul_tensor_7 = None
        sum_dim_int_list_2: "f32[96]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [0, 2, 3])
        convert_element_type_default_3: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.float32);  arg80_1 = None
        sub_tensor_3: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_default_3, arg291_1);  convert_element_type_default_3 = arg291_1 = None
        mul_tensor_8: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor_3)
        sum_dim_int_list_3: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.4912308673469386e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.4912308673469386e-06);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[96]" = torch.ops.aten.mul.Tensor(arg81_1, arg81_1)
        mul_tensor_12: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[96]" = torch.ops.aten.mul.Tensor(arg81_1, arg5_1);  arg81_1 = arg5_1 = None
        unsqueeze_default_15: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_default, mul_tensor_14);  convert_element_type_default = mul_tensor_14 = None
        sub_tensor_5: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        convert_element_type_default_4: "f16[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_tensor_15, torch.float16);  mul_tensor_15 = None
        return (convert_element_type_default_2, convert_element_type_default_4)


def _default_make_inputs():
    return [
    torch.randn(38535168, dtype=torch.float16, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # getitem_108
    torch.randn(38535168, dtype=torch.float16, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # getitem_111
    torch.randn(38535168, dtype=torch.float16, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # arg85_1
    torch.randn([], dtype=torch.float16, device='cuda'),
    torch.randn(38535168, dtype=torch.float16, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # arg83_1
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(38535168, dtype=torch.float16, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # arg80_1
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
