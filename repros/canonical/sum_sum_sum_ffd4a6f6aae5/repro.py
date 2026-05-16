"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g53
Pattern hash: ffd4a6f6aae5
Shape hash: e91d1fe0
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_114: "f16[128, 96, 56, 56]", getitem_117: "f16[128, 96, 56, 56]", convert_element_type_169: "f32[128, 96, 56, 56]", arg77_1: "f16[128, 96, 56, 56]", arg292_1: "f32[1, 96, 1, 1]", arg78_1: "f32[96]", arg4_1: "f32[96]", full: "f16[]", arg75_1: "f16[128, 96, 56, 56]", arg293_1: "f32[1, 96, 1, 1]", arg76_1: "f32[96]", arg3_1: "f32[96]", arg72_1: "f16[128, 96, 56, 56]", arg294_1: "f32[1, 96, 1, 1]", arg73_1: "f32[96]", arg2_1: "f32[96]"):
        # No stacktrace found for following nodes
        add_tensor: "f16[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(getitem_114, getitem_117);  getitem_114 = getitem_117 = None
        sum_dim_int_list: "f32[96]" = torch.ops.aten.sum.dim_IntList(convert_element_type_169, [0, 2, 3])
        convert_element_type_default: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.float32)
        sub_tensor: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_default, arg292_1);  convert_element_type_default = arg292_1 = None
        mul_tensor: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_169, sub_tensor)
        sum_dim_int_list_1: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[96]" = torch.ops.aten.mul.Tensor(arg78_1, arg78_1)
        mul_tensor_4: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[96]" = torch.ops.aten.mul.Tensor(arg78_1, arg4_1);  arg78_1 = arg4_1 = None
        unsqueeze_default_6: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_169, mul_tensor_6);  convert_element_type_169 = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        convert_element_type_default_1: "f16[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.float16);  mul_tensor_7 = None
        add_tensor_1: "f16[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_1);  add_tensor = convert_element_type_default_1 = None
        le_scalar: "b8[128, 96, 56, 56]" = torch.ops.aten.le.Scalar(arg77_1, 0);  arg77_1 = None
        where_self: "f16[128, 96, 56, 56]" = torch.ops.aten.where.self(le_scalar, full, add_tensor_1);  le_scalar = full = add_tensor_1 = None
        convert_element_type_default_2: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(where_self, torch.float32);  where_self = None
        sum_dim_int_list_2: "f32[96]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 2, 3])
        convert_element_type_default_3: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(arg75_1, torch.float32);  arg75_1 = None
        sub_tensor_3: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_default_3, arg293_1);  convert_element_type_default_3 = arg293_1 = None
        mul_tensor_8: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, sub_tensor_3)
        sum_dim_int_list_3: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.4912308673469386e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.4912308673469386e-06);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[96]" = torch.ops.aten.mul.Tensor(arg76_1, arg76_1)
        mul_tensor_12: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[96]" = torch.ops.aten.mul.Tensor(arg76_1, arg3_1);  arg76_1 = arg3_1 = None
        unsqueeze_default_15: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, mul_tensor_14);  mul_tensor_14 = None
        sub_tensor_5: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        convert_element_type_default_4: "f16[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_tensor_15, torch.float16);  mul_tensor_15 = None
        sum_dim_int_list_4: "f32[96]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 2, 3])
        convert_element_type_default_5: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.float32);  arg72_1 = None
        sub_tensor_6: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_default_5, arg294_1);  convert_element_type_default_5 = arg294_1 = None
        mul_tensor_16: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, sub_tensor_6)
        sum_dim_int_list_5: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 2, 3]);  mul_tensor_16 = None
        mul_tensor_17: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 2.4912308673469386e-06);  sum_dim_int_list_4 = None
        unsqueeze_default_18: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_19: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 2);  unsqueeze_default_18 = None
        unsqueeze_default_20: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 3);  unsqueeze_default_19 = None
        mul_tensor_18: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, 2.4912308673469386e-06);  sum_dim_int_list_5 = None
        mul_tensor_19: "f32[96]" = torch.ops.aten.mul.Tensor(arg73_1, arg73_1)
        mul_tensor_20: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        unsqueeze_default_21: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_20, 0);  mul_tensor_20 = None
        unsqueeze_default_22: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 2);  unsqueeze_default_21 = None
        unsqueeze_default_23: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 3);  unsqueeze_default_22 = None
        mul_tensor_21: "f32[96]" = torch.ops.aten.mul.Tensor(arg73_1, arg2_1);  arg73_1 = arg2_1 = None
        unsqueeze_default_24: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_21, 0);  mul_tensor_21 = None
        unsqueeze_default_25: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 2);  unsqueeze_default_24 = None
        unsqueeze_default_26: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 3);  unsqueeze_default_25 = None
        mul_tensor_22: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_23);  sub_tensor_6 = unsqueeze_default_23 = None
        sub_tensor_7: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, mul_tensor_22);  convert_element_type_default_2 = mul_tensor_22 = None
        sub_tensor_8: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_7, unsqueeze_default_20);  sub_tensor_7 = unsqueeze_default_20 = None
        mul_tensor_23: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_8, unsqueeze_default_26);  sub_tensor_8 = unsqueeze_default_26 = None
        convert_element_type_default_6: "f16[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_tensor_23, torch.float16);  mul_tensor_23 = None
        return (convert_element_type_default_4, convert_element_type_default_6)


def _default_make_inputs():
    return [
    torch.randn(38535168, dtype=torch.float16, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # getitem_114
    torch.randn(38535168, dtype=torch.float16, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # getitem_117
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # convert_element_type_169
    torch.randn(38535168, dtype=torch.float16, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # arg77_1
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float16, device='cuda'),
    torch.randn(38535168, dtype=torch.float16, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # arg75_1
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(38535168, dtype=torch.float16, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # arg72_1
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
