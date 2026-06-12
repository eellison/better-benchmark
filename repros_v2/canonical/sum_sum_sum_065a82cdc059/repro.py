"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train
Pattern hash: 065a82cdc059
Shape hash: f665655f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 96, 56, 56]", arg1_1: "bf16[128, 96, 56, 56]", arg2_1: "f32[128, 96, 56, 56]", arg3_1: "bf16[128, 96, 56, 56]", arg4_1: "f32[1, 96, 1, 1]", arg5_1: "f32[96]", arg6_1: "f32[96]", arg7_1: "bf16[]", arg8_1: "bf16[128, 96, 56, 56]", arg9_1: "f32[1, 96, 1, 1]", arg10_1: "f32[96]", arg11_1: "f32[96]", arg12_1: "bf16[128, 96, 56, 56]", arg13_1: "f32[1, 96, 1, 1]", arg14_1: "f32[96]", arg15_1: "f32[96]"):
        # No stacktrace found for following nodes
        add: "bf16[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        sum_1: "f32[96]" = torch.ops.aten.sum.dim_IntList(arg2_1, [0, 2, 3])
        convert_element_type: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32)
        sub: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type, arg4_1);  convert_element_type = arg4_1 = None
        mul: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(arg2_1, sub)
        sum_2: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3]);  mul = None
        mul_1: "f32[96]" = torch.ops.aten.mul.Tensor(sum_1, 2.4912308673469386e-06)
        unsqueeze: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1, 0);  mul_1 = None
        unsqueeze_1: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_2: "f32[96]" = torch.ops.aten.mul.Tensor(sum_2, 2.4912308673469386e-06)
        mul_3: "f32[96]" = torch.ops.aten.mul.Tensor(arg5_1, arg5_1)
        mul_4: "f32[96]" = torch.ops.aten.mul.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze_3: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_4: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_5: "f32[96]" = torch.ops.aten.mul.Tensor(arg5_1, arg6_1);  arg6_1 = None
        unsqueeze_6: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_7: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_6: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_5);  sub = unsqueeze_5 = None
        sub_1: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(arg2_1, mul_6);  arg2_1 = mul_6 = None
        sub_2: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2);  sub_1 = unsqueeze_2 = None
        mul_7: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_8);  sub_2 = unsqueeze_8 = None
        mul_8: "f32[96]" = torch.ops.aten.mul.Tensor(sum_2, arg5_1);  sum_2 = arg5_1 = None
        convert_element_type_1: "bf16[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        add_1: "bf16[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(add, convert_element_type_1);  add = convert_element_type_1 = None
        le: "b8[128, 96, 56, 56]" = torch.ops.aten.le.Scalar(arg3_1, 0);  arg3_1 = None
        where: "bf16[128, 96, 56, 56]" = torch.ops.aten.where.self(le, arg7_1, add_1);  le = arg7_1 = add_1 = None
        convert_element_type_2: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sum_3: "f32[96]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2, 3])
        convert_element_type_3: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        sub_3: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_3, arg9_1);  convert_element_type_3 = arg9_1 = None
        mul_9: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_3)
        sum_4: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_9, [0, 2, 3]);  mul_9 = None
        mul_10: "f32[96]" = torch.ops.aten.mul.Tensor(sum_3, 2.4912308673469386e-06)
        unsqueeze_9: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_10, 0);  mul_10 = None
        unsqueeze_10: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        mul_11: "f32[96]" = torch.ops.aten.mul.Tensor(sum_4, 2.4912308673469386e-06)
        mul_12: "f32[96]" = torch.ops.aten.mul.Tensor(arg10_1, arg10_1)
        mul_13: "f32[96]" = torch.ops.aten.mul.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_12: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_13, 0);  mul_13 = None
        unsqueeze_13: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        unsqueeze_14: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None
        mul_14: "f32[96]" = torch.ops.aten.mul.Tensor(arg10_1, arg11_1);  arg11_1 = None
        unsqueeze_15: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_14, 0);  mul_14 = None
        unsqueeze_16: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
        unsqueeze_17: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
        mul_15: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_14);  sub_3 = unsqueeze_14 = None
        sub_4: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_15);  mul_15 = None
        sub_5: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_4, unsqueeze_11);  sub_4 = unsqueeze_11 = None
        mul_16: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_17);  sub_5 = unsqueeze_17 = None
        mul_17: "f32[96]" = torch.ops.aten.mul.Tensor(sum_4, arg10_1);  sum_4 = arg10_1 = None
        convert_element_type_4: "bf16[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        sum_5: "f32[96]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2, 3])
        convert_element_type_5: "f32[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None
        sub_6: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_5, arg13_1);  convert_element_type_5 = arg13_1 = None
        mul_18: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_6)
        sum_6: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_18, [0, 2, 3]);  mul_18 = None
        mul_19: "f32[96]" = torch.ops.aten.mul.Tensor(sum_5, 2.4912308673469386e-06)
        unsqueeze_18: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_19, 0);  mul_19 = None
        unsqueeze_19: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
        unsqueeze_20: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 3);  unsqueeze_19 = None
        mul_20: "f32[96]" = torch.ops.aten.mul.Tensor(sum_6, 2.4912308673469386e-06)
        mul_21: "f32[96]" = torch.ops.aten.mul.Tensor(arg14_1, arg14_1)
        mul_22: "f32[96]" = torch.ops.aten.mul.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
        unsqueeze_21: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_22, 0);  mul_22 = None
        unsqueeze_22: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 2);  unsqueeze_21 = None
        unsqueeze_23: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 3);  unsqueeze_22 = None
        mul_23: "f32[96]" = torch.ops.aten.mul.Tensor(arg14_1, arg15_1);  arg15_1 = None
        unsqueeze_24: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_23, 0);  mul_23 = None
        unsqueeze_25: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
        unsqueeze_26: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 3);  unsqueeze_25 = None
        mul_24: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_23);  sub_6 = unsqueeze_23 = None
        sub_7: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_24);  convert_element_type_2 = mul_24 = None
        sub_8: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_7, unsqueeze_20);  sub_7 = unsqueeze_20 = None
        mul_25: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_26);  sub_8 = unsqueeze_26 = None
        mul_26: "f32[96]" = torch.ops.aten.mul.Tensor(sum_6, arg14_1);  sum_6 = arg14_1 = None
        convert_element_type_6: "bf16[128, 96, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        return (sum_1, mul_8, sum_3, mul_17, convert_element_type_4, sum_5, mul_26, convert_element_type_6)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
