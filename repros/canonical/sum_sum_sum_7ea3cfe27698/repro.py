"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 7ea3cfe27698
Shape hash: 71ddf97f
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
    def forward(self, arg0_1: "bf16[128, 1280, 8, 8]", arg1_1: "bf16[128, 1280, 8, 8]", arg2_1: "bf16[128, 1280, 8, 8]", arg3_1: "bf16[128, 1280, 8, 8]", arg4_1: "bf16[128, 1280, 8, 8]", arg5_1: "bf16[128, 192, 8, 8]", arg6_1: "f32[1, 192, 1, 1]", arg7_1: "f32[1, 192, 1, 1]", arg8_1: "f32[192]", arg9_1: "f32[192]", arg10_1: "bf16[]", arg11_1: "bf16[128, 320, 8, 8]", arg12_1: "f32[1, 320, 1, 1]", arg13_1: "f32[1, 320, 1, 1]", arg14_1: "f32[320]", arg15_1: "f32[320]"):
        # No stacktrace found for following nodes
        avg_pool2d_backward: "bf16[128, 1280, 8, 8]" = torch.ops.aten.avg_pool2d_backward.default(arg0_1, arg1_1, [3, 3], [1, 1], [1, 1], False, True, None);  arg0_1 = arg1_1 = None
        add: "bf16[128, 1280, 8, 8]" = torch.ops.aten.add.Tensor(avg_pool2d_backward, arg2_1);  avg_pool2d_backward = arg2_1 = None
        add_1: "bf16[128, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        add_2: "bf16[128, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_1, arg4_1);  add_1 = arg4_1 = None
        slice_1: "bf16[128, 320, 8, 8]" = torch.ops.aten.slice.Tensor(add_2, 1, 0, 320)
        slice_2: "bf16[128, 192, 8, 8]" = torch.ops.aten.slice.Tensor(add_2, 1, 320, 512)
        sub: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(arg5_1, arg6_1)
        mul: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub, arg7_1);  sub = None
        unsqueeze: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1)
        unsqueeze_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[128, 192, 8, 8]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        relu: "bf16[128, 192, 8, 8]" = torch.ops.aten.relu.default(convert_element_type);  convert_element_type = None
        le: "b8[128, 192, 8, 8]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where: "bf16[128, 192, 8, 8]" = torch.ops.aten.where.self(le, arg10_1, slice_2);  le = slice_2 = None
        convert_element_type_1: "f32[128, 192, 8, 8]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[192]" = torch.ops.aten.squeeze.dims(arg6_1, [0, 2, 3]);  arg6_1 = None
        unsqueeze_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sum_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [0, 2, 3])
        convert_element_type_2: "f32[128, 192, 8, 8]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        sub_1: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_6);  convert_element_type_2 = unsqueeze_6 = None
        mul_2: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub_1)
        sum_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 2, 3]);  mul_2 = None
        mul_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_1, 0.0001220703125)
        unsqueeze_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_3, 0);  mul_3 = None
        unsqueeze_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_2, 0.0001220703125)
        squeeze_1: "f32[192]" = torch.ops.aten.squeeze.dims(arg7_1, [0, 2, 3]);  arg7_1 = None
        mul_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_7: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_1, arg8_1);  arg8_1 = None
        unsqueeze_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_7, 0);  mul_7 = None
        unsqueeze_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_8: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_12);  sub_1 = unsqueeze_12 = None
        sub_2: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_8);  convert_element_type_1 = mul_8 = None
        sub_3: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_9);  sub_2 = unsqueeze_9 = None
        mul_9: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_15);  sub_3 = unsqueeze_15 = None
        mul_10: "f32[192]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_3: "bf16[128, 192, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        sub_4: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(arg11_1, arg12_1)
        mul_11: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_4, arg13_1);  sub_4 = None
        unsqueeze_16: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1)
        unsqueeze_17: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_12: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_17);  mul_11 = unsqueeze_17 = None
        unsqueeze_18: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_19: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_4: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_19);  mul_12 = unsqueeze_19 = None
        convert_element_type_4: "bf16[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        relu_1: "bf16[128, 320, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_4);  convert_element_type_4 = None
        le_1: "b8[128, 320, 8, 8]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_1: "bf16[128, 320, 8, 8]" = torch.ops.aten.where.self(le_1, arg10_1, slice_1);  le_1 = arg10_1 = slice_1 = None
        convert_element_type_5: "f32[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        squeeze_2: "f32[320]" = torch.ops.aten.squeeze.dims(arg12_1, [0, 2, 3]);  arg12_1 = None
        unsqueeze_20: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
        unsqueeze_21: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 2);  unsqueeze_20 = None
        unsqueeze_22: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 3);  unsqueeze_21 = None
        sum_3: "f32[320]" = torch.ops.aten.sum.dim_IntList(convert_element_type_5, [0, 2, 3])
        convert_element_type_6: "f32[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float32);  arg11_1 = None
        sub_5: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_6, unsqueeze_22);  convert_element_type_6 = unsqueeze_22 = None
        mul_13: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_5, sub_5)
        sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_13, [0, 2, 3]);  mul_13 = None
        mul_14: "f32[320]" = torch.ops.aten.mul.Tensor(sum_3, 0.0001220703125)
        unsqueeze_23: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_14, 0);  mul_14 = None
        unsqueeze_24: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 2);  unsqueeze_23 = None
        unsqueeze_25: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 3);  unsqueeze_24 = None
        mul_15: "f32[320]" = torch.ops.aten.mul.Tensor(sum_4, 0.0001220703125)
        squeeze_3: "f32[320]" = torch.ops.aten.squeeze.dims(arg13_1, [0, 2, 3]);  arg13_1 = None
        mul_16: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
        mul_17: "f32[320]" = torch.ops.aten.mul.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        unsqueeze_26: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_17, 0);  mul_17 = None
        unsqueeze_27: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 2);  unsqueeze_26 = None
        unsqueeze_28: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 3);  unsqueeze_27 = None
        mul_18: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_3, arg14_1);  arg14_1 = None
        unsqueeze_29: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_18, 0);  mul_18 = None
        unsqueeze_30: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, 2);  unsqueeze_29 = None
        unsqueeze_31: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 3);  unsqueeze_30 = None
        mul_19: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_28);  sub_5 = unsqueeze_28 = None
        sub_6: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_5, mul_19);  convert_element_type_5 = mul_19 = None
        sub_7: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(sub_6, unsqueeze_25);  sub_6 = unsqueeze_25 = None
        mul_20: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_31);  sub_7 = unsqueeze_31 = None
        mul_21: "f32[320]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_3);  sum_4 = squeeze_3 = None
        convert_element_type_7: "bf16[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        return (add_2, sum_1, mul_10, convert_element_type_3, sum_3, mul_21, convert_element_type_7)



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
