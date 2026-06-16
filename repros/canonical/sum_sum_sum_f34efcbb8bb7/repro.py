"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: f34efcbb8bb7
Shape hash: 32065934
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
    def forward(self, arg0_1: "bf16[128, 2048, 8, 8]", arg1_1: "bf16[128, 2048, 8, 8]", arg2_1: "bf16[128, 2048, 8, 8]", arg3_1: "bf16[128, 2048, 8, 8]", arg4_1: "bf16[128, 2048, 8, 8]", arg5_1: "bf16[128, 192, 8, 8]", arg6_1: "f32[1, 192, 1, 1]", arg7_1: "f32[1, 192, 1, 1]", arg8_1: "f32[192]", arg9_1: "f32[192]", arg10_1: "bf16[]", arg11_1: "bf16[128, 384, 8, 8]", arg12_1: "f32[1, 384, 1, 1]", arg13_1: "f32[1, 384, 1, 1]", arg14_1: "f32[384]", arg15_1: "f32[384]", arg16_1: "bf16[128, 384, 8, 8]", arg17_1: "f32[1, 384, 1, 1]", arg18_1: "f32[1, 384, 1, 1]", arg19_1: "f32[384]", arg20_1: "f32[384]", arg21_1: "bf16[128, 384, 8, 8]", arg22_1: "f32[1, 384, 1, 1]", arg23_1: "f32[1, 384, 1, 1]", arg24_1: "f32[384]", arg25_1: "f32[384]", arg26_1: "bf16[128, 384, 8, 8]", arg27_1: "f32[1, 384, 1, 1]", arg28_1: "f32[1, 384, 1, 1]", arg29_1: "f32[384]", arg30_1: "f32[384]", arg31_1: "bf16[128, 320, 8, 8]", arg32_1: "f32[1, 320, 1, 1]", arg33_1: "f32[1, 320, 1, 1]", arg34_1: "f32[320]", arg35_1: "f32[320]"):
        # No stacktrace found for following nodes
        avg_pool2d_backward: "bf16[128, 2048, 8, 8]" = torch.ops.aten.avg_pool2d_backward.default(arg0_1, arg1_1, [3, 3], [1, 1], [1, 1], False, True, None);  arg0_1 = arg1_1 = None
        add: "bf16[128, 2048, 8, 8]" = torch.ops.aten.add.Tensor(avg_pool2d_backward, arg2_1);  avg_pool2d_backward = arg2_1 = None
        add_1: "bf16[128, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        add_2: "bf16[128, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_1, arg4_1);  add_1 = arg4_1 = None
        slice_1: "bf16[128, 320, 8, 8]" = torch.ops.aten.slice.Tensor(add_2, 1, 0, 320)
        slice_2: "bf16[128, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_2, 1, 320, 1088)
        slice_3: "bf16[128, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_2, 1, 1088, 1856)
        slice_4: "bf16[128, 192, 8, 8]" = torch.ops.aten.slice.Tensor(add_2, 1, 1856, 2048);  add_2 = None
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
        where: "bf16[128, 192, 8, 8]" = torch.ops.aten.where.self(le, arg10_1, slice_4);  le = slice_4 = None
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
        slice_5: "bf16[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 384)
        slice_6: "bf16[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_3, 1, 384, 768);  slice_3 = None
        sub_4: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg11_1, arg12_1)
        mul_11: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_4, arg13_1);  sub_4 = None
        unsqueeze_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1)
        unsqueeze_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_12: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_17);  mul_11 = unsqueeze_17 = None
        unsqueeze_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_4: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_19);  mul_12 = unsqueeze_19 = None
        convert_element_type_4: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        relu_1: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_4);  convert_element_type_4 = None
        le_1: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_1: "bf16[128, 384, 8, 8]" = torch.ops.aten.where.self(le_1, arg10_1, slice_6);  le_1 = slice_6 = None
        convert_element_type_5: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        squeeze_2: "f32[384]" = torch.ops.aten.squeeze.dims(arg12_1, [0, 2, 3]);  arg12_1 = None
        unsqueeze_20: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
        unsqueeze_21: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 2);  unsqueeze_20 = None
        unsqueeze_22: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 3);  unsqueeze_21 = None
        sum_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(convert_element_type_5, [0, 2, 3])
        convert_element_type_6: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float32);  arg11_1 = None
        sub_5: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_6, unsqueeze_22);  convert_element_type_6 = unsqueeze_22 = None
        mul_13: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_5, sub_5)
        sum_4: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_13, [0, 2, 3]);  mul_13 = None
        mul_14: "f32[384]" = torch.ops.aten.mul.Tensor(sum_3, 0.0001220703125)
        unsqueeze_23: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_14, 0);  mul_14 = None
        unsqueeze_24: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 2);  unsqueeze_23 = None
        unsqueeze_25: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 3);  unsqueeze_24 = None
        mul_15: "f32[384]" = torch.ops.aten.mul.Tensor(sum_4, 0.0001220703125)
        squeeze_3: "f32[384]" = torch.ops.aten.squeeze.dims(arg13_1, [0, 2, 3]);  arg13_1 = None
        mul_16: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
        mul_17: "f32[384]" = torch.ops.aten.mul.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        unsqueeze_26: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_17, 0);  mul_17 = None
        unsqueeze_27: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 2);  unsqueeze_26 = None
        unsqueeze_28: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 3);  unsqueeze_27 = None
        mul_18: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_3, arg14_1);  arg14_1 = None
        unsqueeze_29: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_18, 0);  mul_18 = None
        unsqueeze_30: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, 2);  unsqueeze_29 = None
        unsqueeze_31: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 3);  unsqueeze_30 = None
        mul_19: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_28);  sub_5 = unsqueeze_28 = None
        sub_6: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_5, mul_19);  convert_element_type_5 = mul_19 = None
        sub_7: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_6, unsqueeze_25);  sub_6 = unsqueeze_25 = None
        mul_20: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_31);  sub_7 = unsqueeze_31 = None
        mul_21: "f32[384]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_3);  sum_4 = squeeze_3 = None
        convert_element_type_7: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        sub_8: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg16_1, arg17_1)
        mul_22: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_8, arg18_1);  sub_8 = None
        unsqueeze_32: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1)
        unsqueeze_33: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_23: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_33);  mul_22 = unsqueeze_33 = None
        unsqueeze_34: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_35: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_5: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_35);  mul_23 = unsqueeze_35 = None
        convert_element_type_8: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        relu_2: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_8);  convert_element_type_8 = None
        le_2: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_2: "bf16[128, 384, 8, 8]" = torch.ops.aten.where.self(le_2, arg10_1, slice_5);  le_2 = slice_5 = None
        convert_element_type_9: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        squeeze_4: "f32[384]" = torch.ops.aten.squeeze.dims(arg17_1, [0, 2, 3]);  arg17_1 = None
        unsqueeze_36: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_4, 0);  squeeze_4 = None
        unsqueeze_37: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
        unsqueeze_38: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 3);  unsqueeze_37 = None
        sum_5: "f32[384]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [0, 2, 3])
        convert_element_type_10: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.float32);  arg16_1 = None
        sub_9: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_10, unsqueeze_38);  convert_element_type_10 = unsqueeze_38 = None
        mul_24: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_9, sub_9)
        sum_6: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_24, [0, 2, 3]);  mul_24 = None
        mul_25: "f32[384]" = torch.ops.aten.mul.Tensor(sum_5, 0.0001220703125)
        unsqueeze_39: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_25, 0);  mul_25 = None
        unsqueeze_40: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 2);  unsqueeze_39 = None
        unsqueeze_41: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 3);  unsqueeze_40 = None
        mul_26: "f32[384]" = torch.ops.aten.mul.Tensor(sum_6, 0.0001220703125)
        squeeze_5: "f32[384]" = torch.ops.aten.squeeze.dims(arg18_1, [0, 2, 3]);  arg18_1 = None
        mul_27: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
        mul_28: "f32[384]" = torch.ops.aten.mul.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
        unsqueeze_42: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_28, 0);  mul_28 = None
        unsqueeze_43: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        unsqueeze_44: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 3);  unsqueeze_43 = None
        mul_29: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_5, arg19_1);  arg19_1 = None
        unsqueeze_45: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_29, 0);  mul_29 = None
        unsqueeze_46: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
        unsqueeze_47: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 3);  unsqueeze_46 = None
        mul_30: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_44);  sub_9 = unsqueeze_44 = None
        sub_10: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_9, mul_30);  convert_element_type_9 = mul_30 = None
        sub_11: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_10, unsqueeze_41);  sub_10 = unsqueeze_41 = None
        mul_31: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_47);  sub_11 = unsqueeze_47 = None
        mul_32: "f32[384]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_5);  sum_6 = squeeze_5 = None
        convert_element_type_11: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        slice_7: "bf16[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_2, 1, 0, 384)
        slice_8: "bf16[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_2, 1, 384, 768);  slice_2 = None
        sub_12: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg21_1, arg22_1)
        mul_33: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_12, arg23_1);  sub_12 = None
        unsqueeze_48: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1)
        unsqueeze_49: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_34: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_49);  mul_33 = unsqueeze_49 = None
        unsqueeze_50: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_51: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_6: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_51);  mul_34 = unsqueeze_51 = None
        convert_element_type_12: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        relu_3: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_12);  convert_element_type_12 = None
        le_3: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_3: "bf16[128, 384, 8, 8]" = torch.ops.aten.where.self(le_3, arg10_1, slice_8);  le_3 = slice_8 = None
        convert_element_type_13: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        squeeze_6: "f32[384]" = torch.ops.aten.squeeze.dims(arg22_1, [0, 2, 3]);  arg22_1 = None
        unsqueeze_52: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_53: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 2);  unsqueeze_52 = None
        unsqueeze_54: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 3);  unsqueeze_53 = None
        sum_7: "f32[384]" = torch.ops.aten.sum.dim_IntList(convert_element_type_13, [0, 2, 3])
        convert_element_type_14: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.float32);  arg21_1 = None
        sub_13: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_14, unsqueeze_54);  convert_element_type_14 = unsqueeze_54 = None
        mul_35: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_13, sub_13)
        sum_8: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_35, [0, 2, 3]);  mul_35 = None
        mul_36: "f32[384]" = torch.ops.aten.mul.Tensor(sum_7, 0.0001220703125)
        unsqueeze_55: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_36, 0);  mul_36 = None
        unsqueeze_56: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 2);  unsqueeze_55 = None
        unsqueeze_57: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, 3);  unsqueeze_56 = None
        mul_37: "f32[384]" = torch.ops.aten.mul.Tensor(sum_8, 0.0001220703125)
        squeeze_7: "f32[384]" = torch.ops.aten.squeeze.dims(arg23_1, [0, 2, 3]);  arg23_1 = None
        mul_38: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_39: "f32[384]" = torch.ops.aten.mul.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
        unsqueeze_58: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_39, 0);  mul_39 = None
        unsqueeze_59: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 2);  unsqueeze_58 = None
        unsqueeze_60: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 3);  unsqueeze_59 = None
        mul_40: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_7, arg24_1);  arg24_1 = None
        unsqueeze_61: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_40, 0);  mul_40 = None
        unsqueeze_62: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None
        unsqueeze_63: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, 3);  unsqueeze_62 = None
        mul_41: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_60);  sub_13 = unsqueeze_60 = None
        sub_14: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_13, mul_41);  convert_element_type_13 = mul_41 = None
        sub_15: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_14, unsqueeze_57);  sub_14 = unsqueeze_57 = None
        mul_42: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_63);  sub_15 = unsqueeze_63 = None
        mul_43: "f32[384]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_7);  sum_8 = squeeze_7 = None
        convert_element_type_15: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None
        sub_16: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg26_1, arg27_1)
        mul_44: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_16, arg28_1);  sub_16 = None
        unsqueeze_64: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1)
        unsqueeze_65: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_45: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_65);  mul_44 = unsqueeze_65 = None
        unsqueeze_66: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_67: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_7: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_45, unsqueeze_67);  mul_45 = unsqueeze_67 = None
        convert_element_type_16: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        relu_4: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_16);  convert_element_type_16 = None
        le_4: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_4: "bf16[128, 384, 8, 8]" = torch.ops.aten.where.self(le_4, arg10_1, slice_7);  le_4 = slice_7 = None
        convert_element_type_17: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        squeeze_8: "f32[384]" = torch.ops.aten.squeeze.dims(arg27_1, [0, 2, 3]);  arg27_1 = None
        unsqueeze_68: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_8, 0);  squeeze_8 = None
        unsqueeze_69: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, 2);  unsqueeze_68 = None
        unsqueeze_70: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 3);  unsqueeze_69 = None
        sum_9: "f32[384]" = torch.ops.aten.sum.dim_IntList(convert_element_type_17, [0, 2, 3])
        convert_element_type_18: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.float32);  arg26_1 = None
        sub_17: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_18, unsqueeze_70);  convert_element_type_18 = unsqueeze_70 = None
        mul_46: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_17, sub_17)
        sum_10: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_46, [0, 2, 3]);  mul_46 = None
        mul_47: "f32[384]" = torch.ops.aten.mul.Tensor(sum_9, 0.0001220703125)
        unsqueeze_71: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_47, 0);  mul_47 = None
        unsqueeze_72: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
        unsqueeze_73: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None
        mul_48: "f32[384]" = torch.ops.aten.mul.Tensor(sum_10, 0.0001220703125)
        squeeze_9: "f32[384]" = torch.ops.aten.squeeze.dims(arg28_1, [0, 2, 3]);  arg28_1 = None
        mul_49: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_9, squeeze_9)
        mul_50: "f32[384]" = torch.ops.aten.mul.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
        unsqueeze_74: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_50, 0);  mul_50 = None
        unsqueeze_75: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, 2);  unsqueeze_74 = None
        unsqueeze_76: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 3);  unsqueeze_75 = None
        mul_51: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_9, arg29_1);  arg29_1 = None
        unsqueeze_77: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_51, 0);  mul_51 = None
        unsqueeze_78: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None
        unsqueeze_79: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 3);  unsqueeze_78 = None
        mul_52: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_76);  sub_17 = unsqueeze_76 = None
        sub_18: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_17, mul_52);  convert_element_type_17 = mul_52 = None
        sub_19: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_18, unsqueeze_73);  sub_18 = unsqueeze_73 = None
        mul_53: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_79);  sub_19 = unsqueeze_79 = None
        mul_54: "f32[384]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_9);  sum_10 = squeeze_9 = None
        convert_element_type_19: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        sub_20: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(arg31_1, arg32_1)
        mul_55: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_20, arg33_1);  sub_20 = None
        unsqueeze_80: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1)
        unsqueeze_81: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_56: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_81);  mul_55 = unsqueeze_81 = None
        unsqueeze_82: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_83: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_8: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_83);  mul_56 = unsqueeze_83 = None
        convert_element_type_20: "bf16[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        relu_5: "bf16[128, 320, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_20);  convert_element_type_20 = None
        le_5: "b8[128, 320, 8, 8]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_5: "bf16[128, 320, 8, 8]" = torch.ops.aten.where.self(le_5, arg10_1, slice_1);  le_5 = arg10_1 = slice_1 = None
        convert_element_type_21: "f32[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        squeeze_10: "f32[320]" = torch.ops.aten.squeeze.dims(arg32_1, [0, 2, 3]);  arg32_1 = None
        unsqueeze_84: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_10, 0);  squeeze_10 = None
        unsqueeze_85: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 2);  unsqueeze_84 = None
        unsqueeze_86: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 3);  unsqueeze_85 = None
        sum_11: "f32[320]" = torch.ops.aten.sum.dim_IntList(convert_element_type_21, [0, 2, 3])
        convert_element_type_22: "f32[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.float32);  arg31_1 = None
        sub_21: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_22, unsqueeze_86);  convert_element_type_22 = unsqueeze_86 = None
        mul_57: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_21, sub_21)
        sum_12: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_57, [0, 2, 3]);  mul_57 = None
        mul_58: "f32[320]" = torch.ops.aten.mul.Tensor(sum_11, 0.0001220703125)
        unsqueeze_87: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_58, 0);  mul_58 = None
        unsqueeze_88: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
        unsqueeze_89: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 3);  unsqueeze_88 = None
        mul_59: "f32[320]" = torch.ops.aten.mul.Tensor(sum_12, 0.0001220703125)
        squeeze_11: "f32[320]" = torch.ops.aten.squeeze.dims(arg33_1, [0, 2, 3]);  arg33_1 = None
        mul_60: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_11, squeeze_11)
        mul_61: "f32[320]" = torch.ops.aten.mul.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
        unsqueeze_90: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_61, 0);  mul_61 = None
        unsqueeze_91: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 2);  unsqueeze_90 = None
        unsqueeze_92: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
        mul_62: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_11, arg34_1);  arg34_1 = None
        unsqueeze_93: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_62, 0);  mul_62 = None
        unsqueeze_94: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
        unsqueeze_95: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, 3);  unsqueeze_94 = None
        mul_63: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_92);  sub_21 = unsqueeze_92 = None
        sub_22: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_21, mul_63);  convert_element_type_21 = mul_63 = None
        sub_23: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(sub_22, unsqueeze_89);  sub_22 = unsqueeze_89 = None
        mul_64: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_95);  sub_23 = unsqueeze_95 = None
        mul_65: "f32[320]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_11);  sum_12 = squeeze_11 = None
        convert_element_type_23: "bf16[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        return (sum_1, mul_10, convert_element_type_3, sum_3, mul_21, convert_element_type_7, sum_5, mul_32, convert_element_type_11, sum_7, mul_43, convert_element_type_15, sum_9, mul_54, convert_element_type_19, sum_11, mul_65, convert_element_type_23)



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
