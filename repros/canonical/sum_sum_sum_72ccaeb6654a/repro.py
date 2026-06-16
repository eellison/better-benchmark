"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 72ccaeb6654a
Shape hash: a91d4c93
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
    def forward(self, arg0_1: "bf16[128, 288, 35, 35]", arg1_1: "bf16[128, 288, 35, 35]", arg2_1: "bf16[128, 288, 35, 35]", arg3_1: "bf16[128, 288, 35, 35]", arg4_1: "bf16[128, 288, 35, 35]", arg5_1: "bf16[128, 64, 35, 35]", arg6_1: "f32[1, 64, 1, 1]", arg7_1: "f32[1, 64, 1, 1]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "bf16[]", arg11_1: "bf16[128, 96, 35, 35]", arg12_1: "f32[1, 96, 1, 1]", arg13_1: "f32[1, 96, 1, 1]", arg14_1: "f32[96]", arg15_1: "f32[96]", arg16_1: "bf16[128, 64, 35, 35]", arg17_1: "f32[1, 64, 1, 1]", arg18_1: "f32[1, 64, 1, 1]", arg19_1: "f32[64]", arg20_1: "f32[64]", arg21_1: "bf16[128, 64, 35, 35]", arg22_1: "f32[1, 64, 1, 1]", arg23_1: "f32[1, 64, 1, 1]", arg24_1: "f32[64]", arg25_1: "f32[64]"):
        # No stacktrace found for following nodes
        avg_pool2d_backward: "bf16[128, 288, 35, 35]" = torch.ops.aten.avg_pool2d_backward.default(arg0_1, arg1_1, [3, 3], [1, 1], [1, 1], False, True, None);  arg0_1 = arg1_1 = None
        add: "bf16[128, 288, 35, 35]" = torch.ops.aten.add.Tensor(avg_pool2d_backward, arg2_1);  avg_pool2d_backward = arg2_1 = None
        add_1: "bf16[128, 288, 35, 35]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        add_2: "bf16[128, 288, 35, 35]" = torch.ops.aten.add.Tensor(add_1, arg4_1);  add_1 = arg4_1 = None
        slice_1: "bf16[128, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_2, 1, 0, 64)
        slice_2: "bf16[128, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_2, 1, 64, 128)
        slice_3: "bf16[128, 96, 35, 35]" = torch.ops.aten.slice.Tensor(add_2, 1, 128, 224)
        slice_4: "bf16[128, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_2, 1, 224, 288);  add_2 = None
        sub: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(arg5_1, arg6_1)
        mul: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub, arg7_1);  sub = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1)
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        relu: "bf16[128, 64, 35, 35]" = torch.ops.aten.relu.default(convert_element_type);  convert_element_type = None
        le: "b8[128, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where: "bf16[128, 64, 35, 35]" = torch.ops.aten.where.self(le, arg10_1, slice_4);  le = slice_4 = None
        convert_element_type_1: "f32[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(arg6_1, [0, 2, 3]);  arg6_1 = None
        unsqueeze_4: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sum_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [0, 2, 3])
        convert_element_type_2: "f32[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        sub_1: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_6);  convert_element_type_2 = unsqueeze_6 = None
        mul_2: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub_1)
        sum_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 2, 3]);  mul_2 = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(sum_1, 6.3775510204081635e-06)
        unsqueeze_7: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_3, 0);  mul_3 = None
        unsqueeze_8: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, 6.3775510204081635e-06)
        squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(arg7_1, [0, 2, 3]);  arg7_1 = None
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, arg8_1);  arg8_1 = None
        unsqueeze_13: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_7, 0);  mul_7 = None
        unsqueeze_14: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_8: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_12);  sub_1 = unsqueeze_12 = None
        sub_2: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_8);  convert_element_type_1 = mul_8 = None
        sub_3: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_9);  sub_2 = unsqueeze_9 = None
        mul_9: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_15);  sub_3 = unsqueeze_15 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_3: "bf16[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        sub_4: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(arg11_1, arg12_1)
        mul_11: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_4, arg13_1);  sub_4 = None
        unsqueeze_16: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1)
        unsqueeze_17: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_12: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_17);  mul_11 = unsqueeze_17 = None
        unsqueeze_18: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_19: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_4: "f32[128, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_19);  mul_12 = unsqueeze_19 = None
        convert_element_type_4: "bf16[128, 96, 35, 35]" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        relu_1: "bf16[128, 96, 35, 35]" = torch.ops.aten.relu.default(convert_element_type_4);  convert_element_type_4 = None
        le_1: "b8[128, 96, 35, 35]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_1: "bf16[128, 96, 35, 35]" = torch.ops.aten.where.self(le_1, arg10_1, slice_3);  le_1 = slice_3 = None
        convert_element_type_5: "f32[128, 96, 35, 35]" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        squeeze_2: "f32[96]" = torch.ops.aten.squeeze.dims(arg12_1, [0, 2, 3]);  arg12_1 = None
        unsqueeze_20: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
        unsqueeze_21: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 2);  unsqueeze_20 = None
        unsqueeze_22: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 3);  unsqueeze_21 = None
        sum_3: "f32[96]" = torch.ops.aten.sum.dim_IntList(convert_element_type_5, [0, 2, 3])
        convert_element_type_6: "f32[128, 96, 35, 35]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float32);  arg11_1 = None
        sub_5: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convert_element_type_6, unsqueeze_22);  convert_element_type_6 = unsqueeze_22 = None
        mul_13: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(convert_element_type_5, sub_5)
        sum_4: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_13, [0, 2, 3]);  mul_13 = None
        mul_14: "f32[96]" = torch.ops.aten.mul.Tensor(sum_3, 6.3775510204081635e-06)
        unsqueeze_23: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_14, 0);  mul_14 = None
        unsqueeze_24: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 2);  unsqueeze_23 = None
        unsqueeze_25: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 3);  unsqueeze_24 = None
        mul_15: "f32[96]" = torch.ops.aten.mul.Tensor(sum_4, 6.3775510204081635e-06)
        squeeze_3: "f32[96]" = torch.ops.aten.squeeze.dims(arg13_1, [0, 2, 3]);  arg13_1 = None
        mul_16: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
        mul_17: "f32[96]" = torch.ops.aten.mul.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        unsqueeze_26: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_17, 0);  mul_17 = None
        unsqueeze_27: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 2);  unsqueeze_26 = None
        unsqueeze_28: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 3);  unsqueeze_27 = None
        mul_18: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_3, arg14_1);  arg14_1 = None
        unsqueeze_29: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_18, 0);  mul_18 = None
        unsqueeze_30: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, 2);  unsqueeze_29 = None
        unsqueeze_31: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 3);  unsqueeze_30 = None
        mul_19: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_28);  sub_5 = unsqueeze_28 = None
        sub_6: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convert_element_type_5, mul_19);  convert_element_type_5 = mul_19 = None
        sub_7: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_6, unsqueeze_25);  sub_6 = unsqueeze_25 = None
        mul_20: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_31);  sub_7 = unsqueeze_31 = None
        mul_21: "f32[96]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_3);  sum_4 = squeeze_3 = None
        convert_element_type_7: "bf16[128, 96, 35, 35]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        sub_8: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(arg16_1, arg17_1)
        mul_22: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_8, arg18_1);  sub_8 = None
        unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1)
        unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_23: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_33);  mul_22 = unsqueeze_33 = None
        unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_5: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_35);  mul_23 = unsqueeze_35 = None
        convert_element_type_8: "bf16[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        relu_2: "bf16[128, 64, 35, 35]" = torch.ops.aten.relu.default(convert_element_type_8);  convert_element_type_8 = None
        le_2: "b8[128, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_2: "bf16[128, 64, 35, 35]" = torch.ops.aten.where.self(le_2, arg10_1, slice_2);  le_2 = slice_2 = None
        convert_element_type_9: "f32[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(arg17_1, [0, 2, 3]);  arg17_1 = None
        unsqueeze_36: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_4, 0);  squeeze_4 = None
        unsqueeze_37: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
        unsqueeze_38: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 3);  unsqueeze_37 = None
        sum_5: "f32[64]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [0, 2, 3])
        convert_element_type_10: "f32[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.float32);  arg16_1 = None
        sub_9: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convert_element_type_10, unsqueeze_38);  convert_element_type_10 = unsqueeze_38 = None
        mul_24: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(convert_element_type_9, sub_9)
        sum_6: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_24, [0, 2, 3]);  mul_24 = None
        mul_25: "f32[64]" = torch.ops.aten.mul.Tensor(sum_5, 6.3775510204081635e-06)
        unsqueeze_39: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_25, 0);  mul_25 = None
        unsqueeze_40: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 2);  unsqueeze_39 = None
        unsqueeze_41: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 3);  unsqueeze_40 = None
        mul_26: "f32[64]" = torch.ops.aten.mul.Tensor(sum_6, 6.3775510204081635e-06)
        squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(arg18_1, [0, 2, 3]);  arg18_1 = None
        mul_27: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
        mul_28: "f32[64]" = torch.ops.aten.mul.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
        unsqueeze_42: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_28, 0);  mul_28 = None
        unsqueeze_43: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        unsqueeze_44: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 3);  unsqueeze_43 = None
        mul_29: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, arg19_1);  arg19_1 = None
        unsqueeze_45: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_29, 0);  mul_29 = None
        unsqueeze_46: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
        unsqueeze_47: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 3);  unsqueeze_46 = None
        mul_30: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_44);  sub_9 = unsqueeze_44 = None
        sub_10: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convert_element_type_9, mul_30);  convert_element_type_9 = mul_30 = None
        sub_11: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_10, unsqueeze_41);  sub_10 = unsqueeze_41 = None
        mul_31: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_47);  sub_11 = unsqueeze_47 = None
        mul_32: "f32[64]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_5);  sum_6 = squeeze_5 = None
        convert_element_type_11: "bf16[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        sub_12: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(arg21_1, arg22_1)
        mul_33: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_12, arg23_1);  sub_12 = None
        unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1)
        unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_34: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_49);  mul_33 = unsqueeze_49 = None
        unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_6: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_51);  mul_34 = unsqueeze_51 = None
        convert_element_type_12: "bf16[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        relu_3: "bf16[128, 64, 35, 35]" = torch.ops.aten.relu.default(convert_element_type_12);  convert_element_type_12 = None
        le_3: "b8[128, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_3: "bf16[128, 64, 35, 35]" = torch.ops.aten.where.self(le_3, arg10_1, slice_1);  le_3 = arg10_1 = slice_1 = None
        convert_element_type_13: "f32[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(arg22_1, [0, 2, 3]);  arg22_1 = None
        unsqueeze_52: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_53: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 2);  unsqueeze_52 = None
        unsqueeze_54: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 3);  unsqueeze_53 = None
        sum_7: "f32[64]" = torch.ops.aten.sum.dim_IntList(convert_element_type_13, [0, 2, 3])
        convert_element_type_14: "f32[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.float32);  arg21_1 = None
        sub_13: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convert_element_type_14, unsqueeze_54);  convert_element_type_14 = unsqueeze_54 = None
        mul_35: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(convert_element_type_13, sub_13)
        sum_8: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_35, [0, 2, 3]);  mul_35 = None
        mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(sum_7, 6.3775510204081635e-06)
        unsqueeze_55: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_36, 0);  mul_36 = None
        unsqueeze_56: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 2);  unsqueeze_55 = None
        unsqueeze_57: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, 3);  unsqueeze_56 = None
        mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(sum_8, 6.3775510204081635e-06)
        squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(arg23_1, [0, 2, 3]);  arg23_1 = None
        mul_38: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_39: "f32[64]" = torch.ops.aten.mul.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
        unsqueeze_58: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_39, 0);  mul_39 = None
        unsqueeze_59: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 2);  unsqueeze_58 = None
        unsqueeze_60: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 3);  unsqueeze_59 = None
        mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, arg24_1);  arg24_1 = None
        unsqueeze_61: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_40, 0);  mul_40 = None
        unsqueeze_62: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None
        unsqueeze_63: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, 3);  unsqueeze_62 = None
        mul_41: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_60);  sub_13 = unsqueeze_60 = None
        sub_14: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convert_element_type_13, mul_41);  convert_element_type_13 = mul_41 = None
        sub_15: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_14, unsqueeze_57);  sub_14 = unsqueeze_57 = None
        mul_42: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_63);  sub_15 = unsqueeze_63 = None
        mul_43: "f32[64]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_7);  sum_8 = squeeze_7 = None
        convert_element_type_15: "bf16[128, 64, 35, 35]" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None
        return (sum_1, mul_10, convert_element_type_3, sum_3, mul_21, convert_element_type_7, sum_5, mul_32, convert_element_type_11, sum_7, mul_43, convert_element_type_15)



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
