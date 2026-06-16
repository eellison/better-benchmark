"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train
Pattern hash: c3cfd4211c4e
Shape hash: b12d2d03
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
    def forward(self, arg0_1: "f32[8, 512, 4096]", arg1_1: "f32[8, 512, 4096]", arg2_1: "bf16[4096, 4096]", arg3_1: "f32[8, 512, 4096]", arg4_1: "f32[8, 512, 4096]", arg5_1: "bf16[4096, 4096]", arg6_1: "f32[8, 512, 4096]", arg7_1: "f32[8, 512, 4096]", arg8_1: "bf16[4096, 4096]", arg9_1: "f32[8, 512, 4096]", arg10_1: "f32[8, 512, 4096]", arg11_1: "bf16[4096, 4096]", arg12_1: "f32[8, 512, 4096]", arg13_1: "f32[8, 512, 4096]", arg14_1: "bf16[4096, 4096]", arg15_1: "f32[8, 512, 4096]", arg16_1: "f32[8, 512, 4096]", arg17_1: "bf16[4096, 4096]", arg18_1: "f32[8, 512, 4096]", arg19_1: "f32[8, 512, 4096]", arg20_1: "bf16[4096, 4096]", arg21_1: "f32[8, 512, 4096]", arg22_1: "f32[8, 512, 4096]", arg23_1: "bf16[4096, 4096]", arg24_1: "f32[8, 512, 4096]", arg25_1: "f32[8, 512, 4096]", arg26_1: "bf16[4096, 4096]", arg27_1: "f32[8, 512, 4096]", arg28_1: "f32[8, 512, 4096]", arg29_1: "bf16[4096, 4096]", arg30_1: "f32[8, 512, 4096]", arg31_1: "f32[8, 512, 4096]", arg32_1: "bf16[4096, 4096]", arg33_1: "bf16[4096, 4096]", arg34_1: "f32[8, 512, 4096]", arg35_1: "bf16[4096, 4096]", arg36_1: "bf16[4096, 4096]", arg37_1: "f32[4096]", arg38_1: "f32[8, 512, 4096]", arg39_1: "f32[8, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15):
        # No stacktrace found for following nodes
        mul: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg1_1 = None
        sum_1: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul, [0, 1]);  mul = None
        sum_2: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0, 1]);  arg0_1 = None
        sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg2_1, [0], True, dtype = torch.float32);  arg2_1 = None
        view: "f32[4096]" = torch.ops.aten.view.default(sum_3, _shape_param_0);  sum_3 = _shape_param_0 = None
        convert_element_type: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_1: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        mul_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg3_1, arg4_1);  arg4_1 = None
        sum_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_1, [0, 1]);  mul_1 = None
        sum_5: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg3_1, [0, 1]);  arg3_1 = None
        add: "f32[4096]" = torch.ops.aten.add.Tensor(sum_1, sum_4);  sum_1 = sum_4 = None
        add_1: "f32[4096]" = torch.ops.aten.add.Tensor(sum_2, sum_5);  sum_2 = sum_5 = None
        sum_6: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg5_1, [0], True, dtype = torch.float32);  arg5_1 = None
        view_1: "f32[4096]" = torch.ops.aten.view.default(sum_6, _shape_param_1);  sum_6 = _shape_param_1 = None
        convert_element_type_2: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_3: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        add_2: "f32[4096]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_3);  convert_element_type_1 = convert_element_type_3 = None
        mul_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg6_1, arg7_1);  arg7_1 = None
        sum_7: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 1]);  mul_2 = None
        sum_8: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg6_1, [0, 1]);  arg6_1 = None
        add_3: "f32[4096]" = torch.ops.aten.add.Tensor(add, sum_7);  add = sum_7 = None
        add_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_1, sum_8);  add_1 = sum_8 = None
        sum_9: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg8_1, [0], True, dtype = torch.float32);  arg8_1 = None
        view_2: "f32[4096]" = torch.ops.aten.view.default(sum_9, _shape_param_2);  sum_9 = _shape_param_2 = None
        convert_element_type_4: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_5: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        add_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_5);  add_2 = convert_element_type_5 = None
        mul_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg9_1, arg10_1);  arg10_1 = None
        sum_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1]);  mul_3 = None
        sum_11: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg9_1, [0, 1]);  arg9_1 = None
        add_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_3, sum_10);  add_3 = sum_10 = None
        add_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_4, sum_11);  add_4 = sum_11 = None
        sum_12: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg11_1, [0], True, dtype = torch.float32);  arg11_1 = None
        view_3: "f32[4096]" = torch.ops.aten.view.default(sum_12, _shape_param_3);  sum_12 = _shape_param_3 = None
        convert_element_type_6: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_7: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        add_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_7);  add_5 = convert_element_type_7 = None
        mul_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg12_1, arg13_1);  arg13_1 = None
        sum_13: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_4, [0, 1]);  mul_4 = None
        sum_14: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg12_1, [0, 1]);  arg12_1 = None
        add_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_6, sum_13);  add_6 = sum_13 = None
        add_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_7, sum_14);  add_7 = sum_14 = None
        sum_15: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg14_1, [0], True, dtype = torch.float32);  arg14_1 = None
        view_4: "f32[4096]" = torch.ops.aten.view.default(sum_15, _shape_param_4);  sum_15 = _shape_param_4 = None
        convert_element_type_8: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_9: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        add_11: "f32[4096]" = torch.ops.aten.add.Tensor(add_8, convert_element_type_9);  add_8 = convert_element_type_9 = None
        mul_5: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg15_1, arg16_1);  arg16_1 = None
        sum_16: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_17: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg15_1, [0, 1]);  arg15_1 = None
        add_12: "f32[4096]" = torch.ops.aten.add.Tensor(add_9, sum_16);  add_9 = sum_16 = None
        add_13: "f32[4096]" = torch.ops.aten.add.Tensor(add_10, sum_17);  add_10 = sum_17 = None
        sum_18: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg17_1, [0], True, dtype = torch.float32);  arg17_1 = None
        view_5: "f32[4096]" = torch.ops.aten.view.default(sum_18, _shape_param_5);  sum_18 = _shape_param_5 = None
        convert_element_type_10: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        convert_element_type_11: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_10, torch.float32);  convert_element_type_10 = None
        add_14: "f32[4096]" = torch.ops.aten.add.Tensor(add_11, convert_element_type_11);  add_11 = convert_element_type_11 = None
        mul_6: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg18_1, arg19_1);  arg19_1 = None
        sum_19: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_20: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg18_1, [0, 1]);  arg18_1 = None
        add_15: "f32[4096]" = torch.ops.aten.add.Tensor(add_12, sum_19);  add_12 = sum_19 = None
        add_16: "f32[4096]" = torch.ops.aten.add.Tensor(add_13, sum_20);  add_13 = sum_20 = None
        sum_21: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg20_1, [0], True, dtype = torch.float32);  arg20_1 = None
        view_6: "f32[4096]" = torch.ops.aten.view.default(sum_21, _shape_param_6);  sum_21 = _shape_param_6 = None
        convert_element_type_12: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        convert_element_type_13: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        add_17: "f32[4096]" = torch.ops.aten.add.Tensor(add_14, convert_element_type_13);  add_14 = convert_element_type_13 = None
        mul_7: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg21_1, arg22_1);  arg22_1 = None
        sum_22: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_23: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg21_1, [0, 1]);  arg21_1 = None
        add_18: "f32[4096]" = torch.ops.aten.add.Tensor(add_15, sum_22);  add_15 = sum_22 = None
        add_19: "f32[4096]" = torch.ops.aten.add.Tensor(add_16, sum_23);  add_16 = sum_23 = None
        sum_24: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg23_1, [0], True, dtype = torch.float32);  arg23_1 = None
        view_7: "f32[4096]" = torch.ops.aten.view.default(sum_24, _shape_param_7);  sum_24 = _shape_param_7 = None
        convert_element_type_14: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        convert_element_type_15: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        add_20: "f32[4096]" = torch.ops.aten.add.Tensor(add_17, convert_element_type_15);  add_17 = convert_element_type_15 = None
        mul_8: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg24_1, arg25_1);  arg25_1 = None
        sum_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_8, [0, 1]);  mul_8 = None
        sum_26: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg24_1, [0, 1]);  arg24_1 = None
        add_21: "f32[4096]" = torch.ops.aten.add.Tensor(add_18, sum_25);  add_18 = sum_25 = None
        add_22: "f32[4096]" = torch.ops.aten.add.Tensor(add_19, sum_26);  add_19 = sum_26 = None
        sum_27: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg26_1, [0], True, dtype = torch.float32);  arg26_1 = None
        view_8: "f32[4096]" = torch.ops.aten.view.default(sum_27, _shape_param_8);  sum_27 = _shape_param_8 = None
        convert_element_type_16: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        convert_element_type_17: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_16, torch.float32);  convert_element_type_16 = None
        add_23: "f32[4096]" = torch.ops.aten.add.Tensor(add_20, convert_element_type_17);  add_20 = convert_element_type_17 = None
        mul_9: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg27_1, arg28_1);  arg28_1 = None
        sum_28: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_9, [0, 1]);  mul_9 = None
        sum_29: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg27_1, [0, 1]);  arg27_1 = None
        add_24: "f32[4096]" = torch.ops.aten.add.Tensor(add_21, sum_28);  add_21 = sum_28 = None
        add_25: "f32[4096]" = torch.ops.aten.add.Tensor(add_22, sum_29);  add_22 = sum_29 = None
        sum_30: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg29_1, [0], True, dtype = torch.float32);  arg29_1 = None
        view_9: "f32[4096]" = torch.ops.aten.view.default(sum_30, _shape_param_9);  sum_30 = _shape_param_9 = None
        convert_element_type_18: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        convert_element_type_19: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_18, torch.float32);  convert_element_type_18 = None
        add_26: "f32[4096]" = torch.ops.aten.add.Tensor(add_23, convert_element_type_19);  add_23 = convert_element_type_19 = None
        mul_10: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg30_1, arg31_1);  arg31_1 = None
        sum_31: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_10, [0, 1]);  mul_10 = None
        sum_32: "f32[4096]" = torch.ops.aten.sum.dim_IntList(arg30_1, [0, 1]);  arg30_1 = None
        add_27: "f32[4096]" = torch.ops.aten.add.Tensor(add_24, sum_31);  add_24 = sum_31 = None
        add_28: "f32[4096]" = torch.ops.aten.add.Tensor(add_25, sum_32);  add_25 = sum_32 = None
        sum_33: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(arg32_1, [0], True, dtype = torch.float32);  arg32_1 = None
        view_10: "f32[4096]" = torch.ops.aten.view.default(sum_33, _shape_param_10);  sum_33 = _shape_param_10 = None
        convert_element_type_20: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        convert_element_type_21: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        add_29: "f32[4096]" = torch.ops.aten.add.Tensor(add_26, convert_element_type_21);  add_26 = convert_element_type_21 = None
        view_11: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(arg33_1, _shape_param_11);  arg33_1 = _shape_param_11 = None
        convert_element_type_22: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        add_30: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(arg34_1, convert_element_type_22);  arg34_1 = convert_element_type_22 = None
        view_12: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(arg35_1, _shape_param_12);  arg35_1 = _shape_param_12 = None
        convert_element_type_23: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        add_31: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_30, convert_element_type_23);  add_30 = convert_element_type_23 = None
        view_13: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(arg36_1, _shape_param_13);  arg36_1 = _shape_param_13 = None
        convert_element_type_24: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        add_32: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_31, convert_element_type_24);  add_31 = convert_element_type_24 = None
        mul_11: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_32, arg37_1);  arg37_1 = None
        mul_12: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_11, 4096)
        sum_34: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_11, [2], True)
        mul_13: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_11, arg38_1);  mul_11 = None
        sum_35: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_13, [2], True);  mul_13 = None
        mul_14: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg38_1, sum_35);  sum_35 = None
        sub: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_12, sum_34);  mul_12 = sum_34 = None
        sub_1: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub, mul_14);  sub = mul_14 = None
        mul_15: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg39_1, sub_1);  arg39_1 = sub_1 = None
        mul_16: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_32, arg38_1);  arg38_1 = None
        sum_36: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_16, [0, 1]);  mul_16 = None
        sum_37: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_32, [0, 1]);  add_32 = None
        add_33: "f32[4096]" = torch.ops.aten.add.Tensor(add_27, sum_36);  add_27 = sum_36 = None
        add_34: "f32[4096]" = torch.ops.aten.add.Tensor(add_28, sum_37);  add_28 = sum_37 = None
        convert_element_type_25: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16)
        view_14: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_14);  convert_element_type_25 = _shape_param_14 = None
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_14, [1, 0])
        sum_38: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_14, [0], True, dtype = torch.float32)
        view_15: "f32[4096]" = torch.ops.aten.view.default(sum_38, _shape_param_15);  sum_38 = _shape_param_15 = None
        convert_element_type_26: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_15, torch.bfloat16);  view_15 = None
        convert_element_type_27: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_26, torch.float32);  convert_element_type_26 = None
        add_35: "f32[4096]" = torch.ops.aten.add.Tensor(add_29, convert_element_type_27);  add_29 = convert_element_type_27 = None
        return (mul_15, add_33, add_34, view_14, permute, add_35)



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
