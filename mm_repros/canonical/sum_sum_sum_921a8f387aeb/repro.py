"""
Standalone repro captured via capture_hook.
Label: AllenaiLongformerBase
Pattern hash: 921a8f387aeb
Shape hash: 944fd5b5
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
    def forward(self, arg0_1: "bf16[9437184]", arg1_1: "bf16[6291456]", arg2_1: "bf16[8192, 768]", arg3_1: "bf16[768, 768]", arg4_1: "bf16[6291456]", arg5_1: "bf16[768, 768]", arg6_1: "bf16[768, 768]", arg7_1: "f32[8, 1024, 768]", arg8_1: "f32[768]", arg9_1: "f32[8, 1024, 768]", arg10_1: "f32[8, 1024, 1]", arg11_1: "b8[8, 1024, 768]", arg12_1: "bf16[768, 3072]", arg13_1: "bf16[8192, 3072]", arg14_1: "bf16[8192, 3072]", arg15_1: "bf16[3072, 768]", arg16_1: "bf16[8192, 768]", arg17_1: "f32[768]", arg18_1: "f32[8, 1024, 768]", arg19_1: "f32[8, 1024, 1]", arg20_1: "b8[8, 1024, 768]", arg21_1: "bf16[8192, 768]", arg22_1: "bf16[768, 768]", arg23_1: "bf16[384, 768, 256]", arg24_1: "bf16[384, 64, 768]", arg25_1: "bf16[96, 4, 256, 769]", arg26_1: "bf16[96, 4, 197120]", arg27_1: "b8[8, 1024, 12, 513]", arg28_1: "b8[8, 1024, 1, 1]", arg29_1: "f32[]", arg30_1: "bf16[8, 1024, 12, 513]", arg31_1: "f32[8, 1024, 12, 1]", arg32_1: "f32[8, 1024, 12, 1]", arg33_1: "bf16[8, 256, 12, 257]", arg34_1: "b8[8, 256, 12, 257]", arg35_1: "bf16[]", arg36_1: "bf16[8, 256, 12, 513]", arg37_1: "bf16[8, 1024, 12, 513]", arg38_1: "b8[8, 256, 12, 257]", arg39_1: "bf16[96, 255, 255]", arg40_1: "bf16[96, 255, 513]", arg41_1: "bf16[96, 512, 513]", arg42_1: "bf16[96, 3, 512, 513]", arg43_1: "bf16[96, 3, 256, 256]", arg44_1: "bf16[96, 3, 256, 513]", arg45_1: "bf16[96, 256, 257]", arg46_1: "bf16[96, 256, 513]", arg47_1: "bf16[288, 64, 512]", arg48_1: "bf16[288, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58):
        # No stacktrace found for following nodes
        as_strided: "bf16[96, 1536, 64]" = torch.ops.aten.as_strided.default(arg0_1, _shape_param_0, _shape_param_1, 0);  arg0_1 = _shape_param_0 = _shape_param_1 = None
        constant_pad_nd: "bf16[96, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        view: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_3);  constant_pad_nd = _shape_param_3 = None
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        permute_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute, [1, 0, 2, 3]);  permute = None
        clone: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_1: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        as_strided_1: "bf16[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(arg1_1, _shape_param_5, _shape_param_6, 0);  arg1_1 = _shape_param_5 = _shape_param_6 = None
        view_2: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(as_strided_1, _shape_param_7);  as_strided_1 = _shape_param_7 = None
        view_3: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_2, _shape_param_8);  view_2 = _shape_param_8 = None
        permute_2: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        permute_3: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_2, [1, 0, 2, 3]);  permute_2 = None
        view_4: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(permute_3, _shape_param_9);  permute_3 = _shape_param_9 = None
        div: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(view_4, 8.0);  view_4 = None
        sum_1: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1, [0, 1], True, dtype = torch.float32)
        view_5: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_10);  sum_1 = _shape_param_10 = None
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[8192, 768]" = torch.ops.aten.view.default(view_1, _shape_param_11);  view_1 = _shape_param_11 = None
        permute_4: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_6, [1, 0])
        mm: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_4, arg2_1);  permute_4 = None
        mm_1: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_6, arg3_1);  view_6 = arg3_1 = None
        view_7: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_12);  mm_1 = _shape_param_12 = None
        convert_element_type_1: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        convert_element_type_2: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_3: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        as_strided_2: "bf16[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(arg4_1, _shape_param_13, _shape_param_14, 0);  arg4_1 = _shape_param_13 = _shape_param_14 = None
        view_8: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(as_strided_2, _shape_param_15);  as_strided_2 = _shape_param_15 = None
        view_9: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_8, _shape_param_16);  view_8 = _shape_param_16 = None
        permute_5: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        permute_6: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_5, [1, 0, 2, 3]);  permute_5 = None
        view_10: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(permute_6, _shape_param_17);  permute_6 = _shape_param_17 = None
        sum_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_10, [0, 1], True, dtype = torch.float32)
        view_11: "f32[768]" = torch.ops.aten.view.default(sum_2, _shape_param_18);  sum_2 = _shape_param_18 = None
        convert_element_type_4: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        view_12: "bf16[8192, 768]" = torch.ops.aten.view.default(view_10, _shape_param_19);  view_10 = _shape_param_19 = None
        permute_7: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_12, [1, 0])
        mm_2: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_7, arg2_1);  permute_7 = None
        mm_3: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_12, arg5_1);  view_12 = arg5_1 = None
        view_13: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_3, _shape_param_20);  mm_3 = _shape_param_20 = None
        convert_element_type_5: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        add: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_5);  convert_element_type_1 = convert_element_type_5 = None
        convert_element_type_6: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_7: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        sum_3: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div, [0, 1], True, dtype = torch.float32)
        view_14: "f32[768]" = torch.ops.aten.view.default(sum_3, _shape_param_21);  sum_3 = _shape_param_21 = None
        convert_element_type_8: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_14, torch.bfloat16);  view_14 = None
        view_15: "bf16[8192, 768]" = torch.ops.aten.view.default(div, _shape_param_22);  div = _shape_param_22 = None
        permute_8: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_4: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_8, arg2_1);  permute_8 = arg2_1 = None
        mm_5: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_15, arg6_1);  view_15 = arg6_1 = None
        view_16: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_5, _shape_param_23);  mm_5 = _shape_param_23 = None
        convert_element_type_9: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_16, torch.float32);  view_16 = None
        add_1: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_9);  add = convert_element_type_9 = None
        convert_element_type_10: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_11: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        permute_9: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        add_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(arg7_1, permute_9);  arg7_1 = permute_9 = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_2, arg8_1);  arg8_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_4: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, arg9_1);  mul = None
        sum_5: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg9_1, sum_5);  sum_5 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_4);  mul_1 = sum_4 = None
        sub_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg10_1, sub_1);  arg10_1 = sub_1 = None
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_2, arg9_1);  arg9_1 = None
        sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_2, [0, 1]);  add_2 = None
        convert_element_type_12: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        convert_element_type_13: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        mul_6: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_7: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, mul_6);  convert_element_type_12 = mul_6 = None
        view_17: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_7, _shape_param_24);  mul_7 = _shape_param_24 = None
        mm_6: "bf16[8192, 3072]" = torch.ops.aten.mm.default(view_17, arg12_1);  arg12_1 = None
        permute_10: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_17, [1, 0])
        mm_7: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_10, arg13_1);  permute_10 = arg13_1 = None
        sum_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_17, [0], True, dtype = torch.float32);  view_17 = None
        view_18: "f32[768]" = torch.ops.aten.view.default(sum_8, _shape_param_25);  sum_8 = _shape_param_25 = None
        convert_element_type_14: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_18, torch.bfloat16);  view_18 = None
        view_19: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(mm_6, _shape_param_26);  mm_6 = _shape_param_26 = None
        convert_element_type_15: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_16: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        convert_element_type_17: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        view_20: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(arg14_1, _shape_param_27);  arg14_1 = _shape_param_27 = None
        convert_element_type_18: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_20, torch.float32);  view_20 = None
        mul_8: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.7071067811865476)
        erf: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_3: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        mul_10: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_18, convert_element_type_18)
        mul_11: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_10, -0.5);  mul_10 = None
        exp: "f32[8, 1024, 3072]" = torch.ops.aten.exp.default(mul_11);  mul_11 = None
        mul_12: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_13: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_18, mul_12);  convert_element_type_18 = mul_12 = None
        add_4: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_9, mul_13);  mul_9 = mul_13 = None
        mul_14: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_17, add_4);  convert_element_type_17 = add_4 = None
        convert_element_type_19: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        view_21: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_28);  convert_element_type_19 = _shape_param_28 = None
        mm_8: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_21, arg15_1);  arg15_1 = None
        permute_11: "bf16[3072, 8192]" = torch.ops.aten.permute.default(view_21, [1, 0])
        mm_9: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_11, arg16_1);  permute_11 = arg16_1 = None
        sum_9: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_21, [0], True, dtype = torch.float32);  view_21 = None
        view_22: "f32[3072]" = torch.ops.aten.view.default(sum_9, _shape_param_29);  sum_9 = _shape_param_29 = None
        convert_element_type_20: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_22, torch.bfloat16);  view_22 = None
        view_23: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_8, _shape_param_30);  mm_8 = _shape_param_30 = None
        convert_element_type_21: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_23, torch.float32);  view_23 = None
        add_5: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_4, convert_element_type_21);  mul_4 = convert_element_type_21 = None
        convert_element_type_22: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_23: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        mul_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_5, arg17_1);  arg17_1 = None
        mul_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_15, 768)
        sum_10: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True)
        mul_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_15, arg18_1);  mul_15 = None
        sum_11: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [2], True);  mul_17 = None
        mul_18: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg18_1, sum_11);  sum_11 = None
        sub_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_16, sum_10);  mul_16 = sum_10 = None
        sub_3: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_2, mul_18);  sub_2 = mul_18 = None
        mul_19: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg19_1, sub_3);  arg19_1 = sub_3 = None
        mul_20: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_5, arg18_1);  arg18_1 = None
        sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_20, [0, 1]);  mul_20 = None
        sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_5, [0, 1]);  add_5 = None
        convert_element_type_24: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16)
        convert_element_type_25: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        mul_21: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_22: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, mul_21);  convert_element_type_24 = mul_21 = None
        sum_14: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_22, [0, 1], True, dtype = torch.float32)
        view_24: "f32[768]" = torch.ops.aten.view.default(sum_14, _shape_param_31);  sum_14 = _shape_param_31 = None
        convert_element_type_26: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_24, torch.bfloat16);  view_24 = None
        view_25: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_22, _shape_param_32);  mul_22 = _shape_param_32 = None
        permute_12: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_10: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_12, arg21_1);  permute_12 = arg21_1 = None
        mm_11: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_25, arg22_1);  view_25 = arg22_1 = None
        view_26: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_11, _shape_param_33);  mm_11 = _shape_param_33 = None
        convert_element_type_27: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        convert_element_type_28: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_26, torch.float32);  convert_element_type_26 = None
        permute_13: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(view_26, [1, 0, 2]);  view_26 = None
        view_27: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(permute_13, _shape_param_34);  permute_13 = _shape_param_34 = None
        permute_14: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_27, [1, 0, 2, 3]);  view_27 = None
        permute_15: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_14, [0, 2, 1, 3]);  permute_14 = None
        clone_1: "bf16[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_28: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_35);  clone_1 = _shape_param_35 = None
        view_29: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_28, _shape_param_36);  view_28 = _shape_param_36 = None
        permute_16: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_29, [0, 1, 2, 4, 3]);  view_29 = None
        view_30: "bf16[384, 256, 64]" = torch.ops.aten.view.default(permute_16, _shape_param_37);  permute_16 = _shape_param_37 = None
        bmm: "bf16[384, 768, 64]" = torch.ops.aten.bmm.default(arg23_1, view_30);  arg23_1 = None
        bmm_1: "bf16[384, 256, 768]" = torch.ops.aten.bmm.default(view_30, arg24_1);  view_30 = arg24_1 = None
        view_31: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm, _shape_param_38);  bmm = _shape_param_38 = None
        view_32: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_1, _shape_param_39);  bmm_1 = _shape_param_39 = None
        squeeze: "bf16[96, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_31, 4);  view_31 = None
        squeeze_1: "bf16[96, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_32, 4);  view_32 = None
        slice_scatter: "bf16[96, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(arg25_1, squeeze_1, 3, 0, -1);  arg25_1 = squeeze_1 = None
        view_33: "bf16[96, 4, 196864]" = torch.ops.aten.view.default(slice_scatter, _shape_param_40);  slice_scatter = _shape_param_40 = None
        slice_scatter_1: "bf16[96, 4, 197120]" = torch.ops.aten.slice_scatter.default(arg26_1, view_33, 2, 0, -256);  arg26_1 = view_33 = None
        view_34: "bf16[96, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_1, _shape_param_41);  slice_scatter_1 = _shape_param_41 = None
        constant_pad_nd_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_34, _shape_param_42);  view_34 = _shape_param_42 = None
        view_35: "bf16[18874368]" = torch.ops.aten.view.default(squeeze, [-1]);  squeeze = None
        view_36: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_1, _shape_param_43);  constant_pad_nd_1 = _shape_param_43 = None
        permute_17: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        convert_element_type_29: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        mul_23: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_24: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_17, mul_23);  permute_17 = mul_23 = None
        clone_2: "bf16[8, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_24, memory_format = torch.contiguous_format);  mul_24 = None
        convert_element_type_30: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_2, torch.float32);  clone_2 = None
        where: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(arg28_1, arg29_1, convert_element_type_30);  arg28_1 = arg29_1 = convert_element_type_30 = None
        convert_element_type_31: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.float32);  arg30_1 = None
        clone_3: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_31, memory_format = torch.contiguous_format);  convert_element_type_31 = None
        sub_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_3, arg31_1);  clone_3 = arg31_1 = None
        exp_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_1, arg32_1);  exp_1 = arg32_1 = None
        mul_25: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where, div_1);  where = None
        sum_15: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_25, [-1], True)
        neg: "f32[8, 1024, 12, 513]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma: "f32[8, 1024, 12, 513]" = torch.ops.prims.fma.default(neg, sum_15, mul_25);  neg = sum_15 = mul_25 = None
        convert_element_type_32: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        permute_18: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_32, [0, 2, 1, 3]);  convert_element_type_32 = None
        clone_4: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_4, _shape_param_44);  clone_4 = _shape_param_44 = None
        view_38: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_37, _shape_param_45);  view_37 = _shape_param_45 = None
        permute_19: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_5: "bf16[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format)
        copy: "bf16[8, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_19, clone_5);  permute_19 = clone_5 = None
        permute_20: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(copy, [0, 2, 1, 3]);  copy = None
        view_39: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_20, _shape_param_46);  permute_20 = _shape_param_46 = None
        view_40: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_39, _shape_param_47);  view_39 = _shape_param_47 = None
        permute_21: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        slice_1: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_21, 1, -256, 9223372036854775807)
        slice_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1, 3, -257, 9223372036854775807)
        clone_6: "bf16[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format)
        copy_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2, arg33_1);  slice_2 = None
        slice_scatter_2: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1, copy_1, 3, -257, 9223372036854775807);  slice_1 = copy_1 = None
        slice_scatter_3: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_21, slice_scatter_2, 1, -256, 9223372036854775807);  permute_21 = slice_scatter_2 = None
        permute_22: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_3, [0, 2, 1, 3]);  slice_scatter_3 = None
        view_41: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_22, _shape_param_48);  permute_22 = _shape_param_48 = None
        where_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(arg34_1, arg35_1, clone_6);  arg34_1 = clone_6 = None
        slice_scatter_4: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(arg36_1, where_1, 3, -257, 9223372036854775807);  where_1 = None
        slice_scatter_5: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(arg37_1, slice_scatter_4, 1, -256, 9223372036854775807);  slice_scatter_4 = None
        permute_23: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_5, [0, 2, 1, 3]);  slice_scatter_5 = None
        clone_7: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_42: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_7, _shape_param_49);  clone_7 = _shape_param_49 = None
        add_6: "bf16[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_41, view_42);  view_41 = view_42 = None
        view_43: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(add_6, _shape_param_50);  add_6 = _shape_param_50 = None
        permute_24: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        slice_3: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_24, 1, 0, 256)
        slice_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 257)
        clone_8: "bf16[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_4, memory_format = torch.contiguous_format)
        copy_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_4, arg33_1);  slice_4 = arg33_1 = None
        slice_scatter_6: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_2, 3, 0, 257);  slice_3 = copy_2 = None
        slice_scatter_7: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_24, slice_scatter_6, 1, 0, 256);  permute_24 = slice_scatter_6 = None
        permute_25: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_7, [0, 2, 1, 3]);  slice_scatter_7 = None
        view_44: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_25, _shape_param_51);  permute_25 = _shape_param_51 = None
        where_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(arg38_1, arg35_1, clone_8);  arg38_1 = arg35_1 = clone_8 = None
        slice_scatter_8: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(arg36_1, where_2, 3, 0, 257);  arg36_1 = where_2 = None
        slice_scatter_9: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(arg37_1, slice_scatter_8, 1, 0, 256);  arg37_1 = slice_scatter_8 = None
        permute_26: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_9, [0, 2, 1, 3]);  slice_scatter_9 = None
        clone_9: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_45: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_9, _shape_param_52);  clone_9 = _shape_param_52 = None
        add_7: "bf16[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_44, view_45);  view_44 = view_45 = None
        select: "bf16[96, 256, 513]" = torch.ops.aten.select.int(add_7, 1, 0)
        slice_5: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select, 1, 1, 256)
        slice_6: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_5, 2, 1, 256)
        clone_10: "bf16[96, 255, 255]" = torch.ops.aten.clone.default(slice_6, memory_format = torch.contiguous_format)
        copy_3: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_6, arg39_1);  slice_6 = arg39_1 = None
        slice_scatter_10: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_5, copy_3, 2, 1, 256);  slice_5 = copy_3 = None
        slice_scatter_11: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select, slice_scatter_10, 1, 1, 256);  select = slice_scatter_10 = None
        select_scatter: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_7, slice_scatter_11, 1, 0);  add_7 = slice_scatter_11 = None
        slice_scatter_12: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(arg40_1, clone_10, 2, -255, 9223372036854775807);  arg40_1 = clone_10 = None
        slice_scatter_13: "bf16[96, 512, 513]" = torch.ops.aten.slice_scatter.default(arg41_1, slice_scatter_12, 1, 0, 255);  slice_scatter_12 = None
        select_scatter_1: "bf16[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(arg42_1, slice_scatter_13, 1, 0);  slice_scatter_13 = None
        slice_7: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_8: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 256)
        clone_11: "bf16[96, 3, 256, 256]" = torch.ops.aten.clone.default(slice_8, memory_format = torch.contiguous_format)
        copy_4: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_8, arg43_1);  slice_8 = arg43_1 = None
        slice_scatter_14: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_4, 3, 0, 256);  slice_7 = copy_4 = None
        slice_scatter_15: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_14 = None
        slice_scatter_16: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(arg44_1, clone_11, 3, 257, 9223372036854775807);  clone_11 = None
        slice_scatter_17: "bf16[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(arg42_1, slice_scatter_16, 2, -257, -1);  slice_scatter_16 = None
        add_8: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_1, slice_scatter_17);  select_scatter_1 = slice_scatter_17 = None
        select_1: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, -1)
        slice_9: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807)
        clone_12: "bf16[96, 256, 257]" = torch.ops.aten.clone.default(slice_9, memory_format = torch.contiguous_format)
        copy_5: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_9, arg45_1);  slice_9 = arg45_1 = None
        slice_scatter_18: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_1, copy_5, 2, 256, 9223372036854775807);  select_1 = copy_5 = None
        select_scatter_2: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_18, 1, -1);  slice_scatter_15 = slice_scatter_18 = None
        slice_scatter_19: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(arg46_1, clone_12, 2, 0, 257);  arg46_1 = clone_12 = None
        slice_scatter_20: "bf16[96, 512, 513]" = torch.ops.aten.slice_scatter.default(arg41_1, slice_scatter_19, 1, 256, 9223372036854775807);  arg41_1 = slice_scatter_19 = None
        select_scatter_3: "bf16[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(arg42_1, slice_scatter_20, 1, -1);  slice_scatter_20 = None
        add_9: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_8, select_scatter_3);  add_8 = select_scatter_3 = None
        slice_10: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 0, -1);  select_scatter_2 = None
        slice_11: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_10, 3, 256, 9223372036854775807);  slice_10 = None
        clone_13: "bf16[96, 3, 256, 257]" = torch.ops.aten.clone.default(slice_11, memory_format = torch.contiguous_format);  slice_11 = None
        slice_scatter_21: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(arg44_1, clone_13, 3, 0, 257);  arg44_1 = clone_13 = None
        slice_scatter_22: "bf16[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(arg42_1, slice_scatter_21, 2, 0, 256);  arg42_1 = slice_scatter_21 = None
        add_10: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_9, slice_scatter_22);  add_9 = slice_scatter_22 = None
        view_46: "bf16[96, 3, 513, 512]" = torch.ops.aten.view.default(add_10, _shape_param_53);  add_10 = _shape_param_53 = None
        constant_pad_nd_2: "bf16[96, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_46, [0, 0, 0, -1]);  view_46 = None
        view_47: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_2, _shape_param_54);  constant_pad_nd_2 = _shape_param_54 = None
        permute_27: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_47, [0, 1, 2, 4, 3]);  view_47 = None
        view_48: "bf16[288, 512, 512]" = torch.ops.aten.view.default(permute_27, _shape_param_55);  permute_27 = _shape_param_55 = None
        bmm_2: "bf16[288, 64, 512]" = torch.ops.aten.bmm.default(arg47_1, view_48);  arg47_1 = None
        bmm_3: "bf16[288, 512, 64]" = torch.ops.aten.bmm.default(view_48, arg48_1);  view_48 = arg48_1 = None
        view_49: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_2, _shape_param_56);  bmm_2 = _shape_param_56 = None
        permute_28: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 1, 4, 3, 2]);  view_49 = None
        view_50: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_3, _shape_param_57);  bmm_3 = _shape_param_57 = None
        permute_29: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 4, 2]);  permute_28 = None
        squeeze_2: "bf16[96, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_29, 4);  permute_29 = None
        squeeze_3: "bf16[96, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_50, 4);  view_50 = None
        clone_14: "bf16[96, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        view_51: "bf16[9437184]" = torch.ops.aten.view.default(clone_14, _shape_param_58);  clone_14 = _shape_param_58 = None
        view_52: "bf16[9437184]" = torch.ops.aten.view.default(squeeze_3, [-1]);  squeeze_3 = None
        return (convert_element_type_2, convert_element_type_3, convert_element_type_6, convert_element_type_7, convert_element_type_10, convert_element_type_11, sum_6, sum_7, convert_element_type_15, convert_element_type_16, convert_element_type_22, convert_element_type_23, mul_19, sum_12, sum_13, convert_element_type_27, convert_element_type_28, view_35, view_51, view_52)



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
