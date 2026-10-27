"""
Standalone repro captured via capture_hook.
Label: BlenderbotForConditionalGeneration
Pattern hash: 143fa613455d
Shape hash: 89e5478f
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
    def forward(self, arg0_1: "f32[16, 128, 2560]", arg1_1: "b8[16, 128, 2560]", arg2_1: "bf16[2560, 10240]", arg3_1: "bf16[2048, 10240]", arg4_1: "bf16[2048, 10240]", arg5_1: "bf16[10240, 2560]", arg6_1: "bf16[2048, 2560]", arg7_1: "f32[2560]", arg8_1: "f32[16, 128, 2560]", arg9_1: "f32[16, 128, 1]", arg10_1: "b8[16, 128, 2560]", arg11_1: "bf16[2560, 2560]", arg12_1: "bf16[2048, 2560]", arg13_1: "bf16[16, 32, 128, 128]", arg14_1: "bf16[512, 80, 128]", arg15_1: "bf16[512, 80, 128]", arg16_1: "bf16[512, 128, 80]", arg17_1: "bf16[2560, 2560]", arg18_1: "bf16[2048, 2560]", arg19_1: "bf16[2560, 2560]", arg20_1: "bf16[2560, 2560]", arg21_1: "bf16[2048, 2560]", arg22_1: "f32[2560]", arg23_1: "bf16[2048, 2560]", arg24_1: "b8[16, 128, 2560]", arg25_1: "f32[16, 128, 2560]", arg26_1: "f32[16, 128, 1]", arg27_1: "f32[16, 128, 1]", arg28_1: "bf16[2560, 2560]", arg29_1: "bf16[16, 32, 128, 80]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16)
        convert_element_type_1: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[2048, 2560]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        mm: "bf16[2048, 10240]" = torch.ops.aten.mm.default(view, arg2_1);  arg2_1 = None
        permute: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view, [1, 0])
        mm_1: "bf16[2560, 10240]" = torch.ops.aten.mm.default(permute, arg3_1);  permute = arg3_1 = None
        sum_1: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32);  view = None
        view_1: "f32[2560]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        convert_element_type_2: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        view_2: "bf16[16, 128, 10240]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type_3: "f32[2560, 10240]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_4: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        convert_element_type_5: "f32[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        view_3: "bf16[16, 128, 10240]" = torch.ops.aten.view.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        convert_element_type_6: "f32[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_2: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476)
        erf: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_2);  mul_2 = None
        add: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_3: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_4: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_6)
        mul_5: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_4, -0.5);  mul_4 = None
        exp: "f32[16, 128, 10240]" = torch.ops.aten.exp.default(mul_5);  mul_5 = None
        mul_6: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_7: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_6);  convert_element_type_6 = mul_6 = None
        add_1: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(mul_3, mul_7);  mul_3 = mul_7 = None
        mul_8: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_5, add_1);  convert_element_type_5 = add_1 = None
        convert_element_type_7: "bf16[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_4: "bf16[2048, 10240]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_4);  convert_element_type_7 = _shape_param_4 = None
        mm_2: "bf16[2048, 2560]" = torch.ops.aten.mm.default(view_4, arg5_1);  arg5_1 = None
        permute_1: "bf16[10240, 2048]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_3: "bf16[10240, 2560]" = torch.ops.aten.mm.default(permute_1, arg6_1);  permute_1 = arg6_1 = None
        sum_2: "f32[1, 10240]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[10240]" = torch.ops.aten.view.default(sum_2, _shape_param_5);  sum_2 = _shape_param_5 = None
        convert_element_type_8: "bf16[10240]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(mm_2, _shape_param_6);  mm_2 = _shape_param_6 = None
        convert_element_type_9: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_10: "f32[10240, 2560]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_11: "f32[10240]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        mul_9: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_9, arg7_1);  arg7_1 = None
        mul_10: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_9, 2560)
        sum_3: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True)
        mul_11: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_9, arg8_1);  mul_9 = None
        sum_4: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_11, [2], True);  mul_11 = None
        mul_12: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(arg8_1, sum_4);  sum_4 = None
        sub: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_10, sum_3);  mul_10 = sum_3 = None
        sub_1: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub, mul_12);  sub = mul_12 = None
        mul_13: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(arg9_1, sub_1);  arg9_1 = sub_1 = None
        mul_14: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_9, arg8_1);  arg8_1 = None
        sum_5: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_14, [0, 1]);  mul_14 = None
        sum_6: "f32[2560]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [0, 1]);  convert_element_type_9 = None
        add_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(arg0_1, mul_13);  arg0_1 = mul_13 = None
        convert_element_type_12: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
        convert_element_type_13: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        mul_15: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_16: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_12, mul_15);  convert_element_type_12 = mul_15 = None
        view_7: "bf16[2048, 2560]" = torch.ops.aten.view.default(mul_16, _shape_param_7);  mul_16 = _shape_param_7 = None
        mm_4: "bf16[2048, 2560]" = torch.ops.aten.mm.default(view_7, arg11_1);  arg11_1 = None
        permute_2: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_7, [1, 0])
        mm_5: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_2, arg12_1);  permute_2 = arg12_1 = None
        sum_7: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True, dtype = torch.float32);  view_7 = None
        view_8: "f32[2560]" = torch.ops.aten.view.default(sum_7, _shape_param_8);  sum_7 = _shape_param_8 = None
        convert_element_type_14: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        view_9: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(mm_4, _shape_param_9);  mm_4 = _shape_param_9 = None
        convert_element_type_15: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_16: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        view_10: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_9, _shape_param_10);  view_9 = _shape_param_10 = None
        permute_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_11: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone, _shape_param_11);  clone = _shape_param_11 = None
        expand: "bf16[16, 32, 128, 128]" = torch.ops.aten.expand.default(arg13_1, _shape_param_12);  _shape_param_12 = None
        view_12: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_13);  expand = _shape_param_13 = None
        permute_4: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm: "bf16[512, 128, 80]" = torch.ops.aten.bmm.default(permute_4, view_11);  permute_4 = None
        bmm_1: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_11, arg14_1);  view_11 = arg14_1 = None
        view_13: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(bmm, _shape_param_14);  bmm = _shape_param_14 = None
        view_14: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_15);  bmm_1 = _shape_param_15 = None
        convert_element_type_17: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        convert_element_type_18: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        mul_17: "f32[16, 32, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_17, convert_element_type_18);  convert_element_type_17 = None
        sum_8: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [-1], True)
        neg: "f32[16, 32, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_18);  convert_element_type_18 = None
        fma: "f32[16, 32, 128, 128]" = torch.ops.prims.fma.default(neg, sum_8, mul_17);  neg = sum_8 = mul_17 = None
        convert_element_type_19: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_15: "bf16[512, 128, 128]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_16);  convert_element_type_19 = _shape_param_16 = None
        bmm_2: "bf16[512, 80, 128]" = torch.ops.aten.bmm.default(arg15_1, view_15);  arg15_1 = None
        bmm_3: "bf16[512, 128, 80]" = torch.ops.aten.bmm.default(view_15, arg16_1);  view_15 = arg16_1 = None
        view_16: "bf16[16, 32, 80, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_17);  bmm_2 = _shape_param_17 = None
        view_17: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(bmm_3, _shape_param_18);  bmm_3 = _shape_param_18 = None
        mul_18: "bf16[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(view_16, 0.334370152488211);  view_16 = None
        permute_5: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(mul_18, [0, 1, 3, 2]);  mul_18 = None
        mul_19: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(view_17, 0.334370152488211);  view_17 = None
        permute_6: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        clone_1: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_18: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(clone_1, _shape_param_19);  clone_1 = _shape_param_19 = None
        permute_7: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        view_19: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(permute_7, _shape_param_20);  permute_7 = _shape_param_20 = None
        view_20: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_18, _shape_param_21);  view_18 = _shape_param_21 = None
        mm_6: "bf16[2048, 2560]" = torch.ops.aten.mm.default(view_20, arg17_1);  arg17_1 = None
        permute_8: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_20, [1, 0])
        mm_7: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_8, arg18_1);  permute_8 = None
        sum_9: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_20, [0], True, dtype = torch.float32);  view_20 = None
        view_21: "f32[2560]" = torch.ops.aten.view.default(sum_9, _shape_param_22);  sum_9 = _shape_param_22 = None
        convert_element_type_20: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_21, torch.bfloat16);  view_21 = None
        view_22: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(mm_6, _shape_param_23);  mm_6 = _shape_param_23 = None
        convert_element_type_21: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_22, torch.float32);  view_22 = None
        convert_element_type_22: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_23: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        clone_2: "bf16[16, 128, 2560]" = torch.ops.aten.clone.default(view_19, memory_format = torch.contiguous_format);  view_19 = None
        view_23: "bf16[2048, 2560]" = torch.ops.aten.view.default(clone_2, _shape_param_24);  clone_2 = _shape_param_24 = None
        mm_8: "bf16[2048, 2560]" = torch.ops.aten.mm.default(view_23, arg19_1);  arg19_1 = None
        permute_9: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_23, [1, 0])
        mm_9: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_9, arg18_1);  permute_9 = arg18_1 = None
        sum_10: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_23, [0], True, dtype = torch.float32);  view_23 = None
        view_24: "f32[2560]" = torch.ops.aten.view.default(sum_10, _shape_param_25);  sum_10 = _shape_param_25 = None
        convert_element_type_24: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_24, torch.bfloat16);  view_24 = None
        view_25: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(mm_8, _shape_param_26);  mm_8 = _shape_param_26 = None
        convert_element_type_25: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_25, torch.float32);  view_25 = None
        add_3: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(convert_element_type_21, convert_element_type_25);  convert_element_type_21 = convert_element_type_25 = None
        convert_element_type_26: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_27: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_24, torch.float32);  convert_element_type_24 = None
        permute_10: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(mul_19, [0, 2, 1, 3]);  mul_19 = None
        clone_3: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        view_26: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(clone_3, _shape_param_27);  clone_3 = _shape_param_27 = None
        view_27: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_26, _shape_param_28);  view_26 = _shape_param_28 = None
        mm_10: "bf16[2048, 2560]" = torch.ops.aten.mm.default(view_27, arg20_1);  arg20_1 = None
        permute_11: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_27, [1, 0])
        mm_11: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_11, arg21_1);  permute_11 = arg21_1 = None
        sum_11: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_27, [0], True, dtype = torch.float32);  view_27 = None
        view_28: "f32[2560]" = torch.ops.aten.view.default(sum_11, _shape_param_29);  sum_11 = _shape_param_29 = None
        convert_element_type_28: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_28, torch.bfloat16);  view_28 = None
        view_29: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(mm_10, _shape_param_30);  mm_10 = _shape_param_30 = None
        convert_element_type_29: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        convert_element_type_30: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_31: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32);  convert_element_type_28 = None
        mul_20: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_29, arg22_1);  arg22_1 = None
        mul_21: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_20, 2560)
        sum_12: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True)
        view_30: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(arg23_1, _shape_param_31);  arg23_1 = _shape_param_31 = None
        mul_22: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(arg24_1, view_30);  view_30 = None
        mul_23: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_22, 1.1111111111111112);  mul_22 = None
        add_4: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(arg25_1, mul_23);  arg25_1 = mul_23 = None
        sub_2: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_4, arg26_1);  add_4 = arg26_1 = None
        mul_24: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_2, arg27_1);  sub_2 = None
        mul_25: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_20, mul_24);  mul_20 = None
        sum_13: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_25, [2], True);  mul_25 = None
        mul_26: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_24, sum_13);  sum_13 = None
        sub_3: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_21, sum_12);  mul_21 = sum_12 = None
        sub_4: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_3, mul_26);  sub_3 = mul_26 = None
        div: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(arg27_1, 2560);  arg27_1 = None
        mul_27: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(div, sub_4);  div = sub_4 = None
        mul_28: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_29, mul_24);  mul_24 = None
        sum_14: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_28, [0, 1]);  mul_28 = None
        sum_15: "f32[2560]" = torch.ops.aten.sum.dim_IntList(convert_element_type_29, [0, 1]);  convert_element_type_29 = None
        add_5: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_2, mul_27);  add_2 = mul_27 = None
        convert_element_type_32: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16)
        convert_element_type_33: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.bfloat16);  arg24_1 = None
        mul_29: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_30: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_32, mul_29);  convert_element_type_32 = mul_29 = None
        view_31: "bf16[2048, 2560]" = torch.ops.aten.view.default(mul_30, _shape_param_32);  mul_30 = _shape_param_32 = None
        mm_12: "bf16[2048, 2560]" = torch.ops.aten.mm.default(view_31, arg28_1);  arg28_1 = None
        permute_12: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_31, [1, 0])
        permute_13: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(arg29_1, [0, 2, 1, 3]);  arg29_1 = None
        view_32: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(permute_13, _shape_param_33);  permute_13 = _shape_param_33 = None
        view_33: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_32, _shape_param_34);  view_32 = _shape_param_34 = None
        mm_13: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_12, view_33);  permute_12 = view_33 = None
        sum_16: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_31, [0], True, dtype = torch.float32);  view_31 = None
        view_34: "f32[2560]" = torch.ops.aten.view.default(sum_16, _shape_param_35);  sum_16 = _shape_param_35 = None
        convert_element_type_34: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_34, torch.bfloat16);  view_34 = None
        view_35: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(mm_12, _shape_param_36);  mm_12 = _shape_param_36 = None
        convert_element_type_35: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_36: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_34, torch.float32);  convert_element_type_34 = None
        view_36: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_35, _shape_param_37);  view_35 = _shape_param_37 = None
        permute_14: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_10, convert_element_type_11, sum_5, sum_6, convert_element_type_15, convert_element_type_16, convert_element_type_22, convert_element_type_23, add_3, convert_element_type_26, convert_element_type_27, convert_element_type_30, convert_element_type_31, sum_14, sum_15, add_5, convert_element_type_35, convert_element_type_36, permute_14)



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
