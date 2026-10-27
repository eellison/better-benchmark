"""
Standalone repro captured via capture_hook.
Label: BlenderbotForConditionalGeneration
Pattern hash: 609efc0d16df
Shape hash: 7cb35f96
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
    def forward(self, arg0_1: "f32[16, 128, 2560]", arg1_1: "b8[16, 128, 2560]", arg2_1: "bf16[2560, 10240]", arg3_1: "bf16[2048, 10240]", arg4_1: "bf16[2048, 10240]", arg5_1: "bf16[10240, 2560]", arg6_1: "bf16[2048, 2560]", arg7_1: "f32[2560]", arg8_1: "bf16[2048, 2560]", arg9_1: "b8[16, 128, 2560]", arg10_1: "f32[16, 128, 2560]", arg11_1: "f32[16, 128, 1]", arg12_1: "f32[16, 128, 1]", arg13_1: "bf16[2560, 2560]", arg14_1: "bf16[2048, 2560]", arg15_1: "bf16[16, 32, 128, 128]", arg16_1: "bf16[512, 80, 128]", arg17_1: "bf16[512, 80, 128]", arg18_1: "bf16[512, 128, 80]", arg19_1: "bf16[2560, 2560]", arg20_1: "bf16[2048, 2560]", arg21_1: "bf16[2560, 2560]", arg22_1: "bf16[2560, 2560]", arg23_1: "f32[2560]", arg24_1: "f32[16, 128, 1]", arg25_1: "f32[16, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31):
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
        view_7: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(arg8_1, _shape_param_7);  arg8_1 = _shape_param_7 = None
        mul_11: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(arg9_1, view_7);  view_7 = None
        mul_12: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_11, 1.1111111111111112);  mul_11 = None
        add_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(arg10_1, mul_12);  mul_12 = None
        sub: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_2, arg11_1);  add_2 = arg11_1 = None
        mul_13: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, arg12_1);  sub = None
        mul_14: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_9, mul_13);  mul_9 = None
        sum_4: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_14, [2], True);  mul_14 = None
        mul_15: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_13, sum_4);  sum_4 = None
        sub_1: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_10, sum_3);  mul_10 = sum_3 = None
        sub_2: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_1, mul_15);  sub_1 = mul_15 = None
        div: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(arg12_1, 2560);  arg12_1 = None
        mul_16: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_17: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_13);  mul_13 = None
        sum_5: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_17, [0, 1]);  mul_17 = None
        sum_6: "f32[2560]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [0, 1]);  convert_element_type_9 = None
        add_3: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(arg0_1, mul_16);  arg0_1 = mul_16 = None
        convert_element_type_12: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        convert_element_type_13: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        mul_18: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_19: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_12, mul_18);  convert_element_type_12 = mul_18 = None
        view_8: "bf16[2048, 2560]" = torch.ops.aten.view.default(mul_19, _shape_param_8);  mul_19 = _shape_param_8 = None
        mm_4: "bf16[2048, 2560]" = torch.ops.aten.mm.default(view_8, arg13_1);  arg13_1 = None
        permute_2: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_5: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_2, arg14_1);  permute_2 = arg14_1 = None
        sum_7: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[2560]" = torch.ops.aten.view.default(sum_7, _shape_param_9);  sum_7 = _shape_param_9 = None
        convert_element_type_14: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(mm_4, _shape_param_10);  mm_4 = _shape_param_10 = None
        convert_element_type_15: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_16: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        view_11: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_10, _shape_param_11);  view_10 = _shape_param_11 = None
        permute_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        clone: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_12: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone, _shape_param_12);  clone = _shape_param_12 = None
        expand: "bf16[16, 32, 128, 128]" = torch.ops.aten.expand.default(arg15_1, _shape_param_13);  _shape_param_13 = None
        view_13: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_14);  expand = _shape_param_14 = None
        permute_4: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        bmm: "bf16[512, 128, 80]" = torch.ops.aten.bmm.default(permute_4, view_12);  permute_4 = None
        bmm_1: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_12, arg16_1);  view_12 = arg16_1 = None
        view_14: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(bmm, _shape_param_15);  bmm = _shape_param_15 = None
        view_15: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_16);  bmm_1 = _shape_param_16 = None
        convert_element_type_17: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None
        convert_element_type_18: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.float32);  arg15_1 = None
        mul_20: "f32[16, 32, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_17, convert_element_type_18);  convert_element_type_17 = None
        sum_8: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [-1], True)
        neg: "f32[16, 32, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_18);  convert_element_type_18 = None
        fma: "f32[16, 32, 128, 128]" = torch.ops.prims.fma.default(neg, sum_8, mul_20);  neg = sum_8 = mul_20 = None
        convert_element_type_19: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_16: "bf16[512, 128, 128]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_17);  convert_element_type_19 = _shape_param_17 = None
        bmm_2: "bf16[512, 80, 128]" = torch.ops.aten.bmm.default(arg17_1, view_16);  arg17_1 = None
        bmm_3: "bf16[512, 128, 80]" = torch.ops.aten.bmm.default(view_16, arg18_1);  view_16 = arg18_1 = None
        view_17: "bf16[16, 32, 80, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_18);  bmm_2 = _shape_param_18 = None
        view_18: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(bmm_3, _shape_param_19);  bmm_3 = _shape_param_19 = None
        mul_21: "bf16[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(view_17, 0.334370152488211);  view_17 = None
        permute_5: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(mul_21, [0, 1, 3, 2]);  mul_21 = None
        mul_22: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(view_18, 0.334370152488211);  view_18 = None
        permute_6: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_1: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_19: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(clone_1, _shape_param_20);  clone_1 = _shape_param_20 = None
        permute_7: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        view_20: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(permute_7, _shape_param_21);  permute_7 = _shape_param_21 = None
        view_21: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_19, _shape_param_22);  view_19 = _shape_param_22 = None
        mm_6: "bf16[2048, 2560]" = torch.ops.aten.mm.default(view_21, arg19_1);  arg19_1 = None
        permute_8: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_21, [1, 0])
        mm_7: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_8, arg20_1);  permute_8 = None
        sum_9: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_21, [0], True, dtype = torch.float32);  view_21 = None
        view_22: "f32[2560]" = torch.ops.aten.view.default(sum_9, _shape_param_23);  sum_9 = _shape_param_23 = None
        convert_element_type_20: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_22, torch.bfloat16);  view_22 = None
        view_23: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(mm_6, _shape_param_24);  mm_6 = _shape_param_24 = None
        convert_element_type_21: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_23, torch.float32);  view_23 = None
        convert_element_type_22: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_23: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        clone_2: "bf16[16, 128, 2560]" = torch.ops.aten.clone.default(view_20, memory_format = torch.contiguous_format);  view_20 = None
        view_24: "bf16[2048, 2560]" = torch.ops.aten.view.default(clone_2, _shape_param_25);  clone_2 = _shape_param_25 = None
        mm_8: "bf16[2048, 2560]" = torch.ops.aten.mm.default(view_24, arg21_1);  arg21_1 = None
        permute_9: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_24, [1, 0])
        mm_9: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_9, arg20_1);  permute_9 = None
        sum_10: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_24, [0], True, dtype = torch.float32);  view_24 = None
        view_25: "f32[2560]" = torch.ops.aten.view.default(sum_10, _shape_param_26);  sum_10 = _shape_param_26 = None
        convert_element_type_24: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_25, torch.bfloat16);  view_25 = None
        view_26: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(mm_8, _shape_param_27);  mm_8 = _shape_param_27 = None
        convert_element_type_25: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_26, torch.float32);  view_26 = None
        add_4: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(convert_element_type_21, convert_element_type_25);  convert_element_type_21 = convert_element_type_25 = None
        convert_element_type_26: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_27: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_24, torch.float32);  convert_element_type_24 = None
        permute_10: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(mul_22, [0, 2, 1, 3]);  mul_22 = None
        clone_3: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        view_27: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(clone_3, _shape_param_28);  clone_3 = _shape_param_28 = None
        view_28: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_27, _shape_param_29);  view_27 = _shape_param_29 = None
        mm_10: "bf16[2048, 2560]" = torch.ops.aten.mm.default(view_28, arg22_1);  arg22_1 = None
        permute_11: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_11: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_11, arg20_1);  permute_11 = arg20_1 = None
        sum_11: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True, dtype = torch.float32);  view_28 = None
        view_29: "f32[2560]" = torch.ops.aten.view.default(sum_11, _shape_param_30);  sum_11 = _shape_param_30 = None
        convert_element_type_28: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(mm_10, _shape_param_31);  mm_10 = _shape_param_31 = None
        convert_element_type_29: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        add_5: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_29);  add_4 = convert_element_type_29 = None
        convert_element_type_30: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_31: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32);  convert_element_type_28 = None
        mul_23: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_5, arg23_1);  arg23_1 = None
        mul_24: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_23, 2560)
        sum_12: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_23, [2], True)
        sub_3: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(arg10_1, arg24_1);  arg10_1 = arg24_1 = None
        mul_25: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_3, arg25_1);  sub_3 = None
        mul_26: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_23, mul_25);  mul_23 = None
        sum_13: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_26, [2], True);  mul_26 = None
        mul_27: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_25, sum_13);  sum_13 = None
        sub_4: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_24, sum_12);  mul_24 = sum_12 = None
        sub_5: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_4, mul_27);  sub_4 = mul_27 = None
        div_1: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(arg25_1, 2560);  arg25_1 = None
        mul_28: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_29: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_5, mul_25);  mul_25 = None
        sum_14: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_29, [0, 1]);  mul_29 = None
        sum_15: "f32[2560]" = torch.ops.aten.sum.dim_IntList(add_5, [0, 1]);  add_5 = None
        add_6: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_3, mul_28);  add_3 = mul_28 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_10, convert_element_type_11, sum_5, sum_6, convert_element_type_15, convert_element_type_16, convert_element_type_22, convert_element_type_23, convert_element_type_26, convert_element_type_27, convert_element_type_30, convert_element_type_31, sum_14, sum_15, add_6)



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
