"""
Standalone repro captured via capture_hook.
Label: TrOCRForCausalLM
Pattern hash: 137b81bc55a1
Shape hash: c93e4901
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
    def forward(self, arg0_1: "f32[64, 256, 1024]", arg1_1: "f32[1024]", arg2_1: "f32[64, 256, 1024]", arg3_1: "f32[64, 256, 1]", arg4_1: "b8[64, 256, 1024]", arg5_1: "bf16[1024, 4096]", arg6_1: "bf16[16384, 4096]", arg7_1: "bf16[16384, 4096]", arg8_1: "bf16[4096, 1024]", arg9_1: "bf16[16384, 1024]", arg10_1: "f32[1024]", arg11_1: "f32[64, 256, 1024]", arg12_1: "f32[64, 256, 1]", arg13_1: "b8[64, 256, 1024]", arg14_1: "bf16[1024, 1024]", arg15_1: "bf16[16384, 1024]", arg16_1: "bf16[1024, 256, 256]", arg17_1: "bf16[1024, 64, 256]", arg18_1: "bf16[1024, 256, 256]", arg19_1: "f32[64, 1, 256, 256]", arg20_1: "f32[1024, 256, 1]", arg21_1: "f32[1024, 256, 1]", arg22_1: "bf16[1024, 64, 256]", arg23_1: "bf16[1024, 256, 64]", arg24_1: "bf16[1024, 1024]", arg25_1: "bf16[16384, 1024]", arg26_1: "bf16[1024, 1024]", arg27_1: "bf16[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30):
        # No stacktrace found for following nodes
        mul: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul_1: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sum_1: "f32[64, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = None
        sum_2: "f32[64, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  arg2_1 = sum_2 = None
        sub: "f32[64, 256, 1024]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[64, 256, 1024]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(arg3_1, sub_1);  arg3_1 = sub_1 = None
        convert_element_type: "bf16[64, 256, 1024]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        convert_element_type_1: "bf16[64, 256, 1024]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        mul_5: "bf16[64, 256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_6: "bf16[64, 256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_5);  convert_element_type = mul_5 = None
        view: "bf16[16384, 1024]" = torch.ops.aten.view.default(mul_6, _shape_param_0);  mul_6 = _shape_param_0 = None
        mm: "bf16[16384, 4096]" = torch.ops.aten.mm.default(view, arg5_1);  arg5_1 = None
        permute: "bf16[1024, 16384]" = torch.ops.aten.permute.default(view, [1, 0])
        mm_1: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute, arg6_1);  permute = arg6_1 = None
        sum_3: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32);  view = None
        view_1: "f32[1024]" = torch.ops.aten.view.default(sum_3, _shape_param_1);  sum_3 = _shape_param_1 = None
        convert_element_type_2: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        view_2: "bf16[64, 256, 4096]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type_3: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_4: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        convert_element_type_5: "f32[64, 256, 4096]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        view_3: "bf16[64, 256, 4096]" = torch.ops.aten.view.default(arg7_1, _shape_param_3);  arg7_1 = _shape_param_3 = None
        convert_element_type_6: "f32[64, 256, 4096]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_7: "f32[64, 256, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476)
        erf: "f32[64, 256, 4096]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add: "f32[64, 256, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[64, 256, 4096]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_9: "f32[64, 256, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_6)
        mul_10: "f32[64, 256, 4096]" = torch.ops.aten.mul.Tensor(mul_9, -0.5);  mul_9 = None
        exp: "f32[64, 256, 4096]" = torch.ops.aten.exp.default(mul_10);  mul_10 = None
        mul_11: "f32[64, 256, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_12: "f32[64, 256, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_11);  convert_element_type_6 = mul_11 = None
        add_1: "f32[64, 256, 4096]" = torch.ops.aten.add.Tensor(mul_8, mul_12);  mul_8 = mul_12 = None
        mul_13: "f32[64, 256, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_5, add_1);  convert_element_type_5 = add_1 = None
        convert_element_type_7: "bf16[64, 256, 4096]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        view_4: "bf16[16384, 4096]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_4);  convert_element_type_7 = _shape_param_4 = None
        mm_2: "bf16[16384, 1024]" = torch.ops.aten.mm.default(view_4, arg8_1);  arg8_1 = None
        permute_1: "bf16[4096, 16384]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_3: "bf16[4096, 1024]" = torch.ops.aten.mm.default(permute_1, arg9_1);  permute_1 = arg9_1 = None
        sum_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[4096]" = torch.ops.aten.view.default(sum_4, _shape_param_5);  sum_4 = _shape_param_5 = None
        convert_element_type_8: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[64, 256, 1024]" = torch.ops.aten.view.default(mm_2, _shape_param_6);  mm_2 = _shape_param_6 = None
        convert_element_type_9: "f32[64, 256, 1024]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        add_2: "f32[64, 256, 1024]" = torch.ops.aten.add.Tensor(mul_4, convert_element_type_9);  mul_4 = convert_element_type_9 = None
        convert_element_type_10: "f32[4096, 1024]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_11: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        mul_14: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(add_2, arg10_1);  arg10_1 = None
        mul_15: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_14, 1024)
        sum_5: "f32[64, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_14, [2], True)
        mul_16: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_14, arg11_1);  mul_14 = None
        sum_6: "f32[64, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_16, [2], True);  mul_16 = None
        mul_17: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(arg11_1, sum_6);  sum_6 = None
        sub_2: "f32[64, 256, 1024]" = torch.ops.aten.sub.Tensor(mul_15, sum_5);  mul_15 = sum_5 = None
        sub_3: "f32[64, 256, 1024]" = torch.ops.aten.sub.Tensor(sub_2, mul_17);  sub_2 = mul_17 = None
        mul_18: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(arg12_1, sub_3);  arg12_1 = sub_3 = None
        mul_19: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(add_2, arg11_1);  arg11_1 = None
        sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_19, [0, 1]);  mul_19 = None
        sum_8: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_2, [0, 1]);  add_2 = None
        convert_element_type_12: "bf16[64, 256, 1024]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16)
        convert_element_type_13: "bf16[64, 256, 1024]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        mul_20: "bf16[64, 256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_21: "bf16[64, 256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, mul_20);  convert_element_type_12 = mul_20 = None
        view_7: "bf16[16384, 1024]" = torch.ops.aten.view.default(mul_21, _shape_param_7);  mul_21 = _shape_param_7 = None
        mm_4: "bf16[16384, 1024]" = torch.ops.aten.mm.default(view_7, arg14_1);  arg14_1 = None
        permute_2: "bf16[1024, 16384]" = torch.ops.aten.permute.default(view_7, [1, 0])
        mm_5: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_2, arg15_1);  permute_2 = arg15_1 = None
        sum_9: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True, dtype = torch.float32);  view_7 = None
        view_8: "f32[1024]" = torch.ops.aten.view.default(sum_9, _shape_param_8);  sum_9 = _shape_param_8 = None
        convert_element_type_14: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        view_9: "bf16[64, 256, 1024]" = torch.ops.aten.view.default(mm_4, _shape_param_9);  mm_4 = _shape_param_9 = None
        convert_element_type_15: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_16: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        view_10: "bf16[64, 256, 16, 64]" = torch.ops.aten.view.default(view_9, _shape_param_10);  view_9 = _shape_param_10 = None
        permute_3: "bf16[64, 16, 256, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone: "bf16[64, 16, 256, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_11: "bf16[1024, 256, 64]" = torch.ops.aten.view.default(clone, _shape_param_11);  clone = _shape_param_11 = None
        bmm: "bf16[1024, 256, 64]" = torch.ops.aten.bmm.default(arg16_1, view_11);  arg16_1 = None
        bmm_1: "bf16[1024, 256, 256]" = torch.ops.aten.bmm.default(view_11, arg17_1);  view_11 = arg17_1 = None
        convert_element_type_17: "f32[1024, 256, 256]" = torch.ops.prims.convert_element_type.default(bmm_1, torch.float32);  bmm_1 = None
        view_12: "bf16[64, 16, 256, 256]" = torch.ops.aten.view.default(arg18_1, _shape_param_12);  arg18_1 = _shape_param_12 = None
        add_3: "f32[64, 16, 256, 256]" = torch.ops.aten.add.Tensor(view_12, arg19_1);  view_12 = arg19_1 = None
        view_13: "f32[1024, 256, 256]" = torch.ops.aten.view.default(add_3, _shape_param_13);  add_3 = _shape_param_13 = None
        sub_4: "f32[1024, 256, 256]" = torch.ops.aten.sub.Tensor(view_13, arg20_1);  view_13 = arg20_1 = None
        exp_1: "f32[1024, 256, 256]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div: "f32[1024, 256, 256]" = torch.ops.aten.div.Tensor(exp_1, arg21_1);  exp_1 = arg21_1 = None
        mul_22: "f32[1024, 256, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_17, div);  convert_element_type_17 = None
        sum_10: "f32[1024, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [-1], True)
        neg: "f32[1024, 256, 256]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[1024, 256, 256]" = torch.ops.prims.fma.default(neg, sum_10, mul_22);  neg = sum_10 = mul_22 = None
        view_14: "f32[64, 16, 256, 256]" = torch.ops.aten.view.default(fma, _shape_param_14);  fma = _shape_param_14 = None
        convert_element_type_18: "bf16[64, 16, 256, 256]" = torch.ops.prims.convert_element_type.default(view_14, torch.bfloat16);  view_14 = None
        view_15: "bf16[1024, 256, 256]" = torch.ops.aten.view.default(convert_element_type_18, _shape_param_15);  convert_element_type_18 = _shape_param_15 = None
        bmm_2: "bf16[1024, 64, 256]" = torch.ops.aten.bmm.default(arg22_1, view_15);  arg22_1 = None
        bmm_3: "bf16[1024, 256, 64]" = torch.ops.aten.bmm.default(view_15, arg23_1);  view_15 = arg23_1 = None
        permute_4: "bf16[1024, 256, 64]" = torch.ops.aten.permute.default(bmm_2, [0, 2, 1]);  bmm_2 = None
        view_16: "bf16[64, 16, 256, 64]" = torch.ops.aten.view.default(bmm, _shape_param_16);  bmm = _shape_param_16 = None
        view_17: "bf16[64, 16, 256, 64]" = torch.ops.aten.view.default(permute_4, _shape_param_17);  permute_4 = _shape_param_17 = None
        view_18: "bf16[64, 16, 256, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_18);  bmm_3 = _shape_param_18 = None
        permute_5: "bf16[64, 256, 16, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_1: "bf16[64, 256, 16, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_19: "bf16[64, 256, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_19);  clone_1 = _shape_param_19 = None
        permute_6: "bf16[64, 256, 16, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_2: "bf16[64, 256, 16, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_20: "bf16[64, 256, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_20);  clone_2 = _shape_param_20 = None
        permute_7: "bf16[64, 256, 16, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        view_21: "bf16[64, 256, 1024]" = torch.ops.aten.view.default(permute_7, _shape_param_21);  permute_7 = _shape_param_21 = None
        view_22: "bf16[16384, 1024]" = torch.ops.aten.view.default(view_20, _shape_param_22);  view_20 = _shape_param_22 = None
        mm_6: "bf16[16384, 1024]" = torch.ops.aten.mm.default(view_22, arg24_1);  arg24_1 = None
        permute_8: "bf16[1024, 16384]" = torch.ops.aten.permute.default(view_22, [1, 0])
        mm_7: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_8, arg25_1);  permute_8 = None
        sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True, dtype = torch.float32);  view_22 = None
        view_23: "f32[1024]" = torch.ops.aten.view.default(sum_11, _shape_param_23);  sum_11 = _shape_param_23 = None
        convert_element_type_19: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_23, torch.bfloat16);  view_23 = None
        view_24: "bf16[64, 256, 1024]" = torch.ops.aten.view.default(mm_6, _shape_param_24);  mm_6 = _shape_param_24 = None
        convert_element_type_20: "f32[64, 256, 1024]" = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None
        add_4: "f32[64, 256, 1024]" = torch.ops.aten.add.Tensor(mul_18, convert_element_type_20);  mul_18 = convert_element_type_20 = None
        convert_element_type_21: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_22: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_19, torch.float32);  convert_element_type_19 = None
        clone_3: "bf16[64, 256, 1024]" = torch.ops.aten.clone.default(view_21, memory_format = torch.contiguous_format);  view_21 = None
        view_25: "bf16[16384, 1024]" = torch.ops.aten.view.default(clone_3, _shape_param_25);  clone_3 = _shape_param_25 = None
        mm_8: "bf16[16384, 1024]" = torch.ops.aten.mm.default(view_25, arg26_1);  arg26_1 = None
        permute_9: "bf16[1024, 16384]" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_9: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_9, arg25_1);  permute_9 = None
        sum_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True, dtype = torch.float32);  view_25 = None
        view_26: "f32[1024]" = torch.ops.aten.view.default(sum_12, _shape_param_26);  sum_12 = _shape_param_26 = None
        convert_element_type_23: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_26, torch.bfloat16);  view_26 = None
        view_27: "bf16[64, 256, 1024]" = torch.ops.aten.view.default(mm_8, _shape_param_27);  mm_8 = _shape_param_27 = None
        convert_element_type_24: "f32[64, 256, 1024]" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        add_5: "f32[64, 256, 1024]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_24);  add_4 = convert_element_type_24 = None
        convert_element_type_25: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_26: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_23, torch.float32);  convert_element_type_23 = None
        mul_23: "bf16[64, 256, 1024]" = torch.ops.aten.mul.Tensor(view_19, 0.125);  view_19 = None
        view_28: "bf16[16384, 1024]" = torch.ops.aten.view.default(mul_23, _shape_param_28);  mul_23 = _shape_param_28 = None
        mm_10: "bf16[16384, 1024]" = torch.ops.aten.mm.default(view_28, arg27_1);  arg27_1 = None
        permute_10: "bf16[1024, 16384]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_11: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_10, arg25_1);  permute_10 = arg25_1 = None
        sum_13: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True, dtype = torch.float32);  view_28 = None
        view_29: "f32[1024]" = torch.ops.aten.view.default(sum_13, _shape_param_29);  sum_13 = _shape_param_29 = None
        convert_element_type_27: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[64, 256, 1024]" = torch.ops.aten.view.default(mm_10, _shape_param_30);  mm_10 = _shape_param_30 = None
        convert_element_type_28: "f32[64, 256, 1024]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        add_6: "f32[64, 256, 1024]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_28);  add_5 = convert_element_type_28 = None
        convert_element_type_29: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_30: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_27, torch.float32);  convert_element_type_27 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_10, convert_element_type_11, sum_7, sum_8, convert_element_type_15, convert_element_type_16, convert_element_type_21, convert_element_type_22, convert_element_type_25, convert_element_type_26, add_6, convert_element_type_29, convert_element_type_30)



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
