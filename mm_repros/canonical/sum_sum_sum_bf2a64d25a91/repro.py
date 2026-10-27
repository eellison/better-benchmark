"""
Standalone repro captured via capture_hook.
Label: XGLMForCausalLM
Pattern hash: bf2a64d25a91
Shape hash: 78da7d53
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
    def forward(self, arg0_1: "f32[32, 128, 1024]", arg1_1: "b8[32, 128, 1024]", arg2_1: "bf16[1024, 4096]", arg3_1: "bf16[4096, 4096]", arg4_1: "bf16[4096, 4096]", arg5_1: "bf16[4096, 1024]", arg6_1: "bf16[4096, 1024]", arg7_1: "f32[1024]", arg8_1: "bf16[4096, 1024]", arg9_1: "b8[32, 128, 1024]", arg10_1: "f32[32, 128, 1024]", arg11_1: "f32[32, 128, 1]", arg12_1: "f32[32, 128, 1]", arg13_1: "bf16[1024, 1024]", arg14_1: "bf16[4096, 1024]", arg15_1: "bf16[512, 128, 128]", arg16_1: "bf16[512, 64, 128]", arg17_1: "b8[512, 128, 128]", arg18_1: "bf16[512, 128, 128]", arg19_1: "f32[32, 1, 128, 128]", arg20_1: "f32[512, 128, 1]", arg21_1: "f32[512, 128, 1]", arg22_1: "bf16[512, 64, 128]", arg23_1: "bf16[512, 128, 64]", arg24_1: "bf16[1024, 1024]", arg25_1: "bf16[4096, 1024]", arg26_1: "bf16[1024, 1024]", arg27_1: "bf16[1024, 1024]", arg28_1: "f32[1024]", arg29_1: "f32[32, 128, 1]", arg30_1: "f32[32, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16)
        convert_element_type_1: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        mm: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view, arg2_1);  arg2_1 = None
        permute: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view, [1, 0])
        mm_1: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute, arg3_1);  permute = arg3_1 = None
        sum_1: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32);  view = None
        view_1: "f32[1024]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        convert_element_type_2: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        view_2: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type_3: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_4: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        convert_element_type_5: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        view_3: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        convert_element_type_6: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_2: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476)
        erf: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_2);  mul_2 = None
        add: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_3: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_4: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_6)
        mul_5: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_4, -0.5);  mul_4 = None
        exp: "f32[32, 128, 4096]" = torch.ops.aten.exp.default(mul_5);  mul_5 = None
        mul_6: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_7: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_6);  convert_element_type_6 = mul_6 = None
        add_1: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(mul_3, mul_7);  mul_3 = mul_7 = None
        mul_8: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_5, add_1);  convert_element_type_5 = add_1 = None
        convert_element_type_7: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_4: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_4);  convert_element_type_7 = _shape_param_4 = None
        mm_2: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_4, arg5_1);  arg5_1 = None
        permute_1: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_3: "bf16[4096, 1024]" = torch.ops.aten.mm.default(permute_1, arg6_1);  permute_1 = arg6_1 = None
        sum_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[4096]" = torch.ops.aten.view.default(sum_2, _shape_param_5);  sum_2 = _shape_param_5 = None
        convert_element_type_8: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_2, _shape_param_6);  mm_2 = _shape_param_6 = None
        convert_element_type_9: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_10: "f32[4096, 1024]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_11: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        mul_9: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_9, arg7_1);  arg7_1 = None
        mul_10: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_9, 1024)
        sum_3: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True)
        view_7: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg8_1, _shape_param_7);  arg8_1 = _shape_param_7 = None
        mul_11: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(arg9_1, view_7);  view_7 = None
        mul_12: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_11, 1.1111111111111112);  mul_11 = None
        add_2: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(arg10_1, mul_12);  mul_12 = None
        sub: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_2, arg11_1);  add_2 = arg11_1 = None
        mul_13: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, arg12_1);  sub = None
        mul_14: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_9, mul_13);  mul_9 = None
        sum_4: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_14, [2], True);  mul_14 = None
        mul_15: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_13, sum_4);  sum_4 = None
        sub_1: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_10, sum_3);  mul_10 = sum_3 = None
        sub_2: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_1, mul_15);  sub_1 = mul_15 = None
        div: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(arg12_1, 1024);  arg12_1 = None
        mul_16: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_17: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_13);  mul_13 = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_17, [0, 1]);  mul_17 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [0, 1]);  convert_element_type_9 = None
        add_3: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(arg0_1, mul_16);  arg0_1 = mul_16 = None
        convert_element_type_12: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        convert_element_type_13: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        mul_18: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_19: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, mul_18);  convert_element_type_12 = mul_18 = None
        view_8: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_19, _shape_param_8);  mul_19 = _shape_param_8 = None
        mm_4: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_8, arg13_1);  arg13_1 = None
        permute_2: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_5: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_2, arg14_1);  permute_2 = arg14_1 = None
        sum_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[1024]" = torch.ops.aten.view.default(sum_7, _shape_param_9);  sum_7 = _shape_param_9 = None
        convert_element_type_14: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_4, _shape_param_10);  mm_4 = _shape_param_10 = None
        convert_element_type_15: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_16: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        view_11: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_10, _shape_param_11);  view_10 = _shape_param_11 = None
        permute_3: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        clone: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_12: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_12);  clone = _shape_param_12 = None
        bmm: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(arg15_1, view_12);  arg15_1 = None
        bmm_1: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_12, arg16_1);  view_12 = arg16_1 = None
        convert_element_type_17: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(bmm_1, torch.float32);  bmm_1 = None
        convert_element_type_18: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        mul_20: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_21: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_17, mul_20);  convert_element_type_17 = mul_20 = None
        view_13: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg18_1, _shape_param_13);  arg18_1 = _shape_param_13 = None
        add_4: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_13, arg19_1);  view_13 = arg19_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum: "f32[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_4, full)
        view_14: "f32[512, 128, 128]" = torch.ops.aten.view.default(maximum, _shape_param_14);  maximum = _shape_param_14 = None
        sub_3: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_14, arg20_1);  view_14 = arg20_1 = None
        exp_1: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        div_1: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, arg21_1);  exp_1 = arg21_1 = None
        mul_22: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(mul_21, div_1);  mul_21 = None
        sum_8: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [-1], True)
        neg: "f32[512, 128, 128]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma: "f32[512, 128, 128]" = torch.ops.prims.fma.default(neg, sum_8, mul_22);  neg = sum_8 = mul_22 = None
        view_15: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(fma, _shape_param_15);  fma = _shape_param_15 = None
        div_2: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Scalar(view_15, 2)
        eq: "b8[32, 16, 128, 128]" = torch.ops.aten.eq.Tensor(add_4, full)
        where: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(eq, div_2, view_15);  eq = div_2 = view_15 = None
        lt: "b8[32, 16, 128, 128]" = torch.ops.aten.lt.Tensor(add_4, full);  add_4 = full = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(lt, full_1, where);  lt = full_1 = where = None
        convert_element_type_19: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        view_16: "bf16[512, 128, 128]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_16);  convert_element_type_19 = _shape_param_16 = None
        bmm_2: "bf16[512, 64, 128]" = torch.ops.aten.bmm.default(arg22_1, view_16);  arg22_1 = None
        bmm_3: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(view_16, arg23_1);  view_16 = arg23_1 = None
        permute_4: "bf16[512, 128, 64]" = torch.ops.aten.permute.default(bmm_2, [0, 2, 1]);  bmm_2 = None
        view_17: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm, _shape_param_17);  bmm = _shape_param_17 = None
        view_18: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(permute_4, _shape_param_18);  permute_4 = _shape_param_18 = None
        view_19: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_19);  bmm_3 = _shape_param_19 = None
        permute_5: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        clone_1: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_20: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_20);  clone_1 = _shape_param_20 = None
        permute_6: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_2: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_21: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_21);  clone_2 = _shape_param_21 = None
        permute_7: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        view_22: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(permute_7, _shape_param_22);  permute_7 = _shape_param_22 = None
        view_23: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_21, _shape_param_23);  view_21 = _shape_param_23 = None
        mm_6: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_23, arg24_1);  arg24_1 = None
        permute_8: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_23, [1, 0])
        mm_7: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_8, arg25_1);  permute_8 = None
        sum_9: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_23, [0], True, dtype = torch.float32);  view_23 = None
        view_24: "f32[1024]" = torch.ops.aten.view.default(sum_9, _shape_param_24);  sum_9 = _shape_param_24 = None
        convert_element_type_20: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_24, torch.bfloat16);  view_24 = None
        view_25: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_6, _shape_param_25);  mm_6 = _shape_param_25 = None
        convert_element_type_21: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_25, torch.float32);  view_25 = None
        convert_element_type_22: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_23: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        clone_3: "bf16[32, 128, 1024]" = torch.ops.aten.clone.default(view_22, memory_format = torch.contiguous_format);  view_22 = None
        view_26: "bf16[4096, 1024]" = torch.ops.aten.view.default(clone_3, _shape_param_26);  clone_3 = _shape_param_26 = None
        mm_8: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_26, arg26_1);  arg26_1 = None
        permute_9: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_26, [1, 0])
        mm_9: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_9, arg25_1);  permute_9 = None
        sum_10: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_26, [0], True, dtype = torch.float32);  view_26 = None
        view_27: "f32[1024]" = torch.ops.aten.view.default(sum_10, _shape_param_27);  sum_10 = _shape_param_27 = None
        convert_element_type_24: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_27, torch.bfloat16);  view_27 = None
        view_28: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_8, _shape_param_28);  mm_8 = _shape_param_28 = None
        convert_element_type_25: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        add_5: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_21, convert_element_type_25);  convert_element_type_21 = convert_element_type_25 = None
        convert_element_type_26: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_27: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_24, torch.float32);  convert_element_type_24 = None
        mul_23: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_20, 0.125);  view_20 = None
        view_29: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_23, _shape_param_29);  mul_23 = _shape_param_29 = None
        mm_10: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_29, arg27_1);  arg27_1 = None
        permute_10: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_29, [1, 0])
        mm_11: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_10, arg25_1);  permute_10 = arg25_1 = None
        sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_29, [0], True, dtype = torch.float32);  view_29 = None
        view_30: "f32[1024]" = torch.ops.aten.view.default(sum_11, _shape_param_30);  sum_11 = _shape_param_30 = None
        convert_element_type_28: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_30, torch.bfloat16);  view_30 = None
        view_31: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_10, _shape_param_31);  mm_10 = _shape_param_31 = None
        convert_element_type_29: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        add_6: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_29);  add_5 = convert_element_type_29 = None
        convert_element_type_30: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_31: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32);  convert_element_type_28 = None
        mul_24: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_6, arg28_1);  arg28_1 = None
        mul_25: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_24, 1024)
        sum_12: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_24, [2], True)
        sub_4: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(arg10_1, arg29_1);  arg10_1 = arg29_1 = None
        mul_26: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_4, arg30_1);  sub_4 = None
        mul_27: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_24, mul_26);  mul_24 = None
        sum_13: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True);  mul_27 = None
        mul_28: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_26, sum_13);  sum_13 = None
        sub_5: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_25, sum_12);  mul_25 = sum_12 = None
        sub_6: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_5, mul_28);  sub_5 = mul_28 = None
        div_3: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(arg30_1, 1024);  arg30_1 = None
        mul_29: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(div_3, sub_6);  div_3 = sub_6 = None
        mul_30: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_6, mul_26);  mul_26 = None
        sum_14: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_30, [0, 1]);  mul_30 = None
        sum_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_6, [0, 1]);  add_6 = None
        add_7: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_3, mul_29);  add_3 = mul_29 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_10, convert_element_type_11, sum_5, sum_6, convert_element_type_15, convert_element_type_16, convert_element_type_22, convert_element_type_23, convert_element_type_26, convert_element_type_27, convert_element_type_30, convert_element_type_31, sum_14, sum_15, add_7)



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
