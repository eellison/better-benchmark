"""
Standalone repro captured via capture_hook.
Label: YituTechConvBert
Pattern hash: be00da9d23d2
Shape hash: 9b1b3401
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[32, 512]", arg3_1: "bf16[32, 512, 30522]", arg4_1: "f32[16384, 1]", arg5_1: "f32[16384, 1]", arg6_1: "bf16[32, 512, 30522]", arg7_1: "bf16[30522, 768]", arg8_1: "f32[768]", arg9_1: "bf16[16384, 768]", arg10_1: "f32[32, 512, 1]", arg11_1: "f32[32, 512, 1]", arg12_1: "bf16[768, 768]", arg13_1: "bf16[16384, 768]", arg14_1: "f32[768]", arg15_1: "f32[32, 512, 768]", arg16_1: "f32[32, 512, 1]", arg17_1: "b8[32, 512, 768]", arg18_1: "bf16[768, 3072]", arg19_1: "bf16[16384, 3072]", arg20_1: "bf16[16384, 3072]", arg21_1: "bf16[3072, 768]", arg22_1: "bf16[16384, 768]", arg23_1: "f32[768]", arg24_1: "f32[32, 512, 768]", arg25_1: "f32[32, 512, 1]", arg26_1: "b8[32, 512, 768]", arg27_1: "bf16[768, 768]", arg28_1: "bf16[16384, 768]", arg29_1: "bf16[16384, 384]", arg30_1: "bf16[98304, 9, 64]", arg31_1: "f32[54]", arg32_1: "bf16[16384, 54]", arg33_1: "f32[98304, 1, 1]", arg34_1: "f32[98304, 1, 1]", arg35_1: "bf16[16384, 384]", arg36_1: "bf16[54, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[16384]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[16384, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 30522]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[16384, 30522]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[16384, 30522]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[16384, 30522]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[16384, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = full_1 = None
        mul: "f32[16384, 30522]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "bf16[16384, 30522]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[16384, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_2: "bf16[16384, 30522]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[16384, 30522]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        sub_1: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_3: "bf16[16384, 30522]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[16384, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp: "f32[16384, 30522]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_1: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[16384, 30522]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[16384, 30522]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_3: "bf16[32, 512, 30522]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        add: "bf16[32, 512, 30522]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "bf16[16384, 30522]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        constant_pad_nd: "bf16[16384, 30528]" = torch.ops.aten.constant_pad_nd.default(view_4, _shape_param_5);  _shape_param_5 = None
        constant_pad_nd_1: "bf16[30528, 768]" = torch.ops.aten.constant_pad_nd.default(arg7_1, _shape_param_6);  arg7_1 = _shape_param_6 = None
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        sum_2: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32)
        view_5: "f32[30522]" = torch.ops.aten.view.default(sum_2, _shape_param_7);  sum_2 = _shape_param_7 = None
        convert_element_type_6: "bf16[30522]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_8);  mm = _shape_param_8 = None
        convert_element_type_7: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_8: "f32[30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, arg8_1);  arg8_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_3: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        view_7: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg9_1, _shape_param_9);  arg9_1 = _shape_param_9 = None
        convert_element_type_9: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 0.5)
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 0.7071067811865476)
        erf: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, add_1);  mul_4 = None
        convert_element_type_10: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_11: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_10, torch.float32);  convert_element_type_10 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_11, arg10_1);  convert_element_type_11 = arg10_1 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, arg11_1);  sub_3 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, mul_7);  mul_2 = None
        sum_4: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True);  mul_8 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_4);  sum_4 = None
        sub_4: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_4, mul_9);  sub_4 = mul_9 = None
        div_1: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(arg11_1, 768);  arg11_1 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, mul_7);  mul_7 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_11, [0, 1]);  mul_11 = None
        sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0, 1]);  convert_element_type_7 = None
        convert_element_type_12: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        convert_element_type_13: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        mul_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, convert_element_type_9)
        mul_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_13, -0.5);  mul_13 = None
        exp_1: "f32[32, 512, 768]" = torch.ops.aten.exp.default(mul_14);  mul_14 = None
        mul_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_15);  convert_element_type_9 = mul_15 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_12, mul_16);  mul_12 = mul_16 = None
        mul_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, add_2);  convert_element_type_13 = add_2 = None
        convert_element_type_14: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_8: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_14, _shape_param_10);  convert_element_type_14 = _shape_param_10 = None
        mm_1: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_8, arg12_1);  arg12_1 = None
        permute: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_2: "bf16[768, 768]" = torch.ops.aten.mm.default(permute, arg13_1);  permute = arg13_1 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[768]" = torch.ops.aten.view.default(sum_7, _shape_param_11);  sum_7 = _shape_param_11 = None
        convert_element_type_15: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_12);  mm_1 = _shape_param_12 = None
        convert_element_type_16: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_17: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_18: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_15, torch.float32);  convert_element_type_15 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, arg14_1);  arg14_1 = None
        mul_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, 768)
        sum_8: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True)
        mul_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg15_1);  mul_18 = None
        sum_9: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True);  mul_20 = None
        mul_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg15_1, sum_9);  sum_9 = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_19, sum_8);  mul_19 = sum_8 = None
        sub_7: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_6, mul_21);  sub_6 = mul_21 = None
        mul_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg16_1, sub_7);  arg16_1 = sub_7 = None
        mul_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, arg15_1);  arg15_1 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1]);  mul_23 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_16, [0, 1]);  convert_element_type_16 = None
        convert_element_type_19: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16)
        convert_element_type_20: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        mul_24: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_25: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, mul_24);  convert_element_type_19 = mul_24 = None
        view_11: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_25, _shape_param_13);  mul_25 = _shape_param_13 = None
        mm_3: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_11, arg18_1);  arg18_1 = None
        permute_1: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_4: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_1, arg19_1);  permute_1 = arg19_1 = None
        sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_12: "f32[768]" = torch.ops.aten.view.default(sum_12, _shape_param_14);  sum_12 = _shape_param_14 = None
        convert_element_type_21: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        view_13: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_3, _shape_param_15);  mm_3 = _shape_param_15 = None
        convert_element_type_22: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_23: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_21, torch.float32);  convert_element_type_21 = None
        convert_element_type_24: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        view_14: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg20_1, _shape_param_16);  arg20_1 = _shape_param_16 = None
        convert_element_type_25: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        mul_26: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 0.7071067811865476)
        erf_1: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_3: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_27: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        mul_28: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_25, convert_element_type_25)
        mul_29: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, -0.5);  mul_28 = None
        exp_2: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_29);  mul_29 = None
        mul_30: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_31: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_25, mul_30);  convert_element_type_25 = mul_30 = None
        add_4: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_27, mul_31);  mul_27 = mul_31 = None
        mul_32: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_24, add_4);  convert_element_type_24 = add_4 = None
        convert_element_type_26: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        view_15: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_26, _shape_param_17);  convert_element_type_26 = _shape_param_17 = None
        mm_5: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_15, arg21_1);  arg21_1 = None
        permute_2: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_6: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_2, arg22_1);  permute_2 = arg22_1 = None
        sum_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True, dtype = torch.float32);  view_15 = None
        view_16: "f32[3072]" = torch.ops.aten.view.default(sum_13, _shape_param_18);  sum_13 = _shape_param_18 = None
        convert_element_type_27: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        view_17: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_5, _shape_param_19);  mm_5 = _shape_param_19 = None
        convert_element_type_28: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_22, convert_element_type_28);  mul_22 = convert_element_type_28 = None
        convert_element_type_29: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_30: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_27, torch.float32);  convert_element_type_27 = None
        mul_33: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_5, arg23_1);  arg23_1 = None
        mul_34: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, 768)
        sum_14: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_33, [2], True)
        mul_35: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg24_1);  mul_33 = None
        sum_15: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True);  mul_35 = None
        mul_36: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg24_1, sum_15);  sum_15 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_34, sum_14);  mul_34 = sum_14 = None
        sub_9: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_8, mul_36);  sub_8 = mul_36 = None
        mul_37: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg25_1, sub_9);  arg25_1 = sub_9 = None
        mul_38: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_5, arg24_1);  arg24_1 = None
        sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_38, [0, 1]);  mul_38 = None
        sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_5, [0, 1]);  add_5 = None
        convert_element_type_31: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16)
        convert_element_type_32: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        mul_39: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_40: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, mul_39);  convert_element_type_31 = mul_39 = None
        view_18: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_40, _shape_param_20);  mul_40 = _shape_param_20 = None
        mm_7: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_18, arg27_1);  arg27_1 = None
        permute_3: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_8: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_3, arg28_1);  permute_3 = arg28_1 = None
        sum_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True, dtype = torch.float32);  view_18 = None
        view_19: "f32[768]" = torch.ops.aten.view.default(sum_18, _shape_param_21);  sum_18 = _shape_param_21 = None
        convert_element_type_33: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        view_20: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_7, _shape_param_22);  mm_7 = _shape_param_22 = None
        convert_element_type_34: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        convert_element_type_35: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_33, torch.float32);  convert_element_type_33 = None
        view_21: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_20, _shape_param_23);  view_20 = _shape_param_23 = None
        slice_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_21, 2, 0, 6)
        slice_2: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_21, 2, 6, 12);  view_21 = None
        view_22: "bf16[16384, 384]" = torch.ops.aten.view.default(slice_2, _shape_param_24);  slice_2 = _shape_param_24 = None
        permute_4: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(slice_1, [0, 2, 1, 3]);  slice_1 = None
        clone: "bf16[32, 6, 512, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_23: "bf16[32, 512, 384]" = torch.ops.aten.view.default(arg29_1, _shape_param_25);  arg29_1 = _shape_param_25 = None
        view_24: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view_23, _shape_param_26);  _shape_param_26 = None
        permute_5: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_1: "bf16[16384, 384]" = torch.ops.aten.clone.default(view_22, memory_format = torch.contiguous_format);  view_22 = None
        view_25: "bf16[98304, 64, 1]" = torch.ops.aten.view.default(clone_1, _shape_param_27);  clone_1 = _shape_param_27 = None
        bmm: "bf16[98304, 9, 1]" = torch.ops.aten.bmm.default(arg30_1, view_25);  arg30_1 = None
        convert_element_type_36: "bf16[54]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.bfloat16);  arg31_1 = None
        view_26: "bf16[32, 512, 54]" = torch.ops.aten.view.default(arg32_1, _shape_param_28);  arg32_1 = _shape_param_28 = None
        add_6: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view_26, convert_element_type_36);  view_26 = convert_element_type_36 = None
        view_27: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add_6, _shape_param_29);  add_6 = _shape_param_29 = None
        convert_element_type_37: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        sub_10: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_37, arg33_1);  convert_element_type_37 = arg33_1 = None
        exp_3: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        div_2: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp_3, arg34_1);  exp_3 = arg34_1 = None
        convert_element_type_38: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        mul_41: "f32[98304, 9, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_38, div_2);  convert_element_type_38 = None
        sum_19: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_41, [1], True)
        neg: "f32[98304, 9, 1]" = torch.ops.aten.neg.default(div_2)
        fma: "f32[98304, 9, 1]" = torch.ops.prims.fma.default(neg, sum_19, mul_41);  neg = sum_19 = mul_41 = None
        convert_element_type_39: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_28: "bf16[32, 512, 54]" = torch.ops.aten.view.default(convert_element_type_39, _shape_param_30);  convert_element_type_39 = _shape_param_30 = None
        sum_20: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_28, [0, 1], True, dtype = torch.float32)
        view_29: "f32[54]" = torch.ops.aten.view.default(sum_20, _shape_param_31);  sum_20 = _shape_param_31 = None
        convert_element_type_40: "bf16[54]" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[16384, 54]" = torch.ops.aten.view.default(view_28, _shape_param_32);  view_28 = _shape_param_32 = None
        permute_6: "bf16[54, 16384]" = torch.ops.aten.permute.default(view_30, [1, 0])
        mm_9: "bf16[54, 384]" = torch.ops.aten.mm.default(permute_6, arg35_1);  permute_6 = arg35_1 = None
        mm_10: "bf16[16384, 384]" = torch.ops.aten.mm.default(view_30, arg36_1);  view_30 = arg36_1 = None
        view_31: "bf16[32, 512, 384]" = torch.ops.aten.view.default(mm_10, _shape_param_33);  mm_10 = _shape_param_33 = None
        convert_element_type_41: "f32[54, 384]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_42: "f32[54]" = torch.ops.prims.convert_element_type.default(convert_element_type_40, torch.float32);  convert_element_type_40 = None
        mul_42: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(view_31, view_23);  view_23 = None
        permute_7: "bf16[32, 384, 512]" = torch.ops.aten.permute.default(mul_42, [0, 2, 1]);  mul_42 = None
        sum_21: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_7, [0, 2], True, dtype = torch.float32)
        view_32: "f32[384, 1]" = torch.ops.aten.view.default(sum_21, _shape_param_34);  sum_21 = _shape_param_34 = None
        return (view_4, convert_element_type_8, sum_5, sum_6, convert_element_type_17, convert_element_type_18, sum_10, sum_11, convert_element_type_22, convert_element_type_23, convert_element_type_29, convert_element_type_30, mul_37, sum_16, sum_17, convert_element_type_34, convert_element_type_35, clone, permute_5, view_25, div_2, view_31, convert_element_type_41, convert_element_type_42, permute_7, view_32)



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
